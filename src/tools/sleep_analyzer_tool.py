import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import json
import logging
from typing import Dict, List, Tuple, Optional
import warnings
from langchain.tools import tool, ToolRuntime
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_stage_label(stage_value):
    """
    根据阶段值返回对应的标签
    """
    stage_labels = {
        1: "深睡",
        2: "浅睡", 
        3: "眼动",
        4: "清醒"
    }
    return stage_labels.get(stage_value, "未知")


def convert_numpy_types(obj):
    """
    递归转换numpy/pandas类型为原生Python类型
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.api.types.is_datetime64_any_dtype(type(obj)) or isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
    else:
        return obj


def calculate_bedtime_wakeup_times(night_data, target_date, prev_date):
    """
    根据您提供的逻辑计算就寝时间和起床时间
    """
    print(f"开始计算就寝和起床时间，夜间数据量: {len(night_data)}")
    
    # 按upload_time排序
    sorted_data = night_data.sort_values('upload_time').reset_index(drop=True)
    
    # 标记离床/有效数据状态
    sorted_data['is_off_bed'] = sorted_data['heart_rate'] == 0
    
    print(f"夜间数据时间范围: {sorted_data['upload_time'].min()} 到 {sorted_data['upload_time'].max()}")
    print(f"离床数据点数量: {(sorted_data['is_off_bed'] == True).sum()}, 在床数据点数量: {(sorted_data['is_off_bed'] == False).sum()}")
    
    # 识别「连续有效数据段」（≥3条有效数据）
    valid_segments = []
    current_segment_start = None
    valid_count = 0
    
    for i, row in sorted_data.iterrows():
        if not row['is_off_bed']:  # 有效数据（在床）
            if current_segment_start is None:
                current_segment_start = row['upload_time']
            valid_count += 1
        else:  # 离床，结束当前段
            if current_segment_start is not None and valid_count >= 5:
                # 结束当前有效数据段
                valid_segments.append({
                    'start_time': current_segment_start,
                    'end_time': sorted_data.iloc[i-1]['upload_time'],
                    'count': valid_count,
                    'duration': sorted_data.iloc[i-1]['upload_time'] - current_segment_start
                })
            current_segment_start = None
            valid_count = 0
    
    # 处理最后一个可能的有效段
    if current_segment_start is not None and valid_count >= 5:
        valid_segments.append({
            'start_time': current_segment_start,
            'end_time': sorted_data.iloc[-1]['upload_time'],
            'count': valid_count,
            'duration': sorted_data.iloc[-1]['upload_time'] - current_segment_start
        })
    
    print(f"找到 {len(valid_segments)} 个有效数据段")
    for i, seg in enumerate(valid_segments):
        print(f"  段 {i+1}: {seg['start_time']} - {seg['end_time']}, 数据条数: {seg['count']}, 时长: {seg['duration']}")
    
    # 初步确定上下床时间
    bedtime = None
    wakeup_time = None
    
    if valid_segments:
        # 就寝时间：首个有效数据段的start_time
        bedtime = valid_segments[0]['start_time']
        print(f"初步就寝时间: {bedtime}")
        
        # 起床时间：末次有效数据段的end_time
        wakeup_time = valid_segments[-1]['end_time']
        print(f"初步起床时间: {wakeup_time}")
    
    # 如果没有找到有效段，使用数据边界
    if bedtime is None or wakeup_time is None:
        print("未找到有效数据段，使用数据边界")
        bedtime = sorted_data['upload_time'].min()
        wakeup_time = sorted_data['upload_time'].max()
        print(f"边界就寝时间: {bedtime}, 边界起床时间: {wakeup_time}")
        return bedtime, wakeup_time

    # 修正规则1：时间顺序校验
    if bedtime >= wakeup_time:
        print(f"时间顺序错误: 就寝时间({bedtime}) >= 起床时间({wakeup_time})")
        # 找最长的有效数据段
        if valid_segments:
            longest_segment = max(valid_segments, key=lambda x: x['duration'])
            bedtime = longest_segment['start_time']
            wakeup_time = longest_segment['end_time']
            print(f"使用最长有效段修正: {bedtime} - {wakeup_time}")
        else:
            # 兜底方案
            bedtime = sorted_data['upload_time'].min()
            wakeup_time = sorted_data['upload_time'].max()
            print(f"兜底方案: {bedtime} - {wakeup_time}")
        return bedtime, wakeup_time

    # 修正规则2：起床时间合理性校验（6:00-10:00）
    print(f"验证起床时间合理性: {wakeup_time.time()} 是否在 06:00-10:00 范围内")

    # 如果初步起床时间不在合理范围内，查找目标日期6:00-10:00的合理起床时间
    if wakeup_time.date() == target_date.date() and \
       (wakeup_time.time() < time(6, 0) or wakeup_time.time() > time(10, 0)):
        print(f"初步起床时间不在合理范围，重新查找目标日期内的合理起床时间")
        # 从后往前遍历有效段，找到在目标日期且在合理起床时间范围内的段
        for seg in reversed(valid_segments):
            if seg['end_time'].date() == target_date.date() and \
               time(6, 0) <= seg['end_time'].time() <= time(10, 0):
                wakeup_time = seg['end_time']
                print(f"修正起床时间: {wakeup_time}")
                break

    # 修正规则3：前后15分钟无数据校验（补充验证）
    # 验证就寝时间前15分钟是否无有效数据（确认是"首次上床"）
    bedtime_check_range = bedtime - timedelta(minutes=15)
    data_before_bedtime = sorted_data[sorted_data['upload_time'] < bedtime]
    data_before_check = data_before_bedtime[data_before_bedtime['upload_time'] >= bedtime_check_range]
    
    print(f"就寝时间前15分钟数据条数: {len(data_before_check)}, 有效数据条数: {(data_before_check['is_off_bed'] == False).sum()}")

    # 验证起床时间后15分钟是否无有效数据（确认是"最终起床"）
    wakeup_check_range = wakeup_time + timedelta(minutes=15)
    data_after_wakeup = sorted_data[sorted_data['upload_time'] > wakeup_time]
    data_after_check = data_after_wakeup[data_after_wakeup['upload_time'] <= wakeup_check_range]
    
    print(f"起床时间后15分钟数据条数: {len(data_after_check)}, 有效数据条数: {(data_after_check['is_off_bed'] == False).sum()}")

    print(f"最终确定就寝时间: {bedtime}, 起床时间: {wakeup_time}")

    return bedtime, wakeup_time


def find_valid_sleep_segments(data):
    """
    查找有效的睡眠段
    """
    sorted_data = data.sort_values('upload_time').reset_index(drop=True)
    
    segments = []
    current_segment_start = None
    valid_count = 0
    
    for i, row in sorted_data.iterrows():
        if row['heart_rate'] != 0 and not pd.isna(row['heart_rate']):
            if current_segment_start is None:
                current_segment_start = row['upload_time']
            valid_count += 1
        else:  # 无效数据（离床）
            if current_segment_start is not None and valid_count >= 3:
                segments.append({
                    'start_time': current_segment_start,
                    'end_time': sorted_data.iloc[i-1]['upload_time'],
                    'count': valid_count,
                    'duration': sorted_data.iloc[i-1]['upload_time'] - current_segment_start
                })
            current_segment_start = None
            valid_count = 0
    
    # 处理最后可能的有效数据段
    if current_segment_start is not None and valid_count >= 3:
        segments.append({
            'start_time': current_segment_start,
            'end_time': sorted_data.iloc[-1]['upload_time'],
            'count': valid_count,
            'duration': sorted_data.iloc[-1]['upload_time'] - current_segment_start
        })
    
    return segments


def analyze_single_day_sleep_data(date_str: str, table_name: str = "vital_signs"):
    """
    分析单日睡眠数据
    时间范围：前一天晚上8点到当天早上10点
    
    Args:
        date_str: 日期字符串，格式如 '2024-12-20'
        table_name: 数据库表名，默认为 "vital_signs"
    
    Returns:
        JSON格式的睡眠分析结果
    """
    try:
        
        # 使用新的数据库管理器
        from src.db.database import get_db_manager
        db_manager = get_db_manager()
        
        # 使用新的时间范围：前一天晚上8点到当天早上10点
        df = db_manager.get_sleep_data_for_date_range_and_time(
            table_name,
            date_str,
            start_hour=20,  # 晚上8点
            end_hour=10     # 早上10点
        )
        
        logger.info(f"查询到 {len(df)} 条数据")
        
        if df.empty:
            logger.warning(f"数据库中没有找到 {date_str} 期间的睡眠数据")
            # 返回格式一致但数据为0的结果
            from src.utils.response_handler import SleepAnalysisResponse
            response = SleepAnalysisResponse(
                success=True,
                date=date_str,
                bedtime=f"{date_str} 00:00:00",
                wakeup_time=f"{date_str} 00:00:00",
                time_in_bed_minutes=0,
                sleep_duration_minutes=0,
                sleep_score=0,
                bed_exit_count=0,
                sleep_prep_time_minutes=0,
                sleep_phases={
                    "deep_sleep_minutes": 0,
                    "light_sleep_minutes": 0,
                    "rem_sleep_minutes": 0,
                    "awake_minutes": 0,
                    "deep_sleep_percentage": 0,
                    "light_sleep_percentage": 0,
                    "rem_sleep_percentage": 0,
                    "awake_percentage": 0
                },
                sleep_stage_segments=[],
                average_metrics={
                    "avg_heart_rate": 0,
                    "avg_respiratory_rate": 0,
                    "avg_body_moves_ratio": 0,
                    "avg_heartbeat_interval": 0,
                    "avg_rms_heartbeat_interval": 0
                },
                summary="暂无数据",
                message=f"在{date_str}期间没有找到睡眠数据"
            )
            return response.to_json()
        
        # 转换时间列为datetime格式
        df['upload_time'] = pd.to_datetime(df['upload_time'])
        
        # 分析睡眠数据
        result = analyze_sleep_metrics(df, date_str)
        
        # 将numpy/pandas类型转换为原生Python类型以支持JSON序列化
        result = convert_numpy_types(result)
        
        logger.info(f"睡眠分析完成，结果: {result.get('bedtime', 'N/A')} 到 {result.get('wakeup_time', 'N/A')}")
        print(f"睡眠分析完成，结果: {result.get('bedtime', 'N/A')} 到 {result.get('wakeup_time', 'N/A')}")
        
        # 添加最终结果的详细输出
        if 'bedtime' in result and 'wakeup_time' in result:
            print(f"最终就寝时间: {result['bedtime']}")
            print(f"最终起床时间: {result['wakeup_time']}")
            print(f"卧床时间: {result.get('time_in_bed_minutes', 'N/A')} 分钟")
            print(f"睡眠时长: {result.get('sleep_duration_minutes', 'N/A')} 分钟")
            
            # 计算时间差
            try:
                bedtime_dt = datetime.strptime(result['bedtime'], '%Y-%m-%d %H:%M:%S')
                wakeup_dt = datetime.strptime(result['wakeup_time'], '%Y-%m-%d %H:%M:%S')
                time_diff = wakeup_dt - bedtime_dt
                print(f"时间差: {time_diff}")
            except Exception as e:
                print(f"计算时间差时出错: {e}")
        
        # 使用SleepAnalysisResponse类包装结果
        from src.utils.response_handler import SleepAnalysisResponse
        response = SleepAnalysisResponse(
            success=True,
            date=result.get('date', date_str),
            bedtime=result.get('bedtime', f"{date_str} 00:00:00"),
            wakeup_time=result.get('wakeup_time', f"{date_str} 00:00:00"),
            time_in_bed_minutes=result.get('time_in_bed_minutes', 0),
            sleep_duration_minutes=result.get('sleep_duration_minutes', 0),
            sleep_score=result.get('sleep_score', 0),
            bed_exit_count=result.get('bed_exit_count', 0),
            sleep_prep_time_minutes=result.get('sleep_prep_time_minutes', 0),
            sleep_phases=result.get('sleep_phases', {
                "deep_sleep_minutes": 0,
                "light_sleep_minutes": 0,
                "rem_sleep_minutes": 0,
                "awake_minutes": 0,
                "deep_sleep_percentage": 0,
                "light_sleep_percentage": 0,
                "rem_sleep_percentage": 0,
                "awake_percentage": 0
            }),
            sleep_stage_segments=result.get('sleep_stage_segments', []),
            average_metrics=result.get('average_metrics', {
                "avg_heart_rate": 0,
                "avg_respiratory_rate": 0,
                "avg_body_moves_ratio": 0,
                "avg_heartbeat_interval": 0,
                "avg_rms_heartbeat_interval": 0
            }),
            summary=result.get('summary', '分析完成')
        )
        return response.to_json()
        
    except Exception as e:
        import traceback
        error_msg = f"单日睡眠分析失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        
        from src.utils.response_handler import ApiResponse
        # 返回错误格式但保持一致的结构
        response = ApiResponse.error(
            error=str(e),
            message="睡眠分析失败",
            data={
                "date": date_str,
                "bedtime": f"{date_str} 00:00:00",
                "wakeup_time": f"{date_str} 00:00:00",
                "time_in_bed_minutes": 0,
                "sleep_duration_minutes": 0,
                "sleep_score": 0,
                "bed_exit_count": 0,
                "sleep_prep_time_minutes": 0,
                "sleep_phases": {
                    "deep_sleep_minutes": 0,
                    "light_sleep_minutes": 0,
                    "rem_sleep_minutes": 0,
                    "awake_minutes": 0,
                    "deep_sleep_percentage": 0,
                    "light_sleep_percentage": 0,
                    "rem_sleep_percentage": 0,
                    "awake_percentage": 0
                },
                "sleep_stage_segments": [],
                "average_metrics": {
                    "avg_heart_rate": 0,
                    "avg_respiratory_rate": 0,
                    "avg_body_moves_ratio": 0,
                    "avg_heartbeat_interval": 0,
                    "avg_rms_heartbeat_interval": 0
                },
                "summary": "分析失败",
                "error": str(e)
            }
        )
        return response.to_json()


def analyze_sleep_metrics(df, date_str):
    """
    分析睡眠指标
    
    Args:
        df: 包含当日及前一天数据的DataFrame
        date_str: 目标日期字符串
    
    Returns:
        包含睡眠分析结果的字典
    """
    logger.info(f"开始分析 {date_str} 的睡眠指标，原始数据量: {len(df)}")
    print(f"开始分析 {date_str} 的睡眠指标，原始数据量: {len(df)}")
    
    # 转换数值列
    numeric_columns = [
        'heart_rate', 'respiratory_rate', 'avg_heartbeat_interval', 
        'rms_heartbeat_interval', 'std_heartbeat_interval', 'arrhythmia_ratio', 'body_moves_ratio'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            # logger.info(f"处理列 {col}，非数字值将被转换为NaN")
            # print(f"处理列 {col}，非数字值将被转换为NaN")
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 数据预处理逻辑
    # 1. 仅保留"数据类型 = 周期数据"的记录（假设所有记录都是周期数据）
    # 2. 筛选有效数据：heart_rate∈[40,100]、respiratory_rate∈[12,20]
    print(f"原始数据量: {len(df)}")
    
    # 移除无效数据
    df = df.dropna(subset=['heart_rate', 'respiratory_rate']).sort_values('upload_time')
    
    if df.empty:
        return {
            "error": "没有有效的生理指标数据",
            "date": date_str
        }
    
    target_date = datetime.strptime(date_str, '%Y-%m-%d')
    
    # 找到目标日期的数据
    target_day_data = df[df['upload_time'].dt.date == target_date.date()].copy()
    
    # 找到前一天的数据
    prev_date = target_date - timedelta(days=1)
    prev_day_data = df[df['upload_time'].dt.date == prev_date.date()].copy()
    
    # 合并两天的数据进行分析
    combined_data = pd.concat([prev_day_data, target_day_data]).sort_values('upload_time')
    
    if combined_data.empty:
        return {
            "error": "目标日期及前一天没有数据",
            "date": date_str
        }
    
    print(f"合并后数据量: {len(combined_data)}")
    
    # 无效时段标记：连续≥15 分钟 "heart_rate=0" 的记录标记为 "离床 / 未监测时段"
    combined_data = combined_data.copy()
    # 标记离床时段 - 优先使用is_person字段，如果没有则使用heart_rate
    if 'is_person' in combined_data.columns:
        combined_data['is_off_bed'] = (combined_data['is_person'] == 0)
    else:
        combined_data['is_off_bed'] = (combined_data['heart_rate'] == 0) | (combined_data['heart_rate'].isna())
    
    # 定义夜间时间段（晚上20:00到次日上午10:00），用于查找睡眠周期
    night_start = pd.Timestamp.combine(prev_date.date(), pd.Timestamp('20:00').time())
    night_end = pd.Timestamp.combine(target_date.date(), pd.Timestamp('10:00').time())
    
    # 筛选出夜间数据
    night_data = combined_data[(combined_data['upload_time'] >= night_start) & 
                              (combined_data['upload_time'] <= night_end)].copy()
    
    if night_data.empty:
        return {
            "error": "夜间时段没有有效数据",
            "date": date_str
        }
    
    print(f"夜间数据量: {len(night_data)}")
    
    # 按照您提供的逻辑计算核心时段指标
    bedtime, wakeup_time = calculate_bedtime_wakeup_times(night_data, target_date, prev_date)
    
    print(f"计算得到的就寝时间: {bedtime}")
    print(f"计算得到的起床时间: {wakeup_time}")
    
    # 确保就寝时间在起床时间之前
    if bedtime >= wakeup_time:
        # 如果时间不合理，使用备选方法
        print("时间顺序不合理，使用备选方法")
        # 在夜间数据范围内寻找最长时间的有效数据段
        valid_segments = find_valid_sleep_segments(night_data)
        if valid_segments:
            # 选择最长的有效睡眠段
            longest_segment = max(valid_segments, key=lambda x: x['duration'])
            bedtime = longest_segment['start_time']
            wakeup_time = longest_segment['end_time']
            print(f"使用最长有效睡眠段: {bedtime} - {wakeup_time}")
        else:
            # 如果没有找到有效段，使用数据的最小和最大时间
            bedtime = night_data['upload_time'].min()
            wakeup_time = night_data['upload_time'].max()
            print(f"使用数据边界时间: {bedtime} - {wakeup_time}")
    
    # 计算卧床时间
    time_in_bed = wakeup_time - bedtime
    time_in_bed_minutes = time_in_bed.total_seconds() / 60
    
    # 计算睡眠时长（基于睡眠状态）
    sleep_period_data = df[(df['upload_time'] >= bedtime) & (df['upload_time'] <= wakeup_time)].copy()
    # logger.info(f"Sleep period data: {sleep_period_data}")
    # print("Sleep period data print:", sleep_period_data)
    if not sleep_period_data.empty:
        
        # 使用心率和呼吸率来判断睡眠状态
        sleep_period_data['is_sleeping'] = (
            (sleep_period_data['heart_rate'] >= 40) & 
            (sleep_period_data['heart_rate'] <= 70) &  # 睡眠时心率较低
            (sleep_period_data['respiratory_rate'] >= 12) & 
            (sleep_period_data['respiratory_rate'] <= 18) &  # 睡眠时呼吸较慢
            (sleep_period_data['body_moves_ratio'] <= 10)  # 体动较少
        )
        
        # 计算实际睡眠时间，即处于睡眠状态的时间段总和
        sleep_duration_minutes = 0
        
        # 确保数据按时间排序
        sleep_period_data = sleep_period_data.sort_values('upload_time').reset_index(drop=True)
        
        if len(sleep_period_data) > 1:
            # 向量化计算连续睡眠片段（核心优化：替代iterrows）
            # 1. 标记睡眠状态的变化
            sleep_period_data['sleep_change'] = sleep_period_data['is_sleeping'].ne(sleep_period_data['is_sleeping'].shift())
            # 2. 为每个连续片段分配ID
            sleep_period_data['segment_id'] = sleep_period_data['sleep_change'].cumsum()
            # 3. 按片段分组计算时长
            sleep_segments = sleep_period_data[sleep_period_data['is_sleeping']].groupby('segment_id').agg({
                'upload_time': ['min', 'max']
            }).reset_index()
            
            # 4. 计算每个睡眠片段的时长并累加
            if not sleep_segments.empty:
                sleep_segments['duration'] = (sleep_segments['upload_time']['max'] - sleep_segments['upload_time']['min']).dt.total_seconds() / 60
                sleep_duration_minutes = sleep_segments['duration'].sum()
        else:
            # 如果只有一条记录，检查它是否在睡眠状态
            if len(sleep_period_data) == 1 and sleep_period_data.iloc[0]['is_sleeping']:
                # 假设单条记录代表一个标准时间间隔（如15分钟）
                sleep_duration_minutes = 15  # 默认为15分钟
    else:
        sleep_duration_minutes = 0

    # 计算离床次数（根据is_person字段判断）
    if not sleep_period_data.empty and len(sleep_period_data) > 1:
        # 使用is_person字段来判断离床次数：从1变为0再变为1算一次离床
        bed_exit_threshold_minutes = 1  # 离床时长阈值改为1分钟
        
        # 提前检查列是否存在（只查一次）
        if 'is_person' in sleep_period_data.columns:
            sleep_period_data['is_off_bed'] = (sleep_period_data['is_person'] == 0)
        else:
            sleep_period_data['is_off_bed'] = (sleep_period_data['heart_rate'] == 0)
        
        # 向量化标记离床状态变化
        sleep_period_data['off_bed_change'] = sleep_period_data['is_off_bed'].ne(sleep_period_data['is_off_bed'].shift())
        sleep_period_data['off_bed_segment_id'] = sleep_period_data['off_bed_change'].cumsum()
        
        # 按离床片段分组计算时长
        off_bed_segments = sleep_period_data[sleep_period_data['is_off_bed']].groupby('off_bed_segment_id').agg({
            'upload_time': ['min', 'max']
        }).reset_index()
        
        # 计算满足阈值的离床次数
        bed_exit_count = 0
        if not off_bed_segments.empty:
            off_bed_segments['duration'] = (off_bed_segments['upload_time']['max'] - off_bed_segments['upload_time']['min']).dt.total_seconds() / 60
            bed_exit_count = (off_bed_segments['duration'] >= bed_exit_threshold_minutes).sum()
    else:
        bed_exit_count = 0

    # 计算睡眠准备时间（从上床到真正入睡的时间）- 优化逻辑避免0分钟情况
    if not sleep_period_data.empty:
        # 稳定睡眠判定：连续 5 条有效数据满足 "heart_rate 波动≤8 次 / 分 + arrhymia_ratio≤30%"
        stable_records_needed = 5
        
        # 使用滑动窗口计算心率标准差（向量化替代嵌套循环）
        sleep_period_data['hr_std_5'] = sleep_period_data['heart_rate'].rolling(window=stable_records_needed).std()
        
        # 滑动窗口计算心律失常均值（如果列存在）
        if 'arrhythmia_ratio' in sleep_period_data.columns:
            sleep_period_data['arrhythmia_avg_5'] = sleep_period_data['arrhythmia_ratio'].rolling(window=stable_records_needed).mean()
        else:
            sleep_period_data['arrhythmia_avg_5'] = 100  # 无数据默认不满足
        
        # 找到第一个满足稳定睡眠的位置
        stable_mask = (sleep_period_data['hr_std_5'] <= 8) & (sleep_period_data['arrhythmia_avg_5'] <= 30)
        stable_indices = sleep_period_data[stable_mask].index
        
        if not stable_indices.empty:
            first_stable_idx = stable_indices[0]
            stable_sleep_start = sleep_period_data.loc[first_stable_idx, 'upload_time']
            sleep_prep_time = (stable_sleep_start - bedtime).total_seconds() / 60
            # 确保最小值
            sleep_prep_time = max(sleep_prep_time, 5)
        else:
            # 如果没有找到稳定的睡眠开始时间，使用默认值
            sleep_prep_time = 10  # 默认10分钟准备时间
    else:
        sleep_prep_time = 10  # 默认10分钟准备时间

    # 计算平均生理指标（复用已筛选数据）
    if not sleep_period_data.empty:
        avg_heart_rate = sleep_period_data['heart_rate'].mean()
        avg_respiratory_rate = sleep_period_data['respiratory_rate'].mean()
        avg_body_moves = sleep_period_data['body_moves_ratio'].mean() if 'body_moves_ratio' in sleep_period_data.columns else 0
    else:
        avg_heart_rate = 0
        avg_respiratory_rate = 0
        avg_body_moves = 0

    # 分析睡眠阶段（基于您提供的精确标准）
    # 先计算用户的清醒静息基线
    if not sleep_period_data.empty:
        # 计算清醒时的基线心率和呼吸频率（取较高值作为基线参考）
        # 假设心率>70或体动>20的情况为清醒状态
        awake_data = sleep_period_data[
            (sleep_period_data['heart_rate'] > 70) | (sleep_period_data['body_moves_ratio'] > 20)
        ]
        
        if not awake_data.empty:
            baseline_heart_rate = awake_data['heart_rate'].mean()
            baseline_respiratory_rate = awake_data['respiratory_rate'].mean()
        else:
            # 如果没有明显的清醒数据，使用总体平均值
            baseline_heart_rate = sleep_period_data['heart_rate'].mean()
            baseline_respiratory_rate = sleep_period_data['respiratory_rate'].mean()
    else:
        baseline_heart_rate = 70
        baseline_respiratory_rate = 16

    # 初始化各睡眠阶段时长
    deep_sleep_duration = 0
    rem_sleep_duration = 0
    light_sleep_duration = 0
    awake_duration = 0
    
    # 新增：存储睡眠阶段时间段数据
    sleep_stage_segments = []
    
    # 使用之前已筛选的sleep_period_data，而不是再次筛选period_data
    if not sleep_period_data.empty:
        sleep_data = sleep_period_data[
            (sleep_period_data['heart_rate'] >= 40) & 
            (sleep_period_data['heart_rate'] <= 100) &
            (sleep_period_data['respiratory_rate'] >= 8) &  # 扩展呼吸频率范围以包含深睡和REM
            (sleep_period_data['respiratory_rate'] <= 25)   # 扩展呼吸频率范围以包含REM
        ].copy()
        
        if not sleep_data.empty:
            sleep_data = sleep_data.sort_values('upload_time').reset_index(drop=True)
            
            # 向量化计算时间间隔（替代循环）
            time_diffs = sleep_data['upload_time'].diff().dt.total_seconds() / 60
            time_intervals = time_diffs.fillna(0).clip(lower=1)  # 最小间隔为1分钟
            time_intervals.iloc[0] = time_intervals.iloc[1] if len(time_intervals) > 1 else 1  # 第一个值设为第二个值，如果只有一个数据点则设为1
            
            # 计算呼吸稳定性（完全向量化）
            # 使用滚动窗口计算呼吸稳定性，避免逐行循环
            window_size = 5  # 使用5个数据点的窗口
            rolling_stats = sleep_data['respiratory_rate'].rolling(window=window_size, center=True, min_periods=1)
            rr_std = rolling_stats.std()
            rr_mean = rolling_stats.mean()
            sleep_data['respiratory_stability'] = (
                (rr_std / rr_mean * 100)
                .where(rr_mean != 0, 0)  # 避免除零错误
            ).fillna(0)
            
            # 优化睡眠阶段判定逻辑
            sleep_data = calculate_optimized_sleep_stages(sleep_data, baseline_heart_rate)
            
            # 按优先级分配阶段
            # 1. 清醒（最高优先级）
            awake_mask = sleep_data['is_awake_continuous']
            sleep_data.loc[awake_mask, ['stage_value', 'stage_label']] = [4, "清醒"]
            
            # 2. 深睡（次高）
            deep_mask = (~awake_mask) & sleep_data['is_deep_continuous']
            sleep_data.loc[deep_mask, ['stage_value', 'stage_label']] = [1, "深睡"]
            
            # 3. REM（第三优先级）
            rem_mask = (~awake_mask) & (~deep_mask) & sleep_data['is_rem_continuous']
            sleep_data.loc[rem_mask, ['stage_value', 'stage_label']] = [3, "眼动"]
            
            # 4. 浅睡（默认，其余情况）
            light_mask = (~awake_mask) & (~deep_mask) & (~rem_mask)
            sleep_data.loc[light_mask, ['stage_value', 'stage_label']] = [2, "浅睡"]
            
            # 计算各阶段时长
            awake_mask = sleep_data['stage_value'] == 4
            deep_mask = sleep_data['stage_value'] == 1
            rem_mask = sleep_data['stage_value'] == 3
            light_mask = sleep_data['stage_value'] == 2
            
            awake_duration = time_intervals[awake_mask].sum() if awake_mask.any() else 0
            deep_sleep_duration = time_intervals[deep_mask].sum() if deep_mask.any() else 0
            rem_sleep_duration = time_intervals[rem_mask].sum() if rem_mask.any() else 0
            light_sleep_duration = time_intervals[light_mask].sum() if light_mask.any() else 0
            
            # 保存阶段序列用于后续处理（向量化）
            stages_sequence = [
                {
                    'stage_value': int(row['stage_value']),
                    'stage_label': row['stage_label'],
                    'time': row['upload_time'],
                    'time_interval': time_intervals.iloc[idx] if idx < len(time_intervals) else 1
                }
                for idx, row in sleep_data.iterrows()
            ]
            
            # 应用平滑处理以减少碎片化
            smoothed_stages = smooth_sleep_stages(stages_sequence, min_duration_threshold=3)
            logger.debug(f"Smoothed sleep stages: {smoothed_stages}")
            # 计算平滑后的阶段时长和生成时间段数据
            current_stage = None
            current_stage_start_time = None
            current_stage_duration = 0
            
            for stage_info in smoothed_stages:
                stage_value = stage_info['stage_value']
                stage_label = stage_info['stage_label']
                time_interval = stage_info['time_interval']
                
                if current_stage != stage_value:
                    # 如果之前有阶段，保存之前的阶段段
                    if current_stage is not None and current_stage_start_time is not None:
                        sleep_stage_segments.append({
                            "label": get_stage_label(current_stage),
                            "value": str(int(current_stage_duration))  # 持续时间（分钟）作为value
                        })
                    
                    # 开始新的阶段段
                    current_stage = stage_value
                    current_stage_start_time = stage_info['time']
                    current_stage_duration = time_interval
                else:
                    # 延续当前阶段
                    current_stage_duration += time_interval
            
            # 处理最后一个阶段段
            if current_stage is not None and current_stage_start_time is not None:
                sleep_stage_segments.append({
                    "label": get_stage_label(current_stage),
                    "value": str(int(current_stage_duration))  # 持续时间（分钟）作为value
                })
    
    # 计算清醒时长（卧床但不在睡眠状态的时间）
    # 补充清醒时长的计算
    # 注意：这部分已经在上面的睡眠阶段分析中计算过了，不需要重复计算
    # 因为我们已经通过向量化方式计算了awake_duration
    
    # 重新计算总睡眠时长，基于有效睡眠阶段的总和
    total_sleep_time = deep_sleep_duration + light_sleep_duration + rem_sleep_duration
    sleep_duration_minutes = total_sleep_time  # 更新为基于睡眠阶段的总和

    # 确保各项睡眠阶段总和不超过总卧床时间
    total_sleep_phases = deep_sleep_duration + light_sleep_duration + rem_sleep_duration + awake_duration
    time_in_bed_minutes_total = (wakeup_time - bedtime).total_seconds() / 60
    if total_sleep_phases > time_in_bed_minutes_total:
        # 按比例调整
        if total_sleep_phases > 0:
            adjustment_factor = time_in_bed_minutes_total / total_sleep_phases
            deep_sleep_duration *= adjustment_factor
            light_sleep_duration *= adjustment_factor
            rem_sleep_duration *= adjustment_factor
            awake_duration *= adjustment_factor
            sleep_duration_minutes = total_sleep_phases * adjustment_factor

    # 计算各阶段占比
    if sleep_duration_minutes > 0:
        deep_sleep_ratio = (deep_sleep_duration / sleep_duration_minutes) * 100
        light_sleep_ratio = (light_sleep_duration / sleep_duration_minutes) * 100
        rem_sleep_ratio = (rem_sleep_duration / sleep_duration_minutes) * 100
        awake_ratio = (awake_duration / sleep_duration_minutes) * 100
    else:
        # 如果睡眠时长为0，但有卧床时间，仍分配清醒时间
        if time_in_bed_minutes > 0:
            awake_ratio = 100
            deep_sleep_ratio = light_sleep_ratio = rem_sleep_ratio = 0
        else:
            deep_sleep_ratio = light_sleep_ratio = rem_sleep_ratio = awake_ratio = 0

    # 计算睡眠评分（100分制）
    # 评分维度及权重：
    # 睡眠时长达标率（30分）：成人推荐7-9小时
    sleep_hours = sleep_duration_minutes / 60
    if 7 <= sleep_hours <= 9:
        time_score = 30
    else:
        if sleep_hours < 7:
            time_score = max(0, 30 - ((7 - sleep_hours) * 10))  # 每少30分钟扣5分
        else:
            time_score = max(0, 30 - ((sleep_hours - 9) * 5))  # 每多1小时扣3分

    # 深睡占比（25分）
    if deep_sleep_ratio >= 25:
        deep_sleep_score = 25
    elif 20 <= deep_sleep_ratio < 25:
        deep_sleep_score = 20
    elif 15 <= deep_sleep_ratio < 20:
        deep_sleep_score = 15
    elif 10 <= deep_sleep_ratio < 15:
        deep_sleep_score = 10
    elif deep_sleep_ratio >= 10:
        deep_sleep_score = 5
    else:
        deep_sleep_score = 0

    # 入睡效率（20分）
    if sleep_prep_time <= 30:
        efficiency_score = 20
    elif 31 <= sleep_prep_time <= 60:
        efficiency_score = 15
    elif 61 <= sleep_prep_time <= 90:
        efficiency_score = 10
    else:
        efficiency_score = 5

    # 夜间干扰（15分）
    if bed_exit_count == 0 and awake_duration <= 10:
        interference_score = 15
    elif bed_exit_count == 1 or (11 <= awake_duration <= 30):
        interference_score = 10
    elif bed_exit_count == 2 or (31 <= awake_duration <= 60):
        interference_score = 5
    else:
        interference_score = 0

    # 体征稳定性（10分）
    hr_variability = sleep_period_data['heart_rate'].std() if len(sleep_period_data) > 1 else 0
    if hr_variability <= 15:
        stability_score = 10
    elif 16 <= hr_variability <= 25:
        stability_score = 7
    else:
        stability_score = 3

    sleep_score = min(100, round(time_score + deep_sleep_score + efficiency_score + interference_score + stability_score))

    # 返回详细的睡眠分析结果，包含睡眠阶段时间段数据
    result = {
        "date": date_str,
        "bedtime": bedtime.strftime('%Y-%m-%d %H:%M:%S'),
        "wakeup_time": wakeup_time.strftime('%Y-%m-%d %H:%M:%S'),
        "time_in_bed_minutes": round(time_in_bed_minutes, 2),
        "sleep_duration_minutes": round(sleep_duration_minutes, 2),
        "sleep_score": round(sleep_score, 2),
        "bed_exit_count": int(bed_exit_count),
        "sleep_prep_time_minutes": round(sleep_prep_time, 2),
        "sleep_phases": {
            "deep_sleep_minutes": round(deep_sleep_duration, 2),
            "light_sleep_minutes": round(light_sleep_duration, 2),
            "rem_sleep_minutes": round(rem_sleep_duration, 2),
            "awake_minutes": round(awake_duration, 2),
            "deep_sleep_percentage": round(deep_sleep_ratio, 2),
            "light_sleep_percentage": round(light_sleep_ratio, 2),
            "rem_sleep_percentage": round(rem_sleep_ratio, 2),
            "awake_percentage": round(awake_ratio, 2)
        },
        "sleep_stage_segments": sleep_stage_segments,  # 保留：睡眠阶段时间段数据
        "average_metrics": {
            "avg_heart_rate": round(float(avg_heart_rate), 2) if pd.notna(avg_heart_rate) else 0,
            "avg_respiratory_rate": round(float(avg_respiratory_rate), 2) if pd.notna(avg_respiratory_rate) else 0,
            "avg_body_moves_ratio": round(float(avg_body_moves), 2) if pd.notna(avg_body_moves) else 0,
            "avg_heartbeat_interval": round(float(sleep_period_data['avg_heartbeat_interval'].mean()), 2) if 'avg_heartbeat_interval' in sleep_period_data.columns else 0,
            "avg_rms_heartbeat_interval": round(float(sleep_period_data['rms_heartbeat_interval'].mean()), 2) if 'rms_heartbeat_interval' in sleep_period_data.columns else 0
        },
        "summary": f"睡眠质量{'优秀' if sleep_score >= 80 else '良好' if sleep_score >= 60 else '一般' if sleep_score >= 40 else '较差'}"
    }
    
    return result


def smooth_sleep_stages(stages_sequence, min_duration_threshold=3):
    """
    平滑睡眠阶段序列，减少碎片化
    将持续时间少于阈值的小阶段合并到相邻的大阶段中
    """
    if not stages_sequence:
        return stages_sequence
    
    # 第一步：合并相邻的相同阶段
    merged_same_stages = []
    i = 0
    while i < len(stages_sequence):
        current = stages_sequence[i].copy()
        j = i + 1
        
        # 查找连续相同阶段并合并
        while j < len(stages_sequence) and stages_sequence[j]['stage_value'] == current['stage_value']:
            current['time_interval'] += stages_sequence[j]['time_interval']
            j += 1
        
        merged_same_stages.append(current)
        i = j
    
    # 第二步：移除或合并短持续时间的阶段
    if not merged_same_stages:
        return merged_same_stages
    
    result = [merged_same_stages[0]]
    
    i = 1
    while i < len(merged_same_stages):
        current = merged_same_stages[i]
        
        if current['time_interval'] < min_duration_threshold:
            # 当前阶段太短，需要合并
            # 合并到相邻的较短阶段（优先合并到前一个）
            prev = result[-1]
            prev['time_interval'] += current['time_interval']
        else:
            # 当前阶段足够长，添加到结果中
            result.append(current)
        
        i += 1
    
    return result


def calculate_optimized_sleep_stages(sleep_data, baseline_heart_rate):
    """
    优化的睡眠阶段判定函数，解决频繁切换和REM缺失问题
    """
    # 复制数据避免修改原数据
    sleep_data = sleep_data.copy()
    
    # 确保数据按时间排序
    sleep_data = sleep_data.sort_values('upload_time').reset_index(drop=True)
    
    # 将生理指标字段转换为数值类型，处理字符串格式的数据
    physio_fields = ['breath_amp_avg', 'heart_amp_avg', 'breath_freq_std', 'heart_freq_std', 'breath_amp_diff', 'heart_amp_diff']
    for field in physio_fields:
        if field in sleep_data.columns:
            sleep_data[field] = pd.to_numeric(sleep_data[field], errors='coerce')
    
    # 1. 改进清醒判定（引入体动、时间、心率综合判断）
    # 增加时间段判断：白天时段（6:00-22:00）更可能是清醒的
    sleep_data['hour'] = sleep_data['upload_time'].dt.hour
    sleep_data['minute'] = sleep_data['upload_time'].dt.minute
    sleep_data['is_daytime'] = (sleep_data['hour'] >= 6) & (sleep_data['hour'] < 22)
    
    # 更细致的时间段判断：上午时段（6:00-12:00）和下午时段（12:00-18:00）和晚上时段（18:00-22:00）
    # 上午时段（6:00-12:00）最可能是清醒的
    sleep_data['is_morning'] = (sleep_data['hour'] >= 6) & (sleep_data['hour'] < 12)
    # 下午时段（12:00-18:00）
    sleep_data['is_afternoon'] = (sleep_data['hour'] >= 12) & (sleep_data['hour'] < 18)
    # 晚上时段（18:00-22:00）
    sleep_data['is_evening'] = (sleep_data['hour'] >= 18) & (sleep_data['hour'] < 22)
    
    # 利用新增的生理指标字段增强判断
    # 呼吸幅度均值和心跳幅度均值可用于判断生理活跃程度
    if 'breath_amp_avg' in sleep_data.columns and 'heart_amp_avg' in sleep_data.columns:
        # 呼吸和心跳幅度较高通常表明更清醒的状态
        breath_amp_quantile = sleep_data['breath_amp_avg'].dropna()
        heart_amp_quantile = sleep_data['heart_amp_avg'].dropna()
        breath_thresh = breath_amp_quantile.quantile(0.3) if not breath_amp_quantile.empty else 0
        heart_thresh = heart_amp_quantile.quantile(0.3) if not heart_amp_quantile.empty else 0
        
        sleep_data['amp_indicator'] = (
            (sleep_data['breath_amp_avg'].fillna(0) > breath_thresh) |
            (sleep_data['heart_amp_avg'].fillna(0) > heart_thresh)
        )
    else:
        sleep_data['amp_indicator'] = False
    
    # 呼吸和心跳频率标准差可用于判断生理稳定性
    if 'breath_freq_std' in sleep_data.columns and 'heart_freq_std' in sleep_data.columns:
        # 频率标准差较低表明生理更稳定，倾向于深睡
        breath_freq_quantile = sleep_data['breath_freq_std'].dropna()
        heart_freq_quantile = sleep_data['heart_freq_std'].dropna()
        breath_freq_thresh = breath_freq_quantile.quantile(0.7) if not breath_freq_quantile.empty else float('inf')
        heart_freq_thresh = heart_freq_quantile.quantile(0.7) if not heart_freq_quantile.empty else float('inf')
        
        sleep_data['freq_stability'] = (
            (sleep_data['breath_freq_std'].fillna(float('inf')) < breath_freq_thresh) &
            (sleep_data['heart_freq_std'].fillna(float('inf')) < heart_freq_thresh)
        )
    else:
        sleep_data['freq_stability'] = True
    
    # 呼吸和心跳幅度差值可用于判断生理波动
    if 'breath_amp_diff' in sleep_data.columns and 'heart_amp_diff' in sleep_data.columns:
        breath_diff_quantile = sleep_data['breath_amp_diff'].dropna()
        heart_diff_quantile = sleep_data['heart_amp_diff'].dropna()
        breath_diff_thresh = breath_diff_quantile.quantile(0.3) if not breath_diff_quantile.empty else 0
        heart_diff_thresh = heart_diff_quantile.quantile(0.3) if not heart_diff_quantile.empty else 0
        
        sleep_data['amp_diff_indicator'] = (
            (sleep_data['breath_amp_diff'].fillna(0) > breath_diff_thresh) |
            (sleep_data['heart_amp_diff'].fillna(0) > heart_diff_thresh)
        )
    else:
        sleep_data['amp_diff_indicator'] = False
    
    # 白天时段的清醒判定阈值更低，特别是上午时段
    sleep_data['is_awake'] = (
        # 上午时段（6:00-12:00）：非常容易被判定为清醒
        (sleep_data['is_morning'] & (
            (sleep_data['heart_rate'] >= baseline_heart_rate * 0.85) |  # 心率稍微高一点
            (sleep_data['body_moves_ratio'] > 1.5) |  # 或有轻微体动，阈值更低
            (sleep_data['amp_indicator'])  # 或幅度指标表明清醒
        )) |
        # 下午时段（12:00-18:00）：也很容易被判定为清醒
        (sleep_data['is_afternoon'] & (
            (sleep_data['heart_rate'] >= baseline_heart_rate * 0.88) |  # 心率稍微高一点
            (sleep_data['body_moves_ratio'] > 2) |  # 或有轻微体动，阈值更低
            (sleep_data['amp_indicator'])  # 或幅度指标表明清醒
        )) |
        # 晚上时段（18:00-22:00）：相对容易被判定为清醒
        (sleep_data['is_evening'] & (
            (sleep_data['heart_rate'] >= baseline_heart_rate * 0.90) |  # 心率稍微高一点
            (sleep_data['body_moves_ratio'] > 2.5) |  # 或有轻微体动，阈值更低
            (sleep_data['amp_indicator'])  # 或幅度指标表明清醒
        )) |
        # 夜晚时段（22:00-6:00）：需要更严格的标准
        ((~sleep_data['is_daytime']) & (
            (sleep_data['heart_rate'] >= baseline_heart_rate * 1.10) |  # 心率需要明显高于基线
            ((sleep_data['heart_rate'] >= baseline_heart_rate * 0.95) &  # 心率接近基线
             (sleep_data['body_moves_ratio'] > 5)) |  # 且有体动才算清醒
            (sleep_data['amp_indicator'])  # 或幅度指标表明清醒
        ))
    )
    
    # 2. 改进深睡判定（动态呼吸阈值，不再写死13）
    # 使用呼吸稳定性作为深睡指标，而不是固定呼吸频率值
    # 计算整体呼吸频率的均值和标准差，以识别相对较低的呼吸频率段
    overall_resp_mean = sleep_data['respiratory_rate'].mean()
    overall_resp_std = sleep_data['respiratory_rate'].std()
    
    # 深睡判定条件优化，结合新增的生理指标
    sleep_data['is_deep'] = (
        (sleep_data['heart_rate'] <= baseline_heart_rate * 0.85) &  # 心率相对较低
        (sleep_data['respiratory_rate'] <= overall_resp_mean * 0.9) &  # 呼吸频率相对较低（相对于个人平均）
        (sleep_data['respiratory_stability'] < 5) &  # 呼吸稳定
        (sleep_data['body_moves_ratio'] <= 3) &  # 体动很少
        (sleep_data['freq_stability']) &  # 生理频率稳定
        (~sleep_data['is_awake'])  # 排除清醒状态
    )
    
    # 3. 基于实际深睡数据计算深睡基线（核心优化：替代固定值）
    # 先筛选出初步判定的深睡数据，计算实际深睡心率均值
    deep_sleep_data = sleep_data[sleep_data['is_deep']]
    if not deep_sleep_data.empty:
        actual_deep_hr = deep_sleep_data['heart_rate'].mean()
    else:
        # 若无深睡数据，用基线75%作为兜底
        actual_deep_hr = baseline_heart_rate * 0.75
    
    # 4. 改进REM判定（基于实际深睡基线，解决逻辑矛盾）
    REM_HR_MULTIPLIER_LOW = 1.10  # 心率比深睡基线高10%
    REM_HR_MULTIPLIER_HIGH = 1.40  # 心率比深睡基线高40%
    REM_RESP_STABILITY_LOW = 10   # 呼吸变异系数>10%
    REM_BODY_MOVES_MAX = 3        # 体动≤3
    
    sleep_data['is_rem'] = (
        (~sleep_data['is_awake']) &  # 排除清醒状态
        (~sleep_data['is_deep']) &   # 排除深睡状态
        (sleep_data['heart_rate'] >= actual_deep_hr * REM_HR_MULTIPLIER_LOW) & 
        (sleep_data['heart_rate'] <= actual_deep_hr * REM_HR_MULTIPLIER_HIGH) &
        (sleep_data['respiratory_stability'] > REM_RESP_STABILITY_LOW) &  # 呼吸不稳定
        (sleep_data['body_moves_ratio'] <= REM_BODY_MOVES_MAX) &  # 体动较少
        (sleep_data['respiratory_rate'] >= 8) &  # 呼吸频率≥8
        (sleep_data['respiratory_rate'] <= 20) &   # 呼吸频率≤20
        (sleep_data['amp_diff_indicator']) &  # 幅度差值表明生理活动
        (~sleep_data['freq_stability'])  # 频率不稳定，符合REM特征
    )
    
    # 5. 浅睡判定（优化：更精准的互斥）
    sleep_data['is_light'] = (
        (~sleep_data['is_awake']) & 
        (~sleep_data['is_deep']) & 
        (~sleep_data['is_rem']) &
        (sleep_data['body_moves_ratio'] <= 10) &  # 体动适中
        (sleep_data['heart_rate'] <= baseline_heart_rate * 0.95)  # 心率低于清醒基线
    )
    
    # ======================== 增加连续验证，避免碎片化 ========================
    # 对每个阶段的布尔值，计算连续满足的分钟数（滑动窗口）
    CONTINUOUS_MINUTES = 2  # 连续验证分钟数
    
    for stage in ['is_awake', 'is_deep', 'is_rem', 'is_light']:
        # 滑动窗口：统计当前及前CONTINUOUS_MINUTES-1行是否都满足该阶段
        sleep_data[f'{stage}_continuous'] = (
            sleep_data[stage].rolling(window=CONTINUOUS_MINUTES, min_periods=1).sum() == CONTINUOUS_MINUTES
        )
    
    # 初始化阶段值和标签
    sleep_data['stage_value'] = 2  # 默认为浅睡
    sleep_data['stage_label'] = "浅睡"
    
    return sleep_data


@tool
def analyze_sleep_by_date(date: str, runtime: ToolRuntime = None, table_name: str = "vital_signs") -> str:
    """
    根据指定日期分析睡眠数据
    
    Args:
        date: 日期字符串，格式如 '2024-12-20'
        runtime: ToolRuntime 运行时上下文
        table_name: 数据库表名，默认为 "vital_signs"
    
    Returns:
        JSON格式的睡眠分析结果，包含：
        - bedtime: 上床时间
        - wakeup_time: 起床时间
        - time_in_bed_minutes: 卧床时间（分钟）
        - sleep_duration_minutes: 睡眠时长（分钟）
        - sleep_score: 睡眠评分（0-100）
        - bed_exit_count: 离床次数
        - sleep_prep_time_minutes: 睡眠准备时间（分钟）
        - sleep_phases: 睡眠阶段分布
        - average_metrics: 平均生理指标
        - summary: 睡眠质量总结

    使用场景:
        - 分析特定日期的睡眠质量
        - 监控个人睡眠模式变化
        - 评估睡眠改善措施的效果
    """
    return analyze_single_day_sleep_data(date, table_name)


def analyze_single_day_sleep_data_with_device(date_str: str, device_sn: str, table_name: str = "vital_signs"):
    """
    使用设备序列号分析单日睡眠数据
    时间范围：前一天晚上8点到当天早上10点
    
    Args:
        date_str: 日期字符串，格式如 '2024-12-20'
        device_sn: 设备序列号
        table_name: 数据库表名，默认为 "vital_signs"
    
    Returns:
        JSON格式的睡眠分析结果
    """
    try:
        from src.utils.response_handler import SleepAnalysisResponse
        
        # 使用新的数据库管理器
        from src.db.database import get_db_manager
        db_manager = get_db_manager()
        
        # 解析日期以构建时间范围
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        prev_date = target_date - timedelta(days=1)
        
        # 构建查询条件：检查前一天晚上20点到当天早上10点的数据
        start_time = prev_date.replace(hour=20, minute=0, second=0, microsecond=0)
        end_time = target_date.replace(hour=10, minute=0, second=0, microsecond=0)
        
        # 构建SQL查询，包含设备过滤条件
        escaped_table_name = f"`{table_name.replace('`', '``')}`"
        query = f"""
        SELECT * 
        FROM {escaped_table_name} 
        WHERE upload_time BETWEEN :start_time AND :end_time
        AND device_sn = :device_sn
        ORDER BY upload_time
        """
        
        params = {
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'device_sn': device_sn
        }
        
        # 执行查询
        df = db_manager.execute_query(query, params)
        
        logger.info(f"查询到 {len(df)} 条设备 {device_sn} 的数据")
        
        if df.empty:
            logger.warning(f"数据库中没有找到 {date_str} 期间的设备 {device_sn} 的数据")
            # 返回格式一致但数据为0的结果
            response = SleepAnalysisResponse(
                success=True,
                date=date_str,
                bedtime=f"{date_str} 00:00:00",
                wakeup_time=f"{date_str} 00:00:00",
                time_in_bed_minutes=0,
                sleep_duration_minutes=0,
                sleep_score=0,
                bed_exit_count=0,
                sleep_prep_time_minutes=0,
                sleep_phases={
                    "deep_sleep_minutes": 0,
                    "light_sleep_minutes": 0,
                    "rem_sleep_minutes": 0,
                    "awake_minutes": 0,
                    "deep_sleep_percentage": 0,
                    "light_sleep_percentage": 0,
                    "rem_sleep_percentage": 0,
                    "awake_percentage": 0
                },
                sleep_stage_segments=[],
                average_metrics={
                    "avg_heart_rate": 0,
                    "avg_respiratory_rate": 0,
                    "avg_body_moves_ratio": 0,
                    "avg_heartbeat_interval": 0,
                    "avg_rms_heartbeat_interval": 0
                },
                summary="暂无数据",
                device_sn=device_sn,
                message=f"设备 {device_sn} 在 {date_str} 期间没有数据"
            )
            return response.to_json()
        
        # 转换时间列为datetime格式
        df['upload_time'] = pd.to_datetime(df['upload_time'])
        
        # 分析睡眠数据
        result = analyze_sleep_metrics(df, date_str)
        
        # 将numpy/pandas类型转换为原生Python类型以支持JSON序列化
        result = convert_numpy_types(result)
        
        # 添加设备序列号到结果中
        result['device_sn'] = device_sn
        
        logger.info(f"睡眠分析完成，结果: {result.get('bedtime', 'N/A')} 到 {result.get('wakeup_time', 'N/A')}")
        print(f"睡眠分析完成，结果: {result.get('bedtime', 'N/A')} 到 {result.get('wakeup_time', 'N/A')}")
        
        # 添加最终结果的详细输出
        if 'bedtime' in result and 'wakeup_time' in result:
            print(f"最终就寝时间: {result['bedtime']}")
            print(f"最终起床时间: {result['wakeup_time']}")
            print(f"卧床时间: {result.get('time_in_bed_minutes', 'N/A')} 分钟")
            print(f"睡眠时长: {result.get('sleep_duration_minutes', 'N/A')} 分钟")
            
            # 计算时间差
            try:
                bedtime_dt = datetime.strptime(result['bedtime'], '%Y-%m-%d %H:%M:%S')
                wakeup_dt = datetime.strptime(result['wakeup_time'], '%Y-%m-%d %H:%M:%S')
                time_diff = wakeup_dt - bedtime_dt
                print(f"时间差: {time_diff}")
            except Exception as e:
                print(f"计算时间差时出错: {e}")
        
        # 使用SleepAnalysisResponse类包装结果
        response = SleepAnalysisResponse(
            success=True,
            date=result.get('date', date_str),
            bedtime=result.get('bedtime', f"{date_str} 00:00:00"),
            wakeup_time=result.get('wakeup_time', f"{date_str} 00:00:00"),
            time_in_bed_minutes=result.get('time_in_bed_minutes', 0),
            sleep_duration_minutes=result.get('sleep_duration_minutes', 0),
            sleep_score=result.get('sleep_score', 0),
            bed_exit_count=result.get('bed_exit_count', 0),
            sleep_prep_time_minutes=result.get('sleep_prep_time_minutes', 0),
            sleep_phases=result.get('sleep_phases', {
                "deep_sleep_minutes": 0,
                "light_sleep_minutes": 0,
                "rem_sleep_minutes": 0,
                "awake_minutes": 0,
                "deep_sleep_percentage": 0,
                "light_sleep_percentage": 0,
                "rem_sleep_percentage": 0,
                "awake_percentage": 0
            }),
            sleep_stage_segments=result.get('sleep_stage_segments', []),
            average_metrics=result.get('average_metrics', {
                "avg_heart_rate": 0,
                "avg_respiratory_rate": 0,
                "avg_body_moves_ratio": 0,
                "avg_heartbeat_interval": 0,
                "avg_rms_heartbeat_interval": 0
            }),
            summary=result.get('summary', '分析完成'),
            device_sn=result.get('device_sn', device_sn)
        )
        return response.to_json()
        
    except Exception as e:
        import traceback
        error_msg = f"单日睡眠分析失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        
        from src.utils.response_handler import ApiResponse
        # 返回错误格式但保持一致的结构
        response = ApiResponse.error(
            error=str(e),
            message="睡眠分析失败",
            data={
                "date": date_str,
                "bedtime": f"{date_str} 00:00:00",
                "wakeup_time": f"{date_str} 00:00:00",
                "time_in_bed_minutes": 0,
                "sleep_duration_minutes": 0,
                "sleep_score": 0,
                "bed_exit_count": 0,
                "sleep_prep_time_minutes": 0,
                "sleep_phases": {
                    "deep_sleep_minutes": 0,
                    "light_sleep_minutes": 0,
                    "rem_sleep_minutes": 0,
                    "awake_minutes": 0,
                    "deep_sleep_percentage": 0,
                    "light_sleep_percentage": 0,
                    "rem_sleep_percentage": 0,
                    "awake_percentage": 0
                },
                "sleep_stage_segments": [],
                "average_metrics": {
                    "avg_heart_rate": 0,
                    "avg_respiratory_rate": 0,
                    "avg_body_moves_ratio": 0,
                    "avg_heartbeat_interval": 0,
                    "avg_rms_heartbeat_interval": 0
                },
                "summary": "分析失败",
                "device_sn": device_sn,
                "error": str(e)
            }
        )
        return response.to_json()


@tool
def store_calculated_sleep_data(sleep_analysis_result: str, runtime: ToolRuntime = None) -> str:
    """
    将计算得出的睡眠分析结果存储到数据库中
    
    Args:
        sleep_analysis_result: JSON格式的睡眠分析结果
        
    Returns:
        JSON格式的存储结果
    """
    try:
        from src.db.database import get_db_manager
        import json
        
        # 解析输入的JSON数据
        if isinstance(sleep_analysis_result, str):
            sleep_data = json.loads(sleep_analysis_result)
        else:
            sleep_data = sleep_analysis_result
        
        # 如果是带有data包装的对象，解包它
        if 'data' in sleep_data:
            sleep_data = sleep_data['data']
        
        # 获取数据库管理器并存储数据
        db_manager = get_db_manager()
        db_manager.store_calculated_sleep_data(sleep_data)
        
        return json.dumps({
            "success": True,
            "message": "睡眠分析数据已成功存储到数据库",
            "date": sleep_data.get('date', 'Unknown'),
            "device_sn": sleep_data.get('device_sn', 'Unknown')
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        import traceback
        error_msg = f"存储睡眠分析数据失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return json.dumps({
            "success": False,
            "error": str(e),
            "message": "存储睡眠分析数据失败"
        }, ensure_ascii=False, indent=2)


@tool
def get_stored_sleep_data(date: str, device_sn: str = None, runtime: ToolRuntime = None) -> str:
    """
    从数据库中获取已存储的睡眠分析数据
    
    Args:
        date: 日期字符串，格式如 '2024-12-20'
        device_sn: 设备序列号（可选）
        
    Returns:
        JSON格式的存储的睡眠数据
    """
    try:
        from src.db.database import get_db_manager
        import json
        
        # 获取数据库管理器并查询数据
        db_manager = get_db_manager()
        result_df = db_manager.get_calculated_sleep_data(date, device_sn)
        
        if result_df.empty:
            return json.dumps({
                "success": True,
                "message": "未找到指定日期的睡眠数据",
                "date": date,
                "device_sn": device_sn,
                "data": None
            }, ensure_ascii=False, indent=2)
        
        # 转换为字典列表格式
        data = result_df.to_dict('records')
        
        return json.dumps({
            "success": True,
            "message": f"找到 {len(data)} 条睡眠数据记录",
            "date": date,
            "device_sn": device_sn,
            "data": data[0] if len(data) == 1 else data
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        import traceback
        error_msg = f"获取睡眠数据失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return json.dumps({
            "success": False,
            "error": str(e),
            "message": "获取睡眠数据失败"
        }, ensure_ascii=False, indent=2)


# 测试函数
if __name__ == '__main__':
    # 示例：分析今天的睡眠数据
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"分析 {today} 的睡眠数据...")
    result = analyze_single_day_sleep_data(today)
    print(result)