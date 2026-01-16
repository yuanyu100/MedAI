import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from langchain.tools import tool
from langchain.tools import ToolRuntime
import os
import json
import logging

# 配置日志
import sys
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # 输出到控制台
        logging.FileHandler('sleep_analyzer.log', encoding='utf-8')  # 同时输出到文件
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # 确保日志级别为DEBUG


def analyze_single_day_sleep_data(date_str: str, table_name: str = "device_data"):
    """
    分析单日睡眠数据
    
    Args:
        date_str: 日期字符串，格式如 '2024-12-20'
        table_name: 数据库表名，默认为 "device_data"
    
    Returns:
        JSON格式的睡眠分析结果
    """
    try:
        
        # 使用新的数据库管理器
        from src.db.database import get_db_manager
        db_manager = get_db_manager()
        
        # 查询指定日期及前一天的数据（因为睡眠可能跨天）
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        prev_date = target_date - timedelta(days=1)
        
        logger.info(f"查询日期范围: {prev_date.strftime('%Y-%m-%d')} 到 {target_date.strftime('%Y-%m-%d')}")
        
        # 使用数据库管理器查询数据
        df = db_manager.get_sleep_data_for_date_range(
            table_name,
            prev_date.strftime('%Y-%m-%d'), 
            target_date.strftime('%Y-%m-%d')
        )
        
        logger.info(f"查询到 {len(df)} 条数据")
        
        if df.empty:
            logger.warning(f"数据库中没有找到 {date_str} 及前一天的数据")
            return json.dumps({
                "error": f"数据库中没有找到 {date_str} 及前一天的数据",
                "date": date_str
            }, ensure_ascii=False, indent=2)
        
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
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        import traceback
        error_msg = f"单日睡眠分析失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg}, ensure_ascii=False, indent=2)


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
            if current_segment_start is not None and valid_count >= 3:
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
    if current_segment_start is not None and valid_count >= 3:
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
        'rms_heartbeat_interval', 'std_heartbeat_interval', 'arrhymia_ratio', 'body_moves_ratio'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            # logger.info(f"处理列 {col}，非数字值将被转换为NaN")
            # print(f"处理列 {col}，非数字值将被转换为NaN")
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 数据预处理逻辑
    # 1. 仅保留“数据类型 = 周期数据”的记录（假设所有记录都是周期数据）
    # 2. 筛选有效数据：heart_rate∈[40,100]、respiratory_rate∈[12,20]
    print(f"原始数据量: {len(df)}")
    df = df[(df['heart_rate'] >= 40) & (df['heart_rate'] <= 100) & 
           (df['respiratory_rate'] >= 12) & (df['respiratory_rate'] <= 20)]
    print(f"筛选后有效数据量: {len(df)}")
    
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
    # 标记离床时段
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
        # 重新计算睡眠状态，使用更精确的逻辑
        sleep_period_data = sleep_period_data.copy()
        
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
        sleep_period_data = sleep_period_data.sort_values('upload_time')
        
        if len(sleep_period_data) > 1:
            # 找到所有连续的睡眠片段并计算其总时长
            sleeping_intervals = []
            current_sleep_start = None
            
            for idx, row in sleep_period_data.iterrows():
                if row['is_sleeping']:
                    if current_sleep_start is None:
                        current_sleep_start = row['upload_time']
                else:
                    if current_sleep_start is not None:
                        # 结束当前睡眠片段
                        sleeping_intervals.append(row['upload_time'] - current_sleep_start)
                        current_sleep_start = None
            
            # 如果最后仍在睡眠状态
            if current_sleep_start is not None:
                sleeping_intervals.append(sleep_period_data.iloc[-1]['upload_time'] - current_sleep_start)
            
            # 累加所有睡眠片段的时长
            for interval in sleeping_intervals:
                sleep_duration_minutes += interval.total_seconds() / 60
        else:
            # 如果只有一条记录，检查它是否在睡眠状态
            if len(sleep_period_data) == 1 and sleep_period_data.iloc[0]['is_sleeping']:
                # 假设单条记录代表一个标准时间间隔（如15分钟）
                sleep_duration_minutes = 15  # 默认为15分钟
    else:
        sleep_duration_minutes = 0

    # 计算离床次数（从睡眠状态转为非睡眠状态的次数）
    if not sleep_period_data.empty and len(sleep_period_data) > 1:
        # 使用原始的离床逻辑：heart_rate=0持续>=15分钟
        off_bed_threshold_minutes = 15
        off_bed_events = 0
        
        # 检查是否有连续的离床时段
        is_currently_off_bed = False
        off_bed_start_time = None
        
        for idx, row in sleep_period_data.iterrows():
            if row['heart_rate'] == 0:
                if not is_currently_off_bed:
                    # 开始离床
                    off_bed_start_time = row['upload_time']
                    is_currently_off_bed = True
            else:
                if is_currently_off_bed:
                    # 结束离床时段
                    off_bed_duration = row['upload_time'] - off_bed_start_time
                    if off_bed_duration.total_seconds() >= off_bed_threshold_minutes * 60:
                        off_bed_events += 1
                    is_currently_off_bed = False
        
        # 检查最后是否还有未结束的离床时段
        if is_currently_off_bed:
            last_duration = sleep_period_data.iloc[-1]['upload_time'] - off_bed_start_time
            if last_duration.total_seconds() >= off_bed_threshold_minutes * 60:
                off_bed_events += 1
                
        bed_exit_count = off_bed_events
    else:
        bed_exit_count = 0

    # 计算睡眠准备时间（从上床到真正入睡的时间）
    if not sleep_period_data.empty:
        # 稳定睡眠判定：连续 5 条有效数据满足 "heart_rate 波动≤8 次 / 分 + arrhymia_ratio≤30%"
        stable_sleep_start = None
        stable_records_needed = 5
        
        for i in range(len(sleep_period_data) - stable_records_needed + 1):
            subset = sleep_period_data.iloc[i:i+stable_records_needed]
            
            # 检查是否满足稳定睡眠条件
            heart_rate_stable = subset['heart_rate'].std() <= 8 if len(subset['heart_rate']) > 1 else True
            avg_arrhymia_ratio = subset['arrhymia_ratio'].mean() if 'arrhymia_ratio' in subset.columns else 100
            arrhythmia_low = avg_arrhymia_ratio <= 30
            
            if heart_rate_stable and arrhythmia_low:
                stable_sleep_start = subset.iloc[0]['upload_time']
                break
        
        if stable_sleep_start:
            sleep_prep_time = (stable_sleep_start - bedtime).total_seconds() / 60
        else:
            sleep_prep_time = 0
    else:
        sleep_prep_time = 0

    # 计算平均生理指标
    period_data = df[(df['upload_time'] >= bedtime) & (df['upload_time'] <= wakeup_time)]
    avg_heart_rate = period_data['heart_rate'].mean()
    avg_respiratory_rate = period_data['respiratory_rate'].mean()
    avg_body_moves = period_data['body_moves_ratio'].mean() if 'body_moves_ratio' in period_data.columns else 0

    # 分析睡眠阶段（基于您提供的精确标准）
    # 先计算用户的清醒静息基线
    if not period_data.empty:
        # 计算清醒时的基线心率和呼吸频率（取较高值作为基线参考）
        # 假设心率>70或体动>20的情况为清醒状态
        awake_data = period_data[
            (period_data['heart_rate'] > 70) | (period_data['body_moves_ratio'] > 20)
        ]
        
        if not awake_data.empty:
            baseline_heart_rate = awake_data['heart_rate'].mean()
            baseline_respiratory_rate = awake_data['respiratory_rate'].mean()
        else:
            # 如果没有明显的清醒数据，使用总体平均值
            baseline_heart_rate = period_data['heart_rate'].mean()
            baseline_respiratory_rate = period_data['respiratory_rate'].mean()
    else:
        baseline_heart_rate = 70
        baseline_respiratory_rate = 16

    # 初始化各睡眠阶段时长
    deep_sleep_duration = 0
    rem_sleep_duration = 0
    light_sleep_duration = 0
    awake_duration = 0
    
    if not period_data.empty:
        # 筛选睡眠时段（有效数据且在睡眠期间）
        sleep_data = period_data[
            (period_data['heart_rate'] >= 40) & 
            (period_data['heart_rate'] <= 100) &
            (period_data['respiratory_rate'] >= 8) &  # 扩展呼吸频率范围以包含深睡和REM
            (period_data['respiratory_rate'] <= 25)   # 扩展呼吸频率范围以包含REM
        ].copy()
        
        if not sleep_data.empty:
            sleep_data = sleep_data.sort_values('upload_time')
            
            # 计算相邻时间点之间的时间间隔
            time_intervals = []
            for i in range(len(sleep_data) - 1):
                current_time = sleep_data.iloc[i]['upload_time']
                next_time = sleep_data.iloc[i + 1]['upload_time']
                interval_minutes = (next_time - current_time).total_seconds() / 60
                time_intervals.append(max(1, interval_minutes))  # 最小间隔为1分钟
            
            if len(time_intervals) > 0:
                time_intervals.append(time_intervals[-1])  # 为最后一个数据点使用相同的时间间隔
            else:
                time_intervals.append(1)  # 如果只有一个数据点，使用1分钟作为间隔
            
            # 检查每个时间点属于哪种睡眠阶段
            for idx, (index, row) in enumerate(sleep_data.iterrows()):
                time_interval = time_intervals[idx] if idx < len(time_intervals) else 1
                
                # 检查是否为清醒状态
                is_awake = (
                    row['heart_rate'] >= baseline_heart_rate * 0.95 or  # 心率接近/高于清醒基线
                    row['body_moves_ratio'] > 15  # 有明显肢体动作
                )
                
                if is_awake:
                    awake_duration += time_interval
                    continue
                
                # 计算心率相对于基线的变化百分比
                hr_change_from_baseline = ((row['heart_rate'] - baseline_heart_rate) / baseline_heart_rate) * 100 if baseline_heart_rate > 0 else 0
                
                # 计算呼吸变异系数（如果有足够的历史数据）
                # 为了简化，这里使用局部窗口计算呼吸稳定性
                respiratory_stability = 0
                if idx > 0 and idx < len(sleep_data) - 1:
                    window_rr = [
                        sleep_data.iloc[max(0, idx-2)]['respiratory_rate'],
                        sleep_data.iloc[idx]['respiratory_rate'],
                        sleep_data.iloc[min(len(sleep_data)-1, idx+2)]['respiratory_rate']
                    ]
                    if len(set(window_rr)) > 1:  # 避免除零错误
                        rr_std = pd.Series(window_rr).std()
                        rr_mean = pd.Series(window_rr).mean()
                        respiratory_stability = (rr_std / rr_mean * 100) if rr_mean > 0 else 0
                    else:
                        respiratory_stability = 0
                else:
                    respiratory_stability = 0  # 无法计算稳定性时设为0

                # 深睡判定：心率比清醒基线低10%-30%，呼吸频率8-12次/分，无肢体动作
                is_deep = (
                    baseline_heart_rate * 0.7 <= row['heart_rate'] <= baseline_heart_rate * 0.9 and  # 心率比基线低10%-30%
                    8 <= row['respiratory_rate'] <= 12 and  # 呼吸频率8-12次/分
                    respiratory_stability < 5 and  # 呼吸变异系数<5%
                    row['body_moves_ratio'] <= 5  # 无肢体动作
                )
                
                # REM判定：心率比深睡基线高20%-40%，呼吸不稳定，无肢体动作
                # 首先需要估算深睡基线心率
                estimated_deep_hr = baseline_heart_rate * 0.8  # 假设深睡心率为清醒基线的80%
                
                is_rem = (
                    row['heart_rate'] >= estimated_deep_hr * 1.2 and  # 心率比深睡基线高20%+
                    respiratory_stability > 15 and  # 呼吸变异系数>15%
                    row['body_moves_ratio'] <= 2 and  # 无肢体动作
                    row['respiratory_rate'] >= 9  # 呼吸频率比深睡高
                )
                
                # 浅睡判定：不满足深睡/REM/清醒特征，但符合浅睡特征
                is_light = (
                    not is_deep and 
                    not is_rem and 
                    not is_awake and
                    60 <= row['heart_rate'] <= baseline_heart_rate * 0.95 and  # 心率介于深睡和清醒之间
                    12 <= row['respiratory_rate'] <= 16 and  # 呼吸频率12-16次/分
                    5 <= respiratory_stability <= 10 and  # 呼吸变异系数5%-10%
                    row['body_moves_ratio'] <= 15  # 偶有轻微肢体动作
                )
                
                # 根据判定结果分配时间到相应阶段
                if is_deep:
                    deep_sleep_duration += time_interval
                elif is_rem:
                    rem_sleep_duration += time_interval
                elif is_light:
                    light_sleep_duration += time_interval
                else:
                    # 如果都不符合，分配给最可能的阶段
                    if row['heart_rate'] < baseline_heart_rate * 0.85 and row['respiratory_rate'] <= 12 and row['body_moves_ratio'] <= 5:
                        deep_sleep_duration += time_interval
                    elif row['heart_rate'] > baseline_heart_rate * 0.9 and (respiratory_stability > 10 or row['body_moves_ratio'] > 10):
                        awake_duration += time_interval
                    elif 60 <= row['heart_rate'] <= baseline_heart_rate * 0.9 and 12 <= row['respiratory_rate'] <= 16:
                        light_sleep_duration += time_interval
                    else:
                        # 默认分配给浅睡
                        light_sleep_duration += time_interval
    
    # 计算清醒时长（卧床但不在睡眠状态的时间）
    # 补充清醒时长的计算
    if not period_data.empty:
        period_sorted = period_data.sort_values('upload_time')
        for i in range(len(period_sorted) - 1):
            current_row = period_sorted.iloc[i]
            next_row = period_sorted.iloc[i + 1]
            time_diff = (next_row['upload_time'] - current_row['upload_time']).total_seconds() / 60
            
            # 检查是否为清醒时段
            is_current_awake = (
                current_row['heart_rate'] >= baseline_heart_rate * 0.95 or
                current_row['body_moves_ratio'] > 15
            )
            
            if is_current_awake:
                awake_duration += max(1, time_diff)  # 至少1分钟
    
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
    hr_variability = period_data['heart_rate'].std() if len(period_data) > 1 else 0
    if hr_variability <= 15:
        stability_score = 10
    elif 16 <= hr_variability <= 25:
        stability_score = 7
    else:
        stability_score = 3

    sleep_score = min(100, round(time_score + deep_sleep_score + efficiency_score + interference_score + stability_score))

    # 返回详细的睡眠分析结果
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
        "average_metrics": {
            "avg_heart_rate": round(float(avg_heart_rate), 2) if pd.notna(avg_heart_rate) else 0,
            "avg_respiratory_rate": round(float(avg_respiratory_rate), 2) if pd.notna(avg_respiratory_rate) else 0,
            "avg_body_moves_ratio": round(float(avg_body_moves), 2) if pd.notna(avg_body_moves) else 0,
            "avg_heartbeat_interval": round(float(period_data['avg_heartbeat_interval'].mean()), 2) if 'avg_heartbeat_interval' in period_data.columns else 0,
            "avg_rms_heartbeat_interval": round(float(period_data['rms_heartbeat_interval'].mean()), 2) if 'rms_heartbeat_interval' in period_data.columns else 0
        },
        "summary": f"睡眠质量{'优秀' if sleep_score >= 80 else '良好' if sleep_score >= 60 else '一般' if sleep_score >= 40 else '较差'}"
    }
    
    return result


@tool
def analyze_sleep_by_date(date: str, runtime: ToolRuntime = None, table_name: str = "device_data") -> str:
    """
    根据指定日期分析睡眠数据
    
    Args:
        date: 日期字符串，格式如 '2024-12-20'
        runtime: ToolRuntime 运行时上下文
        table_name: 数据库表名，默认为 "device_data"
    
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


# 测试函数
if __name__ == '__main__':
    # 示例：分析今天的睡眠数据
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"分析 {today} 的睡眠数据...")
    result = analyze_single_day_sleep_data(today)
    print(result)