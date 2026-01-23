"""
生理指标分析工具 - 分析指定日期的心率、呼吸等生理数据
"""
import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional
import warnings
from langchain.tools import tool, ToolRuntime
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_physiological_metrics(df, date_str):
    """
    分析生理指标
    
    Args:
        df: 包含当日数据的DataFrame
        date_str: 目标日期字符串
    
    Returns:
        包含生理指标分析结果的字典
    """
    logger.info(f"开始分析 {date_str} 的生理指标，原始数据量: {len(df)}")
    
    # 转换数值列
    numeric_columns = [
        'heart_rate', 'respiratory_rate', 'avg_heartbeat_interval', 
        'rms_heartbeat_interval', 'std_heartbeat_interval', 'arrhythmia_ratio', 'body_moves_ratio',
        'respiratory_pause_count', 'avg_pause_duration', 'max_pause_duration'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 对不同类型的记录采用不同的数据清理策略
    # 对于周期数据和状态数据，需要heart_rate和respiratory_rate，且is_person必须为1
    cycle_and_status_data = df[df['data_type'] != '呼吸暂停'].copy()
    apnea_data = df[df['data_type'] == '呼吸暂停'].copy()
    
    # 只对非呼吸暂停数据应用严格的NaN过滤和is_person过滤
    if not cycle_and_status_data.empty:
        # 对于周期数据，只保留is_person为1（有人）的记录
        cycle_and_status_data = cycle_and_status_data[
            (cycle_and_status_data['heart_rate'].notna()) & 
            (cycle_and_status_data['respiratory_rate'].notna()) &
            (cycle_and_status_data['is_person'] == 1)
        ].sort_values('upload_time')
    
    # 合并数据，保留所有呼吸暂停记录
    df = pd.concat([cycle_and_status_data, apnea_data]).sort_values('upload_time')

    if df.empty:
        return {
            "error": "没有有效的生理指标数据",
            "date": date_str
        }
    
    target_date = datetime.strptime(date_str, '%Y-%m-%d')
    
    # 过滤指定日期的数据 - 重点关注前一晚20:00到target_date早上8:00的夜间时段
    # 计算夜间时间段
    prev_evening_start = datetime.combine((target_date - timedelta(days=1)).date(), datetime.min.time().replace(hour=20))  # 前一天20:00
    target_morning_end = datetime.combine(target_date.date(), datetime.min.time().replace(hour=8))  # 当天8:00
    
    # 构建夜间时间段过滤条件
    night_mask = (df['upload_time'] >= prev_evening_start) & (df['upload_time'] <= target_morning_end)
    
    daily_data = df[night_mask].copy()
    
    if daily_data.empty:
        # 如果夜间时段没有数据，尝试扩展到target_date全天数据
        date_mask = df['upload_time'].dt.date == target_date.date()
        daily_data = df[date_mask].copy()

    if daily_data.empty:
        return {
            "error": f"日期 {date_str} 没有数据",
            "date": date_str
        }
    
    # 分析呼吸暂停数据 - 根据data_type为"呼吸暂停"来判断
    apnea_data = daily_data[daily_data['data_type'] == '呼吸暂停'].copy()
    apnea_count = apnea_data['respiratory_pause_count'].sum() if 'respiratory_pause_count' in apnea_data.columns and not apnea_data.empty else 0
    max_apnea_duration = apnea_data['max_pause_duration'].max() if 'max_pause_duration' in apnea_data.columns and not apnea_data.empty else 0
    avg_apnea_duration = apnea_data['avg_pause_duration'].mean() if 'avg_pause_duration' in apnea_data.columns and not apnea_data.empty else 0
    
    # 计算平均呼吸频率和其他指标
    respiratory_data = daily_data['respiratory_rate'].dropna()
    avg_respiratory_rate = respiratory_data.mean() if not respiratory_data.empty else 0
    min_respiratory_rate = respiratory_data.min() if not respiratory_data.empty else 0
    max_respiratory_rate = respiratory_data.max() if not respiratory_data.empty else 0

    # 呼吸健康评分 (简化算法，基于呼吸频率稳定性和暂停情况)
    if not respiratory_data.empty and len(respiratory_data) > 1:
        respiratory_stability = respiratory_data.std()  # 标准差越小越稳定
        respiratory_avg = respiratory_data.mean()
        
        # 评分计算：基于正常范围(12-20)、稳定性等因素
        base_score = 100
        # 偏离正常范围的惩罚
        if respiratory_avg < 12 or respiratory_avg > 20:
            base_score -= 20
        # 不稳定性的惩罚
        if respiratory_stability > 5:
            base_score -= min(30, respiratory_stability)
        # 呼吸暂停的惩罚
        if apnea_count > 0:
            base_score -= min(40, apnea_count * 2)  # 每次暂停扣2分，最多扣40分
        
        respiratory_health_score = max(0, min(100, base_score))
    else:
        respiratory_health_score = 0
    
    # 心率指标分析
    heart_rate_data = daily_data['heart_rate'].dropna()
    avg_heart_rate = heart_rate_data.mean() if not heart_rate_data.empty else 0
    max_heart_rate = heart_rate_data.max() if not heart_rate_data.empty else 0
    min_heart_rate = heart_rate_data.min() if not heart_rate_data.empty else 0
    
    # HRV分数计算 (基于std_heartbeat_interval)
    hrv_score = 0
    if 'std_heartbeat_interval' in daily_data.columns:
        std_hbi_data = daily_data['std_heartbeat_interval'].dropna()
        if not std_hbi_data.empty:
            # HRV分数：基于心跳间期标准差，通常标准差越大HRV越好（在一定范围内）
            avg_std_hbi = std_hbi_data.mean()
            # 简化评分算法：理想HRV范围在50-100之间
            if 50 <= avg_std_hbi <= 100:
                hrv_score = 100
            elif avg_std_hbi < 50:
                hrv_score = min(100, (avg_std_hbi / 50) * 100)
            else:  # avg_std_hbi > 100
                hrv_score = max(0, 100 - ((avg_std_hbi - 100) * 0.5))
            hrv_score = max(0, min(100, hrv_score))
    
    # 返回详细的生理指标分析结果
    result = {
        "date": date_str,
        "respiratory_metrics": {
            "avg_respiratory_rate": round(float(avg_respiratory_rate), 2) if pd.notna(avg_respiratory_rate) else 0,
            "min_respiratory_rate": round(float(min_respiratory_rate), 2) if pd.notna(min_respiratory_rate) else 0,
            "max_respiratory_rate": round(float(max_respiratory_rate), 2) if pd.notna(max_respiratory_rate) else 0,
            "apnea_count": int(apnea_count),
            "max_apnea_duration_seconds": round(max_apnea_duration, 2),
            "avg_apnea_duration_seconds": round(avg_apnea_duration, 2),
            "respiratory_health_score": round(respiratory_health_score, 2)
        },
        "heart_rate_metrics": {
            "avg_heart_rate": round(float(avg_heart_rate), 2) if pd.notna(avg_heart_rate) else 0,
            "hrv_score": round(hrv_score, 2),
            "max_heart_rate": round(float(max_heart_rate), 2) if pd.notna(max_heart_rate) else 0,
            "min_heart_rate": round(float(min_heart_rate), 2) if pd.notna(min_heart_rate) else 0
        },
        "summary": f"生理指标分析完成，共处理 {len(daily_data)} 条数据"
    }
    
    return result


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


def analyze_single_day_physiological_data(date_str: str, table_name: str = "vital_signs"):
    """
    分析单日生理指标数据
    时间范围：前一天晚上8点到当天早上10点
    
    Args:
        date_str: 日期字符串，格式如 '2024-12-20'
        table_name: 数据库表名，默认为 "vital_signs"
    
    Returns:
        JSON格式的生理指标分析结果
    """
    try:
        from src.utils.response_handler import PhysiologicalAnalysisResponse
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
            logger.warning(f"数据库中没有找到 {date_str} 期间的生理指标数据")
            # 返回格式一致但数据为0的结果
            response = PhysiologicalAnalysisResponse(
                success=True,
                date=date_str,
                heart_rate_metrics={
                    "avg_heart_rate": 0,
                    "min_heart_rate": 0,
                    "max_heart_rate": 0,
                    "heart_rate_variability": 0,
                    "heart_rate_stability": 0
                },
                respiratory_metrics={
                    "avg_respiratory_rate": 0,
                    "min_respiratory_rate": 0,
                    "max_respiratory_rate": 0,
                    "respiratory_stability": 0,
                    "apnea_events_per_hour": 0,
                    "apnea_count": 0,
                    "avg_apnea_duration": 0,
                    "max_apnea_duration": 0
                },
                sleep_metrics={
                    "avg_body_moves_ratio": 0,
                    "body_movement_frequency": 0,
                    "sleep_efficiency": 0
                },
                summary="暂无数据",
                message=f"在{date_str}期间没有找到生理指标数据"
            )
            return response.to_json()
        
        # 转换时间列为datetime格式
        df['upload_time'] = pd.to_datetime(df['upload_time'])
        
        # 分析生理指标数据
        result = analyze_physiological_metrics(df, date_str)
        
        # 将numpy/pandas类型转换为原生Python类型以支持JSON序列化
        result = convert_numpy_types(result)
        
        logger.info(f"生理指标分析完成，结果: {result.get('date', 'N/A')}")
        
        # 使用PhysiologicalAnalysisResponse类包装结果
        response = PhysiologicalAnalysisResponse(
            success=True,
            date=result.get('date', date_str),
            heart_rate_metrics=result.get('heart_rate_metrics', {
                "avg_heart_rate": 0,
                "min_heart_rate": 0,
                "max_heart_rate": 0,
                "heart_rate_variability": 0,
                "heart_rate_stability": 0
            }),
            respiratory_metrics=result.get('respiratory_metrics', {
                "avg_respiratory_rate": 0,
                "min_respiratory_rate": 0,
                "max_respiratory_rate": 0,
                "respiratory_stability": 0,
                "apnea_events_per_hour": 0,
                "apnea_count": 0,
                "avg_apnea_duration": 0,
                "max_apnea_duration": 0
            }),
            sleep_metrics=result.get('sleep_metrics', {
                "avg_body_moves_ratio": 0,
                "body_movement_frequency": 0,
                "sleep_efficiency": 0
            }),
            summary=result.get('summary', '分析完成')
        )
        return response.to_json()
        
    except Exception as e:
        import traceback
        error_msg = f"单日生理指标分析失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        
        from src.utils.response_handler import ApiResponse
        # 返回错误格式但保持一致的结构
        response = ApiResponse.error(
            error=str(e),
            message="生理指标分析失败",
            data={
                "date": date_str,
                "heart_rate_metrics": {
                    "avg_heart_rate": 0,
                    "min_heart_rate": 0,
                    "max_heart_rate": 0,
                    "heart_rate_variability": 0,
                    "heart_rate_stability": 0
                },
                "respiratory_metrics": {
                    "avg_respiratory_rate": 0,
                    "min_respiratory_rate": 0,
                    "max_respiratory_rate": 0,
                    "respiratory_stability": 0,
                    "apnea_events_per_hour": 0,
                    "apnea_count": 0,
                    "avg_apnea_duration": 0,
                    "max_apnea_duration": 0
                },
                "sleep_metrics": {
                    "avg_body_moves_ratio": 0,
                    "body_movement_frequency": 0,
                    "sleep_efficiency": 0
                },
                "summary": "分析失败",
                "error": str(e)
            }
        )
        return response.to_json()


def analyze_single_day_physiological_data_with_device(date_str: str, device_sn: str, table_name: str = "vital_signs"):
    """
    使用设备序列号分析单日生理指标数据
    时间范围：前一天晚上8点到当天早上10点
    
    Args:
        date_str: 日期字符串，格式如 '2024-12-20'
        device_sn: 设备序列号
        table_name: 数据库表名，默认为 "vital_signs"
    
    Returns:
        JSON格式的生理指标分析结果
    """
    try:
        from src.utils.response_handler import PhysiologicalAnalysisResponse
        from src.db.database import get_db_manager
        db_manager = get_db_manager()
        
        # 使用新的时间范围：前一天晚上8点到当天早上10点，并包含设备过滤
        df = db_manager.get_sleep_data_for_date_range_and_time(
            table_name,
            date_str,
            start_hour=20,  # 晚上8点
            end_hour=10     # 早上10点
        )
        
        # 过滤特定设备的数据
        df = df[df['device_sn'] == device_sn] if 'device_sn' in df.columns else df
        
        logger.info(f"查询到 {len(df)} 条设备 {device_sn} 的数据")
        
        if df.empty:
            logger.warning(f"数据库中没有找到 {date_str} 期间的设备 {device_sn} 的数据")
            # 返回格式一致但数据为0的结果
            response = PhysiologicalAnalysisResponse(
                success=True,
                date=date_str,
                heart_rate_metrics={
                    "avg_heart_rate": 0,
                    "min_heart_rate": 0,
                    "max_heart_rate": 0,
                    "heart_rate_variability": 0,
                    "heart_rate_stability": 0
                },
                respiratory_metrics={
                    "avg_respiratory_rate": 0,
                    "min_respiratory_rate": 0,
                    "max_respiratory_rate": 0,
                    "respiratory_stability": 0,
                    "apnea_events_per_hour": 0,
                    "apnea_count": 0,
                    "avg_apnea_duration": 0,
                    "max_apnea_duration": 0
                },
                sleep_metrics={
                    "avg_body_moves_ratio": 0,
                    "body_movement_frequency": 0,
                    "sleep_efficiency": 0
                },
                summary="暂无数据",
                device_sn=device_sn,
                message=f"设备 {device_sn} 在 {date_str} 期间没有数据"
            )
            return response.to_json()
        
        # 转换时间列为datetime格式
        df['upload_time'] = pd.to_datetime(df['upload_time'])
        
        # 分析生理指标数据
        result = analyze_physiological_metrics(df, date_str)
        
        # 将numpy/pandas类型转换为原生Python类型以支持JSON序列化
        result = convert_numpy_types(result)
        
        # 添加设备序列号到结果中
        result['device_sn'] = device_sn
        
        logger.info(f"生理指标分析完成，结果: {result.get('date', 'N/A')}")
        
        # 使用PhysiologicalAnalysisResponse类包装结果
        response = PhysiologicalAnalysisResponse(
            success=True,
            date=result.get('date', date_str),
            heart_rate_metrics=result.get('heart_rate_metrics', {
                "avg_heart_rate": 0,
                "min_heart_rate": 0,
                "max_heart_rate": 0,
                "heart_rate_variability": 0,
                "heart_rate_stability": 0
            }),
            respiratory_metrics=result.get('respiratory_metrics', {
                "avg_respiratory_rate": 0,
                "min_respiratory_rate": 0,
                "max_respiratory_rate": 0,
                "respiratory_stability": 0,
                "apnea_events_per_hour": 0,
                "apnea_count": 0,
                "avg_apnea_duration": 0,
                "max_apnea_duration": 0
            }),
            sleep_metrics=result.get('sleep_metrics', {
                "avg_body_moves_ratio": 0,
                "body_movement_frequency": 0,
                "sleep_efficiency": 0
            }),
            summary=result.get('summary', '分析完成'),
            device_sn=result.get('device_sn', device_sn)
        )
        return response.to_json()
        
    except Exception as e:
        import traceback
        error_msg = f"单日生理指标分析失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        
        from src.utils.response_handler import ApiResponse
        # 返回错误格式但保持一致的结构
        response = ApiResponse.error(
            error=str(e),
            message="生理指标分析失败",
            data={
                "date": date_str,
                "heart_rate_metrics": {
                    "avg_heart_rate": 0,
                    "min_heart_rate": 0,
                    "max_heart_rate": 0,
                    "heart_rate_variability": 0,
                    "heart_rate_stability": 0
                },
                "respiratory_metrics": {
                    "avg_respiratory_rate": 0,
                    "min_respiratory_rate": 0,
                    "max_respiratory_rate": 0,
                    "respiratory_stability": 0,
                    "apnea_events_per_hour": 0,
                    "apnea_count": 0,
                    "avg_apnea_duration": 0,
                    "max_apnea_duration": 0
                },
                "sleep_metrics": {
                    "avg_body_moves_ratio": 0,
                    "body_movement_frequency": 0,
                    "sleep_efficiency": 0
                },
                "summary": "分析失败",
                "device_sn": device_sn,
                "error": str(e)
            }
        )
        return response.to_json()


@tool
def analyze_physiological_by_date(date: str, runtime: ToolRuntime = None, table_name: str = "vital_signs") -> str:
    """
    根据指定日期分析生理指标数据
    
    Args:
        date: 日期字符串，格式如 '2024-12-20'
        runtime: ToolRuntime 运行时上下文
        table_name: 数据库表名，默认为 "vital_signs"
    
    Returns:
        JSON格式的生理指标分析结果，包含：
        - respiratory_metrics: 呼吸指标
          * avg_respiratory_rate: 呼吸平均次数/分钟
          * apnea_count: 呼吸暂停次数
          * max_apnea_duration_seconds: 最长呼吸暂停秒数
          * avg_apnea_duration_seconds: 平均呼吸暂停秒数
          * respiratory_health_score: 呼吸健康评分
        - heart_rate_metrics: 心率指标
          * avg_heart_rate: 心率平均次数/分钟
          * hrv_score: HRV分数（100满分）
          * max_heart_rate: 最高心率
          * min_heart_rate: 最低心率
    
    使用场景:
        - 分析特定日期的生理健康指标
        - 监控个人心率和呼吸模式变化
        - 评估呼吸暂停风险
    """
    return analyze_single_day_physiological_data(date, table_name)


def analyze_physiological_trend(date_str: str, table_name: str = "vital_signs"):
    """
    分析生理指标趋势（心率和呼吸率随时间变化）
    
    Args:
        date_str: 日期字符串，格式如 '2024-12-20'
        table_name: 数据库表名，默认为 "vital_signs"
    
    Returns:
        JSON格式的生理指标趋势分析结果
    """
    try:
        from src.db.database import get_db_manager
        db_manager = get_db_manager()
        
        # 尝试解析日期以验证格式
        try:
            # 支持多种日期格式，如 '2026-1-22', '2026-01-22' 等
            parsed_date = datetime.strptime(date_str, '%Y-%m-%d')
            # 重新格式化为标准格式
            date_str = parsed_date.strftime('%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}. Expected format: YYYY-MM-DD")
        
        # 使用新的时间范围：前一天晚上8点到当天早上10点
        df = db_manager.get_sleep_data_for_date_range_and_time(
            table_name,
            date_str,
            start_hour=20,  # 晚上8点
            end_hour=10     # 早上10点
        )
        
        logger.info(f"查询到 {len(df)} 条趋势数据")
        
        if df.empty:
            logger.warning(f"数据库中没有找到 {date_str} 期间的生理指标趋势数据")
            # 返回格式一致但数据为空的结果
            result = {
                "date": date_str,
                "data_points": [],
                "summary": "暂无数据"
            }
            return json.dumps(result, ensure_ascii=False, indent=2)
        
        # 转换时间列为datetime格式
        df['upload_time'] = pd.to_datetime(df['upload_time'])
        
        # 按时间排序
        df = df.sort_values(by='upload_time')
        
        # 过滤只保留data_type为周期数据的记录
        if 'data_type' in df.columns:
            df = df[df['data_type'] == '周期数据']
        
        # 提取时间点和对应的生理指标，按照您期望的格式
        data_points = []
        for _, row in df.iterrows():
            # 提取 HH:MM 格式的时间
            time_str = row['upload_time'].strftime('%H:%M')
            
            # 获取心率和呼吸率，处理 NaN 值
            heart_rate = row['heart_rate']
            respiratory_rate = row['respiratory_rate']
            
            # 如果是 NaN 或者 null，转换为 0
            if pd.isna(heart_rate) or heart_rate is None:
                heart_rate = 0
            else:
                heart_rate = int(heart_rate) if heart_rate != 0 else 0  # 确保是整数
                
            if pd.isna(respiratory_rate) or respiratory_rate is None:
                respiratory_rate = 0
            else:
                respiratory_rate = int(respiratory_rate) if respiratory_rate != 0 else 0  # 确保是整数
                
            data_point = {
                "time": time_str,
                "heart_rate": heart_rate,
                "respiratory_rate": respiratory_rate
            }
            data_points.append(data_point)
        
        # 创建趋势数据
        trend_data = {
            "date": date_str,
            "data_points": data_points,
            "summary": f"共{len(data_points)}个时间点的生理指标趋势数据"
        }
        
        logger.info(f"生理指标趋势分析完成，时间点数量: {len(data_points)}")
        
        return json.dumps(trend_data, ensure_ascii=False, indent=2)
        
    except Exception as e:
        import traceback
        error_msg = f"生理指标趋势分析失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return json.dumps({
            "date": date_str,
            "data_points": [],
            "summary": "分析失败",
            "error": str(e)
        }, ensure_ascii=False, indent=2)


def analyze_physiological_trend_with_device(date_str: str, device_sn: str, table_name: str = "vital_signs"):
    """
    使用设备序列号分析生理指标趋势（心率和呼吸率随时间变化）
    
    Args:
        date_str: 日期字符串，格式如 '2024-12-20'
        device_sn: 设备序列号
        table_name: 数据库表名，默认为 "vital_signs"
    
    Returns:
        JSON格式的生理指标趋势分析结果
    """
    try:
        from src.db.database import get_db_manager
        db_manager = get_db_manager()
        
        # 尝试解析日期以验证格式
        try:
            # 支持多种日期格式，如 '2026-1-22', '2026-01-22' 等
            parsed_date = datetime.strptime(date_str, '%Y-%m-%d')
            # 重新格式化为标准格式
            date_str = parsed_date.strftime('%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}. Expected format: YYYY-MM-DD")
        
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
        
        logger.info(f"查询到 {len(df)} 条设备 {device_sn} 的趋势数据")
        
        if df.empty:
            logger.warning(f"数据库中没有找到 {date_str} 期间的设备 {device_sn} 的生理指标趋势数据")
            # 返回格式一致但数据为空的结果
            result = {
                "date": date_str,
                "device_sn": device_sn,
                "data_points": [],
                "summary": "暂无数据"
            }
            return json.dumps(result, ensure_ascii=False, indent=2)
        
        # 转换时间列为datetime格式
        df['upload_time'] = pd.to_datetime(df['upload_time'])
        
        # 按时间排序
        df = df.sort_values(by='upload_time')
        
        # 过滤只保留data_type为周期数据的记录
        if 'data_type' in df.columns:
            df = df[df['data_type'] == '周期数据']
        
        # 提取时间点和对应的生理指标，按照您期望的格式
        data_points = []
        for _, row in df.iterrows():
            # 提取 HH:MM 格式的时间
            time_str = row['upload_time'].strftime('%H:%M')
            
            # 获取心率和呼吸率，处理 NaN 值
            heart_rate = row['heart_rate']
            respiratory_rate = row['respiratory_rate']
            
            # 如果是 NaN 或者 null，转换为 0
            if pd.isna(heart_rate) or heart_rate is None:
                heart_rate = 0
            else:
                heart_rate = int(heart_rate) if heart_rate != 0 else 0  # 确保是整数
                
            if pd.isna(respiratory_rate) or respiratory_rate is None:
                respiratory_rate = 0
            else:
                respiratory_rate = int(respiratory_rate) if respiratory_rate != 0 else 0  # 确保是整数
                
            data_point = {
                "time": time_str,
                "heart_rate": heart_rate,
                "respiratory_rate": respiratory_rate
            }
            data_points.append(data_point)
        
        # 创建趋势数据
        trend_data = {
            "date": date_str,
            "device_sn": device_sn,
            "data_points": data_points,
            "summary": f"设备 {device_sn} 共{len(data_points)}个时间点的生理指标趋势数据"
        }
        
        logger.info(f"设备 {device_sn} 生理指标趋势分析完成，时间点数量: {len(data_points)}")
        
        return json.dumps(trend_data, ensure_ascii=False, indent=2)
        
    except Exception as e:
        import traceback
        error_msg = f"设备 {device_sn} 生理指标趋势分析失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return json.dumps({
            "date": date_str,
            "device_sn": device_sn,
            "data_points": [],
            "summary": "分析失败",
            "error": str(e)
        }, ensure_ascii=False, indent=2)


def format_physiological_analysis_natural_language(date_str: str, device_sn: str = None, table_name: str = "vital_signs"):
    """
    生成自然语言格式的生理指标分析报告
    
    Args:
        date_str: 日期字符串，格式如 '2024-12-20'
        device_sn: 设备序列号（可选）
        table_name: 数据库表名，默认为 "vital_signs"
    
    Returns:
        自然语言格式的生理指标分析报告
    """
    try:
        from src.db.database import get_db_manager
        db_manager = get_db_manager()
        
        # 解析日期以构建时间范围
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        prev_date = target_date - timedelta(days=1)
        
        # 构建查询条件：检查前一天晚上20点到当天早上10点的数据
        start_time = prev_date.replace(hour=20, minute=0, second=0, microsecond=0)
        end_time = target_date.replace(hour=10, minute=0, second=0, microsecond=0)
        
        # 构建SQL查询
        escaped_table_name = f"`{table_name.replace('`', '``')}`"
        if device_sn:
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
        else:
            query = f"""
            SELECT * 
            FROM {escaped_table_name} 
            WHERE upload_time BETWEEN :start_time AND :end_time
            ORDER BY upload_time
            """
            params = {
                'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S')
            }
        
        # 执行查询
        df = db_manager.execute_query(query, params)
        
        logger.info(f"查询到 {len(df)} 条生理指标数据")
        
        if df.empty:
            return "暂无数据，报告正在生成中"
        
        # 转换时间列为datetime格式
        df['upload_time'] = pd.to_datetime(df['upload_time'])
        
        # 按时间排序
        df = df.sort_values(by='upload_time')
        
        # 进行数据分析
        # 转换数值列
        numeric_columns = ['heart_rate', 'respiratory_rate', 'avg_heartbeat_interval', 
                          'rms_heartbeat_interval', 'std_heartbeat_interval', 'arrhythmia_ratio', 'body_moves_ratio',
                          'respiratory_pause_count', 'avg_pause_duration', 'max_pause_duration']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 对不同类型的记录采用不同的数据清理策略
        cycle_and_status_data = df[df['data_type'] != '呼吸暂停'].copy()
        apnea_data = df[df['data_type'] == '呼吸暂停'].copy()
        
        # 只对非呼吸暂停数据应用严格的NaN过滤和is_person过滤
        if not cycle_and_status_data.empty:
            cycle_and_status_data = cycle_and_status_data[
                (cycle_and_status_data['heart_rate'].notna()) & 
                (cycle_and_status_data['respiratory_rate'].notna()) &
                (cycle_and_status_data['is_person'] == 1)
            ].sort_values('upload_time')
        
        # 合并数据，保留所有呼吸暂停记录
        df = pd.concat([cycle_and_status_data, apnea_data]).sort_values('upload_time')
        
        if df.empty:
            return "暂无有效数据，报告正在生成中"
        
        # 分析数据
        # 获取最早和最晚的时间点
        earliest_time = df['upload_time'].min()
        latest_time = df['upload_time'].max()
        
        # 计算平均值
        avg_heart_rate = df['heart_rate'].mean()
        min_heart_rate = df['heart_rate'].min()
        max_heart_rate = df['heart_rate'].max()
        avg_respiratory_rate = df['respiratory_rate'].mean()
        min_respiratory_rate = df['respiratory_rate'].min()
        max_respiratory_rate = df['respiratory_rate'].max()
        
        # 计算呼吸暂停相关指标
        apnea_count = apnea_data['respiratory_pause_count'].sum() if 'respiratory_pause_count' in apnea_data.columns and not apnea_data.empty else 0
        max_apnea_duration = apnea_data['max_pause_duration'].max() if 'max_pause_duration' in apnea_data.columns and not apnea_data.empty else 0
        avg_apnea_duration = apnea_data['avg_pause_duration'].mean() if 'avg_pause_duration' in apnea_data.columns and not apnea_data.empty else 0
        
        # 生成自然语言描述
        analysis_text = f"根据{date_str}的生理监测数据显示：\n\n"
        analysis_text += f"1. 整体监测时段：从{earliest_time.strftime('%H:%M')}到{latest_time.strftime('%H:%M')}\n\n"
        
        # 心率分析
        analysis_text += f"2. 心率指标：\n"
        analysis_text += f"   - 平均心率：{avg_heart_rate:.1f}次/分钟\n"
        analysis_text += f"   - 最小心率：{min_heart_rate:.1f}次/分钟\n"
        analysis_text += f"   - 最大心率：{max_heart_rate:.1f}次/分钟\n\n"
        
        # 呼吸率分析
        analysis_text += f"3. 呼吸率指标：\n"
        analysis_text += f"   - 平均呼吸率：{avg_respiratory_rate:.1f}次/分钟\n"
        analysis_text += f"   - 最小呼吸率：{min_respiratory_rate:.1f}次/分钟\n"
        analysis_text += f"   - 最大呼吸率：{max_respiratory_rate:.1f}次/分钟\n\n"
        
        # 呼吸暂停分析
        analysis_text += f"4. 呼吸暂停情况：\n"
        analysis_text += f"   - 呼吸暂停次数：{int(apnea_count) if pd.notna(apnea_count) else 0}次\n"
        analysis_text += f"   - 最长呼吸暂停时长：{max_apnea_duration:.1f}秒\n"
        analysis_text += f"   - 平均呼吸暂停时长：{avg_apnea_duration:.1f}秒\n\n"
        
        # 简单的健康评估
        analysis_text += f"5. 健康评估：\n"
        if 60 <= avg_heart_rate <= 100:
            analysis_text += f"   - 平均心率在正常范围内\n"
        elif avg_heart_rate < 60:
            analysis_text += f"   - 平均心率偏低\n"
        else:
            analysis_text += f"   - 平均心率偏高\n"
        
        if 12 <= avg_respiratory_rate <= 20:
            analysis_text += f"   - 平均呼吸率在正常范围内\n"
        elif avg_respiratory_rate < 12:
            analysis_text += f"   - 平均呼吸率偏低\n"
        else:
            analysis_text += f"   - 平均呼吸率偏高\n"
        
        if apnea_count > 5:
            analysis_text += f"   - 呼吸暂停次数较多，建议关注睡眠呼吸健康\n"
        
        analysis_text += f"\n以上是{date_str}的生理指标分析报告。"
        
        return analysis_text
        
    except Exception as e:
        import traceback
        error_msg = f"生成自然语言生理指标分析失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return f"生成分析报告时出现错误：{str(e)}"



# 测试函数
if __name__ == '__main__':
    # 示例：分析今天的生理指标数据
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"分析 {today} 的生理指标数据...")
    result = analyze_single_day_physiological_data(today)
    print(result)