"""
生理指标分析工具 - 分析指定日期的心率、呼吸等生理数据
"""
import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from langchain.tools import tool
from langchain.tools import ToolRuntime
from src.db.database import get_db_manager

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
        'rms_heartbeat_interval', 'std_heartbeat_interval', 'arrhymia_ratio', 'body_moves_ratio'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 移除无效数据
    df = df.dropna(subset=['heart_rate', 'respiratory_rate']).sort_values('upload_time')
    
    if df.empty:
        return {
            "error": "没有有效的生理指标数据",
            "date": date_str
        }
    
    target_date = datetime.strptime(date_str, '%Y-%m-%d')
    
    # 过滤指定日期的数据
    date_mask = df['upload_time'].dt.date == target_date.date()
    daily_data = df[date_mask].copy()
    
    if daily_data.empty:
        return {
            "error": f"日期 {date_str} 没有数据",
            "date": date_str
        }
    
    # 呼吸指标分析
    respiratory_data = daily_data['respiratory_rate'].dropna()
    avg_respiratory_rate = respiratory_data.mean() if not respiratory_data.empty else 0
    
    # 计算呼吸暂停指标
    # 假设呼吸频率为0或低于某个阈值表示呼吸暂停
    apnea_threshold = 5  # 呼吸频率低于5次/分钟认为是呼吸暂停
    apnea_events = daily_data[daily_data['respiratory_rate'] <= apnea_threshold]
    apnea_count = len(apnea_events)
    
    # 计算呼吸暂停持续时间（基于时间间隔）
    apnea_durations = []
    if not apnea_events.empty:
        apnea_events_sorted = apnea_events.sort_values('upload_time')
        for i in range(len(apnea_events_sorted) - 1):
            current_time = apnea_events_sorted.iloc[i]['upload_time']
            next_time = apnea_events_sorted.iloc[i + 1]['upload_time']
            duration = (next_time - current_time).total_seconds()
            # 如果时间间隔小于5分钟，认为是连续的呼吸暂停
            if duration <= 300:  # 5分钟
                apnea_durations.append(duration)
    
    max_apnea_duration = max(apnea_durations) if apnea_durations else 0
    avg_apnea_duration = np.mean(apnea_durations) if apnea_durations else 0
    
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


def analyze_single_day_physiological_data(date_str: str, table_name: str = "device_data"):
    """
    分析单日生理指标数据
    
    Args:
        date_str: 日期字符串，格式如 '2024-12-20'
        table_name: 数据库表名，默认为 "device_data"
    
    Returns:
        JSON格式的生理指标分析结果
    """
    try:
        # 使用数据库管理器
        db_manager = get_db_manager()
        
        # 查询指定日期的数据
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        
        logger.info(f"查询日期: {target_date.strftime('%Y-%m-%d')}")
        
        # 查询指定日期的数据
        query = f"""
        SELECT * FROM {table_name} 
        WHERE DATE(upload_time) = '{target_date.strftime('%Y-%m-%d')}'
        ORDER BY upload_time ASC
        """
        
        df = db_manager.execute_query(query)
        
        logger.info(f"查询到 {len(df)} 条数据")
        
        if df.empty:
            logger.warning(f"数据库中没有找到 {date_str} 的数据")
            return json.dumps({
                "error": f"数据库中没有找到 {date_str} 的数据",
                "date": date_str
            }, ensure_ascii=False, indent=2)
        
        # 转换时间列为datetime格式
        df['upload_time'] = pd.to_datetime(df['upload_time'])
        
        # 分析生理指标数据
        result = analyze_physiological_metrics(df, date_str)
        
        # 将numpy/pandas类型转换为原生Python类型以支持JSON序列化
        result = convert_numpy_types(result)
        
        logger.info(f"生理指标分析完成，结果: {result.get('summary', 'N/A')}")
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        import traceback
        error_msg = f"单日生理指标分析失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg}, ensure_ascii=False, indent=2)


@tool
def analyze_physiological_by_date(date: str, runtime: ToolRuntime = None, table_name: str = "device_data") -> str:
    """
    根据指定日期分析生理指标数据
    
    Args:
        date: 日期字符串，格式如 '2024-12-20'
        runtime: ToolRuntime 运行时上下文
        table_name: 数据库表名，默认为 "device_data"
    
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


# 测试函数
if __name__ == '__main__':
    # 示例：分析今天的生理指标数据
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"分析 {today} 的生理指标数据...")
    result = analyze_single_day_physiological_data(today)
    print(result)