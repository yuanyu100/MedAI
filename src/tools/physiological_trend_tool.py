#!/usr/bin/env python3
"""
生理指标趋势分析工具
用于生成心率和呼吸频率的时间序列数据，用于前端图表展示
"""

import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_physiological_trend_data(date_str: str, table_name: str = "vital_signs") -> str:
    """
    获取指定日期的生理指标趋势数据
    
    Args:
        date_str: 日期字符串，格式如 '2024-12-20'
        table_name: 数据库表名，默认为 "vital_signs"
    
    Returns:
        JSON格式的生理指标趋势数据
    """
    try:
        from src.db.database import get_db_manager
        db_manager = get_db_manager()
        
        # 解析日期
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        prev_date = target_date - timedelta(days=1)
        
        # 构建查询条件：获取从前一天21:00到当天07:00的数据
        start_time = prev_date.replace(hour=21, minute=0, second=0, microsecond=0)
        end_time = target_date.replace(hour=7, minute=0, second=0, microsecond=0)
        
        logger.info(f"查询时间范围: {start_time} 到 {end_time}")
        
        # 查询数据库 - 使用更精确的时间范围查询
        escaped_table_name = f"`{table_name.replace('`', '``')}`"
        query = f"""
        SELECT * 
        FROM {escaped_table_name} 
        WHERE {db_manager.time_col} BETWEEN '{start_time.strftime('%Y-%m-%d %H:%M:%S')}' AND '{end_time.strftime('%Y-%m-%d %H:%M:%S')}'
        ORDER BY {db_manager.time_col} ASC
        """
        df = db_manager.execute_query(query)
        
        logger.info(f"查询到 {len(df)} 条数据")
        
        if df.empty:
            return json.dumps({
                "error": f"数据库中没有找到 {prev_date.strftime('%Y-%m-%d')} 21:00 到 {target_date.strftime('%Y-%m-%d')} 07:00 的数据",
                "date": date_str
            }, ensure_ascii=False, indent=2)
        
        # 转换时间列为datetime格式
        df['upload_time'] = pd.to_datetime(df['upload_time'])
        
        # 由于查询已精确筛选时间范围，直接使用df作为filtered_df
        filtered_df = df.copy()
        
        if filtered_df.empty:
            return json.dumps({
                "error": f"指定时间范围内没有数据: {prev_date.strftime('%Y-%m-%d')} 21:00 到 {target_date.strftime('%Y-%m-%d')} 07:00",
                "date": date_str
            }, ensure_ascii=False, indent=2)
        
        # 按时间排序
        filtered_df = filtered_df.sort_values('upload_time').reset_index(drop=True)
        
        # 生成每5分钟的数据点
        trend_data = generate_5min_trend_data(filtered_df, start_time, end_time)
        
        return json.dumps(trend_data, ensure_ascii=False, indent=2)
    
    except Exception as e:
        import traceback
        error_msg = f"生理指标趋势分析失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg}, ensure_ascii=False, indent=2)


def generate_5min_trend_data(df: pd.DataFrame, start_time: datetime, end_time: datetime) -> Dict:
    """
    生成每5分钟的生理指标趋势数据
    
    Args:
        df: 原始数据DataFrame
        start_time: 开始时间
        end_time: 结束时间
    
    Returns:
        包含趋势数据的字典
    """
    # 创建时间索引，每5分钟一个点
    time_range = pd.date_range(start=start_time, end=end_time, freq='5min')
    
    # 准备返回数据结构
    result = {
        "data_points": []
    }
    
    # 对每个时间点，找到最近的数据或插值
    for time_point in time_range:
        # 找到距离当前时间点最近的原始数据
        time_diff = abs(df['upload_time'] - time_point)
        closest_idx = time_diff.idxmin()
        
        # 检查时间差是否在合理范围内（10分钟内），否则标记为无数据
        if time_diff[closest_idx] > pd.Timedelta(minutes=10):
            # 时间差距过大，认为该时间点无有效数据
            heart_rate = None
            respiratory_rate = None
        else:
            closest_row = df.loc[closest_idx]
            
            # 检查是否有有效的心率和呼吸率数据
            heart_rate = closest_row.get('heart_rate', None)
            respiratory_rate = closest_row.get('respiratory_rate', None)
            
            # 确保数值有效 - 处理数据库中VARCHAR类型的数值
            try:
                if pd.notna(heart_rate) and heart_rate != '' and heart_rate is not None:
                    numeric_heart_rate = float(heart_rate)
                    if numeric_heart_rate >= 0:
                        heart_rate = round(numeric_heart_rate, 1)
                    else:
                        heart_rate = None
                else:
                    heart_rate = None
            except (ValueError, TypeError):
                # 如果转换失败，设置为None
                heart_rate = None
                
            try:
                if pd.notna(respiratory_rate) and respiratory_rate != '' and respiratory_rate is not None:
                    numeric_respiratory_rate = float(respiratory_rate)
                    if numeric_respiratory_rate >= 0:
                        respiratory_rate = round(numeric_respiratory_rate, 1)
                    else:
                        respiratory_rate = None
                else:
                    respiratory_rate = None
            except (ValueError, TypeError):
                # 如果转换失败，设置为None
                respiratory_rate = None
        
        data_point = {
            "time": time_point.strftime('%Y-%m-%d %H:%M'),
            "heart_rate": heart_rate,
            "respiratory_rate": respiratory_rate
        }
        
        result["data_points"].append(data_point)
    
    # 按时间排序
    result["data_points"] = sorted(result["data_points"], key=lambda x: x["time"])
    
    # 添加统计信息
    valid_hr_data = [point["heart_rate"] for point in result["data_points"] if point["heart_rate"] is not None]
    valid_rr_data = [point["respiratory_rate"] for point in result["data_points"] if point["respiratory_rate"] is not None]
    
    result["statistics"] = {
        "heart_rate": {
            "count": len(valid_hr_data),
            "avg": round(sum(valid_hr_data) / len(valid_hr_data), 1) if valid_hr_data else 0,
            "min": min(valid_hr_data) if valid_hr_data else 0,
            "max": max(valid_hr_data) if valid_hr_data else 0
        },
        "respiratory_rate": {
            "count": len(valid_rr_data),
            "avg": round(sum(valid_rr_data) / len(valid_rr_data), 1) if valid_rr_data else 0,
            "min": min(valid_rr_data) if valid_rr_data else 0,
            "max": max(valid_rr_data) if valid_rr_data else 0
        }
    }
    
    return result


def get_physiological_trend_data_by_metric(date_str: str, metric: str, table_name: str = "vital_signs") -> str:
    """
    获取指定生理指标的趋势数据（用于前端分别获取心率或呼吸率数据）
    
    Args:
        date_str: 日期字符串
        metric: 指标类型 ('heart_rate' 或 'respiratory_rate')
        table_name: 数据库表名
    
    Returns:
        JSON格式的指定指标趋势数据
    """
    # 验证指标参数
    valid_metrics = ['heart_rate', 'respiratory_rate']
    if metric not in valid_metrics:
        return json.dumps({
            "metric": metric,
            "unit": "次/分",
            "error": f"无效的指标类型: {metric}. 有效值: {valid_metrics}",
            "date": date_str
        }, ensure_ascii=False, indent=2)
    
    try:
        from src.db.database import get_db_manager
        db_manager = get_db_manager()
        
        # 解析日期
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        prev_date = target_date - timedelta(days=1)
        
        # 构建查询条件：获取从前一天21:00到当天07:00的数据
        start_time = prev_date.replace(hour=21, minute=0, second=0, microsecond=0)
        end_time = target_date.replace(hour=7, minute=0, second=0, microsecond=0)
        
        logger.info(f"查询时间范围: {start_time} 到 {end_time}")
        
        # 查询数据库 - 使用更精确的时间范围查询
        escaped_table_name = f"`{table_name.replace('`', '``')}`"
        query = f"""
        SELECT * 
        FROM {escaped_table_name} 
        WHERE {db_manager.time_col} BETWEEN '{start_time.strftime('%Y-%m-%d %H:%M:%S')}' AND '{end_time.strftime('%Y-%m-%d %H:%M:%S')}'
        ORDER BY {db_manager.time_col} ASC
        """
        df = db_manager.execute_query(query)
        
        logger.info(f"查询到 {len(df)} 条数据")
        
        if df.empty:
            return json.dumps({
                "metric": metric,
                "unit": "次/分" if metric == "heart_rate" else "次/分",
                "error": f"数据库中没有找到 {prev_date.strftime('%Y-%m-%d')} 21:00 到 {target_date.strftime('%Y-%m-%d')} 07:00 的数据",
                "date": date_str
            }, ensure_ascii=False, indent=2)
        
        # 转换时间列为datetime格式
        df['upload_time'] = pd.to_datetime(df['upload_time'])
        
        # 由于查询已精确筛选时间范围，直接使用df作为filtered_df
        filtered_df = df.copy()
        
        if filtered_df.empty:
            return json.dumps({
                "metric": metric,
                "unit": "次/分" if metric == "heart_rate" else "次/分",
                "error": f"指定时间范围内没有数据: {prev_date.strftime('%Y-%m-%d')} 21:00 到 {target_date.strftime('%Y-%m-%d')} 07:00",
                "date": date_str
            }, ensure_ascii=False, indent=2)
        
        # 按时间排序
        filtered_df = filtered_df.sort_values('upload_time').reset_index(drop=True)
        
        # 生成每5分钟的数据点
        trend_data = generate_5min_trend_data_by_metric(filtered_df, start_time, end_time, metric)
        
        return json.dumps(trend_data, ensure_ascii=False, indent=2)
    
    except Exception as e:
        import traceback
        error_msg = f"生理指标趋势分析失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg}, ensure_ascii=False, indent=2)


def generate_5min_trend_data_by_metric(df: pd.DataFrame, start_time: datetime, end_time: datetime, metric: str) -> Dict:
    """
    生成每5分钟的指定生理指标趋势数据
    
    Args:
        df: 原始数据DataFrame
        start_time: 开始时间
        end_time: 结束时间
        metric: 指标类型 ('heart_rate' 或 'respiratory_rate')
    
    Returns:
        包含指定指标趋势数据的字典
    """
    # 创建时间索引，每5分钟一个点
    time_range = pd.date_range(start=start_time, end=end_time, freq='5min')
    
    # 准备返回数据结构
    unit_map = {
        "heart_rate": "次/分",
        "respiratory_rate": "次/分"
    }
    
    result = {
        "metric": metric,
        "unit": unit_map.get(metric, "次/分"),
        "data_points": []
    }
    
    # 对每个时间点，找到最近的数据或插值
    for time_point in time_range:
        # 找到距离当前时间点最近的原始数据
        time_diff = abs(df['upload_time'] - time_point)
        closest_idx = time_diff.idxmin()
        
        # 检查时间差是否在合理范围内（10分钟内），否则标记为无数据
        if time_diff[closest_idx] > pd.Timedelta(minutes=10):
            # 时间差距过大，认为该时间点无有效数据
            value = None
        else:
            closest_row = df.loc[closest_idx]
            
            # 获取指定指标的值
            if metric == "heart_rate":
                value = closest_row.get('heart_rate', None)
            elif metric == "respiratory_rate":
                value = closest_row.get('respiratory_rate', None)
            else:
                value = None
            
            # 确保数值有效 - 处理数据库中VARCHAR类型的数值
            try:
                if pd.notna(value) and value != '' and value is not None:
                    numeric_value = float(value)
                    if numeric_value >= 0:
                        value = round(numeric_value, 1)
                    else:
                        value = None
                else:
                    value = None
            except (ValueError, TypeError):
                # 如果转换失败，设置为None
                value = None
        
        # 无论是否有有效数据，都添加时间点
        data_point = {
            "time": time_point.strftime('%Y-%m-%d %H:%M'),
            "value": value
        }
        
        result["data_points"].append(data_point)
    
    # 按时间排序
    result["data_points"] = sorted(result["data_points"], key=lambda x: x["time"])
    
    # 添加统计信息
    valid_data = [point["value"] for point in result["data_points"] if point["value"] is not None]
    
    if valid_data:
        result["statistics"] = {
            "count": len(valid_data),
            "avg": round(sum(valid_data) / len(valid_data), 1),
            "min": min(valid_data),
            "max": max(valid_data)
        }
    else:
        result["statistics"] = {
            "count": 0,
            "avg": 0,
            "min": 0,
            "max": 0
        }
    
    return result


if __name__ == "__main__":
    # 测试函数
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # 示例：分析今天的生理指标趋势
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"分析 {today} 的生理指标趋势数据...")
    
    result = get_physiological_trend_data(today)
    print(result)