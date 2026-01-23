#!/usr/bin/env python3
"""
睡眠数据检查工具
用于判断是否存在前一天晚上的睡眠数据
"""

import pandas as pd
from datetime import datetime, timedelta, time
import json
import logging
from typing import Dict

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_previous_night_sleep_data(date_str: str, table_name: str = "vital_signs") -> str:
    """
    检查是否存在前一天晚上的睡眠数据
    
    Args:
        date_str: 日期字符串，格式如 '2024-12-20'
        table_name: 数据库表名，默认为 "vital_signs"
    
    Returns:
        JSON格式的检查结果
    """
    try:
        from src.db.database import get_db_manager
        db_manager = get_db_manager()
        
        # 解析日期
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        prev_date = target_date - timedelta(days=1)
        
        # 构建查询条件：检查前一天20:00到当天10:00的数据
        start_time = prev_date.replace(hour=20, minute=0, second=0, microsecond=0)
        end_time = target_date.replace(hour=10, minute=0, second=0, microsecond=0)
        
        logger.info(f"检查时间范围: {start_time} 到 {end_time}")
        
        # 查询数据库
        escaped_table_name = f"`{table_name.replace('`', '``')}`"
        query = f"""
        SELECT COUNT(*) as record_count
        FROM {escaped_table_name} 
        WHERE {db_manager.time_col} BETWEEN '{start_time.strftime('%Y-%m-%d %H:%M:%S')}' 
        AND '{end_time.strftime('%Y-%m-%d %H:%M:%S')}'
        """
        
        result_df = db_manager.execute_query(query)
        
        record_count = int(result_df.iloc[0]['record_count']) if not result_df.empty else 0
        
        logger.info(f"查询到 {record_count} 条数据")
        
        # 构建返回结果
        has_data = record_count > 0
        
        response = {
            "date": date_str,
            "check_period": {
                "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
                "end_time": end_time.strftime('%Y-%m-%d %H:%M:%S')
            },
            "has_sleep_data": has_data,
            "record_count": record_count,
            "message": f"{'存在' if has_data else '不存在'}前一天晚上(20:00-{target_date.strftime('%H:%M')})的睡眠数据"
        }
        
        return json.dumps(response, ensure_ascii=False, indent=2)

    except Exception as e:
        import traceback
        error_msg = f"检查睡眠数据失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return json.dumps({
            "error": error_msg,
            "date": date_str,
            "has_sleep_data": False
        }, ensure_ascii=False, indent=2)


def check_sleep_data_by_time_range(date_str: str, start_hour: int = 20, end_hour: int = 10, table_name: str = "vital_signs") -> str:
    """
    检查指定时间范围内的睡眠数据（可自定义时间范围）
    
    Args:
        date_str: 日期字符串，格式如 '2024-12-20'
        start_hour: 开始小时，默认为20（晚上8点）
        end_hour: 结束小时，默认为10（上午10点）
        table_name: 数据库表名，默认为 "vital_signs"
    
    Returns:
        JSON格式的检查结果
    """
    try:
        from src.db.database import get_db_manager
        db_manager = get_db_manager()
        
        # 解析日期
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        prev_date = target_date - timedelta(days=1)
        
        # 构建查询条件
        start_time = prev_date.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        end_time = target_date.replace(hour=end_hour, minute=0, second=0, microsecond=0)
        
        logger.info(f"检查时间范围: {start_time} 到 {end_time}")
        
        # 查询数据库
        escaped_table_name = f"`{table_name.replace('`', '``')}`"
        query = f"""
        SELECT COUNT(*) as record_count
        FROM {escaped_table_name} 
        WHERE {db_manager.time_col} BETWEEN '{start_time.strftime('%Y-%m-%d %H:%M:%S')}' 
        AND '{end_time.strftime('%Y-%m-%d %H:%M:%S')}'
        """
        
        result_df = db_manager.execute_query(query)
        
        record_count = int(result_df.iloc[0]['record_count']) if not result_df.empty else 0
        
        logger.info(f"查询到 {record_count} 条数据")
        
        # 构建返回结果
        has_data = record_count > 0
        
        response = {
            "date": date_str,
            "check_period": {
                "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
                "end_time": end_time.strftime('%Y-%m-%d %H:%M:%S'),
                "start_hour": start_hour,
                "end_hour": end_hour
            },
            "has_sleep_data": has_data,
            "record_count": record_count,
            "message": f"{'存在' if has_data else '不存在'}{start_hour}:00-{end_hour}:00期间的睡眠数据"
        }
        
        return json.dumps(response, ensure_ascii=False, indent=2)
    
    except Exception as e:
        import traceback
        error_msg = f"检查睡眠数据失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return json.dumps({
            "error": error_msg,
            "date": date_str,
            "has_sleep_data": False
        }, ensure_ascii=False, indent=2)


def check_detailed_sleep_data(date_str: str, table_name: str = "vital_signs") -> str:
    """
    详细检查前一天晚上的睡眠数据，包括是否有生理指标数据
    时间范围：前一天晚上8点到当天早上10点
    
    Args:
        date_str: 日期字符串，格式如 '2024-12-20'
        table_name: 数据库表名，默认为 "vital_signs"
    
    Returns:
        JSON格式的详细检查结果
    """
    try:
        from src.db.database import get_db_manager
        db_manager = get_db_manager()
        
        # 解析日期
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        prev_date = target_date - timedelta(days=1)
        
        # 构建查询条件：检查前一天晚上8点到当天早上10点的数据
        start_time = prev_date.replace(hour=20, minute=0, second=0, microsecond=0)
        end_time = target_date.replace(hour=10, minute=0, second=0, microsecond=0)
        
        logger.info(f"详细检查时间范围: {start_time} 到 {end_time}")
        
        # 查询数据库
        escaped_table_name = f"`{table_name.replace('`', '``')}`"
        query = f"""
        SELECT 
            COUNT(*) as total_records,
            COUNT(CASE WHEN heart_rate IS NOT NULL AND heart_rate > 0 THEN 1 END) as heart_rate_records,
            COUNT(CASE WHEN respiratory_rate IS NOT NULL AND respiratory_rate > 0 THEN 1 END) as respiratory_rate_records,
            MIN({db_manager.time_col}) as first_record_time,
            MAX({db_manager.time_col}) as last_record_time
        FROM {escaped_table_name} 
        WHERE {db_manager.time_col} BETWEEN :start_time AND :end_time
        """
        params = {
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        result_df = db_manager.execute_query(query, params)
        
        if result_df.empty or result_df.iloc[0]['total_records'] == 0:
            # 没有任何数据，返回一致的格式但数据为0
            from src.utils.response_handler import SleepDataCheckResponse
            response = SleepDataCheckResponse(
                success=True,
                date=date_str,
                check_period={
                    "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "end_time": end_time.strftime('%Y-%m-%d %H:%M:%S')
                },
                has_sleep_data=False,
                has_heart_rate_data=False,
                has_respiratory_rate_data=False,
                total_records=0,
                heart_rate_records=0,
                respiratory_rate_records=0,
                first_record_time=None,
                last_record_time=None,
                message=f"不存在{prev_date.strftime('%m-%d')} 20:00-{target_date.strftime('%H:%M')}期间的任何睡眠数据"
            )
        else:
            row = result_df.iloc[0]
            from src.utils.response_handler import SleepDataCheckResponse
            response = SleepDataCheckResponse(
                success=True,
                date=date_str,
                check_period={
                    "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "end_time": end_time.strftime('%Y-%m-%d %H:%M:%S')
                },
                has_sleep_data=int(row['total_records']) > 0,
                has_heart_rate_data=int(row['heart_rate_records']) > 0,
                has_respiratory_rate_data=int(row['respiratory_rate_records']) > 0,
                total_records=int(row['total_records']),
                heart_rate_records=int(row['heart_rate_records']),
                respiratory_rate_records=int(row['respiratory_rate_records']),
                first_record_time=row['first_record_time'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row['first_record_time']) else None,
                last_record_time=row['last_record_time'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row['last_record_time']) else None,
                message=f"存在{int(row['total_records'])}条睡眠数据记录"
            )
        
        return response.to_json()
    
    except Exception as e:
        import traceback
        error_msg = f"详细检查睡眠数据失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        
        from src.utils.response_handler import ApiResponse
        # 返回错误格式但保持一致的结构
        response = ApiResponse.error(
            error=str(e),
            message="检查睡眠数据失败",
            data={
                "date": date_str,
                "check_period": {
                    "start_time": f"{datetime.strptime(date_str, '%Y-%m-%d') - timedelta(days=1)} 20:00:00",
                    "end_time": f"{date_str} 10:00:00"
                },
                "has_sleep_data": False,
                "has_heart_rate_data": False,
                "has_respiratory_rate_data": False,
                "total_records": 0,
                "heart_rate_records": 0,
                "respiratory_rate_records": 0,
                "first_record_time": None,
                "last_record_time": None,
                "message": f"检查失败"
            }
        )
        return response.to_json()


def check_weekly_sleep_data(start_date_str: str, table_name: str = "vital_signs") -> str:
    """
    检查一周的睡眠数据
    
    Args:
        start_date_str: 开始日期字符串，格式如 '2024-12-20' (这将是周一)
        table_name: 数据库表名，默认为 "vital_signs"
    
    Returns:
        JSON格式的周检查结果数组
    """
    try:
        from src.db.database import get_db_manager
        db_manager = get_db_manager()
        
        import datetime
        from datetime import datetime, timedelta, time
        
        # 解析开始日期
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        
        # 确保是周一
        days_since_monday = start_date.weekday()  # Monday is 0
        start_date = start_date - timedelta(days=days_since_monday)
        
        weekly_results = []
        
        # 检查一周的每一天
        for i in range(7):
            current_date = start_date + timedelta(days=i)
            current_date_str = current_date.strftime('%Y-%m-%d')
            
            # 获取前一天日期（用于检查夜间数据）
            prev_date = current_date - timedelta(days=1)
            prev_date_str = prev_date.strftime('%Y-%m-%d')
            
            # 构建查询条件：检查前一天20:00到当天10:00的数据
            start_time = prev_date.replace(hour=20, minute=0, second=0, microsecond=0)
            end_time = current_date.replace(hour=10, minute=0, second=0, microsecond=0)
            
            # 查询数据库
            escaped_table_name = f"`{table_name.replace('`', '``')}`"
            query = f"""
            SELECT COUNT(*) as record_count
            FROM {escaped_table_name} 
            WHERE {db_manager.time_col} BETWEEN '{start_time.strftime('%Y-%m-%d %H:%M:%S')}' 
            AND '{end_time.strftime('%Y-%m-%d %H:%M:%S')}'
            """
            
            result_df = db_manager.execute_query(query)
            
            record_count = int(result_df.iloc[0]['record_count']) if not result_df.empty else 0
            
            # 构建每日结果
            daily_result = {
                "date": current_date_str,
                "prev_night_check": {
                    "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "end_time": end_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "start_hour": 20,
                    "end_hour": 10
                },
                "has_sleep_data": record_count > 0,
                "record_count": record_count,
                "day_of_week": current_date.strftime('%A'),  # 星期几
                "day_of_week_cn": ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][current_date.weekday()],
                "message": f"{'存在' if record_count > 0 else '不存在'}{prev_date.strftime('%m-%d')} 20:00-{current_date.strftime('%H:%M')}期间的睡眠数据"
            }
            
            weekly_results.append(daily_result)
        
        response = {
            "week_start_date": start_date.strftime('%Y-%m-%d'),
            "week_end_date": (start_date + timedelta(days=6)).strftime('%Y-%m-%d'),
            "weekly_summary": {
                "total_days": 7,
                "days_with_data": sum(1 for day in weekly_results if day["has_sleep_data"]),
                "days_without_data": sum(1 for day in weekly_results if not day["has_sleep_data"])
            },
            "daily_results": weekly_results
        }
        
        return json.dumps(response, ensure_ascii=False, indent=2)

    except Exception as e:
        import traceback
        error_msg = f"检查周睡眠数据失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return json.dumps({
            "error": error_msg,
            "week_start_date": start_date_str,
            "daily_results": []
        }, ensure_ascii=False, indent=2)


def check_recent_week_sleep_data_with_device(num_weeks: int = 1, device_sn: str = None, table_name: str = "vital_signs") -> str:
    """
    使用设备序列号检查最近几周的睡眠数据
    
    Args:
        num_weeks: 检查的周数，默认为1周
        device_sn: 设备序列号
        table_name: 数据库表名，默认为 "vital_signs"
    
    Returns:
        JSON格式的多周检查结果
    """
    try:
        import datetime
        from datetime import datetime, timedelta, time
        
        # 获取今天是星期几，然后计算上一个周一
        today = datetime.now()
        days_since_monday = today.weekday()  # Monday is 0
        start_of_current_week = today - timedelta(days=days_since_monday)
        
        # 计算要检查的第一周的开始日期
        first_week_start = start_of_current_week - timedelta(weeks=num_weeks-1)
        start_date_str = first_week_start.strftime('%Y-%m-%d')
        
        results = []
        for week in range(num_weeks):
            week_start = first_week_start + timedelta(weeks=week)
            week_start_str = week_start.strftime('%Y-%m-%d')
            
            # 根据是否有设备序列号来决定使用哪个函数
            if device_sn:
                week_result = check_weekly_sleep_data_with_device(week_start_str, device_sn, table_name)
            else:
                week_result = check_weekly_sleep_data(week_start_str, table_name)
                
            week_data = json.loads(week_result)
            
            results.append(week_data)
        
        response = {
            "period_summary": {
                "num_weeks_checked": num_weeks,
                "start_date": first_week_start.strftime('%Y-%m-%d'),
                "end_date": (start_of_current_week + timedelta(days=6)).strftime('%Y-%m-%d'),
                "device_sn": device_sn
            },
            "weekly_results": results
        }
        
        return json.dumps(response, ensure_ascii=False, indent=2)

    except Exception as e:
        import traceback
        error_msg = f"检查近期周睡眠数据失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return json.dumps({
            "error": error_msg,
            "num_weeks_checked": num_weeks,
            "device_sn": device_sn,
            "weekly_results": []
        }, ensure_ascii=False, indent=2)


def check_recent_week_sleep_data(num_weeks: int = 1, table_name: str = "vital_signs") -> str:
    """
    检查最近几周的睡眠数据
    
    Args:
        num_weeks: 检查的周数，默认为1周
        table_name: 数据库表名，默认为 "vital_signs"
    
    Returns:
        JSON格式的多周检查结果
    """
    try:
        import datetime
        from datetime import datetime, timedelta, time
        
        # 获取今天是星期几，然后计算上一个周一
        today = datetime.now()
        days_since_monday = today.weekday()  # Monday is 0
        start_of_current_week = today - timedelta(days=days_since_monday)
        
        # 计算要检查的第一周的开始日期
        first_week_start = start_of_current_week - timedelta(weeks=num_weeks-1)
        start_date_str = first_week_start.strftime('%Y-%m-%d')
        
        results = []
        for week in range(num_weeks):
            week_start = first_week_start + timedelta(weeks=week)
            week_start_str = week_start.strftime('%Y-%m-%d')
            
            week_result = check_weekly_sleep_data(week_start_str, table_name)
            week_data = json.loads(week_result)
            
            results.append(week_data)
        
        response = {
            "period_summary": {
                "num_weeks_checked": num_weeks,
                "start_date": first_week_start.strftime('%Y-%m-%d'),
                "end_date": (start_of_current_week + timedelta(days=6)).strftime('%Y-%m-%d')
            },
            "weekly_results": results
        }
        
        return json.dumps(response, ensure_ascii=False, indent=2)

    except Exception as e:
        import traceback
        error_msg = f"检查近期周睡眠数据失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return json.dumps({
            "error": error_msg,
            "num_weeks_checked": num_weeks,
            "weekly_results": []
        }, ensure_ascii=False, indent=2)


def check_previous_night_sleep_data_with_device(date_str: str, device_sn: str, table_name: str = "vital_signs") -> str:
    """
    使用设备序列号检查是否存在前一天晚上的睡眠数据
    
    Args:
        date_str: 日期字符串，格式如 '2024-12-20'
        device_sn: 设备序列号
        table_name: 数据库表名，默认为 "vital_signs"
    
    Returns:
        JSON格式的检查结果
    """
    try:
        from src.db.database import get_db_manager
        db_manager = get_db_manager()
        
        # 解析日期
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        prev_date = target_date - timedelta(days=1)
        
        # 构建查询条件：检查前一天20:00到当天10:00的数据
        start_time = prev_date.replace(hour=20, minute=0, second=0, microsecond=0)
        end_time = target_date.replace(hour=10, minute=0, second=0, microsecond=0)
        
        logger.info(f"检查时间范围: {start_time} 到 {end_time}，设备: {device_sn}")
        
        # 查询数据库 - 使用设备过滤
        escaped_table_name = f"`{table_name.replace('`', '``')}`"
        query = f"""
        SELECT COUNT(*) as record_count
        FROM {escaped_table_name} 
        WHERE {db_manager.time_col} BETWEEN '{start_time.strftime('%Y-%m-%d %H:%M:%S')}' 
        AND '{end_time.strftime('%Y-%m-%d %H:%M:%S')}'
        AND device_sn = :device_sn
        """
        params = {'device_sn': device_sn}
        
        result_df = db_manager.execute_query(query, params)
        
        record_count = int(result_df.iloc[0]['record_count']) if not result_df.empty else 0
        
        has_data = record_count > 0
        
        response = {
            "date": date_str,
            "check_period": {
                "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
                "end_time": end_time.strftime('%Y-%m-%d %H:%M:%S')
            },
            "has_sleep_data": has_data,
            "record_count": record_count,
            "device_sn": device_sn,
            "message": f"{'存在' if has_data else '不存在'}{prev_date.strftime('%m-%d')} 20:00-{target_date.strftime('%H:%M')}期间的睡眠数据(设备: {device_sn})"
        }
        
        return json.dumps(response, ensure_ascii=False, indent=2)

    except Exception as e:
        import traceback
        error_msg = f"检查睡眠数据失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return json.dumps({
            "error": error_msg,
            "date": date_str,
            "device_sn": device_sn,
            "has_sleep_data": False
        }, ensure_ascii=False, indent=2)


def check_detailed_sleep_data_with_device(date_str: str, device_sn: str, table_name: str = "vital_signs") -> str:
    """
    使用设备序列号详细检查前一天晚上的睡眠数据，包括是否有生理指标数据
    时间范围：前一天晚上8点到当天早上10点
    
    Args:
        date_str: 日期字符串，格式如 '2024-12-20'
        device_sn: 设备序列号
        table_name: 数据库表名，默认为 "vital_signs"
    
    Returns:
        JSON格式的详细检查结果
    """
    try:
        from src.utils.response_handler import SleepDataCheckResponse
        from src.db.database import get_db_manager
        db_manager = get_db_manager()
        
        # 解析日期
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        prev_date = target_date - timedelta(days=1)
        
        # 构建查询条件：检查前一天晚上8点到当天早上10点的数据
        start_time = prev_date.replace(hour=20, minute=0, second=0, microsecond=0)
        end_time = target_date.replace(hour=10, minute=0, second=0, microsecond=0)
        
        logger.info(f"详细检查时间范围: {start_time} 到 {end_time}，设备: {device_sn}")
        
        # 查询数据库 - 使用设备过滤
        escaped_table_name = f"`{table_name.replace('`', '``')}`"
        query = f"""
        SELECT 
            COUNT(*) as total_records,
            COUNT(CASE WHEN heart_rate IS NOT NULL AND heart_rate > 0 THEN 1 END) as heart_rate_records,
            COUNT(CASE WHEN respiratory_rate IS NOT NULL AND respiratory_rate > 0 THEN 1 END) as respiratory_rate_records,
            MIN({db_manager.time_col}) as first_record_time,
            MAX({db_manager.time_col}) as last_record_time
        FROM {escaped_table_name} 
        WHERE {db_manager.time_col} BETWEEN :start_time AND :end_time
        AND device_sn = :device_sn
        """
        params = {
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'device_sn': device_sn
        }
        
        result_df = db_manager.execute_query(query, params)
        
        if result_df.empty or result_df.iloc[0]['total_records'] == 0:
            # 没有任何数据，返回一致的格式但数据为0
            response = SleepDataCheckResponse(
                success=True,
                date=date_str,
                check_period={
                    "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "end_time": end_time.strftime('%Y-%m-%d %H:%M:%S')
                },
                has_sleep_data=False,
                has_heart_rate_data=False,
                has_respiratory_rate_data=False,
                total_records=0,
                heart_rate_records=0,
                respiratory_rate_records=0,
                first_record_time=None,
                last_record_time=None,
                device_sn=device_sn,
                message=f"不存在{prev_date.strftime('%m-%d')} 20:00-{target_date.strftime('%H:%M')}期间的任何睡眠数据(设备: {device_sn})"
            )
        else:
            row = result_df.iloc[0]
            response = SleepDataCheckResponse(
                success=True,
                date=date_str,
                check_period={
                    "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "end_time": end_time.strftime('%Y-%m-%d %H:%M:%S')
                },
                has_sleep_data=int(row['total_records']) > 0,
                has_heart_rate_data=int(row['heart_rate_records']) > 0,
                has_respiratory_rate_data=int(row['respiratory_rate_records']) > 0,
                total_records=int(row['total_records']),
                heart_rate_records=int(row['heart_rate_records']),
                respiratory_rate_records=int(row['respiratory_rate_records']),
                first_record_time=row['first_record_time'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row['first_record_time']) else None,
                last_record_time=row['last_record_time'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row['last_record_time']) else None,
                device_sn=device_sn,
                message=f"存在{int(row['total_records'])}条睡眠数据记录(设备: {device_sn})"
            )
        
        return response.to_json()
    
    except Exception as e:
        import traceback
        error_msg = f"详细检查睡眠数据失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        
        from src.utils.response_handler import ApiResponse
        # 返回错误格式但保持一致的结构
        response = ApiResponse.error(
            error=str(e),
            message="检查睡眠数据失败",
            data={
                "date": date_str,
                "check_period": {
                    "start_time": f"{datetime.strptime(date_str, '%Y-%m-%d') - timedelta(days=1)} 20:00:00",
                    "end_time": f"{date_str} 10:00:00"
                },
                "has_sleep_data": False,
                "has_heart_rate_data": False,
                "has_respiratory_rate_data": False,
                "total_records": 0,
                "heart_rate_records": 0,
                "respiratory_rate_records": 0,
                "first_record_time": None,
                "last_record_time": None,
                "device_sn": device_sn,
                "message": f"检查失败(设备: {device_sn})"
            }
        )
        return response.to_json()


def check_weekly_sleep_data_with_device(start_date_str: str, device_sn: str, table_name: str = "vital_signs") -> str:
    """
    使用设备序列号检查一周的睡眠数据
    
    Args:
        start_date_str: 开始日期字符串，格式如 '2024-12-20' (这将是周一)
        device_sn: 设备序列号
        table_name: 数据库表名，默认为 "vital_signs"
    
    Returns:
        JSON格式的周检查结果数组
    """
    try:
        from src.db.database import get_db_manager
        db_manager = get_db_manager()
        
        import datetime
        from datetime import datetime, timedelta, time
        
        # 解析开始日期
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        
        # 确保是周一
        days_since_monday = start_date.weekday()  # Monday is 0
        start_date = start_date - timedelta(days=days_since_monday)
        
        weekly_results = []
        
        # 检查一周的每一天
        for i in range(7):
            current_date = start_date + timedelta(days=i)
            current_date_str = current_date.strftime('%Y-%m-%d')
            
            # 获取前一天日期（用于检查夜间数据）
            prev_date = current_date - timedelta(days=1)
            prev_date_str = prev_date.strftime('%Y-%m-%d')
            
            # 构建查询条件：检查前一天20:00到当天10:00的数据
            start_time = prev_date.replace(hour=20, minute=0, second=0, microsecond=0)
            end_time = current_date.replace(hour=10, minute=0, second=0, microsecond=0)
            
            # 查询数据库 - 使用设备过滤
            escaped_table_name = f"`{table_name.replace('`', '``')}`"
            query = f"""
            SELECT COUNT(*) as record_count
            FROM {escaped_table_name} 
            WHERE {db_manager.time_col} BETWEEN :start_time 
            AND :end_time
            AND device_sn = :device_sn
            """
            params = {
                'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                'device_sn': device_sn
            }
            
            result_df = db_manager.execute_query(query, params)
            
            record_count = int(result_df.iloc[0]['record_count']) if not result_df.empty else 0
            
            # 构建每日结果
            daily_result = {
                "date": current_date_str,
                "prev_night_check": {
                    "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "end_time": end_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "start_hour": 20,
                    "end_hour": 10
                },
                "has_sleep_data": record_count > 0,
                "record_count": record_count,
                "device_sn": device_sn,
                "day_of_week": current_date.strftime('%A'),  # 星期几
                "day_of_week_cn": ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][current_date.weekday()],
                "message": f"{'存在' if record_count > 0 else '不存在'}{prev_date.strftime('%m-%d')} 20:00-{current_date.strftime('%H:%M')}期间的睡眠数据(设备: {device_sn})"
            }
            
            weekly_results.append(daily_result)
        
        response = {
            "week_start_date": start_date.strftime('%Y-%m-%d'),
            "week_end_date": (start_date + timedelta(days=6)).strftime('%Y-%m-%d'),
            "device_sn": device_sn,
            "weekly_summary": {
                "total_days": 7,
                "days_with_data": sum(1 for day in weekly_results if day["has_sleep_data"]),
                "days_without_data": sum(1 for day in weekly_results if not day["has_sleep_data"])
            },
            "daily_results": weekly_results
        }
        
        return json.dumps(response, ensure_ascii=False, indent=2)

    except Exception as e:
        import traceback
        error_msg = f"检查周睡眠数据失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return json.dumps({
            "error": error_msg,
            "week_start_date": start_date_str,
            "device_sn": device_sn,
            "daily_results": []
        }, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 测试函数
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # 示例：检查今天的前一天晚上是否有睡眠数据
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"检查 {today} 的前一天晚上睡眠数据...")
    
    result = check_previous_night_sleep_data(today)
    print(result)