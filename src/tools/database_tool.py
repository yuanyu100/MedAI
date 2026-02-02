"""
数据库查询工具，允许AI代理查询数据库以获取信息
"""
import json
from typing import Dict, Any, List
from langchain.tools import tool
from src.db.database import get_db_manager


@tool
def query_database_tables(runtime = None) -> str:
    """
    查询数据库中可用的表信息
    
    Returns:
        JSON格式的表信息列表
    """
    try:
        db_manager = get_db_manager()
        
        # 查询所有表
        tables_query = "SHOW TABLES"
        tables_df = db_manager.execute_query(tables_query)
        
        tables = []
        for _, row in tables_df.iterrows():
            table_name = list(row.values())[0]  # 获取表名
            tables.append(table_name)
        
        # 为每个表获取基本信息
        table_info = []
        for table_name in tables:
            try:
                schema_df = db_manager.get_table_schema(table_name)
                columns = []
                for _, col in schema_df.iterrows():
                    # 确保使用正确的列访问方式
                    col_dict = col.to_dict()  # 转换为字典以确保兼容性
                    columns.append({
                        "column_name": col_dict.get('Field', ''),
                        "data_type": col_dict.get('Type', ''),
                        "nullable": col_dict.get('Null', 'NO') == 'YES',
                        "key": col_dict.get('Key', '')
                    })
                
                # 获取表的记录数
                count_query = f"SELECT COUNT(*) as count FROM {table_name}"
                count_df = db_manager.execute_query(count_query)
                # 确保正确访问计数值
                if not count_df.empty and 'count' in count_df.columns:
                    record_count = int(count_df.iloc[0]['count'])
                else:
                    record_count = 0
                
                table_info.append({
                    "table_name": table_name,
                    "columns": columns,
                    "record_count": record_count
                })
            except Exception as e:
                table_info.append({
                    "table_name": table_name,
                    "error": str(e)
                })
        
        return json.dumps({
            "success": True,
            "tables": table_info
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False, indent=2)


@tool
def query_table_data(table_name: str, limit: int = 10, runtime = None) -> str:
    """
    查询指定表的数据
    
    Args:
        table_name: 表名
        limit: 返回记录数量限制，默认10条
    
    Returns:
        JSON格式的查询结果
    """
    try:
        db_manager = get_db_manager()
        
        # 验证表是否存在
        tables_query = "SHOW TABLES"
        tables_df = db_manager.execute_query(tables_query)
        available_tables = [list(row.values())[0] for _, row in tables_df.iterrows()]
        
        if table_name not in available_tables:
            return json.dumps({
                "success": False,
                "error": f"表 '{table_name}' 不存在"
            }, ensure_ascii=False, indent=2)
        
        # 查询表数据
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        result_df = db_manager.execute_query(query)
        
        # 转换为字典列表
        records = result_df.to_dict('records')
        
        return json.dumps({
            "success": True,
            "table_name": table_name,
            "record_count": len(records),
            "data": records
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False, indent=2)


@tool
def get_available_sleep_dates(runtime = None) -> str:
    """
    获取数据库中可用的睡眠数据日期列表
    
    Returns:
        JSON格式的日期列表
    """
    try:
        db_manager = get_db_manager()
        # 支持多个表名，优先使用 vital_signs，如果不存在则使用 device_data
        table_names_to_try = ["vital_signs", "device_data"]
        dates = []
        
        for table_name in table_names_to_try:
            try:
                dates = db_manager.get_available_dates(table_name)
                if dates:  # 如果找到了日期数据，就停止尝试其他表
                    break
            except:
                continue
        
        # 确保 dates 是列表格式
        if isinstance(dates, (list, tuple)):
            dates_list = list(dates)
        else:
            dates_list = []
        
        return json.dumps({
            "success": True,
            "dates": dates_list,
            "count": len(dates_list),
            "active_table": table_name if dates_list else "none"
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False, indent=2)


@tool  
def query_sleep_data_for_date(date: str, table_name: str = "vital_signs", runtime = None) -> str:
    """
    查询指定日期的睡眠数据
    
    Args:
        date: 日期字符串，格式如 '2024-12-20'
        table_name: 表名，默认为 'vital_signs'
    
    Returns:
        JSON格式的查询结果
    """
    try:
        db_manager = get_db_manager()
        
        # 如果没有指定表或表不存在，则尝试使用默认表
        if table_name == "vital_signs":
            # 检查 vital_signs 表是否存在，如果不存在则使用 device_data
            tables_query = "SHOW TABLES"
            tables_df = db_manager.execute_query(tables_query)
            available_tables = [list(row.values())[0] for _, row in tables_df.iterrows()]
            
            if "vital_signs" not in available_tables:
                table_name = "device_data"
        
        # 获取前一天和当天的数据（因为睡眠可能跨天）
        from datetime import datetime, timedelta
        target_date = datetime.strptime(date, '%Y-%m-%d')
        prev_date = target_date - timedelta(days=1)
        
        result_df = db_manager.get_sleep_data_for_date_range(
            table_name, 
            prev_date.strftime('%Y-%m-%d'), 
            target_date.strftime('%Y-%m-%d')
        )
        
        # 转换为字典列表
        records = result_df.to_dict('records')
        
        return json.dumps({
            "success": True,
            "date": date,
            "previous_date": prev_date.strftime('%Y-%m-%d'),
            "record_count": len(records),
            "data": records,
            "queried_table": table_name
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False, indent=2)


@tool
def create_vital_signs_table(runtime = None) -> str:
    """
    创建 vital_signs 表
    
    Returns:
        JSON格式的结果信息
    """
    try:
        db_manager = get_db_manager()
        db_manager.create_vital_signs_table()
        
        return json.dumps({
            "success": True,
            "message": "vital_signs 表创建成功"
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False, indent=2)


@tool
def insert_vital_signs_data(data: dict, runtime = None) -> str:
    """
    插入生命体征数据到 vital_signs 表
    
    Args:
        data: 包含生命体征数据的字典
    
    Returns:
        JSON格式的结果信息
    """
    try:
        db_manager = get_db_manager()
        
        # 验证数据一致性
        db_manager.validate_data_consistency(data)
        
        # 插入数据
        db_manager.insert_vital_signs_data(data)
        
        return json.dumps({
            "success": True,
            "message": "生命体征数据插入成功",
            "data_type": db_manager._determine_data_type(data)
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False, indent=2)


# 工具列表
DATABASE_TOOLS = [
    query_database_tables,
    query_table_data,
    get_available_sleep_dates,
    query_sleep_data_for_date,
    create_vital_signs_table,
    insert_vital_signs_data
]