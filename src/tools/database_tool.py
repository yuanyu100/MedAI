"""
数据库查询工具，允许AI代理查询数据库以获取信息
"""
import json
from typing import Dict, Any, List
from langchain.tools import tool
from langchain.tools import ToolRuntime
from src.db.database import get_db_manager


@tool
def query_database_tables(runtime: ToolRuntime = None) -> str:
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
def query_table_data(table_name: str, limit: int = 10, runtime: ToolRuntime = None) -> str:
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
def get_available_sleep_dates(runtime: ToolRuntime = None) -> str:
    """
    获取数据库中可用的睡眠数据日期列表
    
    Returns:
        JSON格式的日期列表
    """
    try:
        db_manager = get_db_manager()
        dates = db_manager.get_available_dates("device_data")
        
        # 确保 dates 是列表格式
        if isinstance(dates, (list, tuple)):
            dates_list = list(dates)
        else:
            dates_list = []
        
        return json.dumps({
            "success": True,
            "dates": dates_list,
            "count": len(dates_list)
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False, indent=2)


@tool  
def query_sleep_data_for_date(date: str, runtime: ToolRuntime = None) -> str:
    """
    查询指定日期的睡眠数据
    
    Args:
        date: 日期字符串，格式如 '2024-12-20'
    
    Returns:
        JSON格式的查询结果
    """
    try:
        db_manager = get_db_manager()
        
        # 获取前一天和当天的数据（因为睡眠可能跨天）
        from datetime import datetime, timedelta
        target_date = datetime.strptime(date, '%Y-%m-%d')
        prev_date = target_date - timedelta(days=1)
        
        result_df = db_manager.get_sleep_data_for_date_range(
            "device_data", 
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
            "data": records
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
    query_sleep_data_for_date
]