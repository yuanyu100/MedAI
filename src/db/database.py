"""
数据库管理模块
"""
import os
import pandas as pd
from sqlalchemy import create_engine
from typing import Optional


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        # 从环境变量获取数据库配置
        host = os.getenv("MYSQL_HOST", "rm-bp1486k88hwq56s9x6o.mysql.rds.aliyuncs.com")
        port = os.getenv("MYSQL_PORT", "3306")
        user = os.getenv("MYSQL_USER", "thinglinksDB")
        password = os.getenv("MYSQL_PASSWORD", "thinglinksDB#2026")
        database = os.getenv("MYSQL_DATABASE", "thinglinks")
        
        # 创建数据库引擎
        self.connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"
        self.engine = create_engine(self.connection_string)
        
        # 默认时间列名
        self.time_col = 'upload_time'
    
    def execute_query(self, query: str, params: Optional[dict] = None):
        """执行查询"""
        if params:
            # 使用字符串格式化替代参数绑定，因为pandas.read_sql与SQLAlchemy参数绑定不兼容
            formatted_query = query
            for key, value in params.items():
                # 防止SQL注入，对值进行转义
                if isinstance(value, str):
                    escaped_value = value.replace("'", "''")
                    formatted_query = formatted_query.replace(f":{key}", f"'{escaped_value}'")
                else:
                    formatted_query = formatted_query.replace(f":{key}", str(value))
            return pd.read_sql(formatted_query, self.engine)
        else:
            return pd.read_sql(query, self.engine)
    
    def get_table_schema(self, table_name: str):
        """获取表结构"""
        query = f"DESCRIBE {table_name}"
        return self.execute_query(query)
    
    def get_available_dates(self, table_name: str):
        """获取表中可用的日期"""
        query = f"""
        SELECT DISTINCT DATE({self.time_col}) as date 
        FROM {table_name} 
        ORDER BY date DESC
        """
        result = self.execute_query(query)
        # 使用 values 属性替代 iterrows() 以避免 numpy 相关错误
        if 'date' in result.columns and not result.empty:
            return [str(date_val) for date_val in result['date'].values if pd.notna(date_val)]
        else:
            return []
    
    def get_sleep_data_for_date_range(self, table_name: str, start_date: str, end_date: str):
        """获取指定日期范围内的睡眠数据"""
        query = f"""
        SELECT * 
        FROM {table_name} 
        WHERE DATE({self.time_col}) BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY {self.time_col} ASC
        """
        return self.execute_query(query)


# 全局数据库管理器实例
_db_manager = None


def get_db_manager():
    """获取数据库管理器实例"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager