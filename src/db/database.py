"""
数据库管理模块
"""
import pandas as pd
import pymysql
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime, timedelta
import logging
import json
import os
from typing import Optional, Dict, Any


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        # 从环境变量获取数据库配置
        host = os.getenv("MYSQL_HOST", "rm-bp1486k88hwq56s9x6o.mysql.rds.aliyuncs.com")
        port = os.getenv("MYSQL_PORT", "3306")
        user = os.getenv("MYSQL_USER", "geronova")
        password = os.getenv("MYSQL_PASSWORD", "EIN#2026geronova")
        database = os.getenv("MYSQL_DATABASE", "geronova")
        
        # 创建数据库引擎
        self.connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"
        self.engine = create_engine(self.connection_string)
        
        # 默认时间列名
        self.time_col = 'upload_time'
    
    def execute_query(self, query: str, params: Optional[dict] = None):
        """执行查询"""
        import pandas as pd
        if params:
            # 使用SQLAlchemy的text函数和参数绑定
            from sqlalchemy import text
            stmt = text(query)
            # 使用连接执行查询并返回结果
            with self.engine.connect() as conn:
                result = conn.execute(stmt, params)
                # 将结果转换为DataFrame
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
        else:
            # 使用连接执行查询，避免pandas直接使用engine时可能触发的SQLite检查
            from sqlalchemy import text
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                # 将结果转换为DataFrame
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
    
    def execute_command(self, command: str, params: Optional[dict] = None):
        """执行命令（INSERT, UPDATE, DELETE, CREATE等）"""
        from sqlalchemy import text
        with self.engine.connect() as conn:
            trans = conn.begin()  # 开始事务
            try:
                if params:
                    # 使用SQLAlchemy的text函数和参数绑定
                    stmt = text(command)
                    conn.execute(stmt, params)
                else:
                    conn.execute(text(command))
                trans.commit()  # 提交事务
            except Exception as e:
                trans.rollback()  # 发生错误时回滚
                raise e
    
    def get_table_schema(self, table_name: str):
        """获取表结构"""
        # 转义表名以防止SQL语法错误
        escaped_table_name = f"`{table_name.replace('`', '``')}`"
        query = f"DESCRIBE {escaped_table_name}"
        return self.execute_query(query)
    
    def get_available_dates(self, table_name: str):
        """获取表中可用的日期"""
        # 转义表名以防止SQL语法错误
        escaped_table_name = f"`{table_name.replace('`', '``')}`"
        query = f"""
        SELECT DISTINCT DATE({self.time_col}) as date 
        FROM {escaped_table_name} 
        ORDER BY date DESC
        """
        result = self.execute_query(query)
        # 使用 values 属性替代 iterrows() 以避免 numpy 相关错误
        if 'date' in result.columns and not result.empty:
            return [str(date_val) for date_val in result['date'].values if pd.notna(date_val)]
        else:
            return []
    
    def get_sleep_data_for_date_range(self, table_name: str, start_date: str, end_date: str, device_sn: str = None):
        """获取指定日期范围内的睡眠数据"""
        # 转义表名以防止SQL语法错误
        escaped_table_name = f"`{table_name.replace('`', '``')}`"
        
        if device_sn:
            # 如果提供了设备序列号，则在查询中加入设备过滤条件
            query = f"""
            SELECT * 
            FROM {escaped_table_name} 
            WHERE DATE({self.time_col}) BETWEEN '{start_date}' AND '{end_date}'
            AND device_sn = :device_sn
            ORDER BY {self.time_col} ASC
            """
            params = {'device_sn': device_sn}
            return self.execute_query(query, params)
        else:
            # 原来的查询逻辑，不包含设备过滤
            query = f"""
            SELECT * 
            FROM {escaped_table_name} 
            WHERE DATE({self.time_col}) BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY {self.time_col} ASC
            """
            return self.execute_query(query)
    
    def get_sleep_data_for_date_range_and_device(self, table_name: str, start_date: str, end_date: str, device_sn: str):
        """获取指定日期范围和设备的睡眠数据"""
        # 转义表名以防止SQL语法错误
        escaped_table_name = f"`{table_name.replace('`', '``')}`"
        
        query = f"""
        SELECT * 
        FROM {escaped_table_name} 
        WHERE DATE({self.time_col}) BETWEEN '{start_date}' AND '{end_date}'
        AND device_sn = :device_sn
        ORDER BY {self.time_col} ASC
        """
        params = {'device_sn': device_sn}
        return self.execute_query(query, params)
    
    def get_sleep_data_for_date_range_and_time(self, table_name: str, date_str: str, start_hour: int = 20, end_hour: int = 10):
        """
        获取指定日期范围内特定时间段的睡眠数据（默认为前一天晚上8点到当天早上10点）
        
        Args:
            table_name: 数据库表名
            date_str: 日期字符串，格式如 '2024-12-20'
            start_hour: 开始小时（默认20，即晚上8点）
            end_hour: 结束小时（默认10，即早上10点）
        
        Returns:
            DataFrame: 包含睡眠数据的DataFrame
        """
        # 解析日期
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        prev_date = target_date - timedelta(days=1)
        
        # 构建查询条件：检查前一天晚上start_hour点到当天早上end_hour点的数据
        start_time = prev_date.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        end_time = target_date.replace(hour=end_hour, minute=0, second=0, microsecond=0)
        
        # 构建SQL查询
        escaped_table_name = f"`{table_name.replace('`', '``')}`"
        query = f"""
        SELECT * 
        FROM {escaped_table_name} 
        WHERE {self.time_col} BETWEEN :start_time AND :end_time
        ORDER BY {self.time_col}
        """
        
        params = {
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 执行查询
        result_df = self.execute_query(query, params)
        return result_df
    
    def get_sleep_data_for_date_range_and_time_with_device(self, table_name: str, date_str: str, device_sn: str, start_hour: int = 20, end_hour: int = 10):
        """
        获取指定日期范围内特定时间段和特定设备的睡眠数据（默认为前一天晚上8点到当天早上10点）
        
        Args:
            table_name: 数据库表名
            date_str: 日期字符串，格式如 '2024-12-20'
            device_sn: 设备序列号
            start_hour: 开始小时（默认20，即晚上8点）
            end_hour: 结束小时（默认10，即早上10点）
        
        Returns:
            DataFrame: 包含睡眠数据的DataFrame
        """
        # 解析日期
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        prev_date = target_date - timedelta(days=1)
        
        # 构建查询条件：检查前一天晚上start_hour点到当天早上end_hour点的数据
        start_time = prev_date.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        end_time = target_date.replace(hour=end_hour, minute=0, second=0, microsecond=0)
        
        # 构建SQL查询，包含设备过滤条件
        escaped_table_name = f"`{table_name.replace('`', '``')}`"
        query = f"""
        SELECT * 
        FROM {escaped_table_name} 
        WHERE {self.time_col} BETWEEN :start_time AND :end_time
        AND device_sn = :device_sn
        ORDER BY {self.time_col}
        """
        
        params = {
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'device_sn': device_sn
        }
        
        # 执行查询
        result_df = self.execute_query(query, params)
        return result_df
    
    def create_vital_signs_table(self):
        """创建 vital_signs 表"""
        # 替换 SERIAL 为 AUTO_INCREMENT，适用于 MySQL
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS vital_signs (
            id INT AUTO_INCREMENT PRIMARY KEY,        -- 主键，自增ID
            data_type VARCHAR(50),                    -- 数据类型：周期数据/状态/呼吸暂停
            upload_time TIMESTAMP,                    -- 上传时间
            device_name VARCHAR(100),                 -- 来源设备名称
            device_sn VARCHAR(100),                   -- 来源设备SN码
            organization VARCHAR(100),                -- 所属组织
            is_person INTEGER,                        -- 是否有人：1=有人，0=无人
            heart_rate VARCHAR(20),                   -- 心率（次/分钟）
            respiratory_rate VARCHAR(20),             -- 呼吸频率（次/分钟）
            avg_heartbeat_interval VARCHAR(20),       -- 心跳间期平均值（毫秒）
            rms_heartbeat_interval VARCHAR(20),       -- 心跳间期均方根值（毫秒）
            std_heartbeat_interval VARCHAR(20),       -- 心跳间期标准差（毫秒）
            arrhythmia_ratio VARCHAR(20),             -- 心跳间期紊乱比例（%）
            body_moves_ratio VARCHAR(20),             -- 体动次数占比（%）
            respiratory_pause_count VARCHAR(20),      -- 呼吸暂停次数
            avg_pause_duration VARCHAR(20),           -- 平均呼吸暂停时长（秒）
            max_pause_duration VARCHAR(20),           -- 最大呼吸暂停时长（秒）
            breath_amp_average VARCHAR(32) DEFAULT NULL COMMENT '呼吸幅度均值',
            heart_amp_average VARCHAR(32) DEFAULT NULL COMMENT '心跳幅度均值',
            breath_freq_std VARCHAR(32) DEFAULT NULL COMMENT '呼吸频率标准差',
            heart_freq_std VARCHAR(32) DEFAULT NULL COMMENT '心跳频率标准差',
            breath_amp_diff VARCHAR(32) DEFAULT NULL COMMENT '呼吸幅度差值',
            heart_amp_diff VARCHAR(32) DEFAULT NULL COMMENT '心跳幅度差值',
            person_id BIGINT DEFAULT NULL COMMENT '用户Id',
            is_situp_alarm TINYINT(1) NOT NULL DEFAULT '0' COMMENT '是否坐起告警，0否1是',
            is_off_bed_alarm TINYINT(1) NOT NULL DEFAULT '0' COMMENT '是否离床告警，0否1是'
        );
        """
        self.execute_command(create_table_sql)
        
        # 创建索引以提高查询性能
        self.create_indexes(table_name)
    
    def insert_vital_signs_data(self, data):
        """插入生命体征数据"""
        # 根据数据内容判断 data_type
        data_type = self._determine_data_type(data)
        
        # 准备插入数据
        insert_sql = """
        INSERT INTO vital_signs (
            data_type, upload_time, device_name, device_sn, organization, 
            is_person, heart_rate, respiratory_rate, avg_heartbeat_interval, 
            rms_heartbeat_interval, std_heartbeat_interval, arrhythmia_ratio, 
            body_moves_ratio, respiratory_pause_count, avg_pause_duration, max_pause_duration,
            breath_amp_average, heart_amp_average, breath_freq_std, heart_freq_std,
            breath_amp_diff, heart_amp_diff, person_id, is_situp_alarm, is_off_bed_alarm
        ) VALUES (
            :data_type, :upload_time, :device_name, :device_sn, :organization,
            :is_person, :heart_rate, :respiratory_rate, :avg_heartbeat_interval,
            :rms_heartbeat_interval, :std_heartbeat_interval, :arrhythmia_ratio,
            :body_moves_ratio, :respiratory_pause_count, :avg_pause_duration, :max_pause_duration,
            :breath_amp_average, :heart_amp_average, :breath_freq_std, :heart_freq_std,
            :breath_amp_diff, :heart_amp_diff, :person_id, :is_situp_alarm, :is_off_bed_alarm
        )
        """
        
        params = {
            'data_type': data_type,
            'upload_time': data.get('upload_time'),
            'device_name': data.get('device_name', ''),
            'device_sn': data.get('device_sn', ''),
            'organization': data.get('organization', ''),
            'is_person': data.get('is_person', 0),
            'heart_rate': data.get('heart_rate', ''),
            'respiratory_rate': data.get('respiratory_rate', ''),
            'avg_heartbeat_interval': data.get('avg_heartbeat_interval', ''),
            'rms_heartbeat_interval': data.get('rms_heartbeat_interval', ''),
            'std_heartbeat_interval': data.get('std_heartbeat_interval', ''),
            'arrhythmia_ratio': data.get('arrhythmia_ratio', ''),
            'body_moves_ratio': data.get('body_moves_ratio', ''),
            'respiratory_pause_count': data.get('respiratory_pause_count', ''),
            'avg_pause_duration': data.get('avg_pause_duration', ''),
            'max_pause_duration': data.get('max_pause_duration', ''),
            'breath_amp_average': data.get('breath_amp_average', None),
            'heart_amp_average': data.get('heart_amp_average', None),
            'breath_freq_std': data.get('breath_freq_std', None),
            'heart_freq_std': data.get('heart_freq_std', None),
            'breath_amp_diff': data.get('breath_amp_diff', None),
            'heart_amp_diff': data.get('heart_amp_diff', None),
            'person_id': data.get('person_id', None),
            'is_situp_alarm': data.get('is_situp_alarm', 0),
            'is_off_bed_alarm': data.get('is_off_bed_alarm', 0)
        }
        
        self.execute_command(insert_sql, params)
    
    def _determine_data_type(self, data):
        """根据数据内容确定数据类型"""
        # 检查数据内容中是否包含 "呼吸暂停" 关键词
        data_content_str = ' '.join(str(v) for v in data.values() if v is not None)
        
        if '呼吸暂停' in data_content_str:
            return '呼吸暂停'
        elif data.get('heart_rate') is not None and data.get('respiratory_rate') is not None:
            # 如果有心率和呼吸率数据，则为周期数据
            return '周期数据'
        elif 'is_person' in data and data.get('heart_rate') is None:
            # 如果只有 is_person 字段有效，则为状态数据
            return '状态'
        else:
            return '周期数据'  # 默认为周期数据
    
    def validate_data_consistency(self, data):
        """验证数据一致性，确保 is_person 和 heart_rate 的一致性"""
        is_person = data.get('is_person')
        heart_rate = data.get('heart_rate')
        
        if is_person is not None and heart_rate is not None:
            # 将 heart_rate 转换为数字进行比较
            try:
                hr_value = float(heart_rate) if heart_rate != '' and heart_rate is not None else 0
            except ValueError:
                hr_value = 0
                
            is_person_int = int(is_person) if is_person is not None else 0
            
            # 检查一致性
            if is_person_int == 1 and hr_value == 0:
                raise ValueError("错误：有人状态下心率不应为0")
            elif is_person_int == 0 and hr_value > 0:
                raise ValueError("错误：无人状态下心率应为0")
        
        return True
    
    def create_indexes(self, table_name: str):
        """为表创建索引以提高查询性能"""
        try:
            # 为upload_time字段创建索引，提高日期范围查询性能
            escaped_table_name = f"`{table_name.replace('`', '``')}`"
            index_query = f"CREATE INDEX IF NOT EXISTS idx_{table_name}_upload_time ON {escaped_table_name} ({self.time_col})"
            self.execute_command(index_query)
            
            # 为DATE(upload_time)创建索引，提高日期查询性能
            index_date_query = f"CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON {escaped_table_name} (DATE({self.time_col}))"
            self.execute_command(index_date_query)
        except Exception as e:
            # 如果索引创建失败（可能是权限问题等），不影响主流程
            print(f"创建索引时出错（可忽略）: {e}")
        

# 全局数据库管理器实例
_db_manager = None


def get_db_manager():
    """获取数据库管理器实例"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
