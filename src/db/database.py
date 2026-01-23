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


    def create_calculated_sleep_data_table(self):
        """创建用于存储计算得出的睡眠数据的表"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS calculated_sleep_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            date DATE NOT NULL COMMENT '日期',
            device_sn VARCHAR(100) COMMENT '设备序列号',
            bedtime DATETIME COMMENT '就寝时间',
            wakeup_time DATETIME COMMENT '起床时间',
            time_in_bed_minutes DECIMAL(10,2) DEFAULT 0 COMMENT '卧床时间（分钟）',
            sleep_duration_minutes DECIMAL(10,2) DEFAULT 0 COMMENT '睡眠时长（分钟）',
            sleep_score INT DEFAULT 0 COMMENT '睡眠评分',
            bed_exit_count INT DEFAULT 0 COMMENT '离床次数',
            sleep_prep_time_minutes INT DEFAULT 0 COMMENT '睡眠准备时间（分钟）',
            deep_sleep_minutes INT DEFAULT 0 COMMENT '深睡时长（分钟）',
            light_sleep_minutes INT DEFAULT 0 COMMENT '浅睡时长（分钟）',
            rem_sleep_minutes INT DEFAULT 0 COMMENT 'REM睡眠时长（分钟）',
            awake_minutes INT DEFAULT 0 COMMENT '清醒时长（分钟）',
            deep_sleep_percentage DECIMAL(5,2) DEFAULT 0 COMMENT '深睡占比（百分比）',
            light_sleep_percentage DECIMAL(5,2) DEFAULT 0 COMMENT '浅睡占比（百分比）',
            rem_sleep_percentage DECIMAL(5,2) DEFAULT 0 COMMENT 'REM睡眠占比（百分比）',
            awake_percentage DECIMAL(5,2) DEFAULT 0 COMMENT '清醒占比（百分比）',
            avg_heart_rate DECIMAL(5,2) DEFAULT 0 COMMENT '平均心率',
            avg_respiratory_rate DECIMAL(5,2) DEFAULT 0 COMMENT '平均呼吸率',
            min_heart_rate DECIMAL(5,2) DEFAULT 0 COMMENT '最低心率',
            max_heart_rate DECIMAL(5,2) DEFAULT 0 COMMENT '最高心率',
            min_respiratory_rate DECIMAL(5,2) DEFAULT 0 COMMENT '最低呼吸率',
            max_respiratory_rate DECIMAL(5,2) DEFAULT 0 COMMENT '最高呼吸率',
            apnea_count DECIMAL(10,2) DEFAULT 0 COMMENT '呼吸暂停次数',
            max_apnea_duration_seconds DECIMAL(10,2) DEFAULT 0 COMMENT '最长呼吸暂停时长（秒）',
            avg_apnea_duration_seconds DECIMAL(10,2) DEFAULT 0 COMMENT '平均呼吸暂停时长（秒）',
            respiratory_health_score DECIMAL(5,2) DEFAULT 0 COMMENT '呼吸健康评分',
            hrv_score DECIMAL(5,2) DEFAULT 0 COMMENT 'HRV分数',
            heart_rate_variability DECIMAL(10,2) DEFAULT 0 COMMENT '心率变异性原始值',
            heart_rate_stability DECIMAL(10,2) DEFAULT 0 COMMENT '心率稳定性评分',
            respiratory_stability DECIMAL(10,2) DEFAULT 0 COMMENT '呼吸稳定性评分',
            apnea_events_per_hour DECIMAL(10,2) DEFAULT 0 COMMENT '每小时呼吸暂停事件数',
            avg_body_moves_ratio DECIMAL(10,2) DEFAULT 0 COMMENT '平均体动占比',
            body_movement_frequency DECIMAL(10,2) DEFAULT 0 COMMENT '体动频率',
            sleep_efficiency DECIMAL(5,2) DEFAULT 0 COMMENT '睡眠效率百分比',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
            UNIQUE KEY uk_date_device (date, device_sn)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
        self.execute_command(create_table_sql)
        
        # 为已存在的表添加新列（如果列不存在）
        self._add_column_if_not_exists('calculated_sleep_data', 'heart_rate_variability', 'DECIMAL(10,2) DEFAULT 0 COMMENT \'心率变异性原始值\'')
        self._add_column_if_not_exists('calculated_sleep_data', 'heart_rate_stability', 'DECIMAL(10,2) DEFAULT 0 COMMENT \'心率稳定性评分\'')
        self._add_column_if_not_exists('calculated_sleep_data', 'respiratory_stability', 'DECIMAL(10,2) DEFAULT 0 COMMENT \'呼吸稳定性评分\'')
        self._add_column_if_not_exists('calculated_sleep_data', 'apnea_events_per_hour', 'DECIMAL(10,2) DEFAULT 0 COMMENT \'每小时呼吸暂停事件数\'')
        self._add_column_if_not_exists('calculated_sleep_data', 'avg_body_moves_ratio', 'DECIMAL(10,2) DEFAULT 0 COMMENT \'平均体动占比\'')
        self._add_column_if_not_exists('calculated_sleep_data', 'body_movement_frequency', 'DECIMAL(10,2) DEFAULT 0 COMMENT \'体动频率\'')
        self._add_column_if_not_exists('calculated_sleep_data', 'sleep_efficiency', 'DECIMAL(5,2) DEFAULT 0 COMMENT \'睡眠效率百分比\'')
    
    def _add_column_if_not_exists(self, table_name: str, column_name: str, column_definition: str):
        """如果列不存在则添加列"""
        try:
            check_sql = f"""
            SELECT COUNT(*) as cnt FROM information_schema.columns 
            WHERE table_schema = DATABASE() 
            AND table_name = '{table_name}' 
            AND column_name = '{column_name}'
            """
            result = self.execute_query(check_sql)
            if result.iloc[0]['cnt'] == 0:
                alter_sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
                self.execute_command(alter_sql)
        except Exception as e:
            # 忽略添加列时的错误（可能是列已存在）
            pass


    def create_sleep_stage_segments_table(self):
        """创建用于存储睡眠阶段细分数据的表"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS sleep_stage_segments (
            id INT AUTO_INCREMENT PRIMARY KEY,
            date DATE NOT NULL COMMENT '日期',
            device_sn VARCHAR(100) COMMENT '设备序列号',
            segment_order INT NOT NULL COMMENT '时间段顺序',
            label VARCHAR(20) NOT NULL COMMENT '睡眠阶段标签',
            value VARCHAR(20) NOT NULL COMMENT '时间段值（分钟）',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
            UNIQUE KEY uk_date_device_order (date, device_sn, segment_order)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
        self.execute_command(create_table_sql)


    def store_sleep_stage_segments(self, date: str, device_sn: str, segments: list):
        """存储睡眠阶段细分数据"""
        # 首先创建表（如果不存在）
        self.create_sleep_stage_segments_table()
        
        # 如果segments为空，则不执行任何操作
        if not segments:
            return
        
        # 清除旧的细分数据
        delete_sql = """
        DELETE FROM sleep_stage_segments 
        WHERE date = :date AND device_sn = :device_sn
        """
        delete_params = {'date': date, 'device_sn': device_sn}
        self.execute_command(delete_sql, delete_params)
        
        # 插入新的细分数据
        if segments:
            for idx, segment in enumerate(segments):
                insert_sql = """
                INSERT INTO sleep_stage_segments (
                    date, device_sn, segment_order, label, value
                ) VALUES (
                    :date, :device_sn, :segment_order, :label, :value
                )
                """
                insert_params = {
                    'date': date,
                    'device_sn': device_sn,
                    'segment_order': idx + 1,  # 从1开始计数
                    'label': segment.get('label', ''),
                    'value': str(segment.get('value', '0'))
                }
                self.execute_command(insert_sql, insert_params)


    def get_sleep_stage_segments(self, date: str, device_sn: str = None):
        """获取睡眠阶段细分数据"""
        # 首先确保表存在
        self.create_sleep_stage_segments_table()
        
        if device_sn:
            query = """
            SELECT * FROM sleep_stage_segments 
            WHERE date = :date AND device_sn = :device_sn
            ORDER BY segment_order ASC
            """
            params = {'date': date, 'device_sn': device_sn}
        else:
            query = """
            SELECT * FROM sleep_stage_segments 
            WHERE date = :date
            ORDER BY segment_order ASC
            """
            params = {'date': date}
        
        result = self.execute_query(query, params)
        return result


    def store_calculated_sleep_data(self, sleep_data: dict):
        """存储计算得出的睡眠数据"""
        # 首先创建表（如果不存在）
        self.create_calculated_sleep_data_table()
        
        # 准备插入数据
        insert_sql = """
        INSERT INTO calculated_sleep_data (
            date, device_sn, bedtime, wakeup_time, 
            time_in_bed_minutes, sleep_duration_minutes, sleep_score, bed_exit_count, 
            sleep_prep_time_minutes, deep_sleep_minutes, light_sleep_minutes, 
            rem_sleep_minutes, awake_minutes, deep_sleep_percentage, 
            light_sleep_percentage, rem_sleep_percentage, awake_percentage,
            avg_heart_rate, avg_respiratory_rate, min_heart_rate, max_heart_rate,
            min_respiratory_rate, max_respiratory_rate, apnea_count,
            max_apnea_duration_seconds, avg_apnea_duration_seconds, respiratory_health_score, hrv_score,
            heart_rate_variability, heart_rate_stability, respiratory_stability,
            apnea_events_per_hour, avg_body_moves_ratio, body_movement_frequency, sleep_efficiency
        ) VALUES (
            :date, :device_sn, :bedtime, :wakeup_time,
            :time_in_bed_minutes, :sleep_duration_minutes, :sleep_score, :bed_exit_count,
            :sleep_prep_time_minutes, :deep_sleep_minutes, :light_sleep_minutes,
            :rem_sleep_minutes, :awake_minutes, :deep_sleep_percentage,
            :light_sleep_percentage, :rem_sleep_percentage, :awake_percentage,
            :avg_heart_rate, :avg_respiratory_rate, :min_heart_rate, :max_heart_rate,
            :min_respiratory_rate, :max_respiratory_rate, :apnea_count,
            :max_apnea_duration_seconds, :avg_apnea_duration_seconds, :respiratory_health_score, :hrv_score,
            :heart_rate_variability, :heart_rate_stability, :respiratory_stability,
            :apnea_events_per_hour, :avg_body_moves_ratio, :body_movement_frequency, :sleep_efficiency
        )
        ON DUPLICATE KEY UPDATE
            -- 睡眠分析专属字段: 只在 bedtime 不为NULL时更新（表示这是/sleep-analysis调用）
            bedtime = IF(VALUES(bedtime) IS NOT NULL, VALUES(bedtime), bedtime),
            wakeup_time = IF(VALUES(bedtime) IS NOT NULL, VALUES(wakeup_time), wakeup_time),
            time_in_bed_minutes = IF(VALUES(bedtime) IS NOT NULL, VALUES(time_in_bed_minutes), time_in_bed_minutes),
            sleep_duration_minutes = IF(VALUES(bedtime) IS NOT NULL, VALUES(sleep_duration_minutes), sleep_duration_minutes),
            sleep_score = IF(VALUES(bedtime) IS NOT NULL, VALUES(sleep_score), sleep_score),
            bed_exit_count = IF(VALUES(bedtime) IS NOT NULL, VALUES(bed_exit_count), bed_exit_count),
            sleep_prep_time_minutes = IF(VALUES(bedtime) IS NOT NULL, VALUES(sleep_prep_time_minutes), sleep_prep_time_minutes),
            deep_sleep_minutes = IF(VALUES(bedtime) IS NOT NULL, VALUES(deep_sleep_minutes), deep_sleep_minutes),
            light_sleep_minutes = IF(VALUES(bedtime) IS NOT NULL, VALUES(light_sleep_minutes), light_sleep_minutes),
            rem_sleep_minutes = IF(VALUES(bedtime) IS NOT NULL, VALUES(rem_sleep_minutes), rem_sleep_minutes),
            awake_minutes = IF(VALUES(bedtime) IS NOT NULL, VALUES(awake_minutes), awake_minutes),
            deep_sleep_percentage = IF(VALUES(bedtime) IS NOT NULL, VALUES(deep_sleep_percentage), deep_sleep_percentage),
            light_sleep_percentage = IF(VALUES(bedtime) IS NOT NULL, VALUES(light_sleep_percentage), light_sleep_percentage),
            rem_sleep_percentage = IF(VALUES(bedtime) IS NOT NULL, VALUES(rem_sleep_percentage), rem_sleep_percentage),
            awake_percentage = IF(VALUES(bedtime) IS NOT NULL, VALUES(awake_percentage), awake_percentage),
            -- 生理分析专属字段: 只在 heart_rate_variability != 0 时更新（表示这是/physiological-analysis调用）
            heart_rate_variability = IF(VALUES(heart_rate_variability) != 0, VALUES(heart_rate_variability), heart_rate_variability),
            heart_rate_stability = IF(VALUES(heart_rate_variability) != 0, VALUES(heart_rate_stability), heart_rate_stability),
            respiratory_stability = IF(VALUES(heart_rate_variability) != 0, VALUES(respiratory_stability), respiratory_stability),
            apnea_events_per_hour = IF(VALUES(heart_rate_variability) != 0, VALUES(apnea_events_per_hour), apnea_events_per_hour),
            body_movement_frequency = IF(VALUES(heart_rate_variability) != 0, VALUES(body_movement_frequency), body_movement_frequency),
            sleep_efficiency = IF(VALUES(heart_rate_variability) != 0, VALUES(sleep_efficiency), sleep_efficiency),
            -- 共享字段: 只要新值不为0就更新（两个接口都可能提供）
            avg_heart_rate = IF(VALUES(avg_heart_rate) != 0, VALUES(avg_heart_rate), avg_heart_rate),
            avg_respiratory_rate = IF(VALUES(avg_respiratory_rate) != 0, VALUES(avg_respiratory_rate), avg_respiratory_rate),
            min_heart_rate = IF(VALUES(min_heart_rate) != 0, VALUES(min_heart_rate), min_heart_rate),
            max_heart_rate = IF(VALUES(max_heart_rate) != 0, VALUES(max_heart_rate), max_heart_rate),
            min_respiratory_rate = IF(VALUES(min_respiratory_rate) != 0, VALUES(min_respiratory_rate), min_respiratory_rate),
            max_respiratory_rate = IF(VALUES(max_respiratory_rate) != 0, VALUES(max_respiratory_rate), max_respiratory_rate),
            apnea_count = IF(VALUES(apnea_count) != 0, VALUES(apnea_count), apnea_count),
            max_apnea_duration_seconds = IF(VALUES(max_apnea_duration_seconds) != 0, VALUES(max_apnea_duration_seconds), max_apnea_duration_seconds),
            avg_apnea_duration_seconds = IF(VALUES(avg_apnea_duration_seconds) != 0, VALUES(avg_apnea_duration_seconds), avg_apnea_duration_seconds),
            respiratory_health_score = IF(VALUES(respiratory_health_score) != 0, VALUES(respiratory_health_score), respiratory_health_score),
            hrv_score = IF(VALUES(hrv_score) != 0, VALUES(hrv_score), hrv_score),
            avg_body_moves_ratio = IF(VALUES(avg_body_moves_ratio) != 0, VALUES(avg_body_moves_ratio), avg_body_moves_ratio),
            updated_at = CURRENT_TIMESTAMP
        """
        
        # 提取睡眠数据中的各个字段
        date = sleep_data.get('date')
        device_sn = sleep_data.get('device_sn', '')
        bedtime = sleep_data.get('bedtime')
        wakeup_time = sleep_data.get('wakeup_time')
        time_in_bed_minutes = sleep_data.get('time_in_bed_minutes', 0)
        sleep_duration_minutes = sleep_data.get('sleep_duration_minutes', 0)
        sleep_score = sleep_data.get('sleep_score', 0)
        bed_exit_count = sleep_data.get('bed_exit_count', 0)
        sleep_prep_time_minutes = sleep_data.get('sleep_prep_time_minutes', 0)
        
        # 从sleep_phases中提取数据
        sleep_phases = sleep_data.get('sleep_phases', {})
        deep_sleep_minutes = sleep_phases.get('deep_sleep_minutes', 0)
        light_sleep_minutes = sleep_phases.get('light_sleep_minutes', 0)
        rem_sleep_minutes = sleep_phases.get('rem_sleep_minutes', 0)
        awake_minutes = sleep_phases.get('awake_minutes', 0)
        deep_sleep_percentage = sleep_phases.get('deep_sleep_percentage', 0)
        light_sleep_percentage = sleep_phases.get('light_sleep_percentage', 0)
        rem_sleep_percentage = sleep_phases.get('rem_sleep_percentage', 0)
        awake_percentage = sleep_phases.get('awake_percentage', 0)
        
        # 从average_metrics中提取数据
        average_metrics = sleep_data.get('average_metrics', {})
        avg_heart_rate = average_metrics.get('avg_heart_rate', 0)
        avg_respiratory_rate = average_metrics.get('avg_respiratory_rate', 0)
        min_heart_rate = average_metrics.get('min_heart_rate', 0)
        max_heart_rate = average_metrics.get('max_heart_rate', 0)
        
        # 从呼吸指标respiratory_metrics中提取数据（如果有的话）
        respiratory_metrics = sleep_data.get('respiratory_metrics', {})
        min_respiratory_rate = respiratory_metrics.get('min_respiratory_rate', 0)
        max_respiratory_rate = respiratory_metrics.get('max_respiratory_rate', 0)
        apnea_count = respiratory_metrics.get('apnea_count', 0)
        max_apnea_duration_seconds = respiratory_metrics.get('max_apnea_duration_seconds', 0) or respiratory_metrics.get('max_apnea_duration', 0)
        avg_apnea_duration_seconds = respiratory_metrics.get('avg_apnea_duration_seconds', 0) or respiratory_metrics.get('avg_apnea_duration', 0)
        respiratory_health_score = respiratory_metrics.get('respiratory_health_score', 0)
        respiratory_stability = respiratory_metrics.get('respiratory_stability', 0)
        apnea_events_per_hour = respiratory_metrics.get('apnea_events_per_hour', 0)
        
        # 从心率指标heart_rate_metrics中提取数据（如果有的话）
        heart_rate_metrics = sleep_data.get('heart_rate_metrics', {})
        hrv_score = heart_rate_metrics.get('hrv_score', 0)
        heart_rate_variability = heart_rate_metrics.get('heart_rate_variability', 0)
        heart_rate_stability = heart_rate_metrics.get('heart_rate_stability', 0)
        # 也尝试从 heart_rate_metrics 中获取心率数据（PhysiologicalAnalysisResponse的结构）
        if not avg_heart_rate and heart_rate_metrics:
            avg_heart_rate = heart_rate_metrics.get('avg_heart_rate', 0)
            min_heart_rate = heart_rate_metrics.get('min_heart_rate', 0)
            max_heart_rate = heart_rate_metrics.get('max_heart_rate', 0)
        
        # 从睡眠指标sleep_metrics中提取数据（PhysiologicalAnalysisResponse特有）
        sleep_metrics = sleep_data.get('sleep_metrics', {})
        avg_body_moves_ratio = sleep_metrics.get('avg_body_moves_ratio', 0) or average_metrics.get('avg_body_moves_ratio', 0)
        body_movement_frequency = sleep_metrics.get('body_movement_frequency', 0)
        sleep_efficiency = sleep_metrics.get('sleep_efficiency', 0)
        
        params = {
            'date': date,
            'device_sn': device_sn,
            'bedtime': bedtime,
            'wakeup_time': wakeup_time,
            'time_in_bed_minutes': time_in_bed_minutes,
            'sleep_duration_minutes': sleep_duration_minutes,
            'sleep_score': sleep_score,
            'bed_exit_count': bed_exit_count,
            'sleep_prep_time_minutes': sleep_prep_time_minutes,
            'deep_sleep_minutes': deep_sleep_minutes,
            'light_sleep_minutes': light_sleep_minutes,
            'rem_sleep_minutes': rem_sleep_minutes,
            'awake_minutes': awake_minutes,
            'deep_sleep_percentage': deep_sleep_percentage,
            'light_sleep_percentage': light_sleep_percentage,
            'rem_sleep_percentage': rem_sleep_percentage,
            'awake_percentage': awake_percentage,
            'avg_heart_rate': avg_heart_rate,
            'avg_respiratory_rate': avg_respiratory_rate,
            'min_heart_rate': min_heart_rate,
            'max_heart_rate': max_heart_rate,
            'min_respiratory_rate': min_respiratory_rate,
            'max_respiratory_rate': max_respiratory_rate,
            'apnea_count': apnea_count,
            'max_apnea_duration_seconds': max_apnea_duration_seconds,
            'avg_apnea_duration_seconds': avg_apnea_duration_seconds,
            'respiratory_health_score': respiratory_health_score,
            'hrv_score': hrv_score,
            'heart_rate_variability': heart_rate_variability,
            'heart_rate_stability': heart_rate_stability,
            'respiratory_stability': respiratory_stability,
            'apnea_events_per_hour': apnea_events_per_hour,
            'avg_body_moves_ratio': avg_body_moves_ratio,
            'body_movement_frequency': body_movement_frequency,
            'sleep_efficiency': sleep_efficiency
        }
        
        # 执行INSERT语句存储主表数据
        self.execute_command(insert_sql, params)
        
        # 如果有sleep_stage_segments，也存储它们到单独的表
        sleep_stage_segments = sleep_data.get('sleep_stage_segments', [])
        if sleep_stage_segments and date and device_sn:
            self.store_sleep_stage_segments(date, device_sn, sleep_stage_segments)


    def get_calculated_sleep_data(self, date: str, device_sn: str = None):
        """获取计算得出的睡眠数据，包括睡眠阶段细分数据"""
        # 首先确保表存在
        self.create_calculated_sleep_data_table()
        
        if device_sn:
            query = """
            SELECT * FROM calculated_sleep_data 
            WHERE date = :date AND device_sn = :device_sn
            """
            params = {'date': date, 'device_sn': device_sn}
        else:
            query = """
            SELECT * FROM calculated_sleep_data 
            WHERE date = :date
            """
            params = {'date': date}
        
        result = self.execute_query(query, params)
        
        # 如果有结果，还需要获取对应的睡眠阶段细分数据
        if not result.empty:
            import pandas as pd
                
            # 为每一行添加睡眠阶段细分数据
            records = result.to_dict('records')
            for record in records:
                record_device_sn = record.get('device_sn')
                segments_df = self.get_sleep_stage_segments(date, record_device_sn)
                    
                if not segments_df.empty:
                    segments_list = []
                    for _, seg_row in segments_df.iterrows():
                        segment = {
                            'label': seg_row['label'],
                            'value': str(seg_row['value'])
                        }
                        segments_list.append(segment)
                    record['sleep_stage_segments'] = segments_list
                else:
                    record['sleep_stage_segments'] = []
                
            # 将处理后的数据转换回DataFrame
            if records:
                result = pd.DataFrame([records[0]]) if len(records) == 1 else pd.DataFrame(records)
            
        return result
        

# 全局数据库管理器实例
_db_manager = None


def get_db_manager():
    """获取数据库管理器实例"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
