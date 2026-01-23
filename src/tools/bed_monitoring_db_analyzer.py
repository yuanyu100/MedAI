import pandas as pd
import re
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import os
from sqlalchemy import create_engine, text
from langchain.tools import tool


class BedMonitoringDBAnalyzer:
    """病床监护数据库数据分析器 - 从数据库读取数据进行分析"""
    
    def __init__(self, table_name: str = "vital_signs", connection_string: str = None):
        """
        Args:
            table_name: 数据库表名，默认为 "vital_signs"
            connection_string: 数据库连接字符串，如果未提供则使用环境变量配置
        """
        if connection_string is None:
            # 从环境变量构建连接字符串
            host = os.getenv("MYSQL_HOST", "rm-bp1486k88hwq56s9x6o.mysql.rds.aliyuncs.com")
            port = os.getenv("MYSQL_PORT", "3306")
            user = os.getenv("MYSQL_USER", "thinglinksDB")
            password = os.getenv("MYSQL_PASSWORD", "thinglinksDB#2026")
            database = os.getenv("MYSQL_DATABASE", "thinglinks")
            
            connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"
        
        self.engine = create_engine(connection_string)
        self.table_name = table_name
        self.df = self._load_data_from_db()
        self.parsed_df = self._parse_all_data()
        
    def _load_data_from_db(self) -> pd.DataFrame:
        """从数据库加载数据"""
        # 首先尝试查询表的所有列，以确定实际的列名
        schema_query = f"DESCRIBE {self.table_name}"
        
        # 使用连接执行查询，避免pandas直接使用engine时可能触发的SQLite检查
        with self.engine.connect() as conn:
            result = conn.execute(text(schema_query))
            # 将结果转换为DataFrame
            schema_df = pd.DataFrame(result.fetchall(), columns=result.keys())
            columns = schema_df['Field'].tolist()
        
        # 确定实际的列名
        time_col = 'upload_time'
        type_col = 'data_type'
        
        # 尝试匹配常見的列名
        for col in columns:
            if 'time' in col.lower() or '时间' in col:
                time_col = col
            elif 'type' in col.lower() or '类型' in col:
                type_col = col
                
        # 额外检查是否包含中文字段名的变种
        for col in columns:
            if col in ['上传时间', 'upload_time', 'time', 'datetime', 'date_time', 'timestamp']:
                time_col = col
            elif col in ['数据类型', 'data_type', 'type', 'data_type_col']:
                type_col = col
        
        # 存储实际的列名
        self.time_col = time_col
        self.type_col = type_col
        
        # 执行查询 - 适配新的vital_signs表结构
        if 'vital_signs' in self.table_name.lower():
            # 新的vital_signs表结构，直接使用各字段
            query = f"""
            SELECT *, 
                   {time_col} as upload_time,
                   {type_col} as data_type
            FROM {self.table_name} 
            ORDER BY {time_col} ASC
            """
        else:
            # 旧的表结构，仍然使用原始解析方式
            query = f"SELECT * FROM {self.table_name} ORDER BY {time_col} ASC"
        
        # 使用连接执行查询，避免pandas直接使用engine时可能触发的SQLite检查
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            # 将结果转换为DataFrame
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df
        
    def _parse_all_data(self) -> pd.DataFrame:
        """解析所有数据"""
        # 检查是否是新的vital_signs表结构
        if 'vital_signs' in self.table_name.lower():
            return self._parse_vital_signs_data()
        else:
            # 使用原有的解析方式
            return self._parse_legacy_data()
    
    def _parse_vital_signs_data(self) -> pd.DataFrame:
        """解析vital_signs表数据"""
        # 将DataFrame中的列映射到内部使用的列名
        parsed_df = pd.DataFrame()
        
        # 时间字段
        if 'upload_time' in self.df.columns:
            parsed_df['upload_time'] = pd.to_datetime(self.df['upload_time'])
        elif self.time_col in self.df.columns:
            parsed_df['upload_time'] = pd.to_datetime(self.df[self.time_col])
        else:
            # 如果没有找到时间字段，使用当前时间
            parsed_df['upload_time'] = pd.to_datetime(['now'] * len(self.df))
        
        # 数据类型字段
        if 'data_type' in self.df.columns:
            parsed_df['data_type'] = self.df['data_type']
        elif self.type_col in self.df.columns:
            parsed_df['data_type'] = self.df[self.type_col]
        else:
            parsed_df['data_type'] = '未知'
        
        # 解析各个生命体征字段
        field_mappings = {
            'heart_rate': 'heart_rate',
            'respiratory_rate': 'respiration_rate',
            'avg_heartbeat_interval': 'hr_interval_avg',
            'rms_heartbeat_interval': 'hr_interval_rmsd',
            'std_heartbeat_interval': 'hr_interval_std',
            'arrhythmia_ratio': 'hr_irregular_ratio',
            'body_moves_ratio': 'body_move_ratio',
            'respiratory_pause_count': 'apnea_count',
            'is_person': 'is_in_bed',
            # 新增字段
            'breath_amp_average': 'breath_amp_avg',
            'heart_amp_average': 'heart_amp_avg',
            'breath_freq_std': 'breath_freq_std',
            'heart_freq_std': 'heart_freq_std',
            'breath_amp_diff': 'breath_amp_diff',
            'heart_amp_diff': 'heart_amp_diff',
            'person_id': 'person_id',
            'is_situp_alarm': 'is_situp_alarm',
            'is_off_bed_alarm': 'is_off_bed_alarm'
        }
        
        for db_field, internal_field in field_mappings.items():
            if db_field in self.df.columns:
                parsed_df[internal_field] = pd.to_numeric(self.df[db_field], errors='coerce')
            else:
                parsed_df[internal_field] = None
        
        # 特殊处理：如果is_person字段存在，将其转换为布尔值
        if 'is_person' in self.df.columns:
            parsed_df['is_in_bed'] = self.df['is_person'].astype(bool)
        
        # 排序
        parsed_df = parsed_df.sort_values('upload_time').reset_index(drop=True)
        
        return parsed_df
    
    def _parse_legacy_data(self) -> pd.DataFrame:
        """解析旧格式数据"""
        # 确保所需的列存在
        if self.content_col not in self.df.columns:
            raise KeyError(f"列 '{self.content_col}' 不存在于数据中。可用列: {list(self.df.columns)}")
        
        parsed_data = self.df[self.content_col].apply(self._parse_data_content_from_db)
        parsed_df = pd.DataFrame(parsed_data.tolist())
        
        # 添加时间戳和数据类型
        parsed_df['upload_time'] = pd.to_datetime(self.df[self.time_col])
        parsed_df['data_type'] = self.df[self.type_col]
        parsed_df['original_content'] = self.df[self.content_col]
        
        # 排序
        parsed_df = parsed_df.sort_values('upload_time').reset_index(drop=True)
        
        return parsed_df

    def _parse_data_content_from_db(self, content: str) -> Dict[str, Optional[int]]:
        """解析病床监护数据库数据内容字符串
        
        Args:
            content: 数据内容字符串，如"心率:60次/分钟;呼吸:15次/分钟;..."
        
        Returns:
            解析后的生理指标字典
        """
        data = {}
        
        # 提取心率
        hr_match = re.search(r'心率:(\d+)次/分钟', content)
        data['heart_rate'] = int(hr_match.group(1)) if hr_match else None
        
        # 提取呼吸
        rr_match = re.search(r'呼吸:(\d+)次/分钟', content)
        data['respiration_rate'] = int(rr_match.group(1)) if rr_match else None
        
        # 提取心跳间期平均值
        hr_avg_match = re.search(r'心跳间期平均值:(\d+)毫秒', content)
        data['hr_interval_avg'] = int(hr_avg_match.group(1)) if hr_avg_match else None
        
        # 提取心跳间期均方根值
        hr_rmsd_match = re.search(r'心跳间期均方根值:(\d+)毫秒', content)
        data['hr_interval_rmsd'] = int(hr_rmsd_match.group(1)) if hr_rmsd_match else None
        
        # 提取心跳间期标准差
        hr_std_match = re.search(r'心跳间期标准差:(\d+)毫秒', content)
        data['hr_interval_std'] = int(hr_std_match.group(1)) if hr_std_match else None
        
        # 提取心跳间期紊乱比例
        hr_irregular_match = re.search(r'心跳间期紊乱比例:(\d+)%', content)
        data['hr_irregular_ratio'] = int(hr_irregular_match.group(1)) if hr_irregular_match else None
        
        # 提取体动次数的占比
        body_move_match = re.search(r'体动次数的占比:(\d+)%', content)
        data['body_move_ratio'] = int(body_move_match.group(1)) if body_move_match else None
        
        # 提取呼吸暂停次数
        apnea_match = re.search(r'呼吸暂停次数:(\d+)次', content)
        data['apnea_count'] = int(apnea_match.group(1)) if apnea_match else 0
        
        return data

    def analyze_bed_occupancy(self) -> Dict[str, Any]:
        """FR 1.0: 病床占用状态分析"""
        
        # 提取状态数据（无人/有人）
        status_df = self.parsed_df[self.parsed_df['data_type'] == '状态'].copy()
        
        if len(status_df) == 0:
            # 如果没有状态数据，通过is_in_bed字段或心率/呼吸判断
            period_df = self.parsed_df[self.parsed_df['data_type'] == '周期数据'].copy()
            
            # 如果有is_in_bed字段，直接使用；否则通过心率/呼吸判断
            if 'is_in_bed' in self.parsed_df.columns and self.parsed_df['is_in_bed'].notna().any():
                period_df = self.parsed_df[
                    (self.parsed_df['data_type'] == '周期数据') | 
                    (self.parsed_df['data_type'] == '呼吸暂停')
                ].copy()
                period_df['is_in_bed'] = period_df['is_in_bed'].astype(bool)
            else:
                period_df['is_in_bed'] = (period_df['heart_rate'] > 0) | (period_df['respiration_rate'] > 0)
        else:
            # 从状态数据推断在床状态
            status_map = {'有人状态': True, '无人状态': False}
            period_df = self.parsed_df[self.parsed_df['data_type'] == '周期数据'].copy()
            
            # 为每条周期数据分配最近的状态
            period_df['is_in_bed'] = False
            for idx in range(len(period_df)):
                # 查找最近的状态记录
                row = period_df.iloc[idx]
                recent_status = status_df[status_df['upload_time'] <= row['upload_time']]
                if len(recent_status) > 0:
                    last_status_series = recent_status.iloc[-1]
                    # 如果last_status_series是Series
                    if hasattr(last_status_series, 'original_content'):
                        last_status = last_status_series['original_content']
                    else:
                        # 如果是标量值
                        last_status = str(last_status_series)
                    period_df.at[idx, 'is_in_bed'] = status_map.get(last_status, True)
        
        # 计算在床/离床时间
        period_df['time_diff'] = period_df['upload_time'].diff().fillna(pd.Timedelta(minutes=1))
        
        # 总卧床时间
        in_bed_segments = period_df[period_df['is_in_bed'] == True]
        total_in_bed_duration = in_bed_segments['time_diff'].sum()
        
        # 总离床时间
        out_of_bed_segments = period_df[period_df['is_in_bed'] == False]
        total_out_of_bed_duration = out_of_bed_segments['time_diff'].sum()
        
        # 检测长离床事件（>15分钟）
        long_out_events = []
        current_out_start = None
        
        for idx in range(len(period_df)):
            row = period_df.iloc[idx]
            if not row['is_in_bed'] and current_out_start is None:
                current_out_start = row['upload_time']
            elif row['is_in_bed'] and current_out_start is not None:
                out_duration = row['upload_time'] - current_out_start
                if out_duration > timedelta(minutes=15):
                    long_out_events.append({
                        'start_time': current_out_start.strftime('%H:%M'),
                        'end_time': row['upload_time'].strftime('%H:%M'),
                        'duration_minutes': int(out_duration.total_seconds() / 60)
                    })
                current_out_start = None
        
        # 夜间离床次数（22:00-06:00）
        night_start = datetime.strptime('22:00', '%H:%M').time()
        night_end = datetime.strptime('06:00', '%H:%M').time()
        
        night_out_periods = period_df[
            (~period_df['is_in_bed']) &
            (
                (period_df['upload_time'].dt.time >= night_start) |
                (period_df['upload_time'].dt.time < night_end)
            )
        ]
        
        # 统计夜间离床事件
        night_out_count = 0
        current_night_out = False
        for idx in range(len(night_out_periods)):
            row = night_out_periods.iloc[idx]
            if not row['is_in_bed'] and not current_night_out:
                night_out_count += 1
                current_night_out = True
            elif row['is_in_bed']:
                current_night_out = False
        
        return {
            'total_in_bed_hours': round(total_in_bed_duration.total_seconds() / 3600, 2),
            'total_out_of_bed_hours': round(total_out_of_bed_duration.total_seconds() / 3600, 2),
            'long_out_of_bed_events': long_out_events,
            'night_out_count': night_out_count,
            'bed_occupancy_rate': round(total_in_bed_duration.total_seconds() / 
                                       (total_in_bed_duration + total_out_of_bed_duration).total_seconds() * 100, 2)
        }
    
    def analyze_vital_signs(self) -> Dict[str, Any]:
        """FR 2.0: 临床级生命体征分析"""
        
        # 过滤有效数据（非零值），包括周期数据和呼吸暂停数据
        valid_data = self.parsed_df[
            ((self.parsed_df['data_type'] == '周期数据') | (self.parsed_df['data_type'] == '呼吸暂停')) &
            ((self.parsed_df['heart_rate'] > 0) | (self.parsed_df['respiration_rate'] > 0))
        ].copy()
        
        if len(valid_data) == 0:
            return {
                'heart_rate_stats': {'error': '没有有效的心率数据'},
                'respiratory_rate_stats': {'error': '没有有效的呼吸频率数据'},
                'abnormal_events': [],
                'hrv_trend': '无法计算，因缺少足够数据'
            }
        
        # 心率统计
        hr_values = valid_data['heart_rate'].dropna()
        if len(hr_values) > 0:
            hr_stats = {
                'min': int(hr_values.min()),
                'max': int(hr_values.max()),
                'avg': round(float(hr_values.mean()), 1),
                'std': round(float(hr_values.std()), 1),
                'range': f"{int(hr_values.min())}-{int(hr_values.max())} bpm"
            }
            
            # 识别心动过速/心动过缓
            tachycardia = len(valid_data[valid_data['heart_rate'] > 100])  # 心动过速 (>100 bpm)
            bradycardia = len(valid_data[valid_data['heart_rate'] < 60])   # 心动过缓 (<60 bpm)
            
            hr_abnormal = f"心动过速事件: {tachycardia}次, 心动过缓事件: {bradycardia}次"
        else:
            hr_stats = {'error': '没有心率数据'}
            hr_abnormal = "无心率数据"
        
        # 呼吸频率统计
        rr_values = valid_data['respiration_rate'].dropna()
        if len(rr_values) > 0:
            rr_stats = {
                'min': int(rr_values.min()),
                'max': int(rr_values.max()),
                'avg': round(float(rr_values.mean()), 1),
                'std': round(float(rr_values.std()), 1),
                'range': f"{int(rr_values.min())}-{int(rr_values.max())} 次/分钟"
            }
            
            # 识别呼吸过速/呼吸过缓
            tachypnea = len(valid_data[valid_data['respiration_rate'] > 20])  # 呼吸过速 (>20 次/分钟)
            bradypnea = len(valid_data[valid_data['respiration_rate'] < 12])  # 呼吸过缓 (<12 次/分钟)
            
            rr_abnormal = f"呼吸过速事件: {tachypnea}次, 呼吸过缓事件: {bradypnea}次"
        else:
            rr_stats = {'error': '没有呼吸频率数据'}
            rr_abnormal = "无呼吸频率数据"
        
        # 异常事件列表
        abnormal_events = []
        if tachycardia > 0:
            abnormal_events.append(f"心动过速 ({hr_stats['max']} bpm)")
        if bradycardia > 0:
            abnormal_events.append(f"心动过缓 ({hr_stats['min']} bpm)")
        if tachypnea > 0:
            abnormal_events.append(f"呼吸过速 ({rr_stats['max']} 次/分钟)")
        if bradypnea > 0:
            abnormal_events.append(f"呼吸过缓 ({rr_stats['min']} 次/分钟)")
        
        return {
            'heart_rate_stats': hr_stats,
            'respiratory_rate_stats': rr_stats,
            'abnormal_events': abnormal_events,
            'hrv_trend': '需要更复杂算法计算HRV趋势，当前版本仅提供基础统计'
        }
    
    def analyze_apnea(self) -> Dict[str, Any]:
        """FR 3.0: 呼吸暂停分析"""
        
        # 过滤包含呼吸暂停数据的记录
        apnea_data = self.parsed_df[
            self.parsed_df['apnea_count'].notna() & 
            (self.parsed_df['apnea_count'] >= 0)
        ].copy()
        
        if len(apnea_data) == 0:
            return {
                'total_apnea_events': 0,
                'ahi_index': 0,
                'risk_grade': '无法评估',
                'significant_events': []
            }
        
        # 总呼吸暂停事件数
        total_apnea_events = int(apnea_data['apnea_count'].sum())
        
        # 计算AHI指数 (Apnea-Hypopnea Index)
        # AHI = 每小时呼吸暂停和低通气事件总数
        total_recording_hours = (apnea_data['upload_time'].max() - apnea_data['upload_time'].min()).total_seconds() / 3600
        if total_recording_hours > 0:
            ahi_index = round(total_apnea_events / total_recording_hours, 2)
        else:
            ahi_index = 0
        
        # 风险分级
        if ahi_index < 5:
            risk_grade = "正常 (<5)"
            risk_level = "正常，无明显呼吸暂停"
        elif ahi_index < 15:
            risk_grade = "轻度 (5-14.9)"
            risk_level = "轻度阻塞性睡眠呼吸暂停，建议就医咨询"
        elif ahi_index < 30:
            risk_grade = "中度 (15-29.9)"
            risk_level = "中度阻塞性睡眠呼吸暂停，需要治疗"
        else:
            risk_grade = "重度 (>30)"
            risk_level = "重度阻塞性睡眠呼吸暂停，需要立即治疗"
        
        # 显著事件
        significant_events = []
        if total_apnea_events > 0:
            max_apnea_in_record = apnea_data['apnea_count'].max()
            if max_apnea_in_record > 0:
                max_apnea_time = apnea_data.loc[apnea_data['apnea_count'].idxmax(), 'upload_time']
                significant_events.append(f"最大单次暂停: {max_apnea_in_record}次 at {max_apnea_time}")
        
        return {
            'total_apnea_events': total_apnea_events,
            'ahi_index': ahi_index,
            'risk_grade': risk_grade,
            'risk_level': risk_level,
            'significant_events': significant_events
        }
    
    def analyze_body_movement(self) -> Dict[str, Any]:
        """FR 4.0: 体动与睡眠行为分析"""
        
        # 过滤包含体动数据的记录
        movement_data = self.parsed_df[
            self.parsed_df['body_move_ratio'].notna() & 
            (self.parsed_df['body_move_ratio'] >= 0)
        ].copy()
        
        if len(movement_data) == 0:
            return {
                'sleep_efficiency': '无法计算',
                'movement_ratio': '无数据',
                'high_movement_periods': [],
                'stability_assessment': '无数据'
            }
        
        # 睡眠效率（这里简化计算，实际可能需要更复杂的算法）
        # 假设体动少代表睡眠效率高
        avg_movement_ratio = movement_data['body_move_ratio'].mean()
        
        # 体动比率
        movement_stats = {
            'avg_ratio': f"{round(avg_movement_ratio, 1)}%",
            'min_ratio': f"{movement_data['body_move_ratio'].min()}%",
            'max_ratio': f"{movement_data['body_move_ratio'].max()}%"
        }
        
        # 高体动时段
        high_movement_threshold = avg_movement_ratio * 1.5  # 设定阈值为平均值的1.5倍
        high_movement_periods = movement_data[
            movement_data['body_move_ratio'] > high_movement_threshold
        ]
        
        high_movement_list = []
        for _, row in high_movement_periods.iterrows():
            high_movement_list.append({
                'time': row['upload_time'].strftime('%H:%M:%S'),
                'ratio': f"{row['body_move_ratio']}%"
            })
        
        # 稳定性评估
        movement_std = movement_data['body_move_ratio'].std()
        if movement_std < 5:
            stability = "睡眠稳定性良好，体动变化较小"
        elif movement_std < 10:
            stability = "睡眠稳定性一般，有一定体动变化"
        else:
            stability = "睡眠稳定性较差，体动变化较大"
        
        return {
            'sleep_efficiency': f"基于体动数据估算: {round(100 - avg_movement_ratio, 1)}%",
            'movement_stats': movement_stats,
            'high_movement_periods': high_movement_list,
            'stability_assessment': stability
        }
    
    def analyze_morning_assessment(self) -> Dict[str, Any]:
        """FR 5.0: 晨间评估"""
        
        # 获取早晨时段（05:00-07:00）的数据
        morning_start = self.parsed_df['upload_time'].dt.time >= datetime.strptime('05:00', '%H:%M').time()
        morning_end = self.parsed_df['upload_time'].dt.time <= datetime.strptime('07:00', '%H:%M').time()
        morning_data = self.parsed_df[morning_start & morning_end].copy()
        
        if len(morning_data) == 0:
            return {
                'morning_hr_trend': '该时段无数据',
                'morning_surge_detected': '无法检测',
                'wake_up_state_evaluation': '无数据'
            }
        
        # 早晨心率趋势
        morning_hr_values = morning_data['heart_rate'].dropna()
        if len(morning_hr_values) > 0:
            morning_avg_hr = morning_hr_values.mean()
            morning_min_hr = morning_hr_values.min()
            morning_max_hr = morning_hr_values.max()
            
            # 检测晨峰现象（早晨心率急剧升高）
            baseline_hr = self.parsed_df['heart_rate'].quantile(0.25)  # 使用25%分位数作为基线
            morning_surge_detected = bool(morning_avg_hr > baseline_hr * 1.2)  # 如果平均晨间心率比基线高20%以上
            
            # 苏醒状态评估
            if morning_surge_detected:
                wake_evaluation = f"检测到晨峰现象，心率从{morning_min_hr}升至{morning_max_hr}，苏醒反应强烈"
            else:
                wake_evaluation = f"苏醒过程平稳，晨间心率变化正常({morning_min_hr}-{morning_max_hr})"
            
            return {
                'morning_hr_trend': {
                    'avg': round(float(morning_avg_hr), 1),
                    'min': int(morning_min_hr),
                    'max': int(morning_max_hr)
                },
                'morning_surge_detected': morning_surge_detected,
                'wake_up_state_evaluation': wake_evaluation
            }
        else:
            return {
                'morning_hr_trend': '该时段无心率数据',
                'morning_surge_detected': '无法检测',
                'wake_up_state_evaluation': '无心率数据'
            }
    
    def generate_full_report(self) -> Dict[str, Any]:
        """生成完整的分析报告"""
        
        # 监测时段
        start_time = self.parsed_df['upload_time'].min().strftime('%Y-%m-%d %H:%M:%S')
        end_time = self.parsed_df['upload_time'].max().strftime('%Y-%m-%d %H:%M:%S')
        
        # 各项分析
        bed_occupancy = self.analyze_bed_occupancy()
        vital_signs = self.analyze_vital_signs()
        apnea = self.analyze_apnea()
        body_movement = self.analyze_body_movement()
        morning = self.analyze_morning_assessment()
        
        # 综合报告
        report = {
            'monitoring_period': {
                'start_time': start_time,
                'end_time': end_time,
                'total_hours': round((self.parsed_df['upload_time'].max() - 
                                     self.parsed_df['upload_time'].min()).total_seconds() / 3600, 2)
            },
            'bed_occupancy': bed_occupancy,
            'vital_signs': vital_signs,
            'apnea_analysis': apnea,
            'body_movement': body_movement,
            'morning_assessment': morning
        }
        
        return report


def analyze_bed_monitoring_from_db(table_name: str = "vital_signs", connection_string: str = None) -> str:
    """
    从数据库分析病床监护数据并生成护理交班报告的JSON格式分析结果
    
    Args:
        table_name: 数据库表名，默认为 "vital_signs"
        connection_string: 数据库连接字符串，如果未提供则使用环境变量配置
    
    Returns:
        包含所有分析结果的JSON字符串
    """
    try:
        analyzer = BedMonitoringDBAnalyzer(table_name, connection_string)
        report = analyzer.generate_full_report()
        return json.dumps(report, ensure_ascii=False, indent=2)
    except Exception as e:
        error_report = {
            'error': str(e),
            'error_type': type(e).__name__,
            'message': '数据库数据分析失败，请检查数据库连接和表结构'
        }
        return json.dumps(error_report, ensure_ascii=False, indent=2)


# 供langchain tool使用的包装函数
@tool
def bed_monitoring_db_analyzer_tool(table_name: str = "vital_signs", connection_string: str = None) -> str:
    """
    病床监护数据库数据分析工具
    
    Args:
        table_name: 数据库表名，默认为 "vital_signs"
        connection_string: 数据库连接字符串，如果未提供则使用环境变量配置
    
    Returns:
        包含以下内容的JSON分析报告:
        - 监测时段统计
        - 病床占用状态（卧床时间、离床时间、长离床事件、夜间离床次数）
        - 生命体征分析（心率、呼吸、异常事件、HRV趋势）
        - 呼吸暂停分析（AHI指数、风险分级、显著事件）
        - 体动与睡眠行为（睡眠效率、体动分析）
        - 晨间评估（晨起心率趋势、苏醒状态）
    """
    return analyze_bed_monitoring_from_db(table_name, connection_string)