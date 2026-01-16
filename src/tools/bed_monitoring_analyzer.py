import pandas as pd  # type: ignore
import re
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


def parse_data_content(content: str) -> Dict[str, Optional[int]]:
    """解析病床监护数据内容字符串
    
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


class BedMonitoringAnalyzer:
    """病床监护数据分析器 - 实现FR 1.0-5.0"""
    
    def __init__(self, file_path: str):
        """
        Args:
            file_path: Excel文件路径
        """
        self.df = pd.read_excel(file_path)
        self.parsed_df = self._parse_all_data()
        
    def _parse_all_data(self) -> pd.DataFrame:  # type: ignore
        """解析所有数据"""
        parsed_data = self.df['数据内容'].apply(parse_data_content)
        parsed_df = pd.DataFrame(parsed_data.tolist())
        
        # 添加时间戳和数据类型
        parsed_df['upload_time'] = pd.to_datetime(self.df['上传时间'])
        parsed_df['data_type'] = self.df['数据类型']
        parsed_df['original_content'] = self.df['数据内容']
        
        # 排序
        parsed_df = parsed_df.sort_values('upload_time').reset_index(drop=True)
        
        return parsed_df
    
    def analyze_bed_occupancy(self) -> Dict[str, Any]:
        """FR 1.0: 病床占用状态分析"""
        
        # 提取状态数据（无人/有人）
        status_df = self.parsed_df[self.parsed_df['data_type'] == '状态'].copy()
        
        if len(status_df) == 0:
            # 如果没有状态数据，通过心率/呼吸是否为0判断
            period_df = self.parsed_df[self.parsed_df['data_type'] == '周期数据'].copy()
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
        
        # 过滤有效数据（非零值）
        valid_data = self.parsed_df[
            (self.parsed_df['heart_rate'] > 0) |
            (self.parsed_df['respiration_rate'] > 0)
        ].copy()
        
        if len(valid_data) == 0:
            return {
                'hr_min': 0, 'hr_max': 0, 'hr_avg': 0,
                'rr_min': 0, 'rr_max': 0, 'rr_avg': 0,
                'hr_abnormalities': [],
                'rr_abnormalities': [],
                'hrv_trend': 'stable'
            }
        
        # 心率统计
        hr_data = valid_data[valid_data['heart_rate'] > 0]['heart_rate']
        hr_min = int(hr_data.min())
        hr_max = int(hr_data.max())
        hr_avg = round(hr_data.mean(), 1)
        
        # 呼吸统计
        rr_data = valid_data[valid_data['respiration_rate'] > 0]['respiration_rate']
        rr_min = int(rr_data.min())
        rr_max = int(rr_data.max())
        rr_avg = round(rr_data.mean(), 1)
        
        # 心率异常检测
        hr_abnormalities = []
        
        # 心动过速（>100 bpm）- 持续检测
        tachycardia_periods = valid_data[valid_data['heart_rate'] > 100]
        for idx in range(len(tachycardia_periods)):
            row = tachycardia_periods.iloc[idx]
            hr_abnormalities.append({
                'type': '心动过速',
                'time': row['upload_time'].strftime('%H:%M'),
                'value': int(row['heart_rate']),
                'severity': 'high' if row['heart_rate'] > 120 else 'moderate'
            })
        
        # 心动过缓（<50 bpm）
        bradycardia_periods = valid_data[valid_data['heart_rate'] < 50]
        for idx in range(len(bradycardia_periods)):
            row = bradycardia_periods.iloc[idx]
            hr_abnormalities.append({
                'type': '心动过缓',
                'time': row['upload_time'].strftime('%H:%M'),
                'value': int(row['heart_rate']),
                'severity': 'high' if row['heart_rate'] < 40 else 'moderate'
            })
        
        # 呼吸异常检测
        rr_abnormalities = []
        
        # Bradypnea（<10 bpm）
        bradypnea_periods = valid_data[valid_data['respiration_rate'] < 10]
        for idx in range(len(bradypnea_periods)):
            row = bradypnea_periods.iloc[idx]
            rr_abnormalities.append({
                'type': '呼吸过缓',
                'time': row['upload_time'].strftime('%H:%M'),
                'value': int(row['respiration_rate']),
                'severity': 'high' if row['respiration_rate'] < 8 else 'moderate'
            })
        
        # Tachypnea（>25 bpm）
        tachypnea_periods = valid_data[valid_data['respiration_rate'] > 25]
        for idx in range(len(tachypnea_periods)):
            row = tachypnea_periods.iloc[idx]
            rr_abnormalities.append({
                'type': '呼吸过速',
                'time': row['upload_time'].strftime('%H:%M'),
                'value': int(row['respiration_rate']),
                'severity': 'high' if row['respiration_rate'] > 30 else 'moderate'
            })
        
        # HRV趋势分析（基于心跳间期紊乱比例）
        hrv_data = valid_data[valid_data['hr_irregular_ratio'].notna()]['hr_irregular_ratio']
        if len(hrv_data) > 0:
            avg_hrv = hrv_data.mean()
            hrv_trend = 'high_stress' if avg_hrv > 50 else 'moderate_stress' if avg_hrv > 20 else 'stable'
        else:
            hrv_trend = 'unknown'
        
        return {
            'hr_min': hr_min,
            'hr_max': hr_max,
            'hr_avg': hr_avg,
            'rr_min': rr_min,
            'rr_max': rr_max,
            'rr_avg': rr_avg,
            'hr_abnormalities': hr_abnormalities[:5],  # 最多返回5个异常
            'rr_abnormalities': rr_abnormalities[:5],
            'hrv_trend': hrv_trend
        }
    
    def analyze_apnea(self) -> Dict[str, Any]:
        """FR 3.0: 呼吸暂停综合分析"""
        
        # 统计总呼吸暂停次数
        valid_data = self.parsed_df[
            (self.parsed_df['heart_rate'] > 0) |
            (self.parsed_df['respiration_rate'] > 0)
        ].copy()
        
        total_apnea_count = valid_data['apnea_count'].sum()
        
        # 识别呼吸暂停事件
        apnea_events = []
        for idx in range(len(valid_data)):
            row = valid_data.iloc[idx]
            if row['apnea_count'] > 0:
                # 关联心率波动
                current_hr = row.get('heart_rate', 0)
                avg_hr = valid_data['heart_rate'].mean()
                
                # 检测心率代偿性升高（>10%）
                hr_compensation = 'yes' if current_hr > avg_hr * 1.1 else 'no'
                
                apnea_events.append({
                    'time': row['upload_time'].strftime('%H:%M'),
                    'count': int(row['apnea_count']),
                    'heart_rate': int(current_hr) if current_hr else 0,
                    'hr_compensation': hr_compensation
                })
        
        # 计算AHI指数（每小时暂停次数）
        monitoring_hours = (self.parsed_df['upload_time'].max() - 
                          self.parsed_df['upload_time'].min()).total_seconds() / 3600
        ahi = round(total_apnea_count / monitoring_hours, 2) if monitoring_hours > 0 else 0
        
        # 风险分级
        if ahi < 5:
            ahi_risk_level = 'normal'
        elif ahi < 15:
            ahi_risk_level = 'mild'
        elif ahi < 30:
            ahi_risk_level = 'moderate'
        else:
            ahi_risk_level = 'severe'
        
        # 识别显著事件（呼吸暂停+心率代偿性升高）
        significant_events = [
            event for event in apnea_events 
            if event['hr_compensation'] == 'yes'
        ]
        
        return {
            'total_apnea_count': int(total_apnea_count),
            'apnea_events': apnea_events,
            'significant_events': significant_events,
            'ahi_index': ahi,
            'ahi_risk_level': ahi_risk_level,
            'monitoring_hours': round(monitoring_hours, 2)
        }
    
    def analyze_body_movement(self) -> Dict[str, Any]:
        """FR 4.0: 体动与睡眠行为分析"""
        
        # 过滤有效数据
        valid_data = self.parsed_df[
            (self.parsed_df['heart_rate'] > 0) |
            (self.parsed_df['respiration_rate'] > 0)
        ].copy()
        
        if len(valid_data) == 0:
            return {
                'sleep_efficiency': 0,
                'body_move_avg': 0,
                'body_move_max': 0,
                'body_move_high_periods': [],
                'body_move_analysis': 'insufficient_data'
            }
        
        # 计算体动统计
        body_move_data = valid_data['body_move_ratio'].dropna()
        body_move_avg = round(body_move_data.mean(), 2)
        body_move_max = int(body_move_data.max())
        
        # 睡眠效率 = 总睡眠时间 / 总卧床时间
        # 这里假设卧床时间中体动<5%的时间为睡眠时间
        sleep_periods = valid_data[valid_data['body_move_ratio'] < 5]
        total_periods = valid_data
        
        if len(total_periods) > 0:
            sleep_efficiency = round(len(sleep_periods) / len(total_periods) * 100, 2)
        else:
            sleep_efficiency = 0
        
        # 识别高体动时期（体动>10%）
        high_body_move_periods = []
        for idx in range(len(valid_data)):
            row = valid_data.iloc[idx]
            if row['body_move_ratio'] > 10:
                high_body_move_periods.append({
                    'time': row['upload_time'].strftime('%H:%M'),
                    'body_move_ratio': int(row['body_move_ratio']),
                    'heart_rate': int(row.get('heart_rate', 0)) if row.get('heart_rate') else 0
                })
        
        # 体动分析
        if body_move_avg > 5:
            body_move_analysis = 'high - possible pain or restlessness'
        elif body_move_avg > 2:
            body_move_analysis = 'moderate - normal sleep transitions'
        else:
            body_move_analysis = 'low - stable sleep'
        
        return {
            'sleep_efficiency': sleep_efficiency,
            'body_move_avg': body_move_avg,
            'body_move_max': body_move_max,
            'body_move_high_periods': high_body_move_periods[:10],
            'body_move_analysis': body_move_analysis
        }
    
    def analyze_morning_assessment(self) -> Dict[str, Any]:
        """FR 5.0: 晨间评估"""
        
        # 筛选晨间时段（05:00-07:00）
        morning_df = self.parsed_df[
            (self.parsed_df['upload_time'].dt.time >= datetime.strptime('05:00', '%H:%M').time()) &
            (self.parsed_df['upload_time'].dt.time <= datetime.strptime('07:00', '%H:%M').time())
        ].copy()
        
        if len(morning_df) == 0:
            return {
                'morning_hr_trend': 'no_data',
                'morning_peak_detected': False,
                'waking_status': 'no_data'
            }
        
        # 有效数据
        valid_morning = morning_df[
            (morning_df['heart_rate'] > 0) |
            (morning_df['respiration_rate'] > 0)
        ].copy()
        
        if len(valid_morning) == 0:
            return {
                'morning_hr_trend': 'no_data',
                'morning_peak_detected': False,
                'waking_status': 'no_data'
            }
        
        # 晨起心率趋势
        hr_data = valid_morning['heart_rate'].dropna()
        if len(hr_data) > 2:
            hr_first_half = hr_data.iloc[:len(hr_data)//2].mean()
            hr_second_half = hr_data.iloc[len(hr_data)//2:].mean()
            
            if hr_second_half > hr_first_half * 1.2:
                morning_hr_trend = 'rising - possible morning hypertension'
            elif hr_second_half < hr_first_half * 0.8:
                morning_hr_trend = 'falling - hypotension risk'
            else:
                morning_hr_trend = 'stable'
        else:
            morning_hr_trend = 'insufficient_data'
        
        # 晨峰现象检测（心率突然升高>20%）
        morning_peak_detected = False
        morning_peak_time = None
        
        for i in range(1, len(hr_data)):
            if hr_data.iloc[i] > hr_data.iloc[i-1] * 1.2:
                morning_peak_detected = True
                morning_peak_time = valid_morning.iloc[i]['upload_time'].strftime('%H:%M')
                break
        
        # 苏醒状态
        avg_morning_hr = hr_data.mean()
        if avg_morning_hr > 100:
            waking_status = 'elevated - morning stress'
        elif avg_morning_hr < 50:
            waking_status = 'low - possible fatigue'
        else:
            waking_status = 'normal'
        
        return {
            'morning_hr_trend': morning_hr_trend,
            'morning_peak_detected': morning_peak_detected,
            'morning_peak_time': morning_peak_time,
            'waking_status': waking_status,
            'morning_avg_hr': round(avg_morning_hr, 1)
        }
    
    def generate_full_report(self) -> str:
        """生成完整的分析报告"""
        
        start_time = self.parsed_df['upload_time'].min().strftime('%Y-%m-%d %H:%M')
        end_time = self.parsed_df['upload_time'].max().strftime('%Y-%m-%d %H:%M')
        
        # 执行所有分析
        bed_occupancy = self.analyze_bed_occupancy()
        vital_signs = self.analyze_vital_signs()
        apnea = self.analyze_apnea()
        body_movement = self.analyze_body_movement()
        morning = self.analyze_morning_assessment()
        
        # 汇总报告
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
        
        return json.dumps(report, ensure_ascii=False, indent=2)


def analyze_bed_monitoring_data(file_path: str) -> str:
    """
    分析病床监护数据并生成护理交班报告的JSON格式分析结果
    
    Args:
        file_path: 病床监护数据Excel文件路径
    
    Returns:
        包含所有分析结果的JSON字符串
    """
    try:
        analyzer = BedMonitoringAnalyzer(file_path)
        report = analyzer.generate_full_report()
        return report
    except Exception as e:
        error_report = {
            'error': str(e),
            'error_type': type(e).__name__,
            'message': '数据分析失败，请检查文件格式和内容'
        }
        return json.dumps(error_report, ensure_ascii=False, indent=2)


# 供langchain tool使用的包装函数
def bed_monitoring_analyzer_tool(file_path: str = None, table_name: str = "device_data", use_database: bool = False) -> str:
    """
    病床监护数据分析工具
    
    Args:
        file_path: 病床监护数据Excel文件路径（当use_database为False时使用）
        table_name: 数据库表名，默认为 "device_data"（当use_database为True时使用）
        use_database: 是否使用数据库数据，默认为False
    
    Returns:
        包含以下内容的JSON分析报告:
        - 监测时段统计
        - 病床占用状态（卧床时间、离床时间、长离床事件、夜间离床次数）
        - 生命体征分析（心率、呼吸、异常事件、HRV趋势）
        - 呼吸暂停分析（AHI指数、风险分级、显著事件）
        - 体动与睡眠行为（睡眠效率、体动分析）
        - 晨间评估（晨起心率趋势、苏醒状态）
    """
    if use_database:
        from .bed_monitoring_db_analyzer import analyze_bed_monitoring_from_db
        return analyze_bed_monitoring_from_db(table_name)
    else:
        if file_path is None:
            error_report = {
                'error': 'file_path is required when use_database is False',
                'message': '请提供Excel文件路径或设置use_database=True以使用数据库'
            }
            return json.dumps(error_report, ensure_ascii=False, indent=2)
        return analyze_bed_monitoring_data(file_path)


if __name__ == '__main__':
    # 测试
    test_file = '/tmp/bed_monitoring_data.xlsx'
    result = analyze_bed_monitoring_data(test_file)
    print(result)
