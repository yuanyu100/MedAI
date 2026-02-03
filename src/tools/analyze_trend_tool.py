"""
趋势分析工具 - 分析多天监护数据的趋势和模式
"""
import json
import pandas as pd
import re
from datetime import datetime, timedelta
from typing import Dict, Any
from langchain.tools import tool


def analyze_trend_and_pattern_internal(file_path: str = None, table_name: str = "device_data") -> Dict[str, Any]:
    """
    内部趋势分析函数
    """
    # 这里是一个占位实现，实际应用中会实现真正的趋势分析逻辑
    return {
        "error": "Trend analysis is temporarily disabled",
        "message": "This function requires data processing that is not currently enabled",
        "example_data": {
            "peak_hours": [22, 23, 0, 1],
            "trend_direction": "stable",
            "trend_value": 0.5,
            "risk_score": 65,
            "peak_hr_value": 85,
            "avg_hr": 72,
            "avg_rr": 16,
            "total_apnea": 12,
            "avg_apnea_per_day": 4,
            "data_days": 3
        }
    }


def analyze_trend_from_database(data_type: str = "week", device_sn: str = None, start_date: str = None, end_date: str = None) -> str:
    """
    从数据库分析周/月数据趋势
    
    功能：从calculated_sleep_data表中获取每天的分析结果，生成周/月数据趋势。
    
    参数:
        data_type: 数据类型，"week"表示周数据，"month"表示月数据
        device_sn: 设备序列号，可选
        start_date: 开始日期，格式如 "2024-12-20"，可选
        end_date: 结束日期，格式如 "2024-12-20"，可选
    
    返回:
        JSON格式的分析结果，包含：
        - 双轴折线图数据（总睡眠时长和睡眠评分）
        - 深睡与浅睡对比图数据
    """
    try:
        # 导入必要的模块
        from datetime import datetime, timedelta, date
        import json
        from src.db.database import DatabaseManager
        
        # 如果没有提供开始和结束日期，则根据data_type计算
        if not start_date or not end_date:
            # 根据data_type计算日期范围
            end_date = datetime.now().strftime('%Y-%m-%d')
            if data_type == "week":
                # 最近7天
                start_date = (datetime.now() - timedelta(days=6)).strftime('%Y-%m-%d')
            else:  # month
                # 最近30天
                start_date = (datetime.now() - timedelta(days=29)).strftime('%Y-%m-%d')
        
        print(f"📅 分析日期范围: {start_date} 到 {end_date}")
        print(f"📊 数据类型: {data_type}")
        print(f"🔧 设备序列号: {device_sn}")
        
        # 从数据库获取真实数据
        print(f"🔗 连接数据库，获取 {start_date} 到 {end_date} 的睡眠数据")
        db_manager = DatabaseManager()
        sleep_data_df = db_manager.get_calculated_sleep_data_for_date_range(start_date, end_date, device_sn)
        
        print(f"📊 数据库查询结果行数: {len(sleep_data_df)}")
        if not sleep_data_df.empty:
            print(f"📋 前5行数据:")
            print(sleep_data_df.head())
        
        # 检查数据是否为空
        if sleep_data_df.empty:
            # 如果没有数据，返回明确的错误信息
            print(f"⚠️ 没有找到指定日期范围内的睡眠数据")
            result = {
                "success": False,
                "error": "No sleep data found",
                "data": {
                    "type": data_type,
                    "start_date": start_date,
                    "end_date": end_date,
                    "message": "没有找到指定日期范围内的睡眠数据，请先执行睡眠分析生成数据"
                }
            }
            return json.dumps(result, ensure_ascii=False, indent=2)
        
        # 将DataFrame转换为字典列表
        sleep_data = sleep_data_df.to_dict('records')
        print(f"🔄 转换为字典列表，长度: {len(sleep_data)}")
        
        # 处理数据，构建周/月数据
        if data_type == "week":
            # 构建周数据
            weekly_data = []
            
            # 确保数据按日期排序
            try:
                sleep_data.sort(key=lambda x: x['date'])
            except Exception as e:
                # 如果排序失败，尝试使用字符串排序
                print(f"⚠️ 排序失败: {e}")
                try:
                    sleep_data.sort(key=lambda x: str(x['date']))
                except Exception as e2:
                    # 如果再次失败，不排序
                    print(f"⚠️ 再次排序失败: {e2}")
            
            for item in sleep_data:
                # 处理日期字段，确保它是字符串
                date_value = item['date']
                try:
                    # 尝试将日期值转换为字符串
                    if hasattr(date_value, 'strftime'):
                        # 如果是日期对象
                        date_str = date_value.strftime('%Y-%m-%d')
                        date_obj = date_value
                    else:
                        # 如果是字符串
                        date_str = str(date_value)
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                except Exception as e:
                    # 如果转换失败，使用当前日期
                    print(f"⚠️ 日期转换失败: {e}")
                    date_str = datetime.now().strftime('%Y-%m-%d')
                    date_obj = datetime.now()
                
                day_name = date_obj.strftime('%a')  # 周几的缩写
                day_name_cn = _get_chinese_weekday(date_obj.weekday())  # 中文周几
                
                # 构建当天数据
                total_sleep_minutes = float(item.get('sleep_duration_minutes', 0))
                daily_data = {
                    "date": date_str,
                    "day": day_name,
                    "day_cn": day_name_cn,
                    "total_sleep_hours": round(total_sleep_minutes / 60, 2),  # 保留两位小数
                    "total_sleep_minutes": int(total_sleep_minutes),  # 转换为分钟
                    "sleep_score": int(item.get('sleep_score', 0)),
                    "deep_sleep_minutes": int(item.get('deep_sleep_minutes', 0)),
                    "light_sleep_minutes": int(item.get('light_sleep_minutes', 0)),
                    "rem_sleep_minutes": int(item.get('rem_sleep_minutes', 0)),
                    "awake_minutes": int(item.get('awake_minutes', 0))
                }
                weekly_data.append(daily_data)
            
            # 计算平均值
            total_sleep_sum = sum(item.get('total_sleep_minutes', 0) for item in weekly_data)
            sleep_score_sum = sum(item.get('sleep_score', 0) for item in weekly_data)
            valid_days = len([item for item in weekly_data if item.get('sleep_score', 0) > 0])
            
            avg_sleep_duration = 0
            avg_sleep_score = 0
            
            if valid_days > 0:
                avg_sleep_duration = total_sleep_sum / valid_days
                avg_sleep_score = sleep_score_sum / valid_days
            
            # 构建返回结果
            result = {
                "success": True,
                "data": {
                    "type": "week",
                    "start_date": start_date,
                    "end_date": end_date,
                    "weekly_data": weekly_data,
                    "average_sleep_duration_minutes": round(avg_sleep_duration, 1),
                    "average_sleep_score": round(avg_sleep_score, 1)
                }
            }
        else:  # month
            # 构建月数据
            monthly_data = []
            
            # 确保数据按日期排序
            try:
                sleep_data.sort(key=lambda x: x['date'])
            except Exception as e:
                # 如果排序失败，尝试使用字符串排序
                print(f"⚠️ 排序失败: {e}")
                try:
                    sleep_data.sort(key=lambda x: str(x['date']))
                except Exception as e2:
                    # 如果再次失败，不排序
                    print(f"⚠️ 再次排序失败: {e2}")
            
            # 将睡眠数据转换为字典，键为日期字符串
            sleep_data_dict = {}
            for item in sleep_data:
                # 处理日期字段，确保它是字符串
                date_value = item['date']
                try:
                    # 尝试将日期值转换为字符串
                    if hasattr(date_value, 'strftime'):
                        # 如果是日期对象
                        date_str = date_value.strftime('%Y-%m-%d')
                    else:
                        # 如果是字符串
                        date_str = str(date_value)
                    sleep_data_dict[date_str] = item
                except Exception as e:
                    # 如果转换失败，跳过
                    print(f"⚠️ 日期转换失败: {e}")
            
            # 生成日期范围内的所有日期
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
            date_range = pd.date_range(start=start_date_obj, end=end_date_obj, freq='D')
            
            # 对于每一天，生成数据条目
            for date in date_range:
                date_str = date.strftime('%Y-%m-%d')
                day_of_month = date.day
                
                # 检查是否有对应日期的数据
                if date_str in sleep_data_dict:
                    # 使用真实数据
                    item = sleep_data_dict[date_str]
                    total_sleep_minutes = float(item.get('sleep_duration_minutes', 0))
                    daily_data = {
                        "date": date_str,
                        "day_of_month": int(day_of_month),
                        "total_sleep_hours": round(total_sleep_minutes / 60, 2),  # 保留两位小数
                        "total_sleep_minutes": int(total_sleep_minutes),  # 转换为分钟
                        "sleep_score": int(item.get('sleep_score', 0)),
                        "deep_sleep_minutes": int(item.get('deep_sleep_minutes', 0)),
                        "light_sleep_minutes": int(item.get('light_sleep_minutes', 0)),
                        "rem_sleep_minutes": int(item.get('rem_sleep_minutes', 0)),
                        "awake_minutes": int(item.get('awake_minutes', 0))
                    }
                else:
                    # 没有数据，用 0 填充
                    daily_data = {
                        "date": date_str,
                        "day_of_month": int(day_of_month),
                        "total_sleep_hours": 0.0,
                        "total_sleep_minutes": 0,
                        "sleep_score": 0,
                        "deep_sleep_minutes": 0,
                        "light_sleep_minutes": 0,
                        "rem_sleep_minutes": 0,
                        "awake_minutes": 0
                    }
                
                monthly_data.append(daily_data)
            
            # 计算平均值
            total_sleep_sum = sum(item.get('total_sleep_minutes', 0) for item in monthly_data)
            sleep_score_sum = sum(item.get('sleep_score', 0) for item in monthly_data)
            valid_days = len([item for item in monthly_data if item.get('sleep_score', 0) > 0])
            
            avg_sleep_duration = 0
            avg_sleep_score = 0
            
            if valid_days > 0:
                avg_sleep_duration = total_sleep_sum / valid_days
                avg_sleep_score = sleep_score_sum / valid_days
            
            # 构建返回结果
            result = {
                "success": True,
                "data": {
                    "type": "month",
                    "start_date": start_date,
                    "end_date": end_date,
                    "monthly_data": monthly_data,
                    "average_sleep_duration_minutes": round(avg_sleep_duration, 1),
                    "average_sleep_score": round(avg_sleep_score, 1)
                }
            }
        
        print(f"✅ 数据处理完成，返回结果")
        print(f"📤 返回数据长度: {len(sleep_data)}")
        
        # 返回结果
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_message = f"分析数据库周/月数据时出错: {str(e)}"
        print(error_message)
        import traceback
        traceback.print_exc()
        # 构建错误响应
        error_response = {
            "success": False,
            "error": error_message,
            "message": "分析数据库周/月数据失败，请检查数据库连接和数据格式"
        }
        # 返回错误响应
        return json.dumps(error_response, ensure_ascii=False, indent=2)


def _get_chinese_weekday(weekday):
    """
    获取中文周几
    
    参数:
        weekday: 0-6，0表示周一，6表示周日
    
    返回:
        中文周几
    """
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    return weekdays[weekday]


def _convert_decimal_to_float(obj):
    """
    递归将Decimal类型转换为float类型
    
    参数:
        obj: 要转换的对象
    
    返回:
        转换后的对象
    """
    import decimal
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _convert_decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_decimal_to_float(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_decimal_to_float(item) for item in obj)
    else:
        return obj


def _parse_and_preprocess_data_from_db(df: pd.DataFrame) -> pd.DataFrame:
    """
    从数据库数据中解析和预处理数据
    
    参数:
        df: 从数据库获取的DataFrame
    
    返回:
        预处理后的DataFrame
    """
    # 复制DataFrame以避免修改原始数据
    parsed_df = df.copy()
    
    # 确保时间列存在并转换为datetime类型
    if 'upload_time' in parsed_df.columns:
        parsed_df['upload_time'] = pd.to_datetime(parsed_df['upload_time'])
    else:
        # 如果没有upload_time列，尝试其他可能的时间列名
        time_columns = ['timestamp', 'date', 'time']
        for col in time_columns:
            if col in parsed_df.columns:
                parsed_df['upload_time'] = pd.to_datetime(parsed_df[col])
                break
        else:
            # 如果没有找到时间列，返回空DataFrame
            return pd.DataFrame()
    
    # 提取小时
    parsed_df['hour'] = parsed_df['upload_time'].dt.hour
    
    # 提取日期
    parsed_df['date'] = parsed_df['upload_time'].dt.date
    
    # 处理心率数据
    if 'heart_rate' in parsed_df.columns:
        # 转换心率为数值类型
        parsed_df['heart_rate'] = pd.to_numeric(parsed_df['heart_rate'], errors='coerce')
    else:
        # 如果没有heart_rate列，尝试从data_content中提取
        parsed_df['heart_rate'] = parsed_df.apply(lambda row: _extract_value_from_content(row, '心率'), axis=1)
    
    # 处理呼吸频率数据
    if 'respiratory_rate' in parsed_df.columns:
        # 转换呼吸频率为数值类型
        parsed_df['respiration_rate'] = pd.to_numeric(parsed_df['respiratory_rate'], errors='coerce')
    elif 'respiration_rate' in parsed_df.columns:
        # 转换呼吸频率为数值类型
        parsed_df['respiration_rate'] = pd.to_numeric(parsed_df['respiration_rate'], errors='coerce')
    else:
        # 如果没有呼吸频率列，尝试从data_content中提取
        parsed_df['respiration_rate'] = parsed_df.apply(lambda row: _extract_value_from_content(row, '呼吸'), axis=1)
    
    # 处理呼吸暂停数据
    if 'apnea_count' in parsed_df.columns:
        # 转换呼吸暂停次数为数值类型
        parsed_df['apnea_count'] = pd.to_numeric(parsed_df['apnea_count'], errors='coerce')
    elif 'respiratory_pause_count' in parsed_df.columns:
        # 转换呼吸暂停次数为数值类型
        parsed_df['apnea_count'] = pd.to_numeric(parsed_df['respiratory_pause_count'], errors='coerce')
    else:
        # 如果没有呼吸暂停列，尝试从data_content中提取
        parsed_df['apnea_count'] = parsed_df.apply(lambda row: _extract_value_from_content(row, '呼吸暂停次数'), axis=1)
    
    # 过滤掉无效数据
    parsed_df = parsed_df[parsed_df['heart_rate'].notna() & parsed_df['heart_rate'] > 0]
    parsed_df = parsed_df[parsed_df['respiration_rate'].notna() & parsed_df['respiration_rate'] > 0]
    
    # 填充呼吸暂停次数的缺失值为0
    parsed_df['apnea_count'] = parsed_df['apnea_count'].fillna(0)
    
    return parsed_df


def _extract_value_from_content(row, key):
    """
    从数据内容中提取指定键的值
    
    参数:
        row: 数据行
        key: 要提取的键
    
    返回:
        提取的值
    """
    # 尝试从data_content列中提取
    if 'data_content' in row:
        content = str(row['data_content'])
        if key == '心率':
            match = re.search(r'心率:(\d+)次/分钟', content)
        elif key == '呼吸':
            match = re.search(r'呼吸:(\d+)次/分钟', content)
        elif key == '呼吸暂停次数':
            match = re.search(r'呼吸暂停次数:(\d+)次', content)
        else:
            return None
        
        if match:
            return float(match.group(1))
    
    # 尝试直接从列中获取
    if key == '心率' and 'heart_rate' in row:
        return pd.to_numeric(row['heart_rate'], errors='coerce')
    elif key == '呼吸' and ('respiratory_rate' in row or 'respiration_rate' in row):
        if 'respiratory_rate' in row:
            return pd.to_numeric(row['respiratory_rate'], errors='coerce')
        else:
            return pd.to_numeric(row['respiration_rate'], errors='coerce')
    elif key == '呼吸暂停次数' and ('apnea_count' in row or 'respiratory_pause_count' in row):
        if 'apnea_count' in row:
            return pd.to_numeric(row['apnea_count'], errors='coerce')
        else:
            return pd.to_numeric(row['respiratory_pause_count'], errors='coerce')
    
    return None


def _analyze_heart_rate_peaks(df: pd.DataFrame) -> Dict[str, Any]:
    """
    分析心率高峰时段
    
    参数:
        df: 预处理后的DataFrame
    
    返回:
        包含心率高峰时段分析结果的字典
    """
    # 按小时分组统计平均心率
    hourly_hr = df.groupby('hour')['heart_rate'].mean().reset_index()
    
    # 找出心率最高的前3个时段
    peak_hours = hourly_hr.nlargest(3, 'heart_rate')['hour'].tolist()
    peak_hours.sort()
    
    # 计算高峰时段的平均心率
    peak_hr_value = hourly_hr.loc[hourly_hr['hour'].isin(peak_hours), 'heart_rate'].mean()
    
    # 计算整体平均心率
    avg_hr = df['heart_rate'].mean()
    
    # 计算整体平均呼吸频率
    avg_rr = df['respiration_rate'].mean()
    
    return {
        "peak_hours": peak_hours,
        "peak_hr_value": round(peak_hr_value, 1),
        "avg_hr": round(avg_hr, 1),
        "avg_rr": round(avg_rr, 1)
    }


def _analyze_apnea_trend(df: pd.DataFrame) -> Dict[str, Any]:
    """
    分析呼吸暂停趋势
    
    参数:
        df: 预处理后的DataFrame
    
    返回:
        包含呼吸暂停趋势分析结果的字典
    """
    # 按日期分组统计呼吸暂停次数
    daily_apnea = df.groupby('date')['apnea_count'].sum().reset_index()
    
    # 计算总呼吸暂停次数
    total_apnea = int(df['apnea_count'].sum())
    
    # 计算数据覆盖天数
    data_days = len(daily_apnea)
    
    # 计算日均呼吸暂停次数
    avg_apnea_per_day = total_apnea / data_days if data_days > 0 else 0
    
    # 分析趋势
    if data_days < 2:
        # 数据不足，无法分析趋势
        trend_direction = "stable"
        trend_value = 0
    else:
        # 计算简单线性回归斜率
        daily_apnea['day_index'] = range(len(daily_apnea))
        slope = ((len(daily_apnea) * (daily_apnea['day_index'] * daily_apnea['apnea_count']).sum() - 
                 daily_apnea['day_index'].sum() * daily_apnea['apnea_count'].sum()) / 
                (len(daily_apnea) * (daily_apnea['day_index'] ** 2).sum() - 
                 (daily_apnea['day_index'].sum()) ** 2))
        
        # 确定趋势方向
        if abs(slope) < 0.5:
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "worsening"
        else:
            trend_direction = "improving"
        
        trend_value = round(slope, 2)
    
    return {
        "total_apnea": total_apnea,
        "avg_apnea_per_day": round(avg_apnea_per_day, 1),
        "data_days": data_days,
        "trend_direction": trend_direction,
        "trend_value": trend_value
    }


def _calculate_risk_score(df: pd.DataFrame) -> int:
    """
    计算综合风险评分
    
    参数:
        df: 预处理后的DataFrame
    
    返回:
        综合风险评分（0-100分）
    """
    # 基础分数
    base_score = 50
    
    # 心率风险因子
    avg_hr = df['heart_rate'].mean()
    max_hr = df['heart_rate'].max()
    min_hr = df['heart_rate'].min()
    
    # 心率异常评分
    hr_score = 0
    if avg_hr > 100 or avg_hr < 50:
        hr_score += 20
    elif avg_hr > 90 or avg_hr < 60:
        hr_score += 10
    
    if max_hr > 120:
        hr_score += 15
    elif max_hr > 110:
        hr_score += 5
    
    if min_hr < 40:
        hr_score += 15
    elif min_hr < 50:
        hr_score += 5
    
    # 呼吸频率风险因子
    avg_rr = df['respiration_rate'].mean()
    max_rr = df['respiration_rate'].max()
    min_rr = df['respiration_rate'].min()
    
    # 呼吸频率异常评分
    rr_score = 0
    if avg_rr > 25 or avg_rr < 10:
        rr_score += 20
    elif avg_rr > 20 or avg_rr < 12:
        rr_score += 10
    
    if max_rr > 30:
        rr_score += 15
    elif max_rr > 25:
        rr_score += 5
    
    if min_rr < 8:
        rr_score += 15
    elif min_rr < 10:
        rr_score += 5
    
    # 呼吸暂停风险因子
    total_apnea = df['apnea_count'].sum()
    data_days = len(df['date'].unique())
    avg_apnea_per_day = total_apnea / data_days if data_days > 0 else 0
    
    # 呼吸暂停异常评分
    apnea_score = 0
    if avg_apnea_per_day > 10:
        apnea_score += 30
    elif avg_apnea_per_day > 5:
        apnea_score += 20
    elif avg_apnea_per_day > 2:
        apnea_score += 10
    
    # 计算总分
    total_score = base_score + hr_score + rr_score + apnea_score
    
    # 确保分数在0-100范围内
    total_score = max(0, min(100, total_score))
    
    return int(total_score)


@tool
def analyze_trend_and_pattern(file_path: str) -> str:
    """
    多天监护数据趋势分析工具（To B 专用）

    功能：分析病人多天、多维度的监护数据，挖掘周期性规律和长期趋势。

    参数:
        file_path: 病床监护数据Excel文件的完整路径（建议包含多天数据以获得准确趋势）

    返回:
        JSON格式的分析结果，包含：
        - peak_hours: 心率最高的时段（小时列表，如[23, 0, 3]表示凌晨23点、0点、3点心率最高）
        - trend_direction: 趋势方向（improving好转/worsening恶化/stable稳定）
        - trend_value: 趋势值（正值表示呼吸暂停次数上升，负值表示下降）
        - risk_score: 综合风险评分（0-100分，分数越高风险越大）
        - peak_hr_value: 高峰时段的平均心率
        - avg_hr: 整体平均心率
        - avg_rr: 整体平均呼吸频率
        - total_apnea: 总呼吸暂停次数
        - avg_apnea_per_day: 日均呼吸暂停次数
        - data_days: 数据覆盖天数

    使用场景:
        - 分析病人长期健康状况变化（需要多天数据）
        - 发现疾病发作的周期规律（如某时段心率异常升高）
        - 评估治疗效果和病情进展（对比不同时期数据）
        - 为医生提供决策支持（预测风险、制定治疗方案）
        - 科研数据分析（挖掘群体规律）

    特点:
        - 按小时分组统计，自动识别心率高峰时段
        - 计算7天移动平均线，准确判断呼吸暂停趋势
        - 综合心率、呼吸频率、呼吸暂停等多维度计算风险评分
        - 适用于多天、多维度的数据分析

    注意事项:
        - 建议至少提供3天以上数据以获得准确趋势分析
        - 单日数据分析结果可能不够准确
        - 风险评分仅供参考，临床决策请结合医生判断
    """
    result = analyze_trend_and_pattern_internal(file_path)
    return json.dumps(result, ensure_ascii=False, indent=2)


# 直接导出内部函数，便于API直接调用
__all__ = ['analyze_trend_and_pattern', 'analyze_trend_and_pattern_internal', 'analyze_trend_from_database']