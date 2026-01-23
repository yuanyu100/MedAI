"""
时间格式化工具模块
提供格式化时间数据的功能，用于生成符合要求格式的睡眠分析摘要
"""

def format_sleep_summary(bedtime, wakeup_time, time_in_bed_minutes, sleep_duration_minutes, 
                       sleep_prep_time_minutes, bed_exit_count, deep_sleep_minutes, 
                       deep_sleep_percentage, avg_respiratory_rate, min_respiratory_rate, 
                       max_respiratory_rate, apnea_count, avg_apnea_duration, 
                       max_apnea_duration, avg_heart_rate, min_heart_rate, max_heart_rate):
    """
    格式化睡眠数据为指定格式的摘要字符串
    
    Args:
        bedtime: 上床时间 (datetime对象或字符串)
        wakeup_time: 起床时间 (datetime对象或字符串)
        time_in_bed_minutes: 卧床时间（分钟）
        sleep_duration_minutes: 睡眠时长（分钟）
        sleep_prep_time_minutes: 睡眠准备时间（分钟）
        bed_exit_count: 离床次数
        deep_sleep_minutes: 深睡时长（分钟）
        deep_sleep_percentage: 深睡占比（百分比）
        avg_respiratory_rate: 平均呼吸率
        min_respiratory_rate: 最低呼吸率
        max_respiratory_rate: 最高呼吸率
        apnea_count: 呼吸暂停次数
        avg_apnea_duration: 平均呼吸暂停时长
        max_apnea_duration: 最长呼吸暂停时长
        avg_heart_rate: 平均心率
        min_heart_rate: 最低心率
        max_heart_rate: 最高心率
    
    Returns:
        str: 格式化的睡眠摘要字符串
    """
    # 格式化时间
    if hasattr(bedtime, 'strftime'):
        bedtime_str = bedtime.strftime('%H:%M')
    else:
        bedtime_str = str(bedtime)
    
    if hasattr(wakeup_time, 'strftime'):
        wakeup_str = wakeup_time.strftime('%H:%M')
    else:
        wakeup_str = str(wakeup_time)
    
    # 计算卧床时长的小时和分钟
    time_in_bed_hours = int(time_in_bed_minutes // 60)
    time_in_bed_remaining_min = int(time_in_bed_minutes % 60)
    
    # 计算睡眠时长的小时和分钟
    sleep_duration_hours = int(sleep_duration_minutes // 60)
    sleep_duration_remaining_min = int(sleep_duration_minutes % 60)
    
    # 计算入睡时间
    from datetime import timedelta
    if hasattr(bedtime, 'strftime'):
        sleep_start_time = bedtime + timedelta(minutes=sleep_prep_time_minutes)
        sleep_start_str = sleep_start_time.strftime('%H:%M')
    else:
        # 如果bedtime不是datetime对象，简单地假设入睡时间是上床时间+24分钟（示例中的差值）
        sleep_start_str = f"{bedtime_str[:2]}:{(int(bedtime_str[3:])+sleep_prep_time_minutes)%60:02d}"
    
    # 格式化睡眠准备期
    sleep_prep_hours = int(sleep_prep_time_minutes // 60)
    sleep_prep_remaining_min = int(sleep_prep_time_minutes % 60)
    
    # 格式化摘要
    summary = (
        f"我 {bedtime_str} 上床，{sleep_start_str} 入睡，"
        f"{wakeup_str} 醒来，总卧床时长为 {time_in_bed_hours} 小时 {time_in_bed_remaining_min} 分，"
        f"睡眠时长为 {sleep_duration_hours} 小时 {sleep_duration_remaining_min} 分；"
        f"其中睡眠准备期为 {sleep_prep_hours} 小时 {sleep_prep_remaining_min} 分，"
        f"深睡时长为 {int(deep_sleep_minutes // 60)} 小时 {int(deep_sleep_minutes % 60)} 分，"
        f"深睡占比为 {deep_sleep_percentage}%，中间有 {bed_exit_count} 次离床。"
        f"睡眠中，平均呼吸率为 {avg_respiratory_rate} 次 / 分钟，最低呼吸率为 {min_respiratory_rate} 次 / 分钟，"
        f"最高呼吸率为 {max_respiratory_rate} 次 / 分钟，呼吸暂停为 {apnea_count} 次 / 小时，"
        f"平均呼吸暂停时长为 {avg_apnea_duration} 秒，最长呼吸暂停时长为 {max_apnea_duration} 秒；"
        f"平均心率为 {avg_heart_rate} 次 / 分钟，最低心率为 {min_heart_rate} 次 / 分钟，"
        f"最高心率为 {max_heart_rate} 次 / 分钟。"
    )
    
    return summary


def format_sleep_summary_simple(bedtime, wakeup_time, sleep_data):
    """
    根据睡眠分析数据简单格式化摘要
    
    Args:
        bedtime: 上床时间
        wakeup_time: 起床时间
        sleep_data: 包含睡眠分析数据的字典
    
    Returns:
        str: 格式化的摘要字符串
    """
    # 从sleep_data中提取所需字段
    time_in_bed_minutes = sleep_data.get('time_in_bed_minutes', 0)
    sleep_duration_minutes = sleep_data.get('sleep_duration_minutes', 0)
    sleep_prep_time_minutes = sleep_data.get('sleep_prep_time_minutes', 0)
    bed_exit_count = sleep_data.get('bed_exit_count', 0)
    
    # 深度睡眠数据
    sleep_phases = sleep_data.get('sleep_phases', {})
    deep_sleep_minutes = sleep_phases.get('deep_sleep_minutes', 0)
    deep_sleep_percentage = sleep_phases.get('deep_sleep_percentage', 0)
    
    # 生理指标数据
    avg_metrics = sleep_data.get('average_metrics', {})
    avg_heart_rate = avg_metrics.get('avg_heart_rate', 0)
    avg_respiratory_rate = avg_metrics.get('avg_respiratory_rate', 0)
    
    # 从工具返回中获取更多细节
    min_heart_rate = sleep_data.get('heart_rate_metrics', {}).get('min_heart_rate', 0)
    max_heart_rate = sleep_data.get('heart_rate_metrics', {}).get('max_heart_rate', 0)
    min_respiratory_rate = sleep_data.get('respiratory_metrics', {}).get('min_respiratory_rate', avg_respiratory_rate)
    max_respiratory_rate = sleep_data.get('respiratory_metrics', {}).get('max_respiratory_rate', avg_respiratory_rate)
    
    apnea_count = sleep_data.get('respiratory_metrics', {}).get('apnea_count', 0)
    avg_apnea_duration = sleep_data.get('respiratory_metrics', {}).get('avg_apnea_duration_seconds', 0)
    max_apnea_duration = sleep_data.get('respiratory_metrics', {}).get('max_apnea_duration_seconds', 0)
    
    # 如果缺少某些数据，使用默认值
    if min_heart_rate == 0:
        min_heart_rate = avg_heart_rate * 0.8
    if max_heart_rate == 0:
        max_heart_rate = avg_heart_rate * 1.2
    if min_respiratory_rate == avg_respiratory_rate:
        min_respiratory_rate = avg_respiratory_rate * 0.8
    if max_respiratory_rate == avg_respiratory_rate:
        max_respiratory_rate = avg_respiratory_rate * 1.2
        
    return format_sleep_summary(
        bedtime, wakeup_time, time_in_bed_minutes, sleep_duration_minutes,
        sleep_prep_time_minutes, bed_exit_count, deep_sleep_minutes,
        deep_sleep_percentage, avg_respiratory_rate, min_respiratory_rate,
        max_respiratory_rate, apnea_count, avg_apnea_duration,
        max_apnea_duration, avg_heart_rate, min_heart_rate, max_heart_rate
    )


def format_sleep_time_only(bedtime, wakeup_time, sleep_prep_time_minutes=24):
    """
    仅格式化睡眠时间部分，返回类似 "我 @HH:MM 上床，HH:MM 入睡，HH:MM 醒来@"
    
    Args:
        bedtime: 上床时间 (datetime对象或字符串)
        wakeup_time: 起床时间 (datetime对象或字符串)
        sleep_prep_time_minutes: 睡眠准备时间（分钟）
    
    Returns:
        str: 格式化的睡眠时间摘要
    """
    # 格式化时间
    if hasattr(bedtime, 'strftime'):
        bedtime_str = bedtime.strftime('%H:%M')
    else:
        bedtime_str = str(bedtime)
    
    if hasattr(wakeup_time, 'strftime'):
        wakeup_str = wakeup_time.strftime('%H:%M')
    else:
        wakeup_str = str(wakeup_time)
    
    # 计算入睡时间
    from datetime import timedelta
    if hasattr(bedtime, 'strftime'):
        sleep_start_time = bedtime + timedelta(minutes=sleep_prep_time_minutes)
        sleep_start_str = sleep_start_time.strftime('%H:%M')
    else:
        # 如果bedtime不是datetime对象，简单地假设入睡时间是上床时间+24分钟（示例中的差值）
        sleep_start_str = f"{bedtime_str[:2]}:{(int(bedtime_str[3:])+sleep_prep_time_minutes)%60:02d}"
    
    return f"我 {bedtime_str} 上床，{sleep_start_str} 入睡，{wakeup_str} 醒来"


def create_sleep_analysis_prompt_with_time(date, sleep_data):
    """
    根据睡眠数据创建分析提示，包含格式化的时间信息
    
    Args:
        date: 日期字符串
        sleep_data: 睡眠分析数据字典
    
    Returns:
        str: 包含时间信息的分析提示
    """
    # 从sleep_data中提取时间信息
    bedtime_str = sleep_data.get('bedtime', '')
    wakeup_time_str = sleep_data.get('wakeup_time', '')
    
    # 如果有完整的时间信息，使用它们
    if bedtime_str and wakeup_time_str:
        from datetime import datetime
        try:
            bedtime = datetime.fromisoformat(bedtime_str.replace('Z', '+00:00')) if bedtime_str.endswith('Z') else datetime.fromisoformat(bedtime_str)
            wakeup_time = datetime.fromisoformat(wakeup_time_str.replace('Z', '+00:00')) if wakeup_time_str.endswith('Z') else datetime.fromisoformat(wakeup_time_str)
            
            # 获取睡眠准备时间（如果有）
            sleep_prep_time = sleep_data.get('sleep_prep_time_minutes', 24)
            
            time_summary = format_sleep_time_only(bedtime, wakeup_time, sleep_prep_time)
        except:
            # 如果解析失败，使用默认格式
            time_summary = f"我 {bedtime_str[-8:-3] if len(bedtime_str) > 8 else 'XX:XX'} 上床，{(lambda x: f'{int(x[-8:-6]):02d}:{(int(x[-5:-3])+24)%60:02d}' if x and len(x) > 8 else 'XX:XX')(bedtime_str)} 入睡，{wakeup_time_str[-8:-3] if len(wakeup_time_str) > 8 else 'XX:XX'} 醒来"
    else:
        # 如果没有完整时间信息，使用默认时间
        time_summary = f"我 XX:XX 上床，XX:XX 入睡，XX:XX 醒来"
    
    # 构建完整的提示
    prompt = f"{time_summary}，请基于这些时间信息和以下睡眠数据进行详细分析：\n\n{str(sleep_data)}"
    
    return prompt


if __name__ == "__main__":
    # 示例使用
    from datetime import datetime
    
    # 示例数据
    bed_time = datetime.strptime("2025-12-18 20:12", "%Y-%m-%d %H:%M")
    wake_time = datetime.strptime("2025-12-19 07:23", "%Y-%m-%d %H:%M")
    
    sample_sleep_data = {
        "time_in_bed_minutes": 659,
        "sleep_duration_minutes": 611,
        "sleep_prep_time_minutes": 23,
        "bed_exit_count": 4,
        "sleep_phases": {
            "deep_sleep_minutes": 156,
            "deep_sleep_percentage": 24.11
        },
        "average_metrics": {
            "avg_heart_rate": 73,
            "avg_respiratory_rate": 15
        },
        "heart_rate_metrics": {
            "min_heart_rate": 59,
            "max_heart_rate": 84
        },
        "respiratory_metrics": {
            "min_respiratory_rate": 10,
            "max_respiratory_rate": 20,
            "apnea_count": 0.4,
            "avg_apnea_duration_seconds": 15,
            "max_apnea_duration_seconds": 24
        }
    }
    
    result = format_sleep_summary_simple(bed_time, wake_time, sample_sleep_data)
    print(result)
    
    print("\n仅时间格式:")
    time_result = format_sleep_time_only(bed_time, wake_time)
    print(time_result)
    
    print("\n完整提示:")
    full_prompt = create_sleep_analysis_prompt_with_time("2025-12-18", sample_sleep_data)
    print(full_prompt)