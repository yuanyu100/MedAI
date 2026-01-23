"""
睡眠分析服务模块
提供高级睡眠分析功能，结合格式化时间信息与智能体分析
"""

from datetime import datetime
import json
from ..tools.sleep_analyzer_tool import analyze_single_day_sleep_data
from ..tools.physiological_analyzer_tool import analyze_single_day_physiological_data
from ..utils.time_formatter import create_sleep_analysis_prompt_with_time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improved_agent import run_improved_agent


def run_sleep_analysis_with_formatted_time(date: str, force_refresh: bool = False):
    """
    运行睡眠分析，使用格式化的时间信息作为提示
    
    Args:
        date: 日期字符串，格式如 '2024-12-20'
        force_refresh: 是否强制刷新，跳过缓存
    
    Returns:
        str: 智能体分析结果
    """
    # 首先获取睡眠数据
    sleep_data_json = analyze_single_day_sleep_data(date)
    
    # 解析睡眠数据
    try:
        sleep_data = json.loads(sleep_data_json)
        
        # 检查是否有错误
        if "error" in sleep_data:
            # 如果睡眠数据获取失败，尝试获取生理数据作为备选
            physio_data_json = analyze_single_day_physiological_data(date)
            physio_data = json.loads(physio_data_json)
            
            if "error" in physio_data:
                return f"数据获取失败：{sleep_data.get('error', '未知错误')} 和 {physio_data.get('error', '未知错误')}"
            else:
                # 使用生理数据构建基本提示
                prompt = f"请分析 {date} 的生理指标数据：\n\n{str(physio_data)}"
        else:
            # 使用睡眠数据创建包含格式化时间的提示
            prompt = create_sleep_analysis_prompt_with_time(date, sleep_data)
    
    except json.JSONDecodeError:
        return f"数据解析失败：无法解析睡眠数据 {sleep_data_json}"
    
    # 调用改进的智能体进行分析
    result = run_improved_agent(date, thread_id=f"sleep_analysis_{date}", force_refresh=force_refresh)
    
    return result


def get_formatted_sleep_time_summary(date: str):
    """
    获取格式化的睡眠时间摘要
    
    Args:
        date: 日期字符串，格式如 '2024-12-20'
    
    Returns:
        str: 格式化的时间摘要
    """
    # 获取睡眠数据
    sleep_data_json = analyze_single_day_sleep_data(date)
    
    try:
        sleep_data = json.loads(sleep_data_json)
        
        if "error" in sleep_data:
            return f"无法获取 {date} 的睡眠时间数据"
        
        # 从数据中提取时间信息
        bedtime_str = sleep_data.get('bedtime', '')
        wakeup_time_str = sleep_data.get('wakeup_time', '')
        
        if bedtime_str and wakeup_time_str:
            from datetime import datetime
            bedtime = datetime.fromisoformat(bedtime_str.replace('Z', '+00:00')) if bedtime_str.endswith('Z') else datetime.fromisoformat(bedtime_str)
            wakeup_time = datetime.fromisoformat(wakeup_time_str.replace('Z', '+00:00')) if wakeup_time_str.endswith('Z') else datetime.fromisoformat(wakeup_time_str)
            
            sleep_prep_time = sleep_data.get('sleep_prep_time_minutes', 24)
            
            from ..utils.time_formatter import format_sleep_time_only
            return format_sleep_time_only(bedtime, wakeup_time, sleep_prep_time)
        else:
            return f"无法从数据中提取 {date} 的时间信息"
            
    except json.JSONDecodeError:
        return f"数据解析失败：无法解析睡眠数据 {sleep_data_json}"


if __name__ == "__main__":
    # 示例使用
    test_date = "2024-12-20"  # 使用一个测试日期
    
    print("获取格式化时间摘要:")
    time_summary = get_formatted_sleep_time_summary(test_date)
    print(time_summary)
    
    print("\n运行完整的睡眠分析:")
    analysis_result = run_sleep_analysis_with_formatted_time(test_date)
    print(analysis_result)