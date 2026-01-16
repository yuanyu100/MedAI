#!/usr/bin/env python3
"""
HTML报告生成工具
"""

from langchain.tools import tool
import json
from datetime import datetime
from typing import Dict, Any


def format_sleep_data_to_html(sleep_data: Dict[str, Any], date: str) -> str:
    """
    将睡眠数据分析结果格式化为HTML格式
    """
    bedtime = sleep_data.get('bedtime', 'N/A')
    wakeup_time = sleep_data.get('wakeup_time', 'N/A')
    time_in_bed_minutes = sleep_data.get('time_in_bed_minutes', 0)
    sleep_duration_minutes = sleep_data.get('sleep_duration_minutes', 0)
    sleep_score = sleep_data.get('sleep_score', 0)
    bed_exit_count = sleep_data.get('bed_exit_count', 0)
    sleep_prep_time_minutes = sleep_data.get('sleep_prep_time_minutes', 0)
    sleep_phases = sleep_data.get('sleep_phases', {})
    average_metrics = sleep_data.get('average_metrics', {})
    
    # 计算睡眠效率
    sleep_efficiency = 0
    if time_in_bed_minutes > 0:
        sleep_efficiency = round((sleep_duration_minutes / time_in_bed_minutes) * 100, 1)
    
    # 获取各睡眠阶段占比
    deep_sleep_percentage = sleep_phases.get('deep_sleep_percentage', 0)
    light_sleep_percentage = sleep_phases.get('light_sleep_percentage', 0)
    rem_sleep_percentage = sleep_phases.get('rem_sleep_percentage', 0)
    awake_percentage = sleep_phases.get('awake_percentage', 0)
    
    # 获取平均指标
    avg_respiratory_rate = average_metrics.get('avg_respiratory_rate', 'N/A')
    avg_heart_rate = average_metrics.get('avg_heart_rate', 'N/A')
    min_heart_rate = average_metrics.get('min_heart_rate', 'N/A')
    max_heart_rate = average_metrics.get('max_heart_rate', 'N/A')
    hrv_score = average_metrics.get('hrv_score', 'N/A')
    
    html_report = f"""<h2>睡眠情况分析</h2>
<p>根据您提供的数据，对<strong>{date}</strong>的睡眠情况进行详细分析：</p>

<h3>睡眠结构分析</h3>
<ul>
  <li>上床时间：<strong>{bedtime}</strong></li>
  <li>起床时间：<strong>{wakeup_time}</strong></li>
  <li>总卧床时长：<strong>{time_in_bed_minutes}分钟</strong></li>
  <li>睡眠时长：<strong>{sleep_duration_minutes}分钟</strong></li>
  <li>睡眠评分：<strong>{sleep_score}分</strong></li>
  <li>离床次数：<strong>{bed_exit_count}次</strong></li>
  <li>睡眠准备时间：<strong>{sleep_prep_time_minutes}分钟</strong></li>
  <li>睡眠效率：<strong>{sleep_efficiency}%</strong>（≥85%为优秀）</li>
</ul>

<h3>睡眠阶段分布</h3>
<ul>
  <li>深睡占比：<strong>{deep_sleep_percentage}%</strong>（理想范围20%-25%）</li>
  <li>浅睡占比：<strong>{light_sleep_percentage}%</strong></li>
  <li>REM睡眠占比：<strong>{rem_sleep_percentage}%</strong></li>
  <li>清醒占比：<strong>{awake_percentage}%</strong></li>
</ul>

<h3>呼吸相关指标</h3>
<ul>
  <li>平均呼吸率：<strong>{avg_respiratory_rate}次/分钟</strong>（正常12-20次/分钟）</li>
</ul>

<h3>心率数据</h3>
<ul>
  <li>平均心率：<strong>{avg_heart_rate}次/分钟</strong>（正常60-100次/分钟）</li>
  <li>最低心率：<strong>{min_heart_rate}次/分钟</strong></li>
  <li>最高心率：<strong>{max_heart_rate}次/分钟</strong></li>
  <li>HRV分数：<strong>{hrv_score}/100</strong></li>
</ul>

<h3>建议与优化方向</h3>
<ul>
  <li><strong>睡眠时长调整：</strong>当前睡眠时长为{sleep_duration_minutes}分钟，成人推荐7-9小时（420-540分钟），{'符合推荐' if 420 <= sleep_duration_minutes <= 540 else '建议调整至推荐范围'}</li>
  <li><strong>深睡质量：</strong>深睡占比{deep_sleep_percentage}%，{'质量良好' if deep_sleep_percentage >= 20 else '偏低，可通过规律作息提升'}</li>
  <li><strong>睡眠效率：</strong>{'睡眠效率优秀' if sleep_efficiency >= 85 else '睡眠效率一般，建议改善睡眠环境'}</li>
</ul>

<p><strong>总结：</strong>根据{date}的睡眠数据分析，整体睡眠质量{'良好' if sleep_score >= 70 else '需要改善'}，建议关注睡眠时长和深睡质量。</p>"""
    
    return html_report


def format_physiological_data_to_html(physiological_data: Dict[str, Any], date: str) -> str:
    """
    将生理指标数据分析结果格式化为HTML格式
    """
    respiratory_metrics = physiological_data.get('respiratory_metrics', {})
    heart_rate_metrics = physiological_data.get('heart_rate_metrics', {})
    
    # 呼吸指标
    avg_respiratory_rate = respiratory_metrics.get('avg_respiratory_rate', 'N/A')
    apnea_count = respiratory_metrics.get('apnea_count', 'N/A')
    max_apnea_duration = respiratory_metrics.get('max_apnea_duration_seconds', 'N/A')
    avg_apnea_duration = respiratory_metrics.get('avg_apnea_duration_seconds', 'N/A')
    respiratory_health_score = respiratory_metrics.get('respiratory_health_score', 'N/A')
    
    # 心率指标
    avg_heart_rate = heart_rate_metrics.get('avg_heart_rate', 'N/A')
    hrv_score = heart_rate_metrics.get('hrv_score', 'N/A')
    max_heart_rate = heart_rate_metrics.get('max_heart_rate', 'N/A')
    min_heart_rate = heart_rate_metrics.get('min_heart_rate', 'N/A')
    
    html_report = f"""<h2>生理指标分析</h2>
<p>根据您提供的数据，对<strong>{date}</strong>的生理指标进行详细分析：</p>

<h3>呼吸相关指标</h3>
<ul>
  <li>平均呼吸率：<strong>{avg_respiratory_rate}次/分钟</strong>（正常12-20次/分钟）</li>
  <li>呼吸暂停次数：<strong>{apnea_count}次</strong></li>
  <li>最长呼吸暂停：<strong>{max_apnea_duration}秒</strong></li>
  <li>平均呼吸暂停：<strong>{avg_apnea_duration}秒</strong></li>
  <li>呼吸健康评分：<strong>{respiratory_health_score}/100</strong></li>
</ul>

<h3>心率数据</h3>
<ul>
  <li>平均心率：<strong>{avg_heart_rate}次/分钟</strong>（正常60-100次/分钟）</li>
  <li>最低心率：<strong>{min_heart_rate}次/分钟</strong></li>
  <li>最高心率：<strong>{max_heart_rate}次/分钟</strong></li>
  <li>HRV分数：<strong>{hrv_score}/100</strong></li>
</ul>

<h3>健康评估</h3>
<ul>
  <li><strong>呼吸健康：</strong>{'呼吸健康良好' if respiratory_health_score >= 80 else '呼吸健康需要注意'}</li>
  <li><strong>呼吸暂停风险：</strong>{'呼吸暂停风险较低' if apnea_count < 5 else '呼吸暂停风险较高，请咨询医生'}</li>
  <li><strong>心率变异性：</strong>{'心率变异性良好' if hrv_score >= 70 else '心率变异性一般'}</li>
</ul>

<p><strong>总结：</strong>根据{date}的生理指标数据分析，{'生理指标整体健康' if all(score != 'N/A' and score >= 70 for score in [respiratory_health_score, hrv_score]) else '部分生理指标需要注意'}，建议定期监测并保持健康生活方式。</p>"""
    
    return html_report


@tool
def generate_html_sleep_report(date: str) -> str:
    """
    生成HTML格式的睡眠分析报告
    
    功能：根据指定日期的睡眠数据，生成专业的HTML格式睡眠分析报告
    
    参数:
        date: 日期字符串，格式如 '2024-12-20'
    
    返回:
        HTML格式的睡眠分析报告字符串
    
    使用场景:
        - 需要生成专业的睡眠分析报告时
        - 需要HTML格式便于展示和分享时
        - 需要综合展示睡眠结构和生理指标时
    """
    from .sleep_analyzer_tool import analyze_single_day_sleep_data
    
    try:
        # 获取睡眠分析数据
        sleep_json_str = analyze_single_day_sleep_data(date)
        sleep_data = json.loads(sleep_json_str)
        
        if "error" in sleep_data:
            return f"错误：{sleep_data['error']}"
        
        # 格式化为HTML
        html_report = format_sleep_data_to_html(sleep_data, date)
        return html_report
        
    except Exception as e:
        return f"生成HTML睡眠报告失败：{str(e)}"


@tool
def generate_html_physiological_report(date: str) -> str:
    """
    生成HTML格式的生理指标分析报告
    
    功能：根据指定日期的生理指标数据，生成专业的HTML格式生理指标分析报告
    
    参数:
        date: 日期字符串，格式如 '2024-12-20'
    
    返回:
        HTML格式的生理指标分析报告字符串
    
    使用场景:
        - 需要生成专业的生理指标分析报告时
        - 需要HTML格式便于展示和分享时
        - 需要综合展示呼吸和心率指标时
    """
    from .physiological_analyzer_tool import analyze_single_day_physiological_data
    
    try:
        # 获取生理指标分析数据
        physio_json_str = analyze_single_day_physiological_data(date)
        physio_data = json.loads(physio_json_str)
        
        if "error" in physio_data:
            return f"错误：{physio_data['error']}"
        
        # 格式化为HTML
        html_report = format_physiological_data_to_html(physio_data, date)
        return html_report
        
    except Exception as e:
        return f"生成HTML生理指标报告失败：{str(e)}"


@tool
def generate_html_report(date: str, report_type: str = "sleep") -> str:
    """
    生成HTML格式的分析报告（通用接口）
    
    参数:
        date: 日期字符串
        report_type: 报告类型 ("sleep" 或 "physiological")
    
    返回:
        HTML格式的分析报告字符串
    """
    if report_type.lower() == "sleep":
        # 注意：这里不能直接调用generate_html_sleep_report，因为它是一个StructuredTool
        # 我们需要使用内部函数逻辑
        from .sleep_analyzer_tool import analyze_single_day_sleep_data
        try:
            sleep_json_str = analyze_single_day_sleep_data(date)
            sleep_data = json.loads(sleep_json_str)
            
            if "error" in sleep_data:
                return f"错误：{sleep_data['error']}"
            
            html_report = format_sleep_data_to_html(sleep_data, date)
            return html_report
        except Exception as e:
            return f"生成HTML睡眠报告失败：{str(e)}"
            
    elif report_type.lower() == "physiological":
        # 注意：这里不能直接调用generate_html_physiological_report，因为它是一个StructuredTool
        from .physiological_analyzer_tool import analyze_single_day_physiological_data
        try:
            physio_json_str = analyze_single_day_physiological_data(date)
            physio_data = json.loads(physio_json_str)
            
            if "error" in physio_data:
                return f"错误：{physio_data['error']}"
            
            html_report = format_physiological_data_to_html(physio_data, date)
            return html_report
        except Exception as e:
            return f"生成HTML生理指标报告失败：{str(e)}"
    else:
        return f"未知的报告类型：{report_type}"