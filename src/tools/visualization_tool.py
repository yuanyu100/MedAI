import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
from datetime import datetime
from typing import Dict, Any, Optional
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 设置中文字体支持
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def setup_chinese_font():
    """设置中文字体"""
    try:
        # 尝试使用系统中的中文字体
        font_path = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont('DroidSansFallback', font_path))
            return 'DroidSansFallback'
        
        # 尝试其他常见字体路径
        alternative_fonts = [
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/System/Library/Fonts/PingFang.ttc',
        ]
        
        for font_file in alternative_fonts:
            if os.path.exists(font_file):
                pdfmetrics.registerFont(TTFont('ChineseFont', font_file))
                return 'ChineseFont'
    except:
        pass
    
    return None  # 使用默认字体


def generate_vital_signs_charts(data: Dict[str, Any], output_dir: str = None) -> Dict[str, str]:
    """
    生成生命体征图表
    
    Args:
        data: 分析数据JSON字符串或字典
        output_dir: 输出目录
    
    Returns:
        生成的图片路径字典
    """
    # 如果输入是字符串，转换为字典
    if isinstance(data, str):
        data = json.loads(data)
    
    # 使用系统临时目录
    if output_dir is None:
        import tempfile
        output_dir = tempfile.gettempdir()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    """
    生成生命体征图表
    
    Args:
        data: 分析数据JSON字符串或字典
        output_dir: 输出目录
    
    Returns:
        生成的图片路径字典
    """
    # 如果输入是字符串，转换为字典
    if isinstance(data, str):
        data = json.loads(data)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = {}
    
    # 1. 心率和呼吸趋势图
    vital_signs = data.get('vital_signs', {})
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 心率图
    hr_range = f"{vital_signs.get('hr_min', 0)} - {vital_signs.get('hr_max', 0)}"
    hr_avg = vital_signs.get('hr_avg', 0)
    ax1.bar(['Min', 'Avg', 'Max'], [vital_signs.get('hr_min', 0), hr_avg, vital_signs.get('hr_max', 0)],
            color=['#3498db', '#2ecc71', '#e74c3c'])
    ax1.set_ylabel('Heart Rate (bpm)')
    ax1.set_title('Heart Rate Statistics')
    ax1.grid(True, alpha=0.3)
    
    for i, v in enumerate([vital_signs.get('hr_min', 0), hr_avg, vital_signs.get('hr_max', 0)]):
        ax1.text(i, v + 1, str(int(v)), ha='center', va='bottom', fontweight='bold')
    
    # 呼吸图
    rr_range = f"{vital_signs.get('rr_min', 0)} - {vital_signs.get('rr_max', 0)}"
    rr_avg = vital_signs.get('rr_avg', 0)
    ax2.bar(['Min', 'Avg', 'Max'], [vital_signs.get('rr_min', 0), rr_avg, vital_signs.get('rr_max', 0)],
            color=['#9b59b6', '#1abc9c', '#f39c12'])
    ax2.set_ylabel('Respiration Rate (bpm)')
    ax2.set_title('Respiration Rate Statistics')
    ax2.grid(True, alpha=0.3)
    
    for i, v in enumerate([vital_signs.get('rr_min', 0), rr_avg, vital_signs.get('rr_max', 0)]):
        ax2.text(i, v + 0.5, str(int(v)), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    vital_signs_path = os.path.join(output_dir, f'vital_signs_{timestamp}.png')
    plt.savefig(vital_signs_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    image_paths['vital_signs'] = vital_signs_path
    
    # 2. 病床占用分析图
    bed_occupancy = data.get('bed_occupancy', {})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 卧床/离床时间对比
    labels = ['In Bed', 'Out of Bed']
    values = [bed_occupancy.get('total_in_bed_hours', 0), bed_occupancy.get('total_out_of_bed_hours', 0)]
    colors = ['#2ecc71', '#e74c3c']
    
    ax1.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title(f'Bed Occupancy Rate: {bed_occupancy.get("bed_occupancy_rate", 0):.1f}%')
    
    # 夜间离床次数
    night_out_count = bed_occupancy.get('night_out_count', 0)
    ax2.bar(['Night Out Count'], [night_out_count], color='#3498db')
    ax2.set_ylabel('Count')
    ax2.set_title('Nighttime Out-of-Bed Events (22:00-06:00)')
    ax2.text(0, night_out_count + 0.1, str(night_out_count), ha='center', va='bottom', fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    bed_occupancy_path = os.path.join(output_dir, f'bed_occupancy_{timestamp}.png')
    plt.savefig(bed_occupancy_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    image_paths['bed_occupancy'] = bed_occupancy_path
    
    # 3. 呼吸暂停分析图
    apnea = data.get('apnea_analysis', {})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # AHI指数和风险等级
    ahi_index = apnea.get('ahi_index', 0)
    risk_level = apnea.get('ahi_risk_level', 'normal')
    
    risk_colors = {
        'normal': '#2ecc71',
        'mild': '#f39c12',
        'moderate': '#e67e22',
        'severe': '#e74c3c'
    }
    
    ax1.bar(['AHI Index'], [ahi_index], color=[risk_colors.get(risk_level, '#95a5a6')])
    ax1.set_ylabel('Events/Hour')
    ax1.set_title(f'Apnea-Hypopnea Index - Risk: {risk_level.upper()}')
    ax1.text(0, ahi_index + 0.5, f'{ahi_index:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    ax1.axhspan(0, 5, alpha=0.2, color='#2ecc71', label='Normal')
    ax1.axhspan(5, 15, alpha=0.2, color='#f39c12', label='Mild')
    ax1.axhspan(15, 30, alpha=0.2, color='#e67e22', label='Moderate')
    ax1.axhspan(30, ahi_index + 5, alpha=0.2, color='#e74c3c', label='Severe')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 呼吸暂停次数
    total_apnea = apnea.get('total_apnea_count', 0)
    significant_events = len(apnea.get('significant_events', []))
    
    ax2.bar(['Total Apnea', 'Significant Events'], [total_apnea, significant_events],
            color=['#3498db', '#e74c3c'])
    ax2.set_ylabel('Count')
    ax2.set_title('Apnea Event Summary')
    for i, v in enumerate([total_apnea, significant_events]):
        ax2.text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    apnea_path = os.path.join(output_dir, f'apnea_analysis_{timestamp}.png')
    plt.savefig(apnea_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    image_paths['apnea_analysis'] = apnea_path
    
    # 4. 体动与睡眠行为图
    body_movement = data.get('body_movement', {})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 睡眠效率
    sleep_efficiency = body_movement.get('sleep_efficiency', 0)
    body_move_avg = body_movement.get('body_move_avg', 0)
    
    ax1.pie([sleep_efficiency, 100 - sleep_efficiency],
            labels=['Sleep', 'Awake/Restless'],
            autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'],
            startangle=90)
    ax1.set_title(f'Sleep Efficiency: {sleep_efficiency:.1f}%')
    
    # 体动分析
    ax2.bar(['Avg Body Move %', 'Max Body Move %'], 
            [body_move_avg, body_movement.get('body_move_max', 0)],
            color=['#9b59b6', '#3498db'])
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Body Movement Analysis')
    for i, v in enumerate([body_move_avg, body_movement.get('body_move_max', 0)]):
        ax2.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    body_move_path = os.path.join(output_dir, f'body_movement_{timestamp}.png')
    plt.savefig(body_move_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    image_paths['body_movement'] = body_move_path
    
    # 5. 晨间评估图
    morning = data.get('morning_assessment', {})
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    morning_hr = morning.get('morning_avg_hr', 0)
    categories = ['<50 (Low)', '50-100 (Normal)', '>100 (High)']
    counts = [0, 0, 0]
    bar_colors = ['#f39c12', '#2ecc71', '#e74c3c']
    
    if morning_hr < 50:
        counts[0] = 1
    elif 50 <= morning_hr <= 100:
        counts[1] = 1
    else:
        counts[2] = 1
    
    ax.bar(categories, counts, color=bar_colors)
    ax.set_ylabel('Status')
    ax.set_title(f'Morning Heart Rate: {morning_hr:.1f} bpm')
    
    # 添加趋势说明
    trend = morning.get('morning_hr_trend', 'no_data')
    waking_status = morning.get('waking_status', 'no_data')
    
    ax.text(0.5, 0.5, f'Trend: {trend}\nStatus: {waking_status}', 
            ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
            transform=ax.transAxes)
    
    plt.tight_layout()
    morning_path = os.path.join(output_dir, f'morning_assessment_{timestamp}.png')
    plt.savefig(morning_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    image_paths['morning_assessment'] = morning_path
    
    return image_paths


def generate_pdf_report(data: Dict[str, Any], image_paths: Dict[str, str], 
                       output_dir: str = None, filename: Optional[str] = None) -> str:
    """
    生成PDF护理交班报告
    
    Args:
        data: 分析数据JSON字符串或字典
        image_paths: 图片路径字典
        output_dir: 输出目录
        filename: PDF文件名（不含扩展名），如果不指定则自动生成
    
    Returns:
        PDF文件路径
    """
    # 如果输入是字符串，转换为字典
    if isinstance(data, str):
        data = json.loads(data)
    
    # 使用系统临时目录
    if output_dir is None:
        import tempfile
        output_dir = tempfile.gettempdir()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    """
    生成PDF护理交班报告
    
    Args:
        data: 分析数据JSON字符串或字典
        image_paths: 图片路径字典
        output_dir: 输出目录
        filename: PDF文件名（不含扩展名），如果不指定则自动生成
    
    Returns:
        PDF文件路径
    """
    # 如果输入是字符串，转换为字典
    if isinstance(data, str):
        data = json.loads(data)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成文件名
    if filename is None:
        timestamp = datetime.now().strftime('%Y-%m-%d')
        filename = f'体征分析_{timestamp}'
    
    pdf_path = os.path.join(output_dir, f'{filename}.pdf')
    
    # 设置中文字体
    chinese_font = setup_chinese_font()
    
    # 创建PDF文档
    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    
    story = []
    styles = getSampleStyleSheet()
    
    # 自定义样式
    if chinese_font:
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontName=chinese_font,
            fontSize=24,
            alignment=TA_CENTER,
            spaceAfter=30
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontName=chinese_font,
            fontSize=16,
            spaceAfter=12
        )
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontName=chinese_font,
            fontSize=10,
            spaceAfter=6
        )
        bold_style = ParagraphStyle(
            'CustomBold',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=10,
            spaceAfter=6
        )
    else:
        title_style = styles['Heading1']
        heading_style = styles['Heading2']
        normal_style = styles['Normal']
        bold_style = styles['Normal']
    
    # 标题
    title = Paragraph("Bed Monitoring Nursing Handover Report", title_style)
    story.append(title)
    story.append(Spacer(1, 12))
    
    # 监测时段信息
    monitoring_period = data.get('monitoring_period', {})
    start_time = monitoring_period.get('start_time', 'N/A')
    end_time = monitoring_period.get('end_time', 'N/A')
    total_hours = monitoring_period.get('total_hours', 0)
    
    period_info = f"""
    <b>Monitoring Period:</b><br/>
    Start Time: {start_time}<br/>
    End Time: {end_time}<br/>
    Total Duration: {total_hours} hours
    """
    story.append(Paragraph(period_info, normal_style))
    story.append(Spacer(1, 20))
    
    # 1. 病床占用状态
    story.append(Paragraph("1. Bed Occupancy Status", heading_style))
    bed_occupancy = data.get('bed_occupancy', {})
    bed_info = f"""
    <b>Total In Bed:</b> {bed_occupancy.get('total_in_bed_hours', 0)} hours<br/>
    <b>Total Out of Bed:</b> {bed_occupancy.get('total_out_of_bed_hours', 0)} hours<br/>
    <b>Bed Occupancy Rate:</b> {bed_occupancy.get('bed_occupancy_rate', 0)}%<br/>
    <b>Night Out-of-Bed Events (22:00-06:00):</b> {bed_occupancy.get('night_out_count', 0)}
    """
    story.append(Paragraph(bed_info, normal_style))
    
    # 添加床占用图表
    if 'bed_occupancy' in image_paths and os.path.exists(image_paths['bed_occupancy']):
        img = Image(image_paths['bed_occupancy'], width=6*inch, height=2.5*inch)
        story.append(img)
        story.append(Spacer(1, 12))
    
    # 长离床事件
    long_out_events = bed_occupancy.get('long_out_of_bed_events', [])
    if long_out_events:
        story.append(Paragraph("<b>Long Out-of-Bed Events (>15 min):</b>", bold_style))
        for event in long_out_events:
            event_str = f"- {event['start_time']} to {event['end_time']} ({event['duration_minutes']} min)"
            story.append(Paragraph(event_str, normal_style))
    
    story.append(Spacer(1, 15))
    
    # 2. 生命体征分析
    story.append(Paragraph("2. Vital Signs Analysis", heading_style))
    vital_signs = data.get('vital_signs', {})
    vital_info = f"""
    <b>Heart Rate:</b> Min {vital_signs.get('hr_min', 0)} / Avg {vital_signs.get('hr_avg', 0)} / Max {vital_signs.get('hr_max', 0)} bpm<br/>
    <b>Respiration Rate:</b> Min {vital_signs.get('rr_min', 0)} / Avg {vital_signs.get('rr_avg', 0)} / Max {vital_signs.get('rr_max', 0)} bpm<br/>
    <b>HRV Trend:</b> {vital_signs.get('hrv_trend', 'unknown')}
    """
    story.append(Paragraph(vital_info, normal_style))
    
    # 添加生命体征图表
    if 'vital_signs' in image_paths and os.path.exists(image_paths['vital_signs']):
        img = Image(image_paths['vital_signs'], width=6*inch, height=4*inch)
        story.append(img)
        story.append(Spacer(1, 12))
    
    # 心率异常事件
    hr_abnormalities = vital_signs.get('hr_abnormalities', [])
    if hr_abnormalities:
        story.append(Paragraph("<b>Heart Rate Abnormalities:</b>", bold_style))
        for i, event in enumerate(hr_abnormalities[:5]):
            event_str = f"{i+1}. {event['time']} - {event['type']} ({event['value']} bpm) - {event['severity']}"
            story.append(Paragraph(event_str, normal_style))
    
    # 呼吸异常事件
    rr_abnormalities = vital_signs.get('rr_abnormalities', [])
    if rr_abnormalities:
        story.append(Paragraph("<b>Respiration Rate Abnormalities:</b>", bold_style))
        for i, event in enumerate(rr_abnormalities[:5]):
            event_str = f"{i+1}. {event['time']} - {event['type']} ({event['value']} bpm) - {event['severity']}"
            story.append(Paragraph(event_str, normal_style))
    
    story.append(Spacer(1, 15))
    story.append(PageBreak())
    
    # 3. 呼吸暂停分析
    story.append(Paragraph("3. Apnea Analysis", heading_style))
    apnea = data.get('apnea_analysis', {})
    apnea_info = f"""
    <b>Total Apnea Events:</b> {apnea.get('total_apnea_count', 0)}<br/>
    <b>AHI Index:</b> {apnea.get('ahi_index', 0)} events/hour<br/>
    <b>Risk Level:</b> {apnea.get('ahi_risk_level', 'unknown').upper()}
    """
    story.append(Paragraph(apnea_info, normal_style))
    
    # 添加呼吸暂停图表
    if 'apnea_analysis' in image_paths and os.path.exists(image_paths['apnea_analysis']):
        img = Image(image_paths['apnea_analysis'], width=6*inch, height=2.5*inch)
        story.append(img)
        story.append(Spacer(1, 12))
    
    # 显著事件
    significant_events = apnea.get('significant_events', [])
    if significant_events:
        story.append(Paragraph("<b>Significant Events (with HR compensation):</b>", bold_style))
        for i, event in enumerate(significant_events[:5]):
            event_str = f"{i+1}. {event['time']} - {event['count']} apnea events, HR: {event['heart_rate']} bpm, Compensation: {event['hr_compensation']}"
            story.append(Paragraph(event_str, normal_style))
    
    story.append(Spacer(1, 15))
    
    # 4. 体动与睡眠行为
    story.append(Paragraph("4. Body Movement & Sleep Behavior", heading_style))
    body_movement = data.get('body_movement', {})
    body_info = f"""
    <b>Sleep Efficiency:</b> {body_movement.get('sleep_efficiency', 0)}%<br/>
    <b>Avg Body Movement:</b> {body_movement.get('body_move_avg', 0)}%<br/>
    <b>Max Body Movement:</b> {body_movement.get('body_move_max', 0)}%<br/>
    <b>Analysis:</b> {body_movement.get('body_move_analysis', 'unknown')}
    """
    story.append(Paragraph(body_info, normal_style))
    
    # 添加体动图表
    if 'body_movement' in image_paths and os.path.exists(image_paths['body_movement']):
        img = Image(image_paths['body_movement'], width=6*inch, height=2.5*inch)
        story.append(img)
        story.append(Spacer(1, 12))
    
    # 高体动时期
    high_body_move_periods = body_movement.get('body_move_high_periods', [])
    if high_body_move_periods:
        story.append(Paragraph("<b>High Body Movement Periods (>10%):</b>", bold_style))
        for i, event in enumerate(high_body_move_periods[:5]):
            event_str = f"{i+1}. {event['time']} - {event['body_move_ratio']}%, HR: {event['heart_rate']} bpm"
            story.append(Paragraph(event_str, normal_style))
    
    story.append(Spacer(1, 15))
    
    # 5. 晨间评估
    story.append(Paragraph("5. Morning Assessment", heading_style))
    morning = data.get('morning_assessment', {})
    morning_info = f"""
    <b>Morning HR Trend:</b> {morning.get('morning_hr_trend', 'no_data')}<br/>
    <b>Morning Peak Detected:</b> {'Yes' if morning.get('morning_peak_detected') else 'No'}<br/>
    <b>Waking Status:</b> {morning.get('waking_status', 'no_data')}
    """
    story.append(Paragraph(morning_info, normal_style))
    
    # 添加晨间评估图表
    if 'morning_assessment' in image_paths and os.path.exists(image_paths['morning_assessment']):
        img = Image(image_paths['morning_assessment'], width=6*inch, height=2.5*inch)
        story.append(img)
        story.append(Spacer(1, 12))
    
    # 报告生成时间
    report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"<b>Report Generated:</b> {report_time}", normal_style))
    
    # 构建PDF
    doc.build(story)
    
    return pdf_path


def generate_visualization_report(data: str, output_dir: str = None) -> str:
    """
    生成完整的可视化报告（图片和PDF）
    
    Args:
        data: 分析数据JSON字符串
        output_dir: 输出目录
    
    Returns:
        包含生成结果的JSON字符串
    """
    # 使用系统临时目录
    if output_dir is None:
        import tempfile
        output_dir = tempfile.gettempdir()
    """
    生成完整的可视化报告（图片和PDF）
    
    Args:
        data: 分析数据JSON字符串
        output_dir: 输出目录
    
    Returns:
        包含生成结果的JSON字符串
    """
    try:
        # 生成图片
        image_paths = generate_vital_signs_charts(data, output_dir)
        
        # 生成PDF
        pdf_path = generate_pdf_report(data, image_paths, output_dir)
        
        result = {
            'success': True,
            'pdf_path': pdf_path,
            'images': image_paths,
            'message': 'Visualization report generated successfully'
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    except Exception as e:
        result = {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__,
            'message': 'Failed to generate visualization report'
        }
        return json.dumps(result, ensure_ascii=False, indent=2)


# 供langchain tool使用的包装函数
def generate_nursing_report_visualization(data: str, runtime=None) -> str:
    """
    生成护理交班报告的可视化图表和PDF文件
    
    Args:
        data: 病床监护分析结果的JSON字符串（来自bed_monitoring_analyzer_tool）
        runtime: ToolRuntime对象（可选）
    
    Returns:
        包含以下内容的JSON字符串:
        - success: 是否成功
        - pdf_path: 生成的PDF文件路径
        - images: 生成的所有图片路径字典
        - message: 操作结果消息
    
    使用示例:
        1. 首先使用 bed_monitoring_analyzer_tool 分析Excel数据
        2. 将分析结果传入本工具生成可视化报告
        3. 生成的PDF和图片将保存到指定目录
    """
    import tempfile
    return generate_visualization_report(data, output_dir=tempfile.gettempdir())
