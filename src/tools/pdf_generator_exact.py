import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
from matplotlib.backends.backend_pdf import PdfPages

# 设置字体和样式
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 8

# 设置中文字体（使用英文标签避免字体问题）
LABEL_CN = {
    'heart_rate': 'Heart Rate',  # 心率
    'respiration_rate': 'Respiration Rate',  # 呼吸频率
    'apnea': 'Apnea',  # 呼吸暂停
}


def parse_data_content(content: str):
    """解析数据内容"""
    import re

    data = {}
    
    hr_match = re.search(r'心率:(\d+)次/分钟', content)
    data['heart_rate'] = int(hr_match.group(1)) if hr_match else 0
    
    rr_match = re.search(r'呼吸:(\d+)次/分钟', content)
    data['respiration_rate'] = int(rr_match.group(1)) if rr_match else 0
    
    apnea_match = re.search(r'呼吸暂停次数:(\d+)次', content)
    data['apnea_count'] = int(apnea_match.group(1)) if apnea_match else 0
    
    body_move_match = re.search(r'体动次数的占比:(\d+)%', content)
    data['body_move_ratio'] = int(body_move_match.group(1)) if body_move_match else 0
    
    return data


def generate_original_style_pdf(file_path: str, output_path: str = None):
    """
    生成原始样式的PDF报告
    
    Args:
        file_path: Excel文件路径
        output_path: 输出PDF路径（可选）
    
    Returns:
        生成的PDF文件路径
    """
    # 读取数据
    df = pd.read_excel(file_path)
    
    # 解析数据
    parsed_data = []
    for idx, row in df.iterrows():
        content = row['数据内容']
        data_dict = parse_data_content(content)
        data_dict['upload_time'] = pd.to_datetime(row['上传时间'])
        data_dict['data_type'] = row['数据类型']
        parsed_data.append(data_dict)
    
    parsed_df = pd.DataFrame(parsed_data)
    parsed_df = parsed_df.sort_values('upload_time').reset_index(drop=True)
    
    # 使用所有数据（包括0值），保留原始时间间隔
    valid_df = parsed_df.copy()
    
    # 生成输出路径（默认保存到 assets 目录）
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = f'assets/{base_name}_夜间统计.pdf'
    
    # 创建PDF
    with PdfPages(output_path) as pdf:
        # 按小时分页
        start_time = valid_df['upload_time'].min()
        end_time = valid_df['upload_time'].max()
        
        # 获取所有小时段
        current_hour = start_time.replace(minute=0, second=0, microsecond=0)
        
        while current_hour < end_time:
            next_hour = current_hour + timedelta(hours=1)
            
            # 获取当前小时的数据
            hour_data = valid_df[
                (valid_df['upload_time'] >= current_hour) &
                (valid_df['upload_time'] < next_hour)
            ].copy()
            
            if len(hour_data) > 0:
                # ===================== 单Y轴重构 =====================
                # 1. 初始化画布：创建1张图，单Y轴显示两条曲线
                fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
                fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.25)
                
                # 2. 提取数据（转换为列表）
                times = hour_data['upload_time'].tolist()
                hr_values = hour_data['heart_rate'].tolist()
                rr_values = hour_data['respiration_rate'].tolist()
                
                # 3. 绘制折线（心率和呼吸频率共用Y轴）
                line1, = ax.plot(times, hr_values, color='#2E86AB', linewidth=2, alpha=0.9, label='Heart Rate', marker='o', markersize=4, markerfacecolor='white', markeredgewidth=1)
                line2, = ax.plot(times, rr_values, color='#A23B72', linewidth=2, alpha=0.9, label='Respiration Rate', marker='s', markersize=4, markerfacecolor='white', markeredgewidth=1)
                
                # 4. 显示数据点数值
                for i, (hr, rr) in enumerate(zip(hr_values, rr_values)):
                    # 心率数值（蓝色）
                    ax.text(times[i], hr, str(hr), fontsize=7, color='#2E86AB', ha='center', va='bottom')
                    # 呼吸频率数值（紫色）
                    ax.text(times[i], rr, str(rr), fontsize=7, color='#A23B72', ha='center', va='top')
                
                # 5. Y轴配置：统一范围0-80
                ax.set_ylim(0, 80)
                ax.set_ylabel('Vital Signs Rate', fontsize=10, fontweight='medium')
                ax.tick_params(axis='y', labelsize=9)
                
                # 6. X轴（时间）配置：避免重叠
                ax.set_xlabel('Time', fontsize=10, fontweight='medium')
                
                # 智能调整时间刻度：如果数据点太多，只显示部分刻度
                num_points = len(times)
                if num_points > 15:
                    # 每隔 n 个点显示一个时间标签
                    step = max(1, num_points // 15)
                    tick_indices = list(range(0, num_points, step))
                    ax.set_xticks([times[i] for i in tick_indices])
                else:
                    ax.set_xticks(times)
                
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.tick_params(axis='x', rotation=45, labelsize=8, pad=5)
                
                # 7. 网格优化
                ax.grid(True, alpha=0.2, axis='y', linestyle='-', linewidth=0.8)
                ax.set_axisbelow(True)
                
                # 8. 标题
                time_range = f"{current_hour.strftime('%Y-%m-%d %H:00')}~{next_hour.strftime('%H:00')}"
                ax.set_title(f'{time_range} Vital Signs Monitor', 
                             fontsize=12, fontweight='bold', pad=10, color='#2C3E50')
                
                # 9. 呼吸暂停事件标记
                apnea_events = hour_data[hour_data['apnea_count'] > 0]
                if len(apnea_events) > 0:
                    for idx, event in apnea_events.iterrows():
                        event_time = event['upload_time']
                        ax.axvline(x=event_time, color='#E74C3C', linestyle='--', alpha=0.8, linewidth=2)
                        if idx == apnea_events.index[0]:
                            ax.text(event_time, 75, 'Apnea Event',
                                   color='#E74C3C', fontsize=9, rotation=90,
                                   verticalalignment='top', fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                # 10. 图例
                ax.legend(loc='upper left', fontsize=9, framealpha=0.9, edgecolor='none')
                
                # 11. 保存
                plt.tight_layout()
                pdf.savefig(fig, dpi=100, bbox_inches='tight')
                plt.close(fig)
            
            current_hour = next_hour
    
    return output_path


# 测试函数
if __name__ == '__main__':
    # 测试
    test_file = 'assets/体征301-2025-12-17夜数据.xlsx'
    output = generate_original_style_pdf(test_file)
    print(f'PDF生成成功: {output}')


# 供langchain tool使用的包装函数
def generate_monitoring_pdf_tool(file_path: str, output_path: str = None) -> str:
    """
    生成原始样式的监护数据PDF报告
    
    Args:
        file_path: 病床监护数据Excel文件的完整路径
        output_path: 输出PDF文件路径（可选，默认保存到assets目录）
    
    Returns:
        生成的PDF文件完整路径
    
    使用场景:
        - 需要按小时查看实时监护曲线时
        - 需要生成原始样式的监护报告时
        - 需要查看心率、呼吸频率的详细变化趋势时
    
    特点:
        - 按小时分页，每页显示1小时的实时数据
        - 使用单Y轴显示：心率曲线和呼吸频率曲线共用Y轴（0-80）
        - 显示数据点数值，便于精确查看
        - 标记呼吸暂停事件
        - X轴时间智能调整，避免重叠
        - 默认保存到 assets 目录，方便直接访问
    """
    try:
        result_path = generate_original_style_pdf(file_path, output_path)
        return result_path
    except Exception as e:
        error_msg = f"PDF生成失败: {str(e)}"
        return error_msg
