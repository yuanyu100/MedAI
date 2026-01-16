import pandas as pd
import json
from datetime import datetime
from langchain.tools import tool
from langchain.tools import ToolRuntime


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


def extract_sleep_info(df):
    """提取睡眠相关信息"""
    info = {}

    # 计算平均体动占比（作为睡眠质量的参考）
    avg_body_move = df['body_move_ratio'].mean()
    info['avg_body_move_ratio'] = round(avg_body_move, 1)

    # 估算深睡时长（简化算法：低体动时段）
    low_move_periods = df[df['body_move_ratio'] < 20]
    if len(low_move_periods) > 0:
        # 假设低体动时段为深睡
        info['deep_sleep_estimate'] = f"约 {len(low_move_periods)} 个低体动时段"
    else:
        info['deep_sleep_estimate'] = "数据不足"

    # 睡眠评分（基于体动频率）
    if avg_body_move < 10:
        info['sleep_score'] = 85  # 优秀
        info['sleep_quality'] = "睡眠质量优秀，体动较少"
    elif avg_body_move < 20:
        info['sleep_score'] = 75  # 良好
        info['sleep_quality'] = "睡眠质量良好，偶有翻身"
    elif avg_body_move < 30:
        info['sleep_score'] = 65  # 一般
        info['sleep_quality'] = "睡眠质量一般，翻身较多"
    else:
        info['sleep_score'] = 55  # 较差
        info['sleep_quality'] = "睡眠质量较差，频繁翻身"

    return info


def extract_heart_rate_info(df):
    """提取心率相关信息"""
    info = {}

    info['avg_hr'] = round(df['heart_rate'].mean(), 1)
    info['min_hr'] = int(df['heart_rate'].min())
    info['max_hr'] = int(df['heart_rate'].max())
    info['hr_range'] = f"{info['min_hr']}-{info['max_hr']} bpm"

    # 心率状态
    if info['avg_hr'] < 60:
        info['hr_status'] = "心率偏慢（心动过缓）"
    elif info['avg_hr'] > 100:
        info['hr_status'] = "心率偏快（心动过速）"
    else:
        info['hr_status'] = "心率正常"

    return info


def extract_respiration_info(df):
    """提取呼吸相关信息"""
    info = {}

    info['avg_rr'] = round(df['respiration_rate'].mean(), 1)
    info['min_rr'] = int(df['respiration_rate'].min())
    info['max_rr'] = int(df['respiration_rate'].max())
    info['rr_range'] = f"{info['min_rr']}-{info['max_rr']} 次/分钟"

    # 呼吸状态
    if info['avg_rr'] < 12:
        info['rr_status'] = "呼吸偏慢（呼吸过缓）"
    elif info['avg_rr'] > 20:
        info['rr_status'] = "呼吸偏快（呼吸过速）"
    else:
        info['rr_status'] = "呼吸频率正常"

    return info


def extract_apnea_info(df):
    """提取呼吸暂停相关信息"""
    info = {}

    total_apnea = df['apnea_count'].sum()
    info['total_apnea'] = int(total_apnea)

    # 计算AHI（每小时暂停次数）
    start_time = df['upload_time'].min()
    end_time = df['upload_time'].max()
    total_hours = (end_time - start_time).total_seconds() / 3600
    if total_hours > 0:
        ahi = round(total_apnea / total_hours, 1)
        info['ahi'] = ahi
    else:
        info['ahi'] = 0

    # 风险评估
    if info['ahi'] < 5:
        info['apnea_risk'] = "正常，无显著呼吸暂停"
    elif info['ahi'] < 15:
        info['apnea_risk'] = "轻度呼吸暂停"
    elif info['ahi'] < 30:
        info['apnea_risk'] = "中度呼吸暂停，建议就医"
    else:
        info['apnea_risk'] = "重度呼吸暂停，需要及时治疗"

    return info


def extract_summary_info(df):
    """提取整体摘要信息"""
    info = {}

    info['data_count'] = len(df)
    info['start_time'] = df['upload_time'].min().strftime('%Y-%m-%d %H:%M')
    info['end_time'] = df['upload_time'].max().strftime('%Y-%m-%d %H:%M')
    duration = df['upload_time'].max() - df['upload_time'].min()
    info['duration_hours'] = round(duration.total_seconds() / 3600, 1)

    # 综合评估
    hr_avg = df['heart_rate'].mean()
    rr_avg = df['respiration_rate'].mean()
    apnea_count = df['apnea_count'].sum()
    body_move_avg = df['body_move_ratio'].mean()

    issues = []
    if hr_avg < 60 or hr_avg > 100:
        issues.append("心率异常")
    if rr_avg < 12 or rr_avg > 20:
        issues.append("呼吸频率异常")
    if apnea_count > 10:
        issues.append("多次呼吸暂停")
    if body_move_avg > 30:
        issues.append("睡眠质量较差")

    if len(issues) == 0:
        info['overall_status'] = "整体状况良好，各项指标正常"
    else:
        info['overall_status'] = f"需要注意：{', '.join(issues)}"

    return info


def match_question_to_category(question: str) -> str:
    """根据问题匹配到相应的数据类别"""
    question_lower = question.lower()

    # 呼吸暂停相关问题（优先级最高，避免被"呼吸"关键词匹配）
    apnea_keywords = ['呼吸暂停', '打呼噜', '憋气', '暂停', '鼾声']
    if any(keyword in question_lower for keyword in apnea_keywords):
        return 'apnea'

    # 睡眠相关问题
    sleep_keywords = ['睡', '睡眠', '质量', '深睡', '浅睡', '翻身', '体动']
    if any(keyword in question_lower for keyword in sleep_keywords):
        return 'sleep'

    # 心率相关问题
    hr_keywords = ['心率', '心跳', '脉搏', '快', '慢', '心动']
    if any(keyword in question_lower for keyword in hr_keywords):
        return 'heart_rate'

    # 呼吸相关问题
    rr_keywords = ['呼吸', '缺氧', '换气', '喘']
    if any(keyword in question_lower for keyword in rr_keywords):
        return 'respiration'

    # 风险/警报问题
    risk_keywords = ['危险', '风险', '警报', '严重', '异常']
    if any(keyword in question_lower for keyword in risk_keywords):
        return 'risk'

    # 总结/汇报问题
    summary_keywords = ['总结', '汇报', '整体', '概括', '情况', '怎么样', '如何']
    if any(keyword in question_lower for keyword in summary_keywords):
        return 'summary'

    return 'summary'  # 默认返回摘要


def format_response(category: str, info: dict) -> str:
    """格式化响应内容"""
    if category == 'sleep':
        return f"""睡眠情况：
- 睡眠评分：{info['sleep_score']}分
- 睡眠质量：{info['sleep_quality']}
- 平均体动占比：{info['avg_body_move_ratio']}%
- 深睡时长估算：{info['deep_sleep_estimate']}"""

    elif category == 'heart_rate':
        return f"""心率情况：
- 平均心率：{info['avg_hr']} bpm
- 心率范围：{info['hr_range']}
- 心率状态：{info['hr_status']}"""

    elif category == 'respiration':
        return f"""呼吸情况：
- 平均呼吸频率：{info['avg_rr']} 次/分钟
- 呼吸范围：{info['rr_range']}
- 呼吸状态：{info['rr_status']}"""

    elif category == 'apnea':
        return f"""呼吸暂停情况：
- 总呼吸暂停次数：{info['total_apnea']}次
- AHI指数：{info['ahi']}（每小时暂停次数）
- 风险评估：{info['apnea_risk']}"""

    elif category == 'risk':
        # 综合风险信息
        risk_info = []
        if 'hr_status' in info and '异常' in info['hr_status']:
            risk_info.append(f"- {info['hr_status']}")
        if 'rr_status' in info and '异常' in info['rr_status']:
            risk_info.append(f"- {info['rr_status']}")
        if 'apnea_risk' in info and '呼吸暂停' in info['apnea_risk']:
            risk_info.append(f"- {info['apnea_risk']}")
        if 'sleep_quality' in info and '较差' in info['sleep_quality']:
            risk_info.append(f"- {info['sleep_quality']}")

        if len(risk_info) > 0:
            return "风险提醒：\n" + "\n".join(risk_info)
        else:
            return "当前没有明显的风险指标，各项指标正常。"

    else:  # summary
        return f"""整体情况：
- 监测时间：{info['start_time']} 至 {info['end_time']}
- 总时长：{info['duration_hours']}小时
- 数据条数：{info['data_count']}条
- 综合评估：{info['overall_status']}"""


def qa_retrieve_internal(table_name: str = "device_data", question: str = "整体情况如何") -> str:
    """
    内部检索函数
    """
    # 问答检索功能暂时禁用，因为数据库查询所有数据不合适
    return "问答检索功能已禁用，因为数据库查询所有数据不合适"


@tool
def qa_retriever(question: str = "整体情况如何", table_name: str = "device_data", runtime: ToolRuntime = None) -> str:
    """
    病床监护数据问答检索工具（To C 专用）

    功能：根据用户的自然语言问题，从病床监护数据库中检索出相关的数据片段。

    Args:
        question: 用户的问题（自然语言字符串）
        table_name: 数据库表名，默认为"device_data"

    Returns:
        与问题相关的数据片段，以简洁易懂的文本形式返回

    支持的问题类型：
        - 睡眠相关：睡得怎么样、睡眠质量、深睡时长、翻身情况等
        - 心率相关：心率多少、心跳快慢、脉搏变化等
        - 呼吸相关：呼吸频率、换气情况、呼吸快慢等
        - 呼吸暂停：呼吸暂停次数、打呼噜、憋气等
        - 风险提醒：危险情况、风险指标、异常警报等
        - 整体总结：整体情况、概括汇报、怎么样等

    使用场景:
        - 患者或家属询问监护数据的具体情况
        - 快速获取某项指标的信息
        - 了解整体健康状况
        - 查看风险提醒

    特点:
        - 支持自然语言提问
        - 自动匹配相关数据类别
        - 返回简洁易懂的回答
        - 适合非专业人士使用

    示例问题：
        - "睡得怎么样"
        - "心率多少"
        - "有呼吸暂停吗"
        - "整体情况如何"
        - "有什么风险"
    """
    result = qa_retrieve_internal(table_name, question)
    return result


# 测试函数
if __name__ == '__main__':
    # 测试
    test_file = 'assets/体征301-2025-12-17夜数据.xlsx'

    test_questions = [
        "睡得怎么样",
        "心率多少",
        "有呼吸暂停吗",
        "整体情况如何",
        "有什么风险"
    ]

    print("=== 问答检索测试 ===\n")
    for question in test_questions:
        print(f"问题：{question}")
        print("回答：")
        result = qa_retrieve_internal(test_file, question)
        print(result)
        print()
        print("-" * 50)
        print()