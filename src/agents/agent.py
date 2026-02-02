import os
import json
from typing import Annotated, List
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain.tools import tool

# from coze_coding_utils.runtime_ctx.context import default_headers
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from storage.memory.memory_saver import get_memory_saver

# 导入病床监护数据分析工具
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))



# 导入趋势分析工具
from tools.analyze_trend_tool import analyze_trend_and_pattern

# 导入问答检索工具
from tools.qa_retriever import qa_retriever

LLM_CONFIG = "config/agent_llm_config.json"

# 默认保留最近 20 轮对话 (40 条消息)
MAX_MESSAGES = 40


def _windowed_messages(old: List[BaseMessage], new: List[BaseMessage]) -> List[BaseMessage]:
    """滑动窗口: 只保留最近 MAX_MESSAGES 条消息"""
    return add_messages(old, new)[-MAX_MESSAGES:]  # type: ignore


class AgentState(MessagesState):
    messages: Annotated[List[BaseMessage], _windowed_messages]  # type: ignore


@tool
def bed_monitoring_db_analyzer_tool(table_name: str = "device_data") -> str:
    """
    病床监护数据库数据分析工具
    
    功能：从数据库分析病床监护数据，生成护理交班报告所需的各种指标和风险评估。
    
    参数:
        table_name: 数据库表名，默认为 "device_data"
    
    返回:
        包含以下内容的JSON分析报告:
        - 监测时段统计（开始时间、结束时间、总时长）
        - 病床占用状态（总卧床时间、总离床时间、长离床事件>15分钟、夜间22:00-06:00离床次数）
        - 生命体征分析（心率范围/平均值、呼吸范围/平均值、异常事件列表、HRV趋势）
        - 呼吸暂停分析（总暂停次数、AHI指数、风险分级、显著事件）
        - 体动与睡眠行为（睡眠效率、体动分析、高体动时段）
        - 晨间评估（05:00-07:00心率趋势、晨峰现象、苏醒状态）
    
    使用场景:
        - 需要从数据库生成护理交班报告时
        - 需要评估患者跌倒风险、心肺功能风险、呼吸暂停风险时
        - 需要分析患者睡眠质量和体动情况时
        - 数据存储在数据库而非Excel文件时
    """
    from tools.bed_monitoring_db_analyzer import analyze_bed_monitoring_from_db
    
    result = analyze_bed_monitoring_from_db(table_name)
    return result

@tool
def trend_analysis_tool(file_path: str) -> str:
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
    from tools.analyze_trend_tool import analyze_trend_and_pattern
    
    result = analyze_trend_and_pattern(file_path)
    return result



def build_agent(ctx=None):
    """构建病床监护数据分析Agent"""
    
    # 优先使用环境变量中的工作空间路径，否则使用当前项目的config目录
    workspace_path = os.getenv("COZE_WORKSPACE_PATH")
    if workspace_path and os.path.exists(os.path.join(workspace_path, LLM_CONFIG)):
        config_path = os.path.join(workspace_path, LLM_CONFIG)
    else:
        # 使用当前项目目录下的config
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), LLM_CONFIG)
        if not os.path.exists(config_path):
            # 如果相对路径不存在，则使用绝对路径
            config_path = os.path.join(os.getcwd(), LLM_CONFIG)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    # 使用用户提供的API密钥
    api_key = os.getenv("QWEN_API_KEY", "sk-2ad6355b98dd43668a5eeb21e50e4642")
    base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    # 根据模型名称选择合适的LLM类
    model_name = cfg['config'].get("model")
    if "qwen" in model_name.lower():
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=cfg['config'].get('temperature', 0.7),
            streaming=True,
            timeout=cfg['config'].get('timeout', 600),
            default_headers=default_headers(ctx) if ctx else {}
        )
    else:
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=cfg['config'].get('temperature', 0.7),
            streaming=True,
            timeout=cfg['config'].get('timeout', 600),
            extra_body={
                "thinking": {
                    "type": cfg['config'].get('thinking', 'disabled')
                }
            },
            default_headers=default_headers(ctx) if ctx else {}
        )
    
    # 构建工具列表
    tools = [
        bed_monitoring_db_analyzer_tool,
        trend_analysis_tool,
        qa_retriever
    ]
    
    return create_agent(
        model=llm,
        system_prompt=cfg.get("sp"),
        tools=tools,
        checkpointer=get_memory_saver(),
        state_schema=AgentState,
    )
