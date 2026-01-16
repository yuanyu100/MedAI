#!/usr/bin/env python3
"""
简化版智能体 - 用于API调用，避免检查点问题
"""

import os
import json
from typing import Annotated, List
from langchain.agents import create_agent
from langchain_community.chat_models import ChatOllama
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain.tools import tool, ToolRuntime
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 导入病床监护数据分析工具
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'tools'))

# 导入PDF生成工具


# 导入趋势分析工具
# from src.tools.analyze_trend_tool import analyze_trend_and_pattern

# 导入问答检索工具
# from src.tools.qa_retriever import qa_retriever

# 导入病床监护分析工具
from src.tools.bed_monitoring_db_analyzer import analyze_bed_monitoring_from_db, bed_monitoring_db_analyzer_tool

# 导入睡眠分析工具
from src.tools.sleep_analyzer_tool import analyze_sleep_by_date

# 导入数据库工具
from src.tools.database_tool import (
    query_database_tables,
    query_table_data,
    get_available_sleep_dates,
    query_sleep_data_for_date
)

# 导入生理指标分析工具
from src.tools.physiological_analyzer_tool import analyze_physiological_by_date

# 导入HTML报告生成工具
from src.tools.html_report_generator import (
    generate_html_sleep_report,
    generate_html_physiological_report,
    generate_html_report
)

# 无法直接对StructuredTool对象应用装饰器，因为它们已经是工具实例

# LLM_CONFIG = "config/updated_agent_llm_config.json"
LLM_CONFIG = "config/agent_llm_config.json"

# 默认保留最近 20 轮对话 (40 条消息)
MAX_MESSAGES = 40


def _windowed_messages(old: List[BaseMessage], new: List[BaseMessage]) -> List[BaseMessage]:
    """滑动窗口: 只保留最近 MAX_MESSAGES 条消息"""
    return add_messages(old, new)[-MAX_MESSAGES:]  # type: ignore


class AgentState(MessagesState):
    messages: Annotated[List[BaseMessage], _windowed_messages]  # type: ignore



import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    logger.info(f"调用 bed_monitoring_db_analyzer_tool, table_name: {table_name}")
    from src.tools.bed_monitoring_db_analyzer import analyze_bed_monitoring_from_db
    
    result = analyze_bed_monitoring_from_db(table_name)
    logger.info(f"bed_monitoring_db_analyzer_tool 返回结果长度: {len(result) if result else 0}")
    return result


@tool

def nursing_report_visualization_tool(data: str) -> str:
    """
    护理交班报告可视化工具
    
    此工具已被移除，请使用其他方法生成可视化报告。
    """
    logger.info(f"调用 nursing_report_visualization_tool, data: {data[:50]}...")
    return "{\"error\": \"nursing_report_visualization_tool has been removed\"}"


# @tool
def trend_analysis_tool(file_path: str) -> str:
    """
    多天监护数据趋势分析工具（To B 专用）
    
    此工具已被移除，因为analyze_trend_and_pattern是一个StructuredTool对象，不能直接调用。
    """
    logger.info(f"调用 trend_analysis_tool, file_path: {file_path}")
    return "{\"error\": \"trend_analysis_tool has been removed due to StructuredTool call issue\"}"


@tool
def monitoring_pdf_tool(file_path: str, output_path: str = None) -> str:
    """
    监护数据原始样式PDF生成工具
    
    此工具已被移除，请使用其他方法生成PDF报告。
    """
    logger.info(f"调用 monitoring_pdf_tool, file_path: {file_path}, output_path: {output_path}")
    return "{\"error\": \"monitoring_pdf_tool has been removed\"}"


def build_simple_agent():
    """构建简化的病床监护数据分析Agent，不使用检查点功能"""
    
    config_path = os.path.join(os.path.dirname(__file__), LLM_CONFIG)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    # 根据配置选择LLM类型
    model_name = cfg['config'].get("model", "qwen-plus")
    llm_type = cfg['config'].get("llm_type", "openai")  # 默认使用openai API
    
    if llm_type.lower() == "ollama":
        # 使用Ollama模型配置
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        
        # 使用Ollama的ChatOllama类
        from langchain_community.chat_models import ChatOllama
        llm = ChatOllama(
            model=model_name,
            base_url=ollama_base_url,
            temperature=cfg['config'].get('temperature', 0.7),
            timeout=cfg['config'].get('timeout', 600)
        )
    else:
        # 使用OpenAI兼容的API配置
        api_key = os.getenv("QWEN_API_KEY", "sk-2ad6355b98dd43668a5eeb21e50e4642")
        base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=cfg['config'].get('temperature', 0.7),
            streaming=True,
            timeout=cfg['config'].get('timeout', 600)
        )
    
    # 构建工具列表
    tools = [
        bed_monitoring_db_analyzer_tool,
#        trend_analysis_tool,
#        qa_retriever,
        # 新增睡眠分析工具
        analyze_sleep_by_date,
        # 新增生理指标分析工具
        analyze_physiological_by_date,
        # 新增HTML报告生成工具
        generate_html_sleep_report,
        generate_html_physiological_report,
        generate_html_report,
        # 新增数据库查询工具
        query_database_tables,
        query_table_data,
        get_available_sleep_dates,
        query_sleep_data_for_date
    ]
    
    # 创建不带检查点的代理
    return create_agent(
        model=llm,
        system_prompt=cfg.get("sp"),
        tools=tools,
        # 不使用检查点功能，避免上下文管理器问题
        state_schema=AgentState,
    )