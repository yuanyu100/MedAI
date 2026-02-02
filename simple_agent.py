#!/usr/bin/env python3
"""
简化版智能体 - 用于API调用，避免检查点问题
"""

import os
import json
from typing import Annotated, List
from langchain.agents import create_agent
try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain.tools import tool
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

# 导入睡眠分期图工具
from src.tools.sleep_stage_chart_tool import sleep_stage_chart_tool


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
    remaining_steps: int = 150  # 为React Agent添加必需字段


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


def build_simple_agent():
    """构建简化的病床监护数据分析Agent，不使用检查点功能"""
    
    config_path = os.path.join(os.path.dirname(__file__), LLM_CONFIG)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    # 根据配置选择LLM类型
    model_name = cfg['config'].get("model", "qwen-plus")
    llm_type = cfg['config'].get("llm_type", "openai")  # 默认使用openai API
    
    if llm_type.lower() == "ollama":
        # 使用Ollama模型配置，通过OpenAI兼容接口
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        
        # 设置虚拟API密钥，因为Ollama不需要真正的API密钥
        os.environ["OPENAI_API_KEY"] = "NA"
        
        # 使用ChatOpenAI连接Ollama，因为Ollama支持OpenAI兼容的API接口
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model_name,
            base_url=ollama_base_url,
            api_key="NA",  # Ollama不需要真正的API密钥
            temperature=cfg['config'].get('temperature', 0.7),
            timeout=cfg['config'].get('timeout', 600),
            max_retries=2,
        )
        
        # 不需要特殊的包装器，因为ChatOpenAI已正确实现了工具绑定方法
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
        # bed_monitoring_db_analyzer_tool,
#        trend_analysis_tool,
#        qa_retriever,
        # 新增睡眠分析工具
        analyze_sleep_by_date,
        # 新增生理指标分析工具
        analyze_physiological_by_date,
        # 新增睡眠分期图工具
        sleep_stage_chart_tool,
        # 新增数据库查询工具
        query_database_tables,
        query_table_data,
        get_available_sleep_dates,
        query_sleep_data_for_date
    ]
    
    # 创建不带检查点的代理
    # 根据LLM类型选择适当的代理创建方法
    if llm_type.lower() == "ollama":
        # 对于Ollama模型，使用React Agent，它更适合处理工具
        from langgraph.prebuilt import create_react_agent
        
        # 创建代理并返回包含graph的字典
        graph = create_react_agent(llm, tools, state_schema=AgentState)
        return {"graph": graph}
    else:
        # 对于其他模型，使用原有方法
        return create_agent(
            model=llm,
            system_prompt=cfg.get("sp"),
            tools=tools,
            # 不使用检查点功能，避免上下文管理器问题
            state_schema=AgentState,
        )