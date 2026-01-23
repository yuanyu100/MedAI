#!/usr/bin/env python3
"""
改进版智能体 - 支持外部系统提示词文件和数据存储
"""

import os
import json
import logging
from datetime import datetime, timedelta, time
from typing import Dict, Any, Optional
import hashlib
import pandas as pd
from typing import Annotated, List
import threading
import openai

try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnablePassthrough


"""
改进版智能体 - 支持外部系统提示词文件和数据存储
"""

import os
import json
import logging
from datetime import datetime, timedelta, time
from typing import Dict, Any, Optional
import hashlib
import pandas as pd
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 导入工具
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'tools'))

# 导入各种分析工具
from src.tools.sleep_analyzer_tool import analyze_sleep_by_date, analyze_single_day_sleep_data, analyze_single_day_sleep_data_with_device
from src.tools.physiological_analyzer_tool import analyze_physiological_by_date, analyze_single_day_physiological_data, analyze_single_day_physiological_data_with_device
from src.tools.database_tool import (
    query_database_tables,
    query_table_data,
    get_available_sleep_dates,
    query_sleep_data_for_date
)

# 数据库相关导入
from src.db.database import get_db_manager

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


def load_system_prompt():
    """加载外部系统提示词文件"""
    prompt_file = os.path.join(os.path.dirname(__file__), 'system_prompt.txt')
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"系统提示词文件未找到: {prompt_file}")
        # 返回默认提示词
        return """
你是一个专业的睡眠健康分析师，专门分析用户的睡眠数据并提供个性化的改善建议。
"""


def save_analysis_result_async(query: str, result: str, date: str = None):
    """异步保存分析结果到数据库"""
    def _save():
        try:
            db_manager = get_db_manager()
            
            # 生成唯一ID
            unique_id = hashlib.md5(f"{query}_{datetime.now().isoformat()}".encode()).hexdigest()
            
            # 清理结果中的特殊字符和emoji以避免数据库错误
            import re
            # 移除所有emoji和其他特殊字符
            cleaned_result = re.sub(r'[\x80-\xFF]{4}|[\U0001F000-\U0001FFFF]', '', result)
            # 进一步清理其他潜在的非法字符
            cleaned_result = cleaned_result.encode('utf-8', errors='ignore').decode('utf-8')
            
            # 准备数据
            analysis_record = {
                'id': unique_id,
                'query': query,
                'result': cleaned_result,
                'date': date or datetime.now().strftime('%Y-%m-%d'),
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 创建表（如果不存在）
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS analysis_results (
                id VARCHAR(64) PRIMARY KEY,
                query TEXT,
                result TEXT,
                date VARCHAR(20),
                created_at DATETIME
            )
            """
            # 使用execute_command执行CREATE TABLE命令
            try:
                db_manager.execute_command(create_table_sql)
            except:
                # 如果创建表失败，继续执行，因为表可能已经存在
                pass
            
            # 尝试执行插入操作，如果出错则跳过
            try:
                # 使用参数化查询，避免SQL注入和格式化问题
                insert_sql = """
                INSERT INTO analysis_results (id, query, result, date, created_at)
                VALUES (:id, :query, :result, :date, :created_at)
                """
                # 准备参数字典
                params = {
                    'id': analysis_record['id'],
                    'query': analysis_record['query'],
                    'result': analysis_record['result'],
                    'date': analysis_record['date'],
                    'created_at': analysis_record['created_at']
                }
                # 使用execute_command执行插入命令
                db_manager.execute_command(insert_sql, params)
                
                logger.info(f"分析结果已异步保存到数据库，ID: {unique_id}")
            except Exception as e:
                logger.warning(f"保存分析结果到数据库时出现警告: {e}")
                # 如果数据库保存失败，至少记录日志
                pass
                
        except Exception as e:
            logger.error(f"保存分析结果到数据库失败: {e}")
    
    # 在后台线程中执行保存操作
    save_thread = threading.Thread(target=_save, daemon=True)
    save_thread.start()


def save_analysis_result(query: str, result: str, date: str = None):
    """保存分析结果到数据库（同步版本，保留向后兼容）"""
    try:
        db_manager = get_db_manager()
        
        # 生成唯一ID
        unique_id = hashlib.md5(f"{query}_{datetime.now().isoformat()}".encode()).hexdigest()
        
        # 清理结果中的特殊字符和emoji以避免数据库错误
        import re
        # 移除所有emoji和其他特殊字符
        cleaned_result = re.sub(r'[\x80-\xFF]{4}|[\U0001F000-\U0001FFFF]', '', result)
        # 进一步清理其他潜在的非法字符
        cleaned_result = cleaned_result.encode('utf-8', errors='ignore').decode('utf-8')
        
        # 准备数据
        analysis_record = {
            'id': unique_id,
            'query': query,
            'result': cleaned_result,
            'date': date or datetime.now().strftime('%Y-%m-%d'),
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 创建表（如果不存在）
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS analysis_results (
            id VARCHAR(64) PRIMARY KEY,
            query TEXT,
            result TEXT,
            date VARCHAR(20),
            created_at DATETIME
        )
        """
        # 使用execute_command执行CREATE TABLE命令
        try:
            db_manager.execute_command(create_table_sql)
        except:
            # 如果创建表失败，继续执行，因为表可能已经存在
            pass
        
        # 尝试执行插入操作，如果出错则跳过
        try:
            # 使用参数化查询，避免SQL注入和格式化问题
            insert_sql = """
            INSERT INTO analysis_results (id, query, result, date, created_at)
            VALUES (:id, :query, :result, :date, :created_at)
            """
            # 准备参数字典
            params = {
                'id': analysis_record['id'],
                'query': analysis_record['query'],
                'result': analysis_record['result'],
                'date': analysis_record['date'],
                'created_at': analysis_record['created_at']
            }
            # 使用execute_command执行插入命令
            db_manager.execute_command(insert_sql, params)
            
            logger.info(f"分析结果已保存到数据库，ID: {unique_id}")
        except Exception as e:
            logger.warning(f"保存分析结果到数据库时出现警告: {e}")
            # 如果数据库保存失败，至少记录日志
            pass
            
    except Exception as e:
        logger.error(f"保存分析结果到数据库失败: {e}")


def load_agent_config(config_file_path=None):
    """加载智能体配置"""
    if config_file_path is None:
        # 默认配置文件路径 - 使用硅基流动配置
        config_file_path = os.path.join(os.path.dirname(__file__), 'config', 'qwen3_8b_siliconflow_config.json')
    
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"智能体配置加载成功: {config_file_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"配置文件未找到: {config_file_path}，使用默认硅基流动配置")
        # 默认使用硅基流动配置
        return {
            "config": {
                "llm_type": "openai",  # 使用openai兼容接口
                "model": "Pro/zai-org/GLM-4.7",  # 使用GLM-4.7模型
                "temperature": 0.3,
                "top_p": 0.9,
                "max_completion_tokens": 8000,
                "timeout": 600,
                "base_url": "https://api.siliconflow.cn/v1",  # 硅基流动API地址
                "api_key": os.getenv("SILICONFLOW_API_KEY", ""),  # 硅基流动API密钥
                "thinking": "disabled"
            },
            "sp": load_system_prompt(),
            "tools": []
        }
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        raise


def build_improved_agent(config_file_path=None):
    """构建改进版智能体，支持从配置文件加载"""
    # 加载配置
    config = load_agent_config(config_file_path)
    
    # 获取系统提示词
    system_prompt = config.get("sp", load_system_prompt())
    
    llm_config = config["config"]
    llm_type = llm_config.get("llm_type", "ollama")

    print(f"LLM Type: {llm_type}")
    
    if llm_type == "ollama":
        # 使用Ollama模型
        model = llm_config.get("model", "qwen3:4b")
        temperature = llm_config.get("temperature", 0.3)
        top_p = llm_config.get("top_p", 0.9)
        timeout = llm_config.get("timeout", 600)
        
        llm = ChatOllama(
            model=model,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            num_predict=llm_config.get("max_completion_tokens", 8000)
        )
        
        # 创建一个简单的工具运行时
        class SimpleToolRuntime:
            def __init__(self):
                self.tools = {}
            
            def register(self, name, func):
                self.tools[name] = func
        
        # 注册工具
        runtime = SimpleToolRuntime()
        
        # 将系统提示词和工具信息整合到一起
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        chain = prompt | llm
        
        # 返回一个可调用对象
        def agent_callable(inputs, config=None):
            if isinstance(inputs, dict):
                if "messages" in inputs:
                    # 处理消息列表，提取第一个消息的内容
                    messages_list = inputs["messages"]
                    if messages_list:
                        first_message = messages_list[0]
                        # 检查是否是HumanMessage对象
                        if hasattr(first_message, 'content'):
                            input_text = first_message.content
                        else:
                            input_text = str(first_message)
                    else:
                        input_text = str(inputs)
                else:
                    input_text = str(inputs)
            else:
                input_text = str(inputs)
            
            response = chain.invoke({"input": input_text})
            return {"messages": [response]}
        
        return agent_callable
    
    elif llm_type == "openai":
        # 使用OpenAI兼容的API（如硅基流动）
        model = llm_config.get("model", "Pro/zai-org/GLM-4.7")
        temperature = llm_config.get("temperature", 0.3)
        top_p = llm_config.get("top_p", 0.9)
        timeout = llm_config.get("timeout", 600)
        base_url = llm_config.get("base_url", "https://api.openai.com/v1")
        api_key = llm_config.get("api_key", os.getenv("OPENAI_API_KEY", ""))
        
        # 设置OpenAI客户端
        client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout
        )
        
        # 创建一个简单的代理函数
        def agent_callable(inputs, config=None):
            if isinstance(inputs, dict):
                if "messages" in inputs:
                    # 如果输入已经是消息格式
                    input_messages = inputs["messages"]
                    # 处理HumanMessage对象
                    messages = []
                    for msg in input_messages:
                        if hasattr(msg, 'content'):  # HumanMessage或其他langchain消息对象
                            messages.append({"role": getattr(msg, 'role', 'user'), "content": msg.content})
                        elif isinstance(msg, dict) and 'content' in msg:
                            # 如果已经是字典格式
                            messages.append(msg)
                        else:
                            # 其他情况转换为字符串
                            messages.append({"role": "user", "content": str(msg)})
                else:
                    # 如果输入是其他格式，提取内容
                    input_text = str(inputs)
                    messages = [{"role": "user", "content": input_text}]
            else:
                input_text = str(inputs)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ]
            
            # 调用API
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=llm_config.get("max_completion_tokens", 8000)
                )
                
                # 返回格式化响应
                content = response.choices[0].message.content
                return {
                    "messages": [{"role": "assistant", "content": content}],
                    "raw_response": response
                }
            except Exception as e:
                logger.error(f"调用API失败: {e}")
                raise
        
        return agent_callable
    
    else:
        raise ValueError(f"不支持的LLM类型: {llm_type}")


def get_cached_analysis(query: str, date: str):
    """获取缓存的分析结果"""
    try:
        db_manager = get_db_manager()
        
        # 统一日期格式 - 确保日期格式与数据库中存储的一致
        # 如果传入的是 "2026-1-22" 格式，将其标准化为 "2026-01-22"
        import re
        if re.match(r'^\d{4}-\d{1,2}-\d{1,2}$', date):
            # 将 "2026-1-22" 格式转换为 "2026-01-22" 格式
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            standardized_date = date_obj.strftime('%Y-%m-%d')
        else:
            standardized_date = date
        
        # 使用命名参数而不是位置参数，以确保与SQLAlchemy兼容
        sql = """
        SELECT result FROM analysis_results 
        WHERE  date = :date
        ORDER BY created_at DESC 
        LIMIT 1
        """
        params = {'query': query, 'date': standardized_date}
        logger.debug(f"SQL: {sql}, Params: {params}")
        
        result_df = db_manager.execute_query(sql, params)
        logger.debug(f"Result DataFrame: {result_df}")
        
        if result_df is not None and not result_df.empty:
            cached_result = result_df.iloc[0]['result']
            logger.info(f"找到缓存的分析结果，日期: {standardized_date}")
            return cached_result
        else:
            logger.info(f"未找到缓存的分析结果，日期: {standardized_date}")
            return None
            
    except Exception as e:
        logger.warning(f"获取缓存分析结果失败: {e}", exc_info=True)
        return None


def run_improved_agent(date: str, thread_id: str = "default-session", force_refresh: bool = False, include_formatted_time: bool = False, formatted_time_input: str = None, config_file_path: str = None, device_sn: str = None):
    """
    运行改进版智能体，支持缓存和结果保存
    """
    # 构建基于日期的查询
    logger.debug(f"Running improved agent for date: {date}")
    print(f"Running improved agent for date: {date}")
    
    # 初始化query变量
    query = f"请分析 {date} 的睡眠数据"
    if device_sn:
        query = f"[设备序列号: {device_sn}] {query}"
    
    # 初始化formatted_time_input变量，确保在整个函数作用域内可用
    formatted_time_input_local = formatted_time_input
    
    if formatted_time_input:
        logger.debug(f"Using formatted time input: {formatted_time_input}") 
        # 如果提供了格式化的睡眠时间信息，直接使用它作为主要输入
        query = f"请直接分析以下睡眠数据，不要调用其他工具。原始数据如下：\n\n{formatted_time_input}\n\n请根据以上睡眠数据，提供专业的睡眠分析和建议。"
    elif include_formatted_time:
        # 如果需要包含格式化的睡眠时间信息
        print(f"正在处理日期 {date} 的格式化时间信息，设备: {device_sn}, 强制刷新: {force_refresh}")
        logger.debug(f"正在处理日期 {date} 的格式化时间信息，设备: {device_sn}, 强制刷新: {force_refresh}")
        formatted_time_str = ""
        try:
            # 获取睡眠数据用于格式化时间
            if device_sn:
                # 如果提供了设备序列号，使用带设备的函数
                logger.debug(f"Using device-specific function to analyze sleep data for device: {device_sn}")
                print(f"正在使用设备 {device_sn} 查询睡眠数据")
                sleep_data_json = analyze_single_day_sleep_data_with_device(date, device_sn)
            else:
                print(f"正在查询通用睡眠数据")
                sleep_data_json = analyze_single_day_sleep_data(date)
            sleep_data_raw = json.loads(sleep_data_json)
            logger.debug(f"Raw sleep data: {sleep_data_raw}")
            
            # 检查是否为包含 success/data 结构的响应
            if isinstance(sleep_data_raw, dict) and 'data' in sleep_data_raw:
                sleep_data = sleep_data_raw.get('data', {})
            else:
                # 如果已经是数据结构，则直接使用
                sleep_data = sleep_data_raw
            
            logger.debug(f"Processed sleep data: {sleep_data}")
            print(f"获取到的睡眠数据: {sleep_data}")

            if "error" not in sleep_data:
                # 检查是否有数据（通过检查关键字段是否为0）
                has_real_data = (
                    sleep_data.get('time_in_bed_minutes', 0) > 0 or 
                    sleep_data.get('sleep_duration_minutes', 0) > 0 or
                    sleep_data.get('average_metrics', {}).get('avg_heart_rate', 0) > 0 or
                    sleep_data.get('average_metrics', {}).get('avg_respiratory_rate', 0) > 0
                )
                
                print(f"数据检查结果 - time_in_bed_minutes: {sleep_data.get('time_in_bed_minutes', 0)}, sleep_duration_minutes: {sleep_data.get('sleep_duration_minutes', 0)}, avg_heart_rate: {sleep_data.get('average_metrics', {}).get('avg_heart_rate', 0)}, avg_respiratory_rate: {sleep_data.get('average_metrics', {}).get('avg_respiratory_rate', 0)}")
                print(f"是否有真实数据: {has_real_data}")
                
                if not has_real_data:
                    # 如果没有实际数据，返回简洁的无数据信息
                    print(f"没有找到 {date} 的真实睡眠数据，返回无数据信息")
                    logger.info(f"没有找到 {date} 的真实睡眠数据，设备: {device_sn}")
                    return "暂无数据分析"
                
                # 从数据中提取时间信息并格式化
                bedtime_str = sleep_data.get('bedtime', '')
                wakeup_time_str = sleep_data.get('wakeup_time', '')
                time_in_bed_minutes = sleep_data.get('time_in_bed_minutes', 0)
                sleep_duration_minutes = sleep_data.get('sleep_duration_minutes', 0)
                sleep_prep_time = sleep_data.get('sleep_prep_time_minutes', 0)
                deep_sleep_minutes = sleep_data.get('sleep_phases', {}).get('deep_sleep_minutes', 0)
                deep_sleep_ratio = sleep_data.get('sleep_phases', {}).get('deep_sleep_percentage', 0)  # 修正键名为 'deep_sleep_percentage'
                bed_exit_count = sleep_data.get('bed_exit_count', 0)
                
                print(f"提取的时间信息 - bedtime: {bedtime_str}, wakeup_time: {wakeup_time_str}, time_in_bed: {time_in_bed_minutes}, sleep_duration: {sleep_duration_minutes}")
                
                # 获取生理指标数据
                if device_sn:
                    # 如果提供了设备序列号，使用带设备的函数
                    print(f"正在使用设备 {device_sn} 查询生理指标数据")
                    physio_data_json = analyze_single_day_physiological_data_with_device(date, device_sn)
                else:
                    print(f"正在查询通用生理指标数据")
                    physio_data_json = analyze_single_day_physiological_data(date)
                            
                physio_data_raw = json.loads(physio_data_json)
                # 检查是否为包含 success/data 结构的响应
                if isinstance(physio_data_raw, dict) and 'data' in physio_data_raw:
                    physio_data = physio_data_raw.get('data', {})
                else:
                    # 如果已经是数据结构，则直接使用
                    physio_data = physio_data_raw
                print(f"获取到的生理指标数据: {physio_data}")
                
                avg_respiratory_rate = 0
                min_respiratory_rate = 0
                max_respiratory_rate = 0
                avg_heart_rate = 0
                min_heart_rate = 0
                max_heart_rate = 0
                apnea_events_per_hour = 0
                max_apnea_duration = 0
                
                if "error" not in physio_data:
                    avg_respiratory_rate = physio_data.get('respiratory_metrics', {}).get('avg_respiratory_rate', 0)
                    min_respiratory_rate = physio_data.get('respiratory_metrics', {}).get('min_respiratory_rate', 0)
                    max_respiratory_rate = physio_data.get('respiratory_metrics', {}).get('max_respiratory_rate', 0)
                    avg_heart_rate = physio_data.get('heart_rate_metrics', {}).get('avg_heart_rate', 0)
                    min_heart_rate = physio_data.get('heart_rate_metrics', {}).get('min_heart_rate', 0)
                    max_heart_rate = physio_data.get('heart_rate_metrics', {}).get('max_heart_rate', 0)
                    apnea_events_per_hour = physio_data.get('respiratory_metrics', {}).get('apnea_events_per_hour', 0)
                    max_apnea_duration = physio_data.get('respiratory_metrics', {}).get('max_apnea_duration', 0)
                
                print(f"生理指标数据 - avg_respiratory_rate: {avg_respiratory_rate}, avg_heart_rate: {avg_heart_rate}")
                
                # 格式化为自然语言描述
                formatted_time_str = f"昨晚我{bedtime_str[11:16]}上床，{bedtime_str[11:16]}入睡，{wakeup_time_str[11:16]}醒来，总卧床时长为{time_in_bed_minutes//60}小时{time_in_bed_minutes%60}分，睡眠时长为{sleep_duration_minutes//60}小时{sleep_duration_minutes%60}分。\n"
                formatted_time_str += f"其中，睡眠准备期为{sleep_prep_time//60}小时{sleep_prep_time%60}分，深睡时长为{deep_sleep_minutes//60}小时{deep_sleep_minutes%60}分，深睡占比为{deep_sleep_ratio:.2f}%，中间有{bed_exit_count}次离床。\n"
                formatted_time_str += f"昨晚的睡眠中，我的平均呼吸率为{avg_respiratory_rate:.1f}次/分钟，最低呼吸率为{min_respiratory_rate:.1f}次/分钟，最高呼吸率为{max_respiratory_rate:.1f}次/分钟，呼吸暂停为{apnea_events_per_hour:.1f}次/小时，平均呼吸暂停时长为0.0秒，最长呼吸暂停时长为{max_apnea_duration:.1f}秒。昨晚的睡眠中，我的平均心率为{avg_heart_rate:.1f}次/分钟，最低心率为{min_heart_rate:.1f}次/分钟，最高心率为{max_heart_rate:.1f}次/分钟。请对我昨晚的睡眠情况进行分析，并给出相关建议。"
                
                formatted_time_input = formatted_time_str

                print(formatted_time_input+"注意！！improved_agent533")
                
                # 更新查询以使用格式化的数据
                query = f"请分析以下睡眠数据:\n\n{formatted_time_input}\n\n请根据以上数据，提供专业的睡眠分析和建议。"
            else:
                # 如果睡眠数据中有错误，检查是否是无数据的情况
                print(f"睡眠数据包含错误: {sleep_data}")
                if "暂无数据" in str(sleep_data) or sleep_data.get('summary') == "暂无数据":
                    logger.warning("暂无数据537")
                    return "暂无数据分析"
                else:
                    # 如果是其他错误，继续处理
                    formatted_time_str = f"请根据睡眠数据，提供专业的睡眠分析和建议。"
                    formatted_time_input = formatted_time_str
                    
                    # 更新查询以使用格式化的数据
                    query = f"请分析以下睡眠数据:\n\n{formatted_time_input}\n\n请根据以上数据，提供专业的睡眠分析和建议。"
        except Exception as e:
            logger.warning(f"获取格式化睡眠时间信息失败: {e}")
            print(f"获取格式化睡眠时间信息失败: {e}")
            # 如果获取格式化时间过程中出现错误，返回简洁的无数据信息
            return "暂无数据分析"
        
        # 如果有格式化的数据输入，使用它更新查询
        if 'formatted_time_input' in locals():
            query = f"请分析以下睡眠数据:\n\n{formatted_time_input}\n\n请根据以上数据，提供专业的睡眠分析和建议。"
    elif not formatted_time_input:
        # 如果既没有格式化时间输入，也不包含格式化时间，使用基本查询
        # query 已经在函数开头初始化
        pass
    
    if not force_refresh:
        cached_result = get_cached_analysis(query, date)
        if cached_result:
            # 检查缓存结果是否为无数据信息
            if "暂无数据分析" in cached_result:
                print("暂无数据分析560")
                return "暂无数据分析"
            return cached_result

    logger.debug(f"Query: {query}")
    
    # 构建智能体
    agent_result = build_improved_agent(config_file_path)
    
    # 准备输入消息
    from langchain_core.messages import HumanMessage
    messages = [HumanMessage(content=query)]
    
    # 配置会话
    config = {"configurable": {"thread_id": thread_id}}
    
    # 调用智能体
    response = agent_result({"messages": messages}, config=config)
    
    # 提取响应内容
    result = []
    messages = response.get('messages', [])
    logger.debug(f"Messages: {messages}")
    
    for msg in messages:
        logger.debug(f"Processing message: {msg}")
        # 修改这里的逻辑以处理字典类型的消息
        if isinstance(msg, dict) and 'content' in msg and msg['content']:
            content = str(msg['content'])
        elif hasattr(msg, 'content') and msg.content:
            content = str(msg.content)
        else:
            continue
            
        # 检查是否是无数据的响应
        if "暂无数据分析" in content or "暂无数据" in content:
            return "暂无数据分析"
            
        # 首先检查是否包含关键词
        if any(keyword in content.lower() for keyword in ['分析', '建议', '总结', '报告', 'recommend', 'summary', 'analysis', 'report']):
            result.append(content)
        # 然后检查是否包含时间信息
        elif any(time_keyword in content.lower() for time_keyword in ['时间', 'hour', 'minute', 'second', '时', '分', '秒']):
            result.append(content)
        # 最后，如果不是错误信息，则添加
        elif 'error' not in content.lower() and '错误' not in content:
            result.append(content)

    # 构建最终结果，将原始数据信息放在结果前面
    analysis_result = "\n".join(result) if result else "暂无数据分析"
    
    # 如果存在formatted_time_input_local，则将其放在分析结果前面
    # if formatted_time_input_local:
    #     # 将原始数据信息作为HTML段落放在分析结果前
    #     html_formatted_input = formatted_time_input_local.replace("\n", "<br>")  # 转换换行为HTML<br>标签
    #     final_result = f"<div class=\"raw-data\">{html_formatted_input}</div><br><br>{analysis_result}"
    # else:
    #     final_result = analysis_result
    final_result = analysis_result

    logger.debug(f"Final result: {final_result[:200]}...")  # 只记录前200个字符
    
    # 保存分析结果到数据库（异步）
    save_analysis_result_async(query, final_result, date)
    
    return final_result
