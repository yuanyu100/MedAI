#!/usr/bin/env python3


import os
import sys
import json
import re
from datetime import datetime, timedelta, time
import logging
import traceback
from typing import Dict, Optional
from contextlib import asynccontextmanager
from langchain.tools import tool, ToolRuntime
import schedule
import threading
import time as time_module
from collections import Counter
import math
import asyncio

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, validator
from typing import List
import uvicorn
from langchain_core.messages import HumanMessage
import pandas as pd
import logging
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 使用改进的智能体
from improved_agent import run_improved_agent

# 导入生理指标趋势工具
from src.tools.physiological_trend_tool import get_physiological_trend_data, get_physiological_trend_data_by_metric
from src.tools.physiological_analyzer_tool import analyze_physiological_trend
# 导入睡眠数据检查工具
from src.tools.sleep_data_checker_tool import (
    check_previous_night_sleep_data,
    check_sleep_data_by_time_range,
    check_detailed_sleep_data
)
# 导入新增的周数据检查函数
from src.tools.sleep_data_checker_tool import check_detailed_sleep_data, check_weekly_sleep_data, check_recent_week_sleep_data
# 导入睡眠分析服务
from src.tools.sleep_analyzer_tool import (
    analyze_single_day_sleep_data,
    analyze_single_day_sleep_data_with_device
)

def convert_to_html(text):
    """将文本转换为HTML格式"""
    if not text:
        return ""
    
    # 将文本按行分割
    lines = text.split('\n')
    html_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            html_lines.append('<br>')
            continue
        
        # 跳过分隔符行（如 ---）
        if line.strip() == '---':
            continue
        
        # 处理标题（以###开头的行）
        if line.startswith('#### '):
            title = line[5:].strip()
            html_lines.append(f'<h4>{title}</h4>')
        elif line.startswith('### '):
            title = line[4:].strip()
            html_lines.append(f'<h3>{title}</h3>')
        # 处理二级标题（以##开头的行）
        elif line.startswith('## '):
            title = line[3:].strip()
            html_lines.append(f'<h2>{title}</h2>')
        # 处理一级标题（以#开头的行）
        elif line.startswith('# '):
            title = line[2:].strip()
            html_lines.append(f'<h1>{title}</h1>')
        # 处理列表项（以数字.开头的行）
        elif re.match(r'^\d+\. ', line):
            # 替换所有**为<strong>，但要处理嵌套情况
            formatted_line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line)
            html_lines.append(f'<p>{formatted_line}</p>')
        # 处理粗体文本（**text**）
        elif '**' in line:
            formatted_line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line)
            html_lines.append(f'<p>{formatted_line}</p>')
        # 处理其他普通文本
        else:
            # 替换任何剩余的**标记
            formatted_line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line)
            html_lines.append(f'<p>{formatted_line}</p>')
    
    return ''.join(html_lines)


def count_words(text):
    """统计文本中的单词数量"""
    if not text:
        return 0
    
    # 移除HTML标签
    clean_text = re.sub(r'<[^>]+>', '', text)
    
    # 分别统计中文字符和英文单词
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', clean_text))
    english_words = len(re.findall(r'\b[a-zA-Z]+\b', clean_text))
    
    return chinese_chars + english_words


def limit_report_length(text, max_words=500):
    """限制报告长度到指定单词数以内"""
    if not text:
        return ""
    
    words_count = count_words(text)
    if words_count <= max_words:
        return text
    
    # 移除HTML标签以便于截断
    clean_text = re.sub(r'<[^>]+>', '', text)
    
    # 按句子分割
    sentences = re.split(r'[。！？.!?]', clean_text)
    
    # 逐步添加句子直到接近限制
    result_parts = []
    current_count = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_word_count = count_words(sentence)
        if current_count + sentence_word_count <= max_words:
            result_parts.append(sentence)
            current_count += sentence_word_count
        else:
            # 计算还能容纳多少词
            remaining_words = max_words - current_count
            if remaining_words > 0:
                # 截断当前句子
                words = re.findall(r'[\u4e00-\u9fff]|\b[a-zA-Z]+\b', sentence)
                truncated_sentence = ''.join(words[:remaining_words])
                result_parts.append(truncated_sentence)
            break
    
    # 将文本重新组合
    result_text = '。'.join(result_parts) + "..."
    
    # 再次转换为HTML格式
    return convert_to_html(result_text)


class AgentRequest(BaseModel):
    """智能体请求模型"""
    date: str  # 日期格式 YYYY-MM-DD
    device_sn: Optional[str] = "210235C9KT3251000013"  # 设备序列号（可选，默认值）
    force_refresh: Optional[bool] = False  # 是否强制刷新，为True时不使用缓存
    include_formatted_time: Optional[bool] = False  # 是否包含格式化的睡眠时间信息
    formatted_time_input: Optional[str] = None  # 直接提供的格式化睡眠时间信息
    thread_id: Optional[str] = "default-session"


class SleepAnalysisWithTimeRequest(BaseModel):
    """使用格式化时间的睡眠分析请求模型"""
    date: str  # 日期格式 YYYY-MM-DD
    device_sn: Optional[str] = "210235C9KT3251000013"  # 设备序列号（可选，默认值）
    force_refresh: Optional[bool] = False  # 是否强制刷新，为True时不使用缓存


class WeeklySleepDataCheckRequest(BaseModel):
    """周睡眠数据检查请求模型"""
    start_date: str = Field(..., description="开始日期，格式如 '2024-12-20'")
    device_sn: Optional[str] = "210235C9KT3251000013"  # 设备序列号（可选，默认值）
    # 移除table_name参数，硬编码为vital_signs


class RecentWeeklySleepDataCheckRequest(BaseModel):
    """近期周睡眠数据检查请求模型"""
    num_weeks: int = Field(1, ge=1, le=4, description="检查的周数，最多4周")
    device_sn: Optional[str] = "210235C9KT3251000013"  # 设备序列号（可选，默认值）
    # 移除table_name参数，硬编码为vital_signs


class AnalysisRequest(BaseModel):
    """数据分析请求模型"""
    file_path: str


class DatabaseAnalysisRequest(BaseModel):
    """数据库分析请求模型"""
    # 移除table_name参数，硬编码为vital_signs


class VisualizationRequest(BaseModel):
    """可视化请求模型"""
    data: str


class PDFTrendRequest(BaseModel):
    """PDF和趋势分析请求模型"""
    file_path: str
    output_path: Optional[str] = None
    # 移除table_name参数，硬编码为vital_signs


class QARequest(BaseModel):
    """问答请求模型"""
    query: str


class SleepAnalysisRequest(BaseModel):
    """睡眠分析请求模型"""
    date: str  # 日期格式 YYYY-MM-DD
    device_sn: Optional[str] = "210235C9KT3251000013"  # 设备序列号（可选，默认值）
    # 移除table_name参数，硬编码为vital_signs


class PhysiologicalAnalysisRequest(BaseModel):
    """生理指标分析请求模型"""
    date: str  # 日期格式 YYYY-MM-DD
    device_sn: Optional[str] = "210235C9KT3251000013"  # 设备序列号（可选，默认值）
    # 移除table_name参数，硬编码为vital_signs


class SleepStageChartRequest(BaseModel):
    """睡眠分期图请求模型"""
    date: str  # 日期格式 YYYY-MM-DD
    # 移除table_name参数，硬编码为vital_signs


class ComprehensiveReportRequest(BaseModel):
    """综合报告请求模型"""
    date: str  # 日期格式 YYYY-MM-DD
    device_sn: Optional[str] = "210235C9KT3251000013"  # 设备序列号（可选，默认值）
    # 移除table_name参数，硬编码为vital_signs


class SleepDataCheckRequest(BaseModel):
    """睡眠数据检查请求模型"""
    date: str  # 日期格式 YYYY-MM-DD
    device_sn: Optional[str] = "210235C9KT3251000013"  # 设备序列号（可选，默认值）
    # 移除table_name参数，硬编码为vital_signs


class SleepAnalysisWithTimeRequest(BaseModel):
    """使用格式化时间的睡眠分析请求模型"""
    date: str  # 日期格式 YYYY-MM-DD
    device_sn: Optional[str] = "210235C9KT3251000013"  # 设备序列号（可选，默认值）
    force_refresh: Optional[bool] = False  # 是否强制刷新，为True时不使用缓存


# ========== Pydantic Response Models for Strong Type Validation ==========

# --- Sleep Analysis Response Models ---
class SleepPhasesModel(BaseModel):
    """睡眠阶段数据模型"""
    deep_sleep_minutes: float = Field(default=0, description="深睡时长(分钟)")
    light_sleep_minutes: float = Field(default=0, description="浅睡时长(分钟)")
    rem_sleep_minutes: float = Field(default=0, description="REM睡眠时长(分钟)")
    awake_minutes: float = Field(default=0, description="清醒时长(分钟)")
    deep_sleep_percentage: float = Field(default=0, ge=0, le=100, description="深睡占比(%)")
    light_sleep_percentage: float = Field(default=0, ge=0, le=100, description="浅睡占比(%)")
    rem_sleep_percentage: float = Field(default=0, ge=0, le=100, description="REM占比(%)")
    awake_percentage: float = Field(default=0, ge=0, le=100, description="清醒占比(%)")

    class Config:
        extra = "allow"  # 允许额外字段


class SleepStageSegmentModel(BaseModel):
    """睡眠阶段分段模型"""
    label: str = Field(..., description="睡眠阶段标签(深睡/浅睡/REM/清醒)")
    value: str = Field(..., description="持续时长(分钟)")

    class Config:
        extra = "allow"


class AverageMetricsModel(BaseModel):
    """平均生理指标模型"""
    avg_heart_rate: float = Field(default=0, ge=0, description="平均心率(次/分钟)")
    avg_respiratory_rate: float = Field(default=0, ge=0, description="平均呼吸率(次/分钟)")
    avg_body_moves_ratio: float = Field(default=0, ge=0, description="平均体动占比(%)")
    avg_heartbeat_interval: float = Field(default=0, ge=0, description="平均心跳间期(ms)")
    avg_rms_heartbeat_interval: float = Field(default=0, ge=0, description="平均心跳间期均方根(ms)")

    class Config:
        extra = "allow"


class SleepAnalysisDataModel(BaseModel):
    """睡眠分析数据模型"""
    date: str = Field(..., description="分析日期")
    bedtime: Optional[str] = Field(default=None, description="上床时间")
    wakeup_time: Optional[str] = Field(default=None, description="起床时间")
    time_in_bed_minutes: float = Field(default=0, ge=0, description="在床时长(分钟)")
    sleep_duration_minutes: float = Field(default=0, ge=0, description="睡眠时长(分钟)")
    sleep_score: int = Field(default=0, ge=0, le=100, description="睡眠评分")
    bed_exit_count: int = Field(default=0, ge=0, description="离床次数")
    sleep_prep_time_minutes: float = Field(default=0, ge=0, description="入睡准备时长(分钟)")
    sleep_phases: Optional[SleepPhasesModel] = Field(default=None, description="睡眠阶段详情")
    sleep_stage_segments: Optional[List[SleepStageSegmentModel]] = Field(default=None, description="睡眠阶段分段")
    average_metrics: Optional[AverageMetricsModel] = Field(default=None, description="平均生理指标")
    summary: str = Field(default="", description="睡眠质量总结")
    device_sn: Optional[str] = Field(default=None, description="设备序列号")

    class Config:
        extra = "allow"


class SleepAnalysisResponseModel(BaseModel):
    """睡眠分析响应模型 - Pydantic强类型校验"""
    success: bool = Field(..., description="请求是否成功")
    data: Optional[SleepAnalysisDataModel] = Field(default=None, description="睡眠分析数据")
    message: Optional[str] = Field(default=None, description="提示信息")
    error: Optional[str] = Field(default=None, description="错误信息")

    class Config:
        extra = "forbid"  # 不允许额外字段，严格校验


# --- Physiological Analysis Response Models ---
class HeartRateMetricsModel(BaseModel):
    """心率指标模型"""
    avg_heart_rate: float = Field(default=0, ge=0, description="平均心率(次/分钟)")
    min_heart_rate: float = Field(default=0, ge=0, description="最低心率(次/分钟)")
    max_heart_rate: float = Field(default=0, ge=0, description="最高心率(次/分钟)")
    heart_rate_variability: float = Field(default=0, ge=0, description="心率变异性")
    heart_rate_stability: float = Field(default=0, ge=0, le=100, description="心率稳定性评分")

    class Config:
        extra = "allow"


class RespiratoryMetricsModel(BaseModel):
    """呼吸指标模型"""
    avg_respiratory_rate: float = Field(default=0, ge=0, description="平均呼吸率(次/分钟)")
    min_respiratory_rate: float = Field(default=0, ge=0, description="最低呼吸率(次/分钟)")
    max_respiratory_rate: float = Field(default=0, ge=0, description="最高呼吸率(次/分钟)")
    respiratory_stability: float = Field(default=0, ge=0, le=100, description="呼吸稳定性评分")
    apnea_events_per_hour: float = Field(default=0, ge=0, description="每小时呼吸暂停次数")
    apnea_count: int = Field(default=0, ge=0, description="呼吸暂停总次数")
    avg_apnea_duration: float = Field(default=0, ge=0, description="平均呼吸暂停时长(秒)")
    max_apnea_duration: float = Field(default=0, ge=0, description="最长呼吸暂停时长(秒)")

    class Config:
        extra = "allow"


class SleepMetricsModel(BaseModel):
    """睡眠质量指标模型"""
    avg_body_moves_ratio: float = Field(default=0, ge=0, description="平均体动占比(%)")
    body_movement_frequency: float = Field(default=0, ge=0, description="体动频率(次/小时)")
    sleep_efficiency: float = Field(default=0, ge=0, le=100, description="睡眠效率(%)")

    class Config:
        extra = "allow"


class PhysiologicalAnalysisDataModel(BaseModel):
    """生理指标分析数据模型"""
    date: str = Field(..., description="分析日期")
    heart_rate_metrics: Optional[HeartRateMetricsModel] = Field(default=None, description="心率指标")
    respiratory_metrics: Optional[RespiratoryMetricsModel] = Field(default=None, description="呼吸指标")
    sleep_metrics: Optional[SleepMetricsModel] = Field(default=None, description="睡眠质量指标")
    summary: str = Field(default="", description="生理指标总结")
    device_sn: Optional[str] = Field(default=None, description="设备序列号")

    class Config:
        extra = "allow"


class PhysiologicalAnalysisResponseModel(BaseModel):
    """生理指标分析响应模型 - Pydantic强类型校验"""
    success: bool = Field(..., description="请求是否成功")
    data: Optional[PhysiologicalAnalysisDataModel] = Field(default=None, description="生理指标分析数据")
    message: Optional[str] = Field(default=None, description="提示信息")
    error: Optional[str] = Field(default=None, description="错误信息")

    class Config:
        extra = "forbid"  # 不允许额外字段，严格校验


# ========== Database Record to Pydantic Model Transformation Functions ==========

def transform_db_record_to_sleep_analysis(db_record: dict, sleep_stage_segments: list = None) -> SleepAnalysisDataModel:
    """
    将数据库平铺记录转换为 SleepAnalysisDataModel 嵌套结构
    
    Args:
        db_record: 数据库返回的平铺字典
        sleep_stage_segments: 睡眠阶段分段列表 (optional)
    
    Returns:
        SleepAnalysisDataModel: 符合Pydantic模型的嵌套结构数据
    """
    # 构建 sleep_phases 嵌套结构
    sleep_phases = SleepPhasesModel(
        deep_sleep_minutes=float(db_record.get('deep_sleep_minutes', 0) or 0),
        light_sleep_minutes=float(db_record.get('light_sleep_minutes', 0) or 0),
        rem_sleep_minutes=float(db_record.get('rem_sleep_minutes', 0) or 0),
        awake_minutes=float(db_record.get('awake_minutes', 0) or 0),
        deep_sleep_percentage=float(db_record.get('deep_sleep_percentage', 0) or 0),
        light_sleep_percentage=float(db_record.get('light_sleep_percentage', 0) or 0),
        rem_sleep_percentage=float(db_record.get('rem_sleep_percentage', 0) or 0),
        awake_percentage=float(db_record.get('awake_percentage', 0) or 0)
    )
    
    # 构建 average_metrics 嵌套结构
    average_metrics = AverageMetricsModel(
        avg_heart_rate=float(db_record.get('avg_heart_rate', 0) or 0),
        avg_respiratory_rate=float(db_record.get('avg_respiratory_rate', 0) or 0),
        avg_body_moves_ratio=float(db_record.get('avg_body_moves_ratio', 0) or 0),
        avg_heartbeat_interval=float(db_record.get('avg_heartbeat_interval', 0) or 0),
        avg_rms_heartbeat_interval=float(db_record.get('avg_rms_heartbeat_interval', 0) or 0)
    )
    
    # 构建 sleep_stage_segments 列表
    segments_list = None
    if sleep_stage_segments:
        segments_list = [
            SleepStageSegmentModel(label=seg['label'], value=str(seg['value']))
            for seg in sleep_stage_segments
        ]
    
    # 辅助函数：将 pandas.Timestamp 转换为字符串
    def safe_str_convert(val):
        if val is None:
            return None
        if hasattr(val, 'strftime'):  # pandas.Timestamp or datetime
            return val.strftime('%Y-%m-%d %H:%M:%S')
        return str(val)
    
    # 构建主数据模型
    return SleepAnalysisDataModel(
        date=str(db_record.get('date', '')),
        bedtime=safe_str_convert(db_record.get('bedtime')),
        wakeup_time=safe_str_convert(db_record.get('wakeup_time')),
        time_in_bed_minutes=float(db_record.get('time_in_bed_minutes', 0) or 0),
        sleep_duration_minutes=float(db_record.get('sleep_duration_minutes', 0) or 0),
        sleep_score=int(db_record.get('sleep_score', 0) or 0),
        bed_exit_count=int(db_record.get('bed_exit_count', 0) or 0),
        sleep_prep_time_minutes=float(db_record.get('sleep_prep_time_minutes', 0) or 0),
        sleep_phases=sleep_phases,
        sleep_stage_segments=segments_list,
        average_metrics=average_metrics,
        summary=str(db_record.get('summary', '')),
        device_sn=db_record.get('device_sn')
    )


def transform_db_record_to_physiological_analysis(db_record: dict) -> PhysiologicalAnalysisDataModel:
    """
    将数据库平铺记录转换为 PhysiologicalAnalysisDataModel 嵌套结构
    
    Args:
        db_record: 数据库返回的平铺字典
    
    Returns:
        PhysiologicalAnalysisDataModel: 符合Pydantic模型的嵌套结构数据
    """
    # 构建 heart_rate_metrics 嵌套结构
    heart_rate_metrics = HeartRateMetricsModel(
        avg_heart_rate=float(db_record.get('avg_heart_rate', 0) or 0),
        min_heart_rate=float(db_record.get('min_heart_rate', 0) or 0),
        max_heart_rate=float(db_record.get('max_heart_rate', 0) or 0),
        heart_rate_variability=float(db_record.get('heart_rate_variability', 0) or 0),
        heart_rate_stability=float(db_record.get('heart_rate_stability', 0) or 0)
    )
    
    # 构建 respiratory_metrics 嵌套结构
    respiratory_metrics = RespiratoryMetricsModel(
        avg_respiratory_rate=float(db_record.get('avg_respiratory_rate', 0) or 0),
        min_respiratory_rate=float(db_record.get('min_respiratory_rate', 0) or 0),
        max_respiratory_rate=float(db_record.get('max_respiratory_rate', 0) or 0),
        respiratory_stability=float(db_record.get('respiratory_stability', 0) or 0),
        apnea_events_per_hour=float(db_record.get('apnea_events_per_hour', 0) or 0),
        apnea_count=int(db_record.get('apnea_count', 0) or 0),
        avg_apnea_duration=float(db_record.get('avg_apnea_duration', 0) or 0),
        max_apnea_duration=float(db_record.get('max_apnea_duration', 0) or 0)
    )
    
    # 构建 sleep_metrics 嵌套结构
    sleep_metrics = SleepMetricsModel(
        avg_body_moves_ratio=float(db_record.get('avg_body_moves_ratio', 0) or 0),
        body_movement_frequency=float(db_record.get('body_movement_frequency', 0) or 0),
        sleep_efficiency=float(db_record.get('sleep_efficiency', 0) or 0)
    )
    
    # 构建主数据模型
    return PhysiologicalAnalysisDataModel(
        date=str(db_record.get('date', '')),
        heart_rate_metrics=heart_rate_metrics,
        respiratory_metrics=respiratory_metrics,
        sleep_metrics=sleep_metrics,
        summary=str(db_record.get('summary', '')),
        device_sn=db_record.get('device_sn')
    )


# 删除重复的SleepAnalysisWithTimeRequest定义

# 为qa_retriever创建一个包装函数
def create_sample_excel():
    """创建示例Excel文件用于QA查询"""
    # 创建一个示例数据集
    sample_data = []
    start_time = datetime.now() - timedelta(hours=24)
    
    for i in range(24 * 4):  # 每15分钟一条记录，共24小时
        current_time = start_time + timedelta(minutes=i*15)
        
        # 模拟不同的数据类型
        if i % 8 == 0:  # 每2小时一条状态数据
            # 状态数据
            status = "有人状态" if i % 16 != 0 else "无人状态"  # 交替有人/无人状态
            sample_data.append({
                '上传时间': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                '数据类型': '状态',
                '数据内容': status
            })
        else:
            # 周期数据
            # 随机生成生理参数
            heart_rate = 60 + (i % 10)  # 心率在60-70之间变化
            respiration_rate = 15 + (i % 5)  # 呼吸频率在15-20之间变化
            body_move_ratio = 2 + (i % 3)  # 体动占比2-5%
            apnea_count = 1 if i % 20 == 0 else 0  # 每20条记录有一次呼吸暂停
            
            data_content = f"心率:{heart_rate}次/分钟;呼吸:{respiration_rate}次/分钟;心跳间期平均值:800毫秒;心跳间期均方根值:50毫秒;心跳间期标准差:40毫秒;心跳间期紊乱比例:15%;体动次数的占比:{body_move_ratio}%;呼吸暂停次数:{apnea_count}次"
            
            sample_data.append({
                '上传时间': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                '数据类型': '周期数据',
                '数据内容': data_content
            })
    
    # 创建DataFrame并保存为Excel
    df = pd.DataFrame(sample_data)
    temp_file = os.path.join(tempfile.gettempdir(), 'sample_qa_data.xlsx')
    df.to_excel(temp_file, index=False)
    return temp_file


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


def qa_retrieve_internal(file_path: str, question: str) -> str:
    """
    内部检索函数
    """
    try:
        # 读取数据
        df = pd.read_excel(file_path)

        # 解析数据内容
        parsed_data = []
        for idx, row in df.iterrows():
            content = row['数据内容']
            data_dict = parse_data_content(content)
            data_dict['upload_time'] = pd.to_datetime(row['上传时间'])
            parsed_data.append(data_dict)

        parsed_df = pd.DataFrame(parsed_data)

        # 匹配问题类别
        category = match_question_to_category(question)

        # 提取相关信息
        info = {}

        if category == 'sleep':
            info = extract_sleep_info(parsed_df)
        elif category == 'heart_rate':
            info = extract_heart_rate_info(parsed_df)
        elif category == 'respiration':
            info = extract_respiration_info(parsed_df)
        elif category == 'apnea':
            info = extract_apnea_info(parsed_df)
        elif category == 'risk':
            # 综合所有风险信息
            info.update(extract_heart_rate_info(parsed_df))
            info.update(extract_respiration_info(parsed_df))
            info.update(extract_apnea_info(parsed_df))
            info.update(extract_sleep_info(parsed_df))
        else:  # summary
            info = extract_summary_info(parsed_df)

        # 格式化响应
        response = format_response(category, info)

        return response

    except Exception as e:
        import traceback
        error_msg = f"检索失败: {str(e)}\n{traceback.format_exc()}"
        return error_msg


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用程序生命周期管理"""
    print("🚀 启动修复版智能体API服务器...")
    # 设置环境变量
    # os.environ.setdefault("QWEN_API_KEY", "sk-2ad6355b98dd43668a5eeb21e50e4642")
    # os.environ.setdefault("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    yield
    # 关闭时的清理


# 创建FastAPI应用
app = FastAPI(
    title="修复版智能病床监控数据分析系统API",
    description="提供智能体和数据分析功能的修复版API接口",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "欢迎使用修复版智能病床监控数据分析系统API",
        "version": "1.0.0",
        "endpoints": {
            "POST /agent/run": "运行智能体（支持device_sn参数）",
            "POST /analysis/database": "分析数据库数据",
            "POST /visualization": "生成可视化报告",
            "POST /trend": "趋势分析",
            "POST /qa": "问答查询",
            "POST /sleep-analysis": "睡眠分析（支持device_sn参数）",
            "POST /physiological-analysis": "生理指标分析（支持device_sn参数）",
            "POST /physiological-trend": "生理指标趋势分析（返回每5分钟的心率和呼吸频率，时间范围：前一晚20:00至当天早上10:00）",
            "POST /sleep-data-check": "睡眠数据检查（检查是否存在前一天晚上的睡眠数据，支持device_sn参数）",
            "POST /weekly-sleep-data-check": "周睡眠数据检查（检查一周内每天的睡眠数据）",
            "POST /recent-weekly-sleep-data-check": "近期周睡眠数据检查（检查最近几周的睡眠数据）",
            "POST /ai-analysis": "AI分析（使用格式化的时间信息作为用户提示，支持device_sn参数）",
            "POST /comprehensive-report": "综合报告（支持device_sn参数）",
            "GET /health": "健康检查"
        }
    }


# @app.post("/agent/run")
async def run_agent_endpoint(request: AgentRequest):
    """运行智能体"""
    try:
        print(f"🤖 运行智能体: {request.date}, 设备: {request.device_sn}, 强制刷新: {request.force_refresh}")
        
        # 运行智能体
        result = run_improved_agent(
            date=request.date,
            thread_id=request.thread_id,
            force_refresh=request.force_refresh,
            include_formatted_time=request.include_formatted_time,
            formatted_time_input=request.formatted_time_input,
            device_sn=request.device_sn  # 传递设备序列号
        )
        
        # 转换为HTML格式
        html_result = convert_to_html(result)
        
        return {"success": True, "data": html_result}
        
    except Exception as e:
        print(f"❌ 智能体运行失败: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


# @app.post("/agent/run-markdown")
async def run_agent_markdown(request: AgentRequest):
    """运行智能体并返回Markdown格式结果"""
    try:
        print(f"🔄 运行智能体并返回Markdown格式，日期: {request.date}, 强制刷新: {request.force_refresh}, 包含格式化时间: {request.include_formatted_time}, 格式化时间输入: {request.formatted_time_input}")
        
        # 使用改进的智能体运行分析，传入日期参数和格式化时间选项
        result = run_improved_agent(request.date, request.thread_id, force_refresh=request.force_refresh, include_formatted_time=request.include_formatted_time, formatted_time_input=request.formatted_time_input)
        
        # 返回纯文本结果，FastAPI会将其作为text/plain响应
        return PlainTextResponse(content=result, media_type="text/markdown")

    except Exception as e:
        print(f"❌ 运行智能体失败: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


@app.post("/ai-analysis")
async def ai_analysis(request: SleepAnalysisWithTimeRequest):
    """AI分析 - 默认从数据库读取预计算结果，force_refresh=True时才重新计算"""
    try:
        print(f"🤖 运行AI分析: {request.date}, 设备: {request.device_sn}, 强制刷新: {request.force_refresh}")

        request.force_refresh = False
        
        # 默认 force_refresh=False，从数据库读取缓存结果
        if not request.force_refresh:
            # 从 analysis_results 表读取已存储的分析结果
            from improved_agent import get_cached_analysis
            
            # 构建查询字符串
            query = f"请分析 {request.date} 的睡眠数据"
            if request.device_sn:
                query = f"[设备序列号: {request.device_sn}] {query}"
            
            # 从数据库获取缓存的分析结果
            cached_result = get_cached_analysis(query, request.date)
            
            if cached_result:
                print(f"✅ 从数据库获取已存储的AI分析结果: {request.date}")
                
                # 检查是否为无数据信息
                if "暂无数据分析" in cached_result:
                    return {
                        "success": True,
                        "data": "<p>当前日期没有可用的睡眠数据。请确保设备已收集相应数据后再进行分析。</p>",
                        "warning": "无可用数据",
                        "has_data": False
                    }
                
                # 将结果转换为HTML格式
                html_result = convert_to_html(cached_result)
                
                # 限制报告长度到500词以内
                limited_html_result = limit_report_length(html_result)
                
                return {
                    "success": True,
                    "data": limited_html_result,
                    "has_data": True
                }
            else:
                # 数据库中没有缓存结果，返回提示信息
                print(f"⚠️ 数据库中没有 {request.date} 的分析结果")
                return {
                    "success": True,
                    "data": "<p>当前日期的分析结果尚未生成。请等待定时任务执行后再查询。</p>",
                    "warning": "分析结果尚未生成",
                    "has_data": False
                }
        
        # force_refresh=True 时，执行实时计算
        print(f"🔄 强制刷新，执行实时AI分析...")
        
        # 首先检查数据可用性
        from src.tools.sleep_data_checker_tool import check_detailed_sleep_data_with_device
        
        if request.device_sn:
            check_result = check_detailed_sleep_data_with_device(request.date, request.device_sn)
        else:
            from src.tools.sleep_data_checker_tool import check_detailed_sleep_data
            check_result = check_detailed_sleep_data(request.date)
        
        check_data = json.loads(check_result)
        has_data = check_data.get('data', {}).get('has_sleep_data', False)
        
        if not has_data:
            print(f"⚠️ 未找到 {request.date} 的睡眠数据，尝试补偿机制...")
            await trigger_data_collection(request.date, request.device_sn)
            
            if request.device_sn:
                check_result = check_detailed_sleep_data_with_device(request.date, request.device_sn)
            else:
                check_result = check_detailed_sleep_data(request.date)
            
            check_data = json.loads(check_result)
            has_data = check_data.get('data', {}).get('has_sleep_data', False)
            
            if not has_data:
                return {
                    "success": True,
                    "data": "<p>当前日期没有可用的睡眠数据。请确保设备已收集相应数据后再进行分析。</p>",
                    "warning": "无可用数据",
                    "has_data": False
                }
        
        # 使用改进的智能体运行分析
        from improved_agent import run_improved_agent
        result = run_improved_agent(
            request.date, 
            thread_id=f"ai_analysis_{request.date}", 
            force_refresh=request.force_refresh,  # 硬编码为False强制不重新计算
            include_formatted_time=True,
            device_sn=request.device_sn
        )
        
        # 将结果转换为HTML格式
        html_result = convert_to_html(result)
        
        # 限制报告长度到500词以内
        limited_html_result = limit_report_length(html_result, max_words=500)
        
        return {
            "success": True,
            "data": limited_html_result,
            "has_data": True
        }

    except Exception as e:
        print(f"❌ AI分析失败: {str(e)}")
        print(traceback.format_exc())
        
        return {
            "success": False,
            "error": str(e),
            "message": "AI分析失败"
        }


async def trigger_data_collection(date: str, device_sn: str = None):
    """触发数据收集补偿机制"""
    print(f"🔄 尝试触发数据收集 for {date}, device: {device_sn}")
    
    # 这里可以实现数据收集的具体逻辑
    # 比如调用数据采集API、从外部设备同步数据等
    # 目前只是占位符
    try:
        # 示例：调用数据库工具检查可用数据
        from src.tools.database_tool import get_available_sleep_dates
        result = get_available_sleep_dates()
        print(f"📊 可用数据日期: {result}")
        
        return True
    except Exception as e:
        print(f"⚠️ 数据收集补偿机制执行失败: {str(e)}")
        return False


def trigger_data_collection_sync(date: str, device_sn: str = None):
    """同步版本的触发数据收集补偿机制，用于定时任务"""
    print(f"🔄 尝试触发数据收集 for {date}, device: {device_sn}")
    
    # 这里可以实现数据收集的具体逻辑
    # 比如调用数据采集API、从外部设备同步数据等
    # 目前只是占位符
    try:
        # 示例：调用数据库工具检查可用数据
        from src.tools.database_tool import get_available_sleep_dates
        result = get_available_sleep_dates()
        print(f"📊 可用数据日期: {result}")
        
        return True
    except Exception as e:
        print(f"⚠️ 数据收集补偿机制执行失败: {str(e)}")
        return False


# @app.post("/analysis/database")
async def analyze_database_data(request: DatabaseAnalysisRequest):
    """分析数据库数据"""
    try:
        print(f"📊 分析数据库表: vital_signs")
        
        # 执行数据库分析
        from src.tools.bed_monitoring_db_analyzer import analyze_bed_monitoring_from_db
        result = analyze_bed_monitoring_from_db("vital_signs")
        # 直接返回工具函数的结果，因为工具函数已经使用ApiResponse格式
        analysis_result = json.loads(result)
        
        # 如果工具返回的是错误格式，需要正确处理
        if analysis_result.get("success") is False:
            # 工具已经返回了完整的错误响应
            return analysis_result
        
        # 如果工具成功，返回其数据部分
        from src.utils.response_handler import ApiResponse
        response = ApiResponse.success(data=analysis_result)
        return response.to_dict()
        
    except Exception as e:
        print(f"❌ 数据库分析失败: {str(e)}")
        
        from src.utils.response_handler import ApiResponse
        response = ApiResponse.error(
            error=str(e), 
            message="数据库分析失败，可能是由于数据库连接问题。请检查数据库配置。",
            data={"recommended_action": "如果您没有可用的数据库，可以使用 /analysis/excel 端点分析Excel文件"}
        )
        return response.to_dict()




# @app.post("/trend")
async def analyze_trend_data(request: PDFTrendRequest):
    """趋势分析"""
    try:
        print(f"📊 趋势分析: {request.file_path}")
        
        # 如果file_path是空的或者默认值，从数据库获取数据
        if not request.file_path or request.file_path == "" or request.file_path == "string":
            print(f"从数据库表 vital_signs 获取数据进行趋势分析")
            # 直接导入内部函数，避免相对导入问题
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'tools'))
            from analyze_trend_tool import analyze_trend_and_pattern_internal
            result = analyze_trend_and_pattern_internal(file_path=None, table_name="vital_signs")
        else:
            # 检查文件是否存在
            import os
            if not os.path.exists(request.file_path):
                raise HTTPException(status_code=404, detail=f"文件不存在: {request.file_path}")

            # 执行趋势分析
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'tools'))
            from analyze_trend_tool import analyze_trend_and_pattern_internal
            result = analyze_trend_and_pattern_internal(file_path=request.file_path, table_name="vital_signs")
        
        # 直接返回工具函数的结果，因为工具函数已经使用ApiResponse格式
        result_dict = json.loads(result)
        
        # 如果工具返回的是错误格式，需要正确处理
        if result_dict.get("success") is False:
            # 工具已经返回了完整的错误响应
            # 但我们需要移除timestamp字段
            filtered_result = {
                "success": result_dict.get("success"),
                "data": result_dict.get("data"),
                "error": result_dict.get("error"),
                "message": result_dict.get("message")
            }
            # 只保留非None的字段
            return {k: v for k, v in filtered_result.items() if v is not None}
        
        # 如果工具成功，返回其数据部分但移除timestamp字段
        filtered_result = {
            "success": True,
            "data": result_dict
        }
        return filtered_result
        
    except Exception as e:
        print(f"❌ 趋势分析失败: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


@app.post("/sleep-analysis", response_model=SleepAnalysisResponseModel)
async def analyze_sleep(request: SleepAnalysisRequest) -> SleepAnalysisResponseModel:
    """睡眠分析 - 使用Pydantic强类型校验返回结果"""
    try:
        print(f"😴 睡眠分析: {request.date}, 设备: {request.device_sn}")
        
        # 首先尝试从数据库获取已存储的分析结果
        from src.db.database import get_db_manager
        db_manager = get_db_manager()
        stored_data_raw = db_manager.get_calculated_sleep_data(request.date, request.device_sn)
        
        # 检查数据库是否有已存储的结果，并检查睡眠分析数据是否已填充
        if not stored_data_raw.empty:
            stored_record = stored_data_raw.to_dict('records')[0]
            
            # 检查bedtime是否不为None（哨兵字段，表示睡眠分析已执行）
            # 如果bedtime为None，说明睡眠分析还没执行过，需要重新计算
            if stored_record.get('bedtime') is not None:
                # 从数据库读取并转换为Pydantic模型结构
                # 获取sleep_stage_segments
                segments_raw = db_manager.get_sleep_stage_segments(request.date, request.device_sn)
                sleep_stage_segments = None
                if not segments_raw.empty:
                    sleep_stage_segments = segments_raw.to_dict('records')
                
                # 使用转换函数将平铺DB记录转换为嵌套Pydantic模型
                data_model = transform_db_record_to_sleep_analysis(stored_record, sleep_stage_segments)
                
                return SleepAnalysisResponseModel(
                    success=True,
                    data=data_model
                )
        
        # 数据库中没有数据，调用分析工具生成新数据
        if request.device_sn:
            result = analyze_single_day_sleep_data_with_device(request.date, request.device_sn, "vital_signs")
        else:
            result = analyze_single_day_sleep_data(request.date, "vital_signs")
        
        result_dict = json.loads(result)
        
        # 如果工具成功，存储结果到数据库
        if result_dict.get("success") and result_dict.get("data"):
            db_manager.store_calculated_sleep_data(result_dict.get("data", {}))
        
        # 返回结果（工具函数已经返回正确的嵌套结构）
        if result_dict.get("success") is False:
            return SleepAnalysisResponseModel(
                success=False,
                error=result_dict.get("error"),
                message=result_dict.get("message")
            )
        
        return SleepAnalysisResponseModel(
            success=True,
            data=result_dict.get("data"),
            message=result_dict.get("message")
        )
        
    except Exception as e:
        print(f"❌ 睡眠分析失败: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


@app.post("/physiological-analysis", response_model=PhysiologicalAnalysisResponseModel)
async def analyze_physiological(request: PhysiologicalAnalysisRequest) -> PhysiologicalAnalysisResponseModel:
    """生理指标分析 - 使用Pydantic强类型校验返回结果"""
    try:
        print(f"📊 生理指标分析: {request.date}, 设备: {request.device_sn}")
        
        # 首先尝试从数据库获取已存储的分析结果
        from src.db.database import get_db_manager
        db_manager = get_db_manager()
        stored_data_raw = db_manager.get_calculated_sleep_data(request.date, request.device_sn)
        
        # 检查数据库是否有已存储的结果，并检查生理指标数据是否已填充
        if not stored_data_raw.empty:
            stored_record = stored_data_raw.to_dict('records')[0]
            
            # 检查heart_rate_variability是否不为0（哨兵字段，表示生理分析已执行）
            if stored_record.get('heart_rate_variability', 0) != 0:
                logger.info("从数据库获取的生理指标数据已存在，不再重新计算")
                # 使用转换函数将平铺DB记录转换为嵌套Pydantic模型
                data_model = transform_db_record_to_physiological_analysis(stored_record)
                
                return PhysiologicalAnalysisResponseModel(
                    success=True,
                    data=data_model
                )
        
        # 数据库中没有数据或生理指标未填充，调用分析工具生成新数据
        if request.device_sn:
            from src.tools.physiological_analyzer_tool import analyze_single_day_physiological_data_with_device
            result = analyze_single_day_physiological_data_with_device(request.date, request.device_sn, "vital_signs")
        else:
            from src.tools.physiological_analyzer_tool import analyze_single_day_physiological_data
            result = analyze_single_day_physiological_data(request.date, "vital_signs")
        
        result_dict = json.loads(result)
        
        # 如果工具成功，存储结果到数据库
        if result_dict.get("success") and result_dict.get("data"):
            logger.info(f"存储生理指标数据到数据库，{result_dict.get("data", {})}, {request.device_sn}")
            db_manager.store_calculated_sleep_data(result_dict.get("data", {}))
        
        # 返回结果（工具函数已经返回正确的嵌套结构）
        if result_dict.get("success") is False:
            return PhysiologicalAnalysisResponseModel(
                success=False,
                error=result_dict.get("error"),
                message=result_dict.get("message")
            )
        
        return PhysiologicalAnalysisResponseModel(
            success=True,
            data=result_dict.get("data"),
            message=result_dict.get("message")
        )
        
    except Exception as e:
        print(f"❌ 生理指标分析失败: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


# @app.post("/qa")
async def qa_query(request: QARequest):
    """问答查询"""
    try:
        print(f"❓ 问答查询: {request.query}")
        
        # 创建示例数据文件
        sample_file = create_sample_excel()
        
        # 执行问答查询（调用内部函数而不是工具装饰的函数）
        result = qa_retrieve_internal(sample_file, request.query)
        
        return {
            "success": True,
            "answer": result
        }
        
    except Exception as e:
        print(f"❌ 问答查询失败: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


# 新增：生理指标趋势分析请求模型
class PhysiologicalTrendRequest(BaseModel):
    """生理指标趋势分析请求模型"""
    date: str  # 日期格式 YYYY-MM-DD
    metric: Optional[str] = None  # 指标类型，可选 'heart_rate' 或 'respiratory_rate'
    device_sn: Optional[str] = None  # 设备序列号（可选）
    # 移除table_name参数，硬编码为vital_signs


@app.post("/physiological-trend")
async def physiological_trend_endpoint(request: PhysiologicalTrendRequest):
    """生理指标趋势分析（心率和呼吸率随时间变化）"""
    try:
        print(f"📊 生理指标趋势分析请求: {request.date}, 设备: {request.device_sn}")
        
        # 根据是否有设备序列号来决定使用哪个函数
        if request.device_sn:
            # 使用带设备过滤的函数
            from src.tools.physiological_analyzer_tool import analyze_physiological_trend_with_device
            result = analyze_physiological_trend_with_device(request.date, request.device_sn)
        else:
            # 使用原有函数
            from src.tools.physiological_analyzer_tool import analyze_physiological_trend
            result = analyze_physiological_trend(request.date)
        result_dict = json.loads(result)
        
        # 直接返回结果但移除timestamp字段
        filtered_result = {
            "success": True,
            "data": result_dict
        }
        return filtered_result
        
    except Exception as e:
        print(f"❌ 生理指标趋势分析失败: {str(e)}")
        print(traceback.format_exc())
        
        # 返回错误响应但移除timestamp字段
        error_result = {
            "success": False,
            "error": str(e),
            "message": "生理指标趋势分析失败"
        }
        return error_result


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "message": "修复版智能体API服务器运行正常",
        "timestamp": datetime.now().isoformat()
    }


# 新增：睡眠数据检查端点
# @app.post("/sleep-data-check")
async def check_sleep_data(request: SleepDataCheckRequest):
    """检查睡眠数据是否存在"""
    try:
        print(f"🔍 检查睡眠数据: {request.date}, 设备: {request.device_sn}")
        
        # 根据是否有设备序列号来决定如何获取数据
        if request.device_sn:
            # 如果提供了设备序列号，使用带设备的函数
            result = check_detailed_sleep_data_with_device(request.date, request.device_sn)
        else:
            # 否则使用普通函数
            result = check_detailed_sleep_data(request.date)
        
        # 解析结果
        result_data = json.loads(result)
        
        # 直接返回结果但移除timestamp字段
        filtered_result = {
            "success": True,
            "data": result_data
        }
        return filtered_result
        
    except Exception as e:
        print(f"❌ 检查睡眠数据失败: {str(e)}")
        print(traceback.format_exc())
        
        # 返回错误响应但移除timestamp字段
        error_result = {
            "success": False,
            "error": str(e),
            "message": "检查睡眠数据失败"
        }
        return error_result


# 新增：周睡眠数据检查端点
@app.post("/weekly-sleep-data-check")
async def check_weekly_sleep_data_endpoint(request: WeeklySleepDataCheckRequest):
    """检查一周的睡眠数据"""
    try:
        print(f"🔍 检查周睡眠数据: {request.start_date}, 设备: {request.device_sn}")
        
        # 根据是否有设备序列号来决定使用哪个函数
        if request.device_sn:
            # 使用带设备过滤的函数
            from src.tools.sleep_data_checker_tool import check_weekly_sleep_data_with_device
            result = check_weekly_sleep_data_with_device(request.start_date, request.device_sn, "vital_signs")
        else:
            # 使用原有函数
            result = check_weekly_sleep_data(request.start_date, "vital_signs")
        
        # 直接返回工具函数的结果，因为工具函数已经使用ApiResponse格式
        result_dict = json.loads(result)
        
        # 如果工具返回的是错误格式，需要正确处理
        if result_dict.get("success") is False:
            # 工具已经返回了完整的错误响应
            # 但我们需要移除timestamp字段
            filtered_result = {
                "success": result_dict.get("success"),
                "data": result_dict.get("data"),
                "error": result_dict.get("error"),
                "message": result_dict.get("message")
            }
            # 只保留非None的字段
            return {k: v for k, v in filtered_result.items() if v is not None}
        
        # 简化返回值，只保留关键信息
        simplified_data = {
            "week_start_date": result_dict.get("week_start_date"),
            "week_end_date": result_dict.get("week_end_date"),
            "weekly_summary": result_dict.get("weekly_summary"),
            "daily_results": [
                {
                    "date": day["date"],
                    "has_sleep_data": day["has_sleep_data"],
                    "record_count": day["record_count"],
                    "day_of_week_cn": day["day_of_week_cn"]
                } for day in result_dict.get("daily_results", [])
            ]
        }
        
        # 构建正确的响应格式，移除timestamp
        filtered_result = {
            "success": True,
            "data": simplified_data
        }
        return filtered_result
        
    except Exception as e:
        print(f"❌ 周睡眠数据检查失败: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


# 新增：近期周睡眠数据检查端点
# @app.post("/recent-weekly-sleep-data-check")
async def check_recent_weekly_sleep_data_endpoint(request: RecentWeeklySleepDataCheckRequest):
    """检查近期几周的睡眠数据"""
    try:
        print(f"🔍 检查近期{request.num_weeks}周睡眠数据, 设备: {request.device_sn}")
        
        # 根据是否有设备序列号来决定使用哪个函数
        if request.device_sn:
            # 使用带设备过滤的函数
            from src.tools.sleep_data_checker_tool import check_recent_week_sleep_data_with_device
            result = check_recent_week_sleep_data_with_device(request.num_weeks, request.device_sn, "vital_signs")
        else:
            # 使用原有函数
            result = check_recent_week_sleep_data(request.num_weeks, "vital_signs")
        
        # 直接返回工具函数的结果，因为工具函数已经使用ApiResponse格式
        result_dict = json.loads(result)
        
        # 如果工具返回的是错误格式，需要正确处理
        if result_dict.get("success") is False:
            # 工具已经返回了完整的错误响应
            # 但我们需要移除timestamp字段
            filtered_result = {
                "success": result_dict.get("success"),
                "data": result_dict.get("data"),
                "error": result_dict.get("error"),
                "message": result_dict.get("message")
            }
            # 只保留非None的字段
            return {k: v for k, v in filtered_result.items() if v is not None}
        
        # 简化返回值，只保留关键信息
        simplified_data = {
            "period_summary": result_dict.get("period_summary"),
            "weekly_results": [
                {
                    "week_start_date": week.get("week_start_date"),
                    "week_end_date": week.get("week_end_date"),
                    "weekly_summary": week.get("weekly_summary"),
                    "daily_results": [
                        {
                            "date": day["date"],
                            "has_sleep_data": day["has_sleep_data"],
                            "record_count": day["record_count"],
                            "day_of_week_cn": day["day_of_week_cn"]
                        } for day in week.get("daily_results", [])
                    ]
                } for week in result_dict.get("weekly_results", [])
            ]
        }
        
        # 构建正确的响应格式，移除timestamp
        filtered_result = {
            "success": True,
            "data": simplified_data
        }
        return filtered_result
        
    except Exception as e:
        print(f"❌ 近期周睡眠数据检查失败: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


@app.post("/comprehensive-report")
async def get_comprehensive_report(request: ComprehensiveReportRequest):
    """获取综合报告 - 包含睡眠和生理指标"""
    try:
        print(f"📋 获取综合报告: {request.date}, 设备: {request.device_sn}")
        
        # 获取睡眠分析数据
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'tools'))
        
        # 根据是否有设备序列号来决定使用哪个函数
        if request.device_sn:
            # 使用带设备过滤的函数
            from src.tools.sleep_analyzer_tool import analyze_single_day_sleep_data_with_device
            sleep_result = analyze_single_day_sleep_data_with_device(request.date, request.device_sn, "vital_signs")
        else:
            # 使用原有函数
            from src.tools.sleep_analyzer_tool import analyze_single_day_sleep_data
            sleep_result = analyze_single_day_sleep_data(request.date, "vital_signs")
        
        # 直接返回工具函数的结果，因为工具函数已经使用ApiResponse格式
        sleep_result_dict = json.loads(sleep_result)
        
        # 如果工具返回的是错误格式，需要正确处理
        if sleep_result_dict.get("success") is False:
            # 工具已经返回了完整的错误响应
            # 但我们需要移除timestamp字段
            filtered_result = {
                "success": sleep_result_dict.get("success"),
                "data": sleep_result_dict.get("data"),
                "error": sleep_result_dict.get("error"),
                "message": sleep_result_dict.get("message")
            }
            # 只保留非None的字段
            return {k: v for k, v in filtered_result.items() if v is not None}
        
        # 获取生理指标分析数据
        if request.device_sn:
            # 使用带设备过滤的函数
            from src.tools.physiological_analyzer_tool import analyze_single_day_physiological_data_with_device
            physio_result = analyze_single_day_physiological_data_with_device(request.date, request.device_sn, "vital_signs")
        else:
            # 使用原有函数
            from src.tools.physiological_analyzer_tool import analyze_single_day_physiological_data
            physio_result = analyze_single_day_physiological_data(request.date, "vital_signs")
        
        # 直接返回工具函数的结果，因为工具函数已经使用ApiResponse格式
        physio_result_dict = json.loads(physio_result)
        
        # 如果工具返回的是错误格式，需要正确处理
        if physio_result_dict.get("success") is False:
            # 工具已经返回了完整的错误响应
            # 但我们需要移除timestamp字段
            filtered_result = {
                "success": physio_result_dict.get("success"),
                "data": physio_result_dict.get("data"),
                "error": physio_result_dict.get("error"),
                "message": physio_result_dict.get("message")
            }
            # 只保留非None的字段
            return {k: v for k, v in filtered_result.items() if v is not None}
        
        # 从工具返回的数据中提取实际数据部分
        sleep_data = sleep_result_dict.get("data", {})
        physio_data = physio_result_dict.get("data", {})
        
        # 整合数据并生成报告
        report_data = generate_comprehensive_report(sleep_data, physio_data, request.date)
        
        # 构建正确的响应格式，移除timestamp
        filtered_result = {
            "success": True,
            "data": report_data
        }
        return filtered_result

    except Exception as e:
        print(f"❌ 综合报告获取失败: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


def generate_comprehensive_report(sleep_data: dict, physio_data: dict, date: str = "unknown"):
    """
    Generate comprehensive report integrating sleep and physiological data
    """
    # Calculate total sleep duration (hours)
    sleep_duration_hours = sleep_data.get('sleep_duration_minutes', 0) / 60
    
    # Deep sleep duration (minutes)
    deep_sleep_minutes = sleep_data.get('sleep_phases', {}).get('deep_sleep_minutes', 0)
    
    # Sleep preparation time (minutes)
    sleep_prep_time_minutes = sleep_data.get('sleep_prep_time_minutes', 0)
    
    # Apnea events per hour
    # Since we don't have explicit total sleep duration to calculate events per hour, 
    # we use a simplified method or assume we can find relevant apnea metrics in the data
    apnea_count = physio_data.get('respiratory_metrics', {}).get('apnea_count', 0)
    # Assume sleep duration as the basis for calculating apnea frequency
    apnea_per_hour = (apnea_count / sleep_duration_hours) if sleep_duration_hours > 0 else 0
    
    # Average heart rate
    avg_heart_rate = physio_data.get('heart_rate_metrics', {}).get('avg_heart_rate', 0)
    
    # Minimum heart rate
    min_heart_rate = physio_data.get('heart_rate_metrics', {}).get('min_heart_rate', 0)
    
    # Maximum heart rate
    max_heart_rate = physio_data.get('heart_rate_metrics', {}).get('max_heart_rate', 0)
    
    # Average respiratory rate
    avg_respiratory_rate = physio_data.get('respiratory_metrics', {}).get('avg_respiratory_rate', 0)
    
    # Evaluation function
    def evaluate_value(value, normal_range, is_higher_better=False):
        """Evaluate if metric is normal"""
        if isinstance(normal_range, tuple):
            lower, upper = normal_range
            if value < lower:
                return "↓", f"<{lower}"
            elif value > upper:
                return "↑", f">{upper}"
            else:
                return "◎", f"{lower}-{upper}"
        else:  # Single threshold comparison
            if is_higher_better:
                if value >= normal_range:
                    return "◎", f">={normal_range}"
                else:
                    return "↓", f"<{normal_range}"
            else:
                if value <= normal_range:
                    return "◎", f"<={normal_range}"
                else:
                    return "↑", f">{normal_range}"
    
    # Generate metric evaluations
    sleep_duration_eval, sleep_duration_ref = evaluate_value(sleep_duration_hours, (6.5, 12))  # 睡眠时长正常范围6.5-12小时
    deep_sleep_eval, deep_sleep_ref = evaluate_value(deep_sleep_minutes, (40, 240))  # 深睡眠正常范围40-240分钟
    sleep_prep_eval, sleep_prep_ref = evaluate_value(sleep_prep_time_minutes, (0, 30))  # 入睡准备时间正常范围0-30分钟
    apnea_eval, apnea_ref = evaluate_value(apnea_per_hour, (0, 5))  # 呼吸暂停正常范围0-5次/小时
    avg_hr_eval, avg_hr_ref = evaluate_value(avg_heart_rate, (55, 70))  # 平均心率正常范围55-70次/分钟
    min_hr_eval, min_hr_ref = evaluate_value(min_heart_rate, 52, is_higher_better=True)  # 最低心率应≥52
    max_hr_eval, max_hr_ref = evaluate_value(max_heart_rate, 85)  # 最高心率应≤85
    avg_resp_eval, avg_resp_ref = evaluate_value(avg_respiratory_rate, (11, 18))  # 平均呼吸频率正常范围11-18次/分钟)
    
    # Return comprehensive report
    report = {
        "date": date,
        "indicators": [
            {
                "name": "总睡眠时长",
                "value": f"{sleep_duration_hours:.1f} 小时",
                "result": sleep_duration_eval,
                "reference": sleep_duration_ref
            },
            {
                "name": "深睡眠时长",
                "value": f"{deep_sleep_minutes} 分钟",
                "result": deep_sleep_eval,
                "reference": f">{deep_sleep_ref.split('>')[-1]}" if '>' in deep_sleep_ref else deep_sleep_ref
            },
            {
                "name": "入睡准备时间",
                "value": f"{sleep_prep_time_minutes} 分钟",
                "result": sleep_prep_eval,
                "reference": sleep_prep_ref.split('<')[-1] if '<' in sleep_prep_ref else sleep_prep_ref
            },
            {
                "name": "呼吸暂停事件",
                "value": f"{apnea_per_hour:.1f} 次/小时",
                "result": apnea_eval,
                "reference": apnea_ref.split('<')[-1] if '<' in apnea_ref else apnea_ref
            },
            {
                "name": "平均心率",
                "value": f"{avg_heart_rate} 次/分钟",
                "result": avg_hr_eval,
                "reference": avg_hr_ref
            },
            {
                "name": "最低心率",
                "value": f"{min_heart_rate} 次/分钟",
                "result": min_hr_eval,
                "reference": min_hr_ref.split('≥')[-1] if '≥' in min_hr_ref else f"≥{min_heart_rate}"
            },
            {
                "name": "最高心率",
                "value": f"{max_heart_rate} 次/分钟",
                "result": max_hr_eval,
                "reference": max_hr_ref.split('≤')[-1] if '≤' in max_hr_ref else f"≤{max_heart_rate}"
            },
            {
                "name": "平均呼吸频率",
                "value": f"{avg_respiratory_rate} 次/分钟",
                "result": avg_resp_eval,
                "reference": avg_resp_ref
            }
        ]
    }
    
    return report


def run_scheduler():
    """运行调度器，在后台执行定时任务"""
    def scheduled_analysis():
        """执行定时分析任务"""
        try:
            print(f"⏰ 执行每日定时AI分析任务: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 获取当前日期
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # 检查当前日期是否有睡眠数据
            from src.tools.sleep_data_checker_tool import check_detailed_sleep_data
            check_result = check_detailed_sleep_data(current_date)
            check_data = json.loads(check_result)
            has_data = check_data.get('data', {}).get('has_sleep_data', False)
            
            if has_data:
                print(f"✅ {current_date} 存在睡眠数据，开始AI分析...")
                
                # 使用改进的智能体运行分析
                from improved_agent import run_improved_agent
                result = run_improved_agent(
                    current_date, 
                    thread_id=f"scheduled_ai_analysis_{current_date}", 
                    force_refresh=False,
                    include_formatted_time=True
                )
                
                print(f"✅ 定时AI分析完成")
            else:
                print(f"⚠️ {current_date} 不存在睡眠数据，跳过AI分析")
                # 尝试触发数据收集
                trigger_data_collection_sync(current_date)
                
        except Exception as e:
            print(f"❌ 定时AI分析任务失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 每天上午10点执行任务
    schedule.every().day.at("10:00").do(scheduled_analysis)
    
    print("⏰ 调度器已启动，等待定时任务执行...")
    while True:
        schedule.run_pending()
        time_module.sleep(60)  # 每分钟检查一次


def start_server(host: str = "0.0.0.0", port: int = 8080, reload: bool = False):
    """启动API服务器"""
    # 启动调度器作为后台线程
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    print(f"🌐 启动修复版API服务器在 {host}:{port}")
    import uvicorn
    if reload:
        # 为了使热重载工作，我们需要使用模块名称而不是app对象
        uvicorn.run("fixed_api_server:app", host=host, port=port, reload=True)
    else:
        uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="启动修复版智能体API服务器")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("-p", "--port", type=int, default=9001, help="服务器端口")
    parser.add_argument("--reload", action="store_true", help="启用热重载模式")
    
    args = parser.parse_args()
    
    print("🚀 启动修复版智能体API服务器...")
    print(f"🌐 访问地址: http://{args.host}:{args.port}")
    print(f"📖 API文档: http://{args.host}:{args.port}/docs")
    if args.reload:
        print("🔥 热重载已启用")
    
    start_server(args.host, args.port, reload=args.reload)
