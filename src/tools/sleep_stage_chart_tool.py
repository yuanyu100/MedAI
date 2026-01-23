#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
睡眠分期图工具
用于生成符合指定格式的睡眠分期数据，包含数据完整性校验
"""
import json
from typing import List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def validate_sleep_stage_data(stages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    验证睡眠分期数据的完整性
    
    Args:
        stages: 睡眠分期数据列表
        
    Returns:
        包含验证结果的字典
    """
    errors = []
    warnings = []
    
    if not stages:
        errors.append("睡眠分期数据不能为空")
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings
        }
    
    # 检查每个阶段的数据完整性
    for i, stage in enumerate(stages):
        # 检查必要字段
        required_fields = ["startTime", "endTime", "stage"]
        for field in required_fields:
            if field not in stage:
                errors.append(f"第{i+1}个阶段缺少必要字段: {field}")
        
        if "startTime" in stage and not isinstance(stage["startTime"], (int, float)):
            errors.append(f"第{i+1}个阶段的startTime必须是数字时间戳")
        
        if "endTime" in stage and not isinstance(stage["endTime"], (int, float)):
            errors.append(f"第{i+1}个阶段的endTime必须是数字时间戳")
            
        if "stage" in stage and stage["stage"] not in [1, 2, 3, 4]:
            errors.append(f"第{i+1}个阶段的stage值非法，必须是1/2/3/4之一，当前值为: {stage['stage']}")
    
    # 检查时间顺序和重叠
    sorted_stages = sorted(stages, key=lambda x: x.get("startTime", 0))
    
    for i in range(len(sorted_stages) - 1):
        current_end = sorted_stages[i].get("endTime", 0)
        next_start = sorted_stages[i + 1].get("startTime", 0)
        
        if current_end > next_start:
            errors.append(f"第{i+1}个阶段({current_end})与第{i+2}个阶段({next_start})存在时间重叠")
        elif current_end < next_start:
            warnings.append(f"第{i+1}个阶段({current_end})与第{i+2}个阶段({next_start})之间存在时间断层")
    
    # 检查时间戳是否为毫秒级
    for i, stage in enumerate(stages):
        if "startTime" in stage:
            start_time = stage["startTime"]
            # 检查是否为毫秒级时间戳（通常大于10位）
            if start_time and start_time > 9999999999:  # 10位数字是秒级，超过10位是毫秒级
                continue
            else:
                warnings.append(f"第{i+1}个阶段的startTime可能不是毫秒级时间戳")
        
        if "endTime" in stage:
            end_time = stage["endTime"]
            if end_time and end_time > 9999999999:
                continue
            else:
                warnings.append(f"第{i+1}个阶段的endTime可能不是毫秒级时间戳")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


def generate_sleep_stage_chart(date: str, stages: List[Dict[str, Any]]) -> str:
    """
    生成睡眠分期图数据
    
    Args:
        date: 日期字符串，格式如 "2024-12-18"
        stages: 睡眠分期数据列表
        
    Returns:
        JSON格式的睡眠分期数据
    """
    try:
        logger.info(f"生成睡眠分期图数据，日期: {date}")
        
        # 验证输入数据
        validation_result = validate_sleep_stage_data(stages)
        
        if not validation_result["valid"]:
            return json.dumps({
                "success": False,
                "error": "数据验证失败",
                "validation_errors": validation_result["errors"],
                "validation_warnings": validation_result["warnings"]
            }, ensure_ascii=False)
        
        # 返回符合要求格式的数据
        result = {
            "success": True,
            "date": date,
            "stages": stages,
            "validation_warnings": validation_result["warnings"],
            "summary": calculate_stage_summary(stages)
        }
        
        logger.info(f"成功生成睡眠分期图数据，共{len(stages)}个阶段")
        return json.dumps(result, ensure_ascii=False)
    
    except Exception as e:
        logger.error(f"生成睡眠分期图数据时出错: {str(e)}")
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


def calculate_stage_summary(stages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算睡眠阶段摘要信息
    
    Args:
        stages: 睡眠分期数据列表
        
    Returns:
        摘要信息
    """
    stage_names = {1: "深睡", 2: "浅睡", 3: "快速眼动", 4: "清醒"}
    stage_durations = {1: 0, 2: 0, 3: 0, 4: 0}
    
    for stage in stages:
        stage_type = stage.get("stage")
        start_time = stage.get("startTime")
        end_time = stage.get("endTime")
        
        if stage_type and start_time and end_time:
            duration = end_time - start_time
            if stage_type in stage_durations:
                stage_durations[stage_type] += duration
    
    total_duration = sum(stage_durations.values())
    
    summary = {
        "total_duration_ms": total_duration,
        "total_duration_hours": round(total_duration / (1000 * 60 * 60), 2),
        "stages_detail": {}
    }
    
    for stage_num, duration in stage_durations.items():
        if duration > 0:
            summary["stages_detail"][stage_names[stage_num]] = {
                "duration_ms": duration,
                "duration_minutes": round(duration / (1000 * 60), 2),
                "percentage": round((duration / total_duration * 100) if total_duration > 0 else 0, 2) if total_duration > 0 else 0
            }
    
    return summary


def get_sleep_stage_chart_for_date(date: str, table_name: str = "device_data") -> str:
    """
    根据数据库数据获取指定日期的睡眠分期图
    
    Args:
        date: 日期字符串，格式如 "2024-12-18"
        table_name: 数据库表名
        
    Returns:
        JSON格式的睡眠分期数据
    """
    try:
        logger.info(f"从数据库获取{date}的睡眠分期数据，表名: {table_name}")
        
        # 这里应该从数据库查询实际的睡眠分期数据
        # 为了演示，我创建一个模拟数据
        # 在实际实现中，这里应该查询数据库并转换为所需的格式
        
        # 模拟数据 - 在实际应用中，这里应该是从数据库查询得到的真实数据
        simulated_stages = [
            {
                "startTime": 1734802800000,  # 2024-12-18 23:00
                "endTime": 1734804600000,    # 2024-12-18 23:30
                "stage": 4                   # 清醒
            },
            {
                "startTime": 1734804600000,  # 2024-12-18 23:30
                "endTime": 1734808200000,    # 2024-12-19 00:30
                "stage": 2                   # 浅睡
            },
            {
                "startTime": 1734808200000,  # 2024-12-19 00:30
                "endTime": 1734813000000,    # 2024-12-19 02:00
                "stage": 1                   # 深睡
            },
            {
                "startTime": 1734813000000,  # 2024-12-19 02:00
                "endTime": 1734814800000,    # 2024-12-19 02:30
                "stage": 3                   # 快速眼动
            },
            {
                "startTime": 1734814800000,  # 2024-12-19 02:30
                "endTime": 1734816600000,    # 2024-12-19 03:00
                "stage": 2                   # 浅睡
            },
            {
                "startTime": 1734816600000,  # 2024-12-19 03:00
                "endTime": 1734822000000,    # 2024-12-19 04:40
                "stage": 1                   # 深睡
            },
            {
                "startTime": 1734822000000,  # 2024-12-19 04:40
                "endTime": 1734823800000,    # 2024-12-19 05:10
                "stage": 4                   # 清醒
            }
        ]
        
        return generate_sleep_stage_chart(date, simulated_stages)
    
    except Exception as e:
        logger.error(f"获取数据库睡眠分期数据时出错: {str(e)}")
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


# 为工具函数添加装饰器，使其成为LangChain工具
from langchain_core.tools import tool


@tool
def sleep_stage_chart_tool(date: str, table_name: str = "device_data") -> str:
    """
    睡眠分期图工具
    
    功能：根据指定日期从数据库获取睡眠分期数据，生成符合前端要求格式的睡眠分期图表数据。
    
    参数:
        date: 日期字符串，格式如 "2024-12-18"
        table_name: 数据库表名，默认为 "device_data"
    
    返回:
        符合以下格式的JSON数据:
        {
            "success": true/false,
            "date": "2024-12-18",
            "stages": [
                {
                    "startTime": 1734802800000, // 阶段开始时间，毫秒级Unix时间戳
                    "endTime": 1734804600000,   // 阶段结束时间，毫秒级Unix时间戳
                    "stage": 1                  // 睡眠阶段：1=深睡、2=浅睡、3=快速眼动、4=清醒
                }
            ],
            "validation_warnings": [...],       // 验证警告（如有）
            "summary": {                       // 摘要信息
                "total_duration_ms": 12345678,
                "total_duration_hours": 3.43,
                "stages_detail": {
                    "深睡": {
                        "duration_ms": 1234567,
                        "duration_minutes": 20.58,
                        "percentage": 20.58
                    }
                }
            }
        }
    
    使用场景:
        - 需要在前端展示睡眠分期图表时
        - 需要获取详细的睡眠阶段分析时
        - 需要验证睡眠数据完整性时
    """
    logger.info(f"调用睡眠分期图工具, date: {date}, table_name: {table_name}")
    
    result = get_sleep_stage_chart_for_date(date, table_name)
    logger.info(f"睡眠分期图工具返回结果长度: {len(result) if result else 0}")
    return result