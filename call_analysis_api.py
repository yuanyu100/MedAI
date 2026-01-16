#!/usr/bin/env python3
"""
调用分析接口测试
"""

import sys
import os
import json
import tempfile
import pandas as pd
from datetime import datetime, timedelta

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))




def test_db_analysis():
    """测试数据库分析"""
    print("\n=== 测试数据库分析 ===")
    
    try:
        from src.tools.bed_monitoring_db_analyzer import bed_monitoring_db_analyzer_tool
        
        print("正在从数据库分析数据...")
        result = bed_monitoring_db_analyzer_tool(table_name="device_data")
        
        # 解析结果
        parsed_result = json.loads(result)
        
        if 'error' in parsed_result:
            print(f"数据库分析失败: {parsed_result['error']}")
            print(f"错误信息: {parsed_result['message']}")
            return False
        
        print("✓ 数据库分析成功!")
        print(f"监测时段: {parsed_result['monitoring_period']['start_time']} 至 {parsed_result['monitoring_period']['end_time']}")
        print(f"总时长: {parsed_result['monitoring_period']['total_hours']} 小时")
        
        # 显示关键指标
        vital_signs = parsed_result.get('vital_signs', {})
        print(f"心率: {vital_signs.get('hr_min', 0)}-{vital_signs.get('hr_max', 0)} bpm (平均: {vital_signs.get('hr_avg', 0)})")
        print(f"呼吸: {vital_signs.get('rr_min', 0)}-{vital_signs.get('rr_max', 0)} 次/分钟 (平均: {vital_signs.get('rr_avg', 0)})")
        
        apnea_analysis = parsed_result.get('apnea_analysis', {})
        print(f"呼吸暂停: {apnea_analysis.get('total_apnea_count', 0)} 次 (AHI指数: {apnea_analysis.get('ahi_index', 0)})")
        
        return True
        
    except Exception as e:
        print(f"数据库分析测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_analysis():
    """测试智能代理分析"""
    print("\n=== 测试智能代理分析 ===")
    
    try:
        from src.agents.agent import build_agent
        from langchain_core.messages import HumanMessage
        
        print("正在构建智能代理...")
        agent = build_agent()
        print("✓ 智能代理构建成功")
        
        # 准备输入
        user_input = "请分析病床监护数据，生成一份完整的护理交班报告。"
        
        print("正在处理请求...")
        messages = [HumanMessage(content=user_input)]
        config = {"configurable": {"thread_id": "test-api-call-1"}}
        
        # 注意：这需要实际的API密钥才能完成，这里仅验证代理构建成功
        print("✓ 智能代理已准备好处理请求")
        print("  （实际执行需要有效的API密钥）")
        
        return True
        
    except Exception as e:
        print(f"智能代理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("智能病床监控数据分析系统 - 分析接口测试")
    print("=" * 60)
    
    # Excel分析已移除
    excel_success = False
    
    # 测试数据库分析
    db_success = test_db_analysis()
    
    # 测试智能代理
    agent_success = test_agent_analysis()
    
    print("\n" + "=" * 60)
    print("测试结果:")

    print(f"数据库分析: {'通过' if db_success else '失败'}")
    print(f"智能代理: {'通过' if agent_success else '失败'}")
    
    if excel_success or db_success:
        print("\n✓ 至少一个分析接口测试通过！")
        print("\n系统可以使用以下方式进行数据分析：")

        print("- 数据库分析：使用bed_monitoring_db_analyzer_tool(table_name='device_data')")
        print("- 智能代理：通过构建的agent进行高级分析")
    else:
        print("\n✗ 所有测试都失败了，请检查系统配置。")


if __name__ == "__main__":
    main()