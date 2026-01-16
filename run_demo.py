#!/usr/bin/env python3
"""
智能病床监控数据分析系统演示脚本
使用Qwen-Plus模型和提供的API密钥
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.agent import build_agent
from langchain_core.messages import HumanMessage
import json
import tempfile


def main():
    """主函数 - 演示完整的分析流程"""
    print("=== 智能病床监控数据分析系统演示 ===\n")

    # 设置环境变量
    os.environ["QWEN_API_KEY"] = "sk-2ad6355b98dd43668a5eeb21e50e4642"
    os.environ["QWEN_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    # 构建Agent
    print("1. 初始化Agent...")
    try:
        agent = build_agent()
        print("✓ Agent初始化完成\n")
    except Exception as e:
        print(f"✗ Agent初始化失败: {str(e)}")
        return 1

    # 获取示例数据文件路径
    data_file = os.path.join(tempfile.gettempdir(), 'full_test_sample_data.xlsx')
    
    if not os.path.exists(data_file):
        print(f"✗ 示例数据文件不存在: {data_file}")
        print("请先运行测试脚本创建示例数据")
        return 1

    print(f"2. 使用数据文件: {data_file}")
    
    user_input = f"""请分析病床监护数据文件 {data_file}，生成一份完整的护理交班报告。

报告需要包含：
1. 风险等级评估（低/中/高）
2. 护理摘要（患者全天的依从性和配合度）
3. 关键事件列表（最多5个）
4. 医生建议（建议医生关注的生理指标异常）
5. 护士提示（给护士的具体建议）

请使用JSON格式输出报告。
"""

    print("3. 开始分析数据...")

    # 调用Agent
    try:
        messages = [HumanMessage(content=user_input)]
        config = {"configurable": {"thread_id": "demo-session-1"}}
        response = agent.invoke({"messages": messages}, config=config)

        print("4. 提取分析结果...\n")
        print("=" * 60)

        # 提取最终结果
        final_response = None
        for msg in response['messages']:
            if hasattr(msg, 'content') and msg.content:
                if isinstance(msg.content, list):
                    # 如果内容是列表，遍历其中的每个元素
                    for item in msg.content:
                        if isinstance(item, dict) and 'text' in item:
                            content_str = item['text']
                        elif isinstance(item, str):
                            content_str = item
                        else:
                            continue
                        
                        if '{' in content_str and '}' in content_str:
                            try:
                                start = content_str.find('{')
                                end = content_str.rfind('}') + 1
                                json_str = content_str[start:end]
                                report = json.loads(json_str)
                                final_response = report
                                break
                            except json.JSONDecodeError:
                                continue
                elif isinstance(msg.content, str):
                    content_str = msg.content
                    if '{' in content_str and '}' in content_str:
                        try:
                            start = content_str.find('{')
                            end = content_str.rfind('}') + 1
                            json_str = content_str[start:end]
                            report = json.loads(json_str)
                            final_response = report
                            break
                        except json.JSONDecodeError:
                            print(content_str)
                
                if final_response:
                    break
        
        if final_response:
            print(json.dumps(final_response, ensure_ascii=False, indent=2))
        else:
            print("未能从响应中提取JSON格式的报告")
            # 打印最后一个消息的内容作为备选
            for msg in reversed(response['messages']):
                if hasattr(msg, 'content'):
                    print(f"最后的消息内容: {msg.content}")
                    break

        print("=" * 60)
        print("\n✓ 分析完成")
        
    except Exception as e:
        print(f"✗ 分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())