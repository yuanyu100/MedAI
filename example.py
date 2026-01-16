#!/usr/bin/env python3
"""
病床监护数据分析示例脚本
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.agent import build_agent
from langchain_core.messages import HumanMessage


def main():
    """主函数"""
    print("=== 病床监护数据分析示例 ===\n")

    # 构建Agent
    print("1. 初始化Agent...")
    agent = build_agent()
    print("✓ Agent初始化完成\n")

    user_input = """请分析病床监护数据库中的数据，生成一份完整的护理交班报告。

报告需要包含：
1. 风险等级评估（低/中/高）
2. 护理摘要（患者全天的依从性和配合度）
3. 关键事件列表（最多5个）
4. 医生建议（建议医生关注的生理指标异常）
5. 护士提示（给护士的具体建议）

请使用JSON格式输出报告。
"""

    print("2. 分析数据...")

    # 调用Agent
    messages = [HumanMessage(content=user_input)]
    config = {"configurable": {"thread_id": "example-session-1"}}
    response = agent.invoke({"messages": messages}, config=config)

    print("3. 提取结果...\n")
    print("=" * 60)

    # 提取最终结果
    for msg in response['messages']:
        if hasattr(msg, 'content') and msg.content:
            import json
            if '{' in msg.content and '}' in msg.content:
                try:
                    start = msg.content.find('{')
                    end = msg.content.rfind('}') + 1
                    json_str = msg.content[start:end]
                    report = json.loads(json_str)
                    print(json.dumps(report, ensure_ascii=False, indent=2))
                except:
                    print(msg.content)
            else:
                print(msg.content)
            break

    print("=" * 60)
    print("\n✓ 分析完成")
    return 0


if __name__ == '__main__':
    sys.exit(main())
