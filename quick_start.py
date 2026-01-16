#!/usr/bin/env python3
"""
智能病床监控数据分析系统 - 快速启动脚本
"""

import os
import sys
import subprocess
import platform

def main():
    print("智能病床监控数据分析系统 - 快速启动")
    print("=" * 45)
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ 错误: 需要Python 3.8或更高版本")
        return 1
    
    print(f"✅ Python版本: {platform.python_version()}")
    
    # 设置环境变量
    os.environ["QWEN_API_KEY"] = "sk-2ad6355b98dd43668a5eeb21e50e4642"
    os.environ["QWEN_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    print("✅ 环境变量已设置")
    
    # 检查依赖
    required_packages = [
        'langgraph', 'langchain', 'langchain-openai', 
        'pandas', 'openpyxl', 'numpy', 'matplotlib', 
        'reportlab', 'pypdf'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    # 特别处理python-dotenv，因为它可能有不同的导入名称
    dotenv_installed = False
    try:
        import dotenv
        dotenv_installed = True
    except ImportError:
        try:
            import python_dotenv
            dotenv_installed = True
        except ImportError:
            pass
    
    if missing_packages:
        print(f"❌ 缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return 1
    
    # 检查dotenv
    if not dotenv_installed:
        try:
            import python_dotenv
        except ImportError:
            missing_packages.append('python-dotenv')
            print(f"❌ 缺少依赖包: {', '.join(missing_packages)}")
            print("请运行: pip install -r requirements.txt")
            return 1
    
    print("✅ 所有依赖包已安装")
    
    print("\n系统已准备就绪！")
    print("\n您可以选择以下操作：")
    print("1. 运行完整测试: python test_run.py")
    print("2. 运行示例: python example.py")
    print("3. 运行演示: python run_demo.py")
    print("4. 查看运行指南: type RUNNING.md (Windows) 或 cat RUNNING.md (Linux/Mac)")
    
    # 提供快速选项
    print("\n" + "-" * 45)
    print("提示: 您可以运行以下命令之一开始使用：")
    print("  python test_run.py          # 执行系统测试")
    print("  python example.py           # 运行示例（需要提供数据文件）")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())