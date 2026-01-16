# 智能病床监控数据分析系统 - 运行指南

## 项目概述

这是一个基于LangGraph和LangChain的智能病床监控数据分析系统，能够自动分析患者监护数据并生成专业的护理交班报告。

## 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置Qwen-Plus模型

在 `config/agent_llm_config.json` 中已配置为使用 `qwen-plus` 模型。

## 运行项目

### 方法一：使用环境变量

```bash
export QWEN_API_KEY="你的API密钥"
export QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
python example.py
```

### 方法二：直接运行（使用内置密钥）

```bash
python example.py
```

## 测试项目

项目包含测试脚本，可以验证各组件是否正常工作：

```bash
python test_run.py
```

## 数据文件格式

系统从数据库获取监护数据，数据应包含以下字段：

- **upload_time**: 数据上传的时间戳
- **data_type**: 数据类型（如"周期数据"、"状态"等）
- **data_content**: 具体的生理参数，格式如：
  - 状态数据：`有人状态` 或 `无人状态`
  - 周期数据：`心率:XX次/分钟;呼吸:XX次/分钟;心跳间期平均值:XXX毫秒;心跳间期均方根值:XX毫秒;心跳间期标准差:XX毫秒;心跳间期紊乱比例:XX%;体动次数的占比:XX%;呼吸暂停次数:X次`

## 功能说明

1. **病床占用状态分析**: 统计卧床/离床时间、长离床事件等
2. **生命体征分析**: 分析心率、呼吸频率等指标
3. **呼吸暂停分析**: 计算AHI指数，评估风险等级
4. **体动与睡眠行为分析**: 评估睡眠质量
5. **晨间评估**: 分析早晨生理指标变化


## API配置

项目已配置为使用通义千问的API服务：

- **模型**: qwen-plus
- **API Key**: 在代码中预设或通过环境变量提供
- **Base URL**: https://dashscope.aliyuncs.com/compatible-mode/v1

## Windows兼容性

项目已适配Windows环境，修复了路径相关问题，并使用系统临时目录存储中间文件。

## 故障排除

1. **API调用超时**: 检查网络连接和API密钥有效性
2. **依赖包错误**: 确认已正确安装所有依赖
3. **文件路径错误**: 确认数据文件路径正确且格式符合要求