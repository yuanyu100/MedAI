# 病床监护数据分析系统

## 项目简介

这是一个基于LangGraph和LangChain的智能病床监护数据分析系统，能够自动分析病床监护数据并生成专业的护理交班报告。

## 核心功能

### 1. 病床占用状态分析（FR 1.0）
- 计算总卧床时间和总离床时间
- 检测长离床事件（>15分钟）
- 统计夜间（22:00-06:00）离床次数
- 计算病床占用率

### 2. 临床级生命体征分析（FR 2.0）
- 心率统计（范围、平均值）
- 呼吸统计（范围、平均值）
- 心动过速/心动过缓检测
- 呼吸异常检测（Bradypnea/Tachypnea）
- HRV（心率变异性）趋势分析

### 3. 呼吸暂停综合分析（FR 3.0）
- 统计呼吸暂停事件
- 计算AHI指数（每小时暂停次数）
- 风险分级（正常/轻度/中度/重度）
- 识别显著事件（呼吸暂停+心率代偿）

### 4. 体动与睡眠行为分析（FR 4.0）
- 计算睡眠效率
- 体动分析（平均/最大体动占比）
- 识别高体动时段
- 评估睡眠稳定性

### 5. 晨间评估（FR 5.0）
- 分析晨起（05:00-07:00）心率趋势
- 检测晨峰现象
- 评估苏醒状态

## 项目结构

```
.
├── config/                          # 配置目录
│   └── agent_llm_config.json         # LLM配置文件
├── src/                             # 源代码目录
│   ├── agents/                       # Agent代码
│   │   └── agent.py                 # 主Agent实现
│   ├── tools/                        # 工具代码
│   │   └── bed_monitoring_analyzer.py  # 病床数据分析工具
│   └── storage/                     # 存储代码
│       └── memory/
│           └── memory_saver.py       # 记忆存储器
├── tests/                           # 测试目录
├── assets/                          # 资源目录
├── docs/                            # 文档目录
├── export.py                        # 导出脚本
└── README.md                        # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- pandas
- openai
- langgraph
- langchain
- langchain-openai
- langchain-core

## 使用方法

### 1. 准备数据文件

使用数据库存储监护数据。

### 2. 运行Agent

```python
from src.agents.agent import build_agent
from langchain_core.messages import HumanMessage

# 构建Agent
agent = build_agent()

# 准备输入消息
user_input = "请分析病床监护数据文件 /path/to/data.xlsx，生成护理交班报告。"
messages = [HumanMessage(content=user_input)]

# 调用Agent（需要提供thread_id）
config = {"configurable": {"thread_id": "session-1"}}
response = agent.invoke({"messages": messages}, config=config)

# 提取结果
for msg in response['messages']:
    if hasattr(msg, 'content') and msg.content:
        print(msg.content)
```

### 3. 直接使用分析工具

```python
from src.tools.bed_monitoring_db_analyzer import analyze_bed_monitoring_from_db

# 从数据库分析数据
result = analyze_bed_monitoring_from_db('device_data')
print(result)
```

### 4. 使用API接口

项目提供了基于FastAPI的RESTful API接口，可以方便地从外部系统调用智能体功能。

启动API服务器：

```bash
python api_server.py
```

API服务器将在 `http://localhost:8000` 启动，提供以下功能：
- 智能体运行：`POST /agent/run`
- Excel数据分析：`POST /analysis/excel`
- 数据库数据分析：`POST /analysis/database`
- 可视化报告生成：`POST /visualization`
- PDF报告生成：`POST /pdf`
- 趋势分析：`POST /trend`
- 问答查询：`POST /qa`

更多API使用详情请参见 [API使用指南](API_USAGE.md)。

## 护理交班报告格式

系统生成的护理交班报告包含以下字段：

```json
{
  "risk_level": "风险等级 (低/中/高)",
  "nursing_summary": "简要描述患者全天的依从性和配合度",
  "critical_events": [
    {
      "type": "事件类型",
      "time": "时间 (HH:MM)",
      "analysis": "可能原因分析"
    }
  ],
  "doctor_notes": "建议医生关注的生理指标异常",
  "nurse_tips": "给护士的建议 (如: 夜间巡视、检查体位)"
}
```

## 护士建议（Nurse Tips）参考
- 夜间22:00-03:00时段重点监测呼吸和心率变化
- 检查患者体位是否影响呼吸
- 评估患者离床活动的必要性和安全性
- 关注患者睡眠质量
- 记录患者晨起状态，监测是否有头晕、乏力等症状
- 必要时提供床栏或离床报警器
- 检查疼痛迹象
- 观察患者情绪，及时安抚

## 医生建议参考
- 关注心动过缓（心率<50）或心动过速（心率>100）事件
- 评估呼吸异常（呼吸<10或>25）的原因
- 检查HRV（心率变异性）高压力状态的潜在原因
- 呼吸暂停AHI指数异常时建议专科评估

## 配置说明

### LLM配置（config/agent_llm_config.json）

```json
{
  "config": {
    "model": "doubao-seed-1-6-251015",
    "temperature": 0.7,
    "top_p": 0.9,
    "max_completion_tokens": 8000,
    "timeout": 600,
    "thinking": "disabled"
  },
  "sp": "系统提示词",
  "tools": ["bed_monitoring_analyzer_tool"]
}
```

## 环境变量

需要设置以下环境变量：

- `COZE_WORKSPACE_PATH`: 项目工作目录
- `COZE_WORKLOAD_IDENTITY_API_KEY`: API密钥
- `COZE_INTEGRATION_MODEL_BASE_URL`: 模型服务Base URL

## 技术栈

- **框架**: LangGraph + LangChain
- **LLM**: 豆包-seed模型
- **数据分析**: pandas
- **语言**: Python 3.12

## 开发指南

### 添加新的分析功能

1. 在`src/tools/bed_monitoring_db_analyzer.py`中添加新的分析方法
2. 在`BedMonitoringAnalyzer`类中实现具体逻辑
3. 在`generate_full_report`方法中集成新分析
4. 更新System Prompt以支持新功能

### 添加新的工具

1. 在`src/tools/`目录下创建新工具文件
2. 使用`@tool`装饰器定义工具函数
3. 在`src/agents/agent.py`中注册工具
4. 更新配置文件中的`tools`列表

## 测试

运行测试脚本：

```bash
python3 test_agent.py
```

## 许可证

MIT License

## 联系方式

如有问题或建议，请联系项目维护者。











I'll explore the project to understand its functionality. Let me start by examining the main components.


让我继续深入了解项目的具体功能和工具。


现在我已经了解了这个项目的主要功能，让我为您总结一下：


# 项目功能概述

这个项目是一个**智能病床监控数据分析系统**，主要用于分析患者的生理监测数据，特别是心率、呼吸、睡眠行为等指标。以下是其主要功能：

## 1. 病床监护数据分析






## 4. 趋势分析
- **长期趋势分析** ([analyze_trend_and_pattern](file:///d:/Code/Python/LangGrapTestMedCoze/3/src/tools/analyze_trend_tool.py#L213-L249))：分析多天数据的趋势，识别心率高峰时段、呼吸暂停趋势和综合风险评分

## 5. 问答检索功能
- **自然语言查询** ([qa_retriever](file:///d:/Code/Python/LangGrapTestMedCoze/3/src/tools/qa_retriever.py#L307-L350))：用户可以用自然语言提问（如"睡得怎么样？"、"心率多少？"），系统会返回相应指标的分析结果

## 6. AI驱动的智能分析
- **LangGraph智能代理**：使用LangGraph框架构建智能代理，整合上述工具进行复杂数据分析任务
- **多轮对话记忆**：支持多轮对话，并保持上下文连贯性

## 7. 数据存储与持久化
- **MySQL数据库支持**：支持将会话数据和状态持久化到MySQL数据库
- **内存回退机制**：当MySQL不可用时，自动回退到内存存储

该项目特别适用于医疗环境中的患者监护数据分析，能够帮助医护人员快速了解患者的生命体征变化趋势，识别潜在的健康风险，并生成专业的护理交班报告。


这个项目是一个**智能病床监控数据分析系统**，主要用于分析患者的生理监测数据。以下是其主要功能：

## 1. 病床监护数据分析
- **病床占用状态分析**：分析患者在床/离床时间、长时间离床事件、夜间离床次数等
- **临床级生命体征分析**：监测心率和呼吸频率，检测异常情况（心动过速、心动过缓、呼吸异常等）
- **呼吸暂停综合分析**：计算AHI指数（呼吸暂停低通气指数），评估呼吸暂停风险等级
- **体动与睡眠行为分析**：评估睡眠效率、体动模式和睡眠质量
- **晨间评估**：分析早晨心率趋势、晨峰现象等





## 4. 趋势分析
- **长期趋势分析**：分析多天数据的趋势，识别心率高峰时段、呼吸暂停趋势和综合风险评分

## 5. 问答检索功能
- **自然语言查询**：用户可以用自然语言提问（如"睡得怎么样？"、"心率多少？"），系统会返回相应指标的分析结果

## 6. AI驱动的智能分析
- **LangGraph智能代理**：使用LangGraph框架构建智能代理，整合上述工具进行复杂数据分析任务
- **多轮对话记忆**：支持多轮对话，并保持上下文连贯性

## 7. 数据存储与持久化
- **MySQL数据库支持**：支持将会话数据和状态持久化到MySQL数据库
- **内存回退机制**：当MySQL不可用时，自动回退到内存存储

该项目特别适用于医疗环境中的患者监护数据分析，能够帮助医护人员快速了解患者的生命体征变化趋势，识别潜在的健康风险，并生成专业的护理交班报告。
