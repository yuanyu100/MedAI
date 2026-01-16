# 使用文档

## 快速开始

### 1. 安装依赖

```bash
pip install pandas openai langgraph langchain langchain-openai langchain-core
```

### 2. 配置环境变量

```bash
export COZE_WORKSPACE_PATH=/path/to/project
export COZE_WORKLOAD_IDENTITY_API_KEY=your_api_key
export COZE_INTEGRATION_MODEL_BASE_URL=https://api.example.com
```

### 3. 运行示例

```python
from src.agents.agent import build_agent
from langchain_core.messages import HumanMessage

# 创建Agent
agent = build_agent()

# 准备输入
input_data = """
请分析病床监护数据文件 /path/to/data.xlsx，生成护理交班报告。
报告需要包含：
1. 风险等级评估
2. 护理摘要
3. 关键事件列表
4. 医生建议
5. 护士提示
"""

# 调用Agent
messages = [HumanMessage(content=input_data)]
config = {"configurable": {"thread_id": "session-1"}}
response = agent.invoke({"messages": messages}, config=config)

# 获取结果
for msg in response['messages']:
    print(msg.content)
```

## 数据格式说明

### Excel数据格式

| 列名 | 说明 | 示例 |
|------|------|------|
| 数据类型 | 周期数据或状态 | 周期数据/状态 |
| 上传时间 | 数据采集时间 | 2025-12-17 18:00:44 |
| 来源设备名称 | 设备名称 | 体征测试301 |
| 来源设备SN码 | 设备序列号 | 210235UHMF3259000005 |
| 所属组织 | 组织名称 | 根组织 |
| 数据内容 | 监测指标 | 心率:60次/分钟;呼吸:15次/分钟;... |

### 数据内容字段解析

- 心率: XX次/分钟
- 呼吸: XX次/分钟
- 心跳间期平均值: XX毫秒
- 心跳间期均方根值: XX毫秒
- 心跳间期标准差: XX毫秒
- 心跳间期紊乱比例: XX%
- 体动次数的占比: XX%
- 呼吸暂停次数: X次

## 护理交班报告示例

```json
{
  "risk_level": "中",
  "nursing_summary": "患者全天依从性一般，总卧床时间10.7小时，夜间无离床行为。存在多次心动过缓和呼吸过缓事件，睡眠效率82.63%。",
  "critical_events": [
    {
      "type": "心动过缓",
      "time": "18:13",
      "analysis": "心率降至49次/分，可能与迷走神经兴奋或药物影响有关"
    },
    {
      "type": "呼吸过缓",
      "time": "22:58",
      "analysis": "呼吸频率降至6次/分，需警惕呼吸抑制风险"
    }
  ],
  "doctor_notes": "建议关注患者心动过缓（最低49次/分）和呼吸过缓（最低6次/分）事件，评估是否与药物、迷走神经张力增高或潜在心肺问题相关。",
  "nurse_tips": "夜间加强巡视，重点监测呼吸和心率变化；检查患者体位是否影响呼吸；评估患者离床活动的安全性。"
}
```

## 常见问题

### Q: 如何修改使用的LLM模型？
A: 编辑`config/agent_llm_config.json`中的`model`字段。

### Q: 如何调整风险等级判定标准？
A: 修改`src/tools/bed_monitoring_analyzer.py`中各个分析方法的判断逻辑。

### Q: 如何添加新的分析维度？
A: 在`BedMonitoringAnalyzer`类中添加新方法，并在`generate_full_report`中调用。

### Q: 支持哪些数据格式？
A: 目前支持Excel格式（.xlsx），包含指定列名的表格数据。

## 性能优化建议

1. 对于大量历史数据，建议分批分析
2. 可以使用缓存机制加速重复分析
3. 考虑使用异步IO提高数据处理速度
