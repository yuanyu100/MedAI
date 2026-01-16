# 智能体API接口使用指南

## 概述

本项目提供了一个基于FastAPI的RESTful API接口，用于访问智能病床监控数据分析系统的各项功能。

## 启动API服务器

```bash
python api_server.py
```

或者指定端口：

```bash
python api_server.py --port 8080
```

服务器默认运行在 `http://localhost:8000`

## API端点

### 1. 健康检查
- **GET** `/health`
- 检查服务器是否正常运行

### 2. 智能体运行
- **POST** `/agent/run`
- 运行智能体并获取分析结果

请求体：
```json
{
  "query": "请输入查询内容",
  "thread_id": "会话ID（可选）",
  "file_path": "文件路径（可选）"
}
```

### 4. 数据库数据分析
- **POST** `/analysis/database`
- 从数据库分析数据

请求体：
```json
{
  "table_name": "数据库表名（默认为'device_data'）"
}
```

### 5. 趋势分析
- **POST** `/trend`
- 分析多天数据的趋势

请求体：
```json
{
  "file_path": "数据文件路径"
}
```

### 8. 问答查询
- **POST** `/qa`
- 自然语言问答查询

请求体：
```json
{
  "query": "查询内容"
}
```

## 使用示例

### Python客户端示例

```python
import requests
import json

# 基础URL
BASE_URL = "http://localhost:8000"

# 1. 健康检查
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# 2. 运行智能体
payload = {
    "query": "请分析病床监护数据文件 /path/to/data.xlsx，生成护理交班报告。",
    "thread_id": "session-1"
}
response = requests.post(f"{BASE_URL}/agent/run", json=payload)
print(response.json())

# 3. Excel数据分析
payload = {
    "file_path": "/path/to/data.xlsx"
}
response = requests.post(f"{BASE_URL}/analysis/excel", json=payload)
print(response.json())
```

### cURL示例

```bash
# 健康检查
curl -X GET http://localhost:8000/health

# 智能体运行
curl -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{"query": "请分析病床监护数据", "thread_id": "session-1"}'

# Excel数据分析
curl -X POST http://localhost:8000/analysis/excel \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/data.xlsx"}'
```

## API文档

API文档自动生成在以下地址：
- http://localhost:8000/docs - 交互式API文档
- http://localhost:8000/redoc - ReDoc格式文档

## 测试API

运行测试客户端：

```bash
python test_api_client.py
```

该脚本会自动测试API服务器的各项功能。

## 错误处理

API返回标准HTTP状态码：
- `200` - 成功
- `400` - 请求错误
- `404` - 资源未找到
- `500` - 服务器内部错误

错误响应格式：
```json
{
  "detail": {
    "success": false,
    "error": "错误信息",
    "traceback": "错误追踪信息（仅开发模式）"
  }
}
```

## 环境配置

API服务器会自动设置所需的环境变量，包括：
- `QWEN_API_KEY`: 模型API密钥
- `QWEN_BASE_URL`: 模型API基础URL

如果需要自定义这些值，可以在启动服务器前设置环境变量。