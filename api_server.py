#!/usr/bin/env python3
"""
智能体API服务器 - 提供对外接口访问智能体功能
"""

import os
import sys
import json
import traceback
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from langchain_core.messages import HumanMessage

from src.tools.bed_monitoring_db_analyzer import analyze_bed_monitoring_from_db

from src.tools.analyze_trend_tool import analyze_trend_and_pattern, analyze_trend_from_database


class AgentRequest(BaseModel):
    """智能体请求模型"""
    query: str
    thread_id: Optional[str] = "default-session"
    file_path: Optional[str] = None


class AnalysisRequest(BaseModel):
    """数据分析请求模型"""
    file_path: str


class DatabaseAnalysisRequest(BaseModel):
    """数据库分析请求模型"""
    table_name: Optional[str] = "device_data"


class VisualizationRequest(BaseModel):
    """可视化请求模型"""
    data: str


class PDFTrendRequest(BaseModel):
    """PDF和趋势分析请求模型"""
    file_path: str
    output_path: Optional[str] = None


class QARequest(BaseModel):
    """问答请求模型"""
    query: str


class TrendAnalysisRequest(BaseModel):
    """趋势分析请求模型"""
    table_name: Optional[str] = "device_data"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    device_sn: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用程序生命周期管理"""
    # 启动时的初始化
    print("🚀 启动智能体API服务器...")
    # 设置环境变量
    os.environ.setdefault("QWEN_API_KEY", "sk-2ad6355b98dd43668a5eeb21e50e4642")
    os.environ.setdefault("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    yield
    # 关闭时的清理


# 创建FastAPI应用
app = FastAPI(
    title="智能病床监控数据分析系统API",
    description="提供智能体和数据分析功能的API接口",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "欢迎使用智能病床监控数据分析系统API",
        "version": "1.0.0",
        "endpoints": {
            "POST /agent/run": "运行智能体",

            "POST /analysis/database": "分析数据库数据",
            "POST /analysis/trend": "分析多天监护数据趋势",
            "POST /visualization": "生成可视化报告",

            "POST /trend": "趋势分析",
            "POST /qa": "问答查询",
            "GET /health": "健康检查"
        }
    }


@app.post("/agent/run")
async def run_agent(request: AgentRequest):
    """运行智能体"""
    try:
        print(f"🔄 运行智能体，查询: {request.query}")
        
        # 智能体功能暂时不可用
        return {
            "success": False,
            "error": "Agent functionality is temporarily disabled",
            "message": "The agent module is not available at this time"
        }
        
    except Exception as e:
        print(f"❌ 运行智能体失败: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


@app.post("/analysis/database")
async def analyze_database_data(request: DatabaseAnalysisRequest):
    """分析数据库数据"""
    try:
        print(f"📊 分析数据库表: {request.table_name}")
        
        # 执行数据库分析
        result = analyze_bed_monitoring_from_db(request.table_name)
        analysis_result = json.loads(result)
        
        return {
            "success": True,
            "data": analysis_result
        }
        
    except Exception as e:
        print(f"❌ 数据库分析失败: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


@app.post("/visualization")
async def generate_visualization(request: VisualizationRequest):
    """生成可视化报告"""
    try:
        print("📈 生成可视化报告")
        
        # 生成可视化报告
        result = generate_nursing_report_visualization(request.data)
        result_dict = json.loads(result)
        
        return {
            "success": result_dict.get('success', False),
            "data": result_dict
        }
        
    except Exception as e:
        print(f"❌ 可视化生成失败: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


@app.post("/trend")
async def analyze_trend_data(request: PDFTrendRequest):
    """趋势分析"""
    try:
        print(f"📊 趋势分析: {request.file_path}")
        
        # 检查文件是否存在
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"文件不存在: {request.file_path}")
        
        # 执行趋势分析
        result = analyze_trend_and_pattern(request.file_path)
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        print(f"❌ 趋势分析失败: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


# 为qa_retriever创建一个包装函数
import tempfile
import os

def create_sample_excel():
    """创建示例Excel文件用于QA查询"""
    import pandas as pd
    from datetime import datetime, timedelta
    
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


class MockRuntime:
    """模拟ToolRuntime对象"""
    def __init__(self):
        self.context = None


@app.post("/qa")
async def qa_query(request: QARequest):
    """问答查询"""
    try:
        print(f"❓ 问答查询: {request.query}")
        
        # 问答功能暂时不可用
        return {
            "success": False,
            "error": "QA functionality is temporarily disabled",
            "message": "The QA module is not available at this time"
        }
        
    except Exception as e:
        print(f"❌ 问答查询失败: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


@app.post("/analysis/trend")
async def analyze_trend_data(request: TrendAnalysisRequest):
    """分析多天监护数据趋势"""
    try:
        print(f"📈 分析多天监护数据趋势")
        print(f"  表名: {request.table_name}")
        print(f"  开始日期: {request.start_date}")
        print(f"  结束日期: {request.end_date}")
        print(f"  设备序列号: {request.device_sn}")
        
        # 执行趋势分析
        result = analyze_trend_from_database(
            table_name=request.table_name,
            start_date=request.start_date,
            end_date=request.end_date,
            device_sn=request.device_sn
        )
        analysis_result = json.loads(result)
        
        return {
            "success": True,
            "data": analysis_result
        }
        
    except Exception as e:
        print(f"❌ 趋势分析失败: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "message": "智能体API服务器运行正常",
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """启动API服务器"""
    print(f"🌐 启动API服务器在 {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="启动智能体API服务器")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("-p", "--port", type=int, default=8000, help="服务器端口")
    
    args = parser.parse_args()
    
    print("🚀 启动智能体API服务器...")
    print(f"🌐 访问地址: http://{args.host}:{args.port}")
    print(f"📖 API文档: http://{args.host}:{args.port}/docs")
    
    start_server(args.host, args.port)