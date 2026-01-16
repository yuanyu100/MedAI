#!/usr/bin/env python3
"""
数据库分析API服务器 - 专注于解决数据库分析端点问题
"""

import os
import sys
import json
import traceback
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from datetime import datetime

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


class DatabaseAnalysisRequest(BaseModel):
    """数据库分析请求模型"""
    table_name: Optional[str] = "device_data"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用程序生命周期管理"""
    print("🚀 启动数据库分析API服务器...")
    yield


# 创建FastAPI应用
app = FastAPI(
    title="数据库分析API服务器",
    description="提供数据库分析功能的API接口",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/analysis/database")
async def analyze_database_data(request: DatabaseAnalysisRequest):
    """分析数据库数据"""
    try:
        print(f"📊 分析数据库表: {request.table_name}")
        
        # 尝试导入并执行数据库分析
        from src.tools.bed_monitoring_db_analyzer import analyze_bed_monitoring_from_db
        
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
        
        # 返回错误信息，但不崩溃服务器
        error_response = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "message": "数据库分析失败，可能是由于数据库连接问题。请检查数据库配置。",
            "recommended_action": "如果您没有可用的数据库，可以使用 /analysis/excel 端点分析Excel文件"
        }
        
        return error_response


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "message": "数据库分析API服务器运行正常",
        "timestamp": datetime.now().isoformat()
    }


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """启动API服务器"""
    print(f"🌐 启动数据库分析API服务器在 {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="启动数据库分析API服务器")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("-p", "--port", type=int, default=8000, help="服务器端口")
    
    args = parser.parse_args()
    
    print("🚀 启动数据库分析API服务器...")
    print(f"🌐 访问地址: http://{args.host}:{args.port}")
    print(f"📖 API文档: http://{args.host}:{args.port}/docs")
    
    start_server(args.host, args.port)