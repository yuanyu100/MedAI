#!/usr/bin/env python3
"""
最终版智能体API服务器 - 提供完整功能的API接口，带错误处理
"""

import os
import sys
import json
import traceback
import tempfile
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from langchain_core.messages import HumanMessage
import pandas as pd

# 延迟导入，避免启动时的错误
def get_agent_module():
    from src.agents.agent import build_agent
    return build_agent

def get_excel_analysis_module():
    # Excel analysis module has been removed
    def dummy_excel_analysis(file_path):
        return '{"error": "Excel analysis has been removed"}'
    return dummy_excel_analysis

def get_db_analysis_module():
    from src.tools.bed_monitoring_db_analyzer import analyze_bed_monitoring_from_db
    return analyze_bed_monitoring_from_db

def get_visualization_module():
    # Visualization module has been removed
    def dummy_visualization(data):
        return '{"error": "nursing_report_visualization_tool has been removed"}'
    return dummy_visualization

def get_pdf_module():
    # PDF module has been removed
    def dummy_pdf_tool(file_path, output_path=None):
        return {"error": "monitoring_pdf_tool has been removed"}
    return dummy_pdf_tool

def get_trend_module():
    from src.tools.analyze_trend_tool import analyze_trend_and_pattern
    return analyze_trend_and_pattern


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


# 为qa_retriever创建一个包装函数
def create_sample_excel():
    """创建示例Excel文件用于QA查询"""
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


def parse_data_content(content: str):
    """解析数据内容"""
    import re

    data = {}

    hr_match = re.search(r'心率:(\d+)次/分钟', content)
    data['heart_rate'] = int(hr_match.group(1)) if hr_match else 0

    rr_match = re.search(r'呼吸:(\d+)次/分钟', content)
    data['respiration_rate'] = int(rr_match.group(1)) if rr_match else 0

    apnea_match = re.search(r'呼吸暂停次数:(\d+)次', content)
    data['apnea_count'] = int(apnea_match.group(1)) if apnea_match else 0

    body_move_match = re.search(r'体动次数的占比:(\d+)%', content)
    data['body_move_ratio'] = int(body_move_match.group(1)) if body_move_match else 0

    return data


def extract_summary_info(df):
    """提取整体摘要信息"""
    info = {}

    info['data_count'] = len(df)
    info['start_time'] = df['upload_time'].min().strftime('%Y-%m-%d %H:%M')
    info['end_time'] = df['upload_time'].max().strftime('%Y-%m-%d %H:%M')
    duration = df['upload_time'].max() - df['upload_time'].min()
    info['duration_hours'] = round(duration.total_seconds() / 3600, 1)

    # 综合评估
    hr_avg = df['heart_rate'].mean()
    rr_avg = df['respiration_rate'].mean()
    apnea_count = df['apnea_count'].sum()
    body_move_avg = df['body_move_ratio'].mean()

    issues = []
    if hr_avg < 60 or hr_avg > 100:
        issues.append("心率异常")
    if rr_avg < 12 or rr_avg > 20:
        issues.append("呼吸频率异常")
    if apnea_count > 10:
        issues.append("多次呼吸暂停")
    if body_move_avg > 30:
        issues.append("睡眠质量较差")

    if len(issues) == 0:
        info['overall_status'] = "整体状况良好，各项指标正常"
    else:
        info['overall_status'] = f"需要注意：{', '.join(issues)}"

    return info


def extract_heart_rate_info(df):
    """提取心率相关信息"""
    info = {}

    info['avg_hr'] = round(df['heart_rate'].mean(), 1)
    info['min_hr'] = int(df['heart_rate'].min())
    info['max_hr'] = int(df['heart_rate'].max())
    info['hr_range'] = f"{info['min_hr']}-{info['max_hr']} bpm"

    # 心率状态
    if info['avg_hr'] < 60:
        info['hr_status'] = "心率偏慢（心动过缓）"
    elif info['avg_hr'] > 100:
        info['hr_status'] = "心率偏快（心动过速）"
    else:
        info['hr_status'] = "心率正常"

    return info


def extract_respiration_info(df):
    """提取呼吸相关信息"""
    info = {}

    info['avg_rr'] = round(df['respiration_rate'].mean(), 1)
    info['min_rr'] = int(df['respiration_rate'].min())
    info['max_rr'] = int(df['respiration_rate'].max())
    info['rr_range'] = f"{info['min_rr']}-{info['max_rr']} 次/分钟"

    # 呼吸状态
    if info['avg_rr'] < 12:
        info['rr_status'] = "呼吸偏慢（呼吸过缓）"
    elif info['avg_rr'] > 20:
        info['rr_status'] = "呼吸偏快（呼吸过速）"
    else:
        info['rr_status'] = "呼吸频率正常"

    return info


def extract_apnea_info(df):
    """提取呼吸暂停相关信息"""
    info = {}

    total_apnea = df['apnea_count'].sum()
    info['total_apnea'] = int(total_apnea)

    # 计算AHI（每小时暂停次数）
    start_time = df['upload_time'].min()
    end_time = df['upload_time'].max()
    total_hours = (end_time - start_time).total_seconds() / 3600
    if total_hours > 0:
        ahi = round(total_apnea / total_hours, 1)
        info['ahi'] = ahi
    else:
        info['ahi'] = 0

    # 风险评估
    if info['ahi'] < 5:
        info['apnea_risk'] = "正常，无显著呼吸暂停"
    elif info['ahi'] < 15:
        info['apnea_risk'] = "轻度呼吸暂停"
    elif info['ahi'] < 30:
        info['apnea_risk'] = "中度呼吸暂停，建议就医"
    else:
        info['apnea_risk'] = "重度呼吸暂停，需要及时治疗"

    return info


def extract_sleep_info(df):
    """提取睡眠相关信息"""
    info = {}

    # 计算平均体动占比（作为睡眠质量的参考）
    avg_body_move = df['body_move_ratio'].mean()
    info['avg_body_move_ratio'] = round(avg_body_move, 1)

    # 估算深睡时长（简化算法：低体动时段）
    low_move_periods = df[df['body_move_ratio'] < 20]
    if len(low_move_periods) > 0:
        # 假设低体动时段为深睡
        info['deep_sleep_estimate'] = f"约 {len(low_move_periods)} 个低体动时段"
    else:
        info['deep_sleep_estimate'] = "数据不足"

    # 睡眠评分（基于体动频率）
    if avg_body_move < 10:
        info['sleep_score'] = 85  # 优秀
        info['sleep_quality'] = "睡眠质量优秀，体动较少"
    elif avg_body_move < 20:
        info['sleep_score'] = 75  # 良好
        info['sleep_quality'] = "睡眠质量良好，偶有翻身"
    elif avg_body_move < 30:
        info['sleep_score'] = 65  # 一般
        info['sleep_quality'] = "睡眠质量一般，翻身较多"
    else:
        info['sleep_score'] = 55  # 较差
        info['sleep_quality'] = "睡眠质量较差，频繁翻身"

    return info


def match_question_to_category(question: str) -> str:
    """根据问题匹配到相应的数据类别"""
    question_lower = question.lower()

    # 呼吸暂停相关问题（优先级最高，避免被"呼吸"关键词匹配）
    apnea_keywords = ['呼吸暂停', '打呼噜', '憋气', '暂停', '鼾声']
    if any(keyword in question_lower for keyword in apnea_keywords):
        return 'apnea'

    # 睡眠相关问题
    sleep_keywords = ['睡', '睡眠', '质量', '深睡', '浅睡', '翻身', '体动']
    if any(keyword in question_lower for keyword in sleep_keywords):
        return 'sleep'

    # 心率相关问题
    hr_keywords = ['心率', '心跳', '脉搏', '快', '慢', '心动']
    if any(keyword in question_lower for keyword in hr_keywords):
        return 'heart_rate'

    # 呼吸相关问题
    rr_keywords = ['呼吸', '缺氧', '换气', '喘']
    if any(keyword in question_lower for keyword in rr_keywords):
        return 'respiration'

    # 风险/警报问题
    risk_keywords = ['危险', '风险', '警报', '严重', '异常']
    if any(keyword in question_lower for keyword in risk_keywords):
        return 'risk'

    # 总结/汇报问题
    summary_keywords = ['总结', '汇报', '整体', '概括', '情况', '怎么样', '如何']
    if any(keyword in question_lower for keyword in summary_keywords):
        return 'summary'

    return 'summary'  # 默认返回摘要


def format_response(category: str, info: dict) -> str:
    """格式化响应内容"""
    if category == 'sleep':
        return f"""睡眠情况：
- 睡眠评分：{info['sleep_score']}分
- 睡眠质量：{info['sleep_quality']}
- 平均体动占比：{info['avg_body_move_ratio']}%
- 深睡时长估算：{info['deep_sleep_estimate']}"""

    elif category == 'heart_rate':
        return f"""心率情况：
- 平均心率：{info['avg_hr']} bpm
- 心率范围：{info['hr_range']}
- 心率状态：{info['hr_status']}"""

    elif category == 'respiration':
        return f"""呼吸情况：
- 平均呼吸频率：{info['avg_rr']} 次/分钟
- 呼吸范围：{info['rr_range']}
- 呼吸状态：{info['rr_status']}"""

    elif category == 'apnea':
        return f"""呼吸暂停情况：
- 总呼吸暂停次数：{info['total_apnea']}次
- AHI指数：{info['ahi']}（每小时暂停次数）
- 风险评估：{info['apnea_risk']}"""

    elif category == 'risk':
        # 综合风险信息
        risk_info = []
        if 'hr_status' in info and '异常' in info['hr_status']:
            risk_info.append(f"- {info['hr_status']}")
        if 'rr_status' in info and '异常' in info['rr_status']:
            risk_info.append(f"- {info['rr_status']}")
        if 'apnea_risk' in info and '呼吸暂停' in info['apnea_risk']:
            risk_info.append(f"- {info['apnea_risk']}")
        if 'sleep_quality' in info and '较差' in info['sleep_quality']:
            risk_info.append(f"- {info['sleep_quality']}")

        if len(risk_info) > 0:
            return "风险提醒：\n" + "\n".join(risk_info)
        else:
            return "当前没有明显的风险指标，各项指标正常。"

    else:  # summary
        return f"""整体情况：
- 监测时间：{info['start_time']} 至 {info['end_time']}
- 总时长：{info['duration_hours']}小时
- 数据条数：{info['data_count']}条
- 综合评估：{info['overall_status']}"""


def qa_retrieve_internal(file_path: str, question: str) -> str:
    """
    内部检索函数
    """
    try:
        # 读取数据
        df = pd.read_excel(file_path)

        # 解析数据内容
        parsed_data = []
        for idx, row in df.iterrows():
            content = row['数据内容']
            data_dict = parse_data_content(content)
            data_dict['upload_time'] = pd.to_datetime(row['上传时间'])
            parsed_data.append(data_dict)

        parsed_df = pd.DataFrame(parsed_data)

        # 匹配问题类别
        category = match_question_to_category(question)

        # 提取相关信息
        info = {}

        if category == 'sleep':
            info = extract_sleep_info(parsed_df)
        elif category == 'heart_rate':
            info = extract_heart_rate_info(parsed_df)
        elif category == 'respiration':
            info = extract_respiration_info(parsed_df)
        elif category == 'apnea':
            info = extract_apnea_info(parsed_df)
        elif category == 'risk':
            # 综合所有风险信息
            info.update(extract_heart_rate_info(parsed_df))
            info.update(extract_respiration_info(parsed_df))
            info.update(extract_apnea_info(parsed_df))
            info.update(extract_sleep_info(parsed_df))
        else:  # summary
            info = extract_summary_info(parsed_df)

        # 格式化响应
        response = format_response(category, info)

        return response

    except Exception as e:
        import traceback
        error_msg = f"检索失败: {str(e)}\n{traceback.format_exc()}"
        return error_msg


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用程序生命周期管理"""
    print("🚀 启动最终版智能体API服务器...")
    # 设置环境变量
    os.environ.setdefault("QWEN_API_KEY", "sk-2ad6355b98dd43668a5eeb21e50e4642")
    os.environ.setdefault("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    yield
    # 关闭时的清理


# 创建FastAPI应用
app = FastAPI(
    title="最终版智能病床监控数据分析系统API",
    description="提供智能体和数据分析功能的完整API接口",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "欢迎使用最终版智能病床监控数据分析系统API",
        "version": "1.0.0",
        "endpoints": {
            "POST /agent/run": "运行智能体",
            "POST /analysis/excel": "分析Excel数据",
            "POST /analysis/database": "分析数据库数据",
            "POST /visualization": "生成可视化报告",
            "POST /pdf": "生成PDF报告",
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
        
        # 构建智能体
        build_agent_fn = get_agent_module()
        agent = build_agent_fn()
        
        # 准备输入消息
        messages = [HumanMessage(content=request.query)]
        
        # 配置会话
        config = {"configurable": {"thread_id": request.thread_id}}
        
        # 调用智能体
        response = agent.invoke({"messages": messages}, config=config)
        
        # 提取响应内容
        result = []
        for msg in response.get('messages', []):
            if hasattr(msg, 'content') and msg.content:
                result.append(str(msg.content))
        
        return {
            "success": True,
            "result": result,
            "thread_id": request.thread_id
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
        analyze_fn = get_db_analysis_module()
        result = analyze_fn(request.table_name)
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


@app.post("/visualization")
async def generate_visualization(request: VisualizationRequest):
    """生成可视化报告"""
    try:
        print("📈 生成可视化报告")
        
        # 生成可视化报告
        visualize_fn = get_visualization_module()
        result = visualize_fn(request.data)
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


@app.post("/pdf")
async def generate_pdf_report(request: PDFTrendRequest):
    """生成PDF报告"""
    try:
        print(f"📄 生成PDF报告: {request.file_path}")
        
        # 检查文件是否存在
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"文件不存在: {request.file_path}")
        
        # 生成PDF报告
        pdf_fn = get_pdf_module()
        result = pdf_fn(request.file_path, request.output_path)
        
        return {
            "success": True,
            "pdf_path": result
        }
        
    except Exception as e:
        print(f"❌ PDF生成失败: {str(e)}")
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
        trend_fn = get_trend_module()
        result = trend_fn(request.file_path)
        
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


@app.post("/qa")
async def qa_query(request: QARequest):
    """问答查询"""
    try:
        print(f"❓ 问答查询: {request.query}")
        
        # 创建示例数据文件
        sample_file = create_sample_excel()
        
        # 执行问答查询（调用内部函数而不是工具装饰的函数）
        result = qa_retrieve_internal(sample_file, request.query)
        
        return {
            "success": True,
            "answer": result
        }
        
    except Exception as e:
        print(f"❌ 问答查询失败: {str(e)}")
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
        "message": "最终版智能体API服务器运行正常",
        "timestamp": datetime.now().isoformat()
    }


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """启动API服务器"""
    print(f"🌐 启动最终版API服务器在 {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="启动最终版智能体API服务器")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("-p", "--port", type=int, default=8000, help="服务器端口")
    
    args = parser.parse_args()
    
    print("🚀 启动最终版智能体API服务器...")
    print(f"🌐 访问地址: http://{args.host}:{args.port}")
    print(f"📖 API文档: http://{args.host}:{args.port}/docs")
    
    start_server(args.host, args.port)