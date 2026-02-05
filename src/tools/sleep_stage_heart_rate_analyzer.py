"""
基于心率和HRV特征的睡眠分期分析工具
根据AASM标准和心率/HRV特征进行睡眠分期
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple, Optional
from langchain_community.tools import tool

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeartRateSleepStageAnalyzer:
    """基于心率和HRV的睡眠阶段分析器"""
    
    # 睡眠阶段常量
    STAGE_AWAKE = 0
    STAGE_N1 = 1
    STAGE_N2 = 2
    STAGE_N3 = 3
    STAGE_REM = 4
    
    # 睡眠阶段标签映射
    STAGE_LABELS = {
        STAGE_AWAKE: "清醒",
        STAGE_N1: "浅睡N1", 
        STAGE_N2: "中睡N2",
        STAGE_N3: "深睡N3",
        STAGE_REM: "眼动REM"
    }
    
    @classmethod
    def get_stage_label(cls, stage_value: int) -> str:
        """
        根据阶段值返回对应的标签
        
        Args:
            stage_value: 睡眠阶段值
            
        Returns:
            睡眠阶段标签
        """
        return cls.STAGE_LABELS.get(stage_value, "未知")
    
    @staticmethod
    def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        准备数据，包括排序和类型转换
        
        Args:
            data: 原始数据
            
        Returns:
            处理后的数据
        """
        # 复制数据避免修改原数据
        processed_data = data.copy()
        
        # 确保数据按时间排序
        processed_data = processed_data.sort_values('upload_time').reset_index(drop=True)
        
        # 将生理指标字段转换为数值类型
        numeric_fields = ['heart_rate', 'respiratory_rate', 'body_moves_ratio']
        for field in numeric_fields:
            if field in processed_data.columns:
                processed_data[field] = pd.to_numeric(processed_data[field], errors='coerce')
        
        return processed_data
    
    @staticmethod
    def _calculate_hrv_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        计算HRV特征
        
        Args:
            data: 原始数据
            
        Returns:
            包含HRV特征的数据
        """
        processed_data = data.copy()
        
        # 计算心率变异性指标
        if 'heart_rate' in processed_data.columns:
            # 计算心率滚动标准差作为HRV的近似
            processed_data['hr_std'] = processed_data['heart_rate'].rolling(window=5, center=True, min_periods=1).std()
            
            # 计算心率滚动平均值
            processed_data['hr_avg'] = processed_data['heart_rate'].rolling(window=5, center=True, min_periods=1).mean()
            
            # 计算心率变化率
            processed_data['hr_change'] = processed_data['heart_rate'].diff().abs()
            processed_data['hr_change_avg'] = processed_data['hr_change'].rolling(window=5, center=True, min_periods=1).mean()
        else:
            processed_data['hr_std'] = 0
            processed_data['hr_avg'] = processed_data.get('heart_rate', 0)
            processed_data['hr_change_avg'] = 0
        
        return processed_data
    
    @classmethod
    def calculate_baseline_heart_rate(cls, data: pd.DataFrame) -> float:
        """
        计算基线心率
        
        Args:
            data: 睡眠数据
            
        Returns:
            基线心率
        """
        if 'heart_rate' not in data.columns:
            return 70  # 默认值
        
        # 过滤有效心率数据
        valid_hr = data['heart_rate'].dropna()
        valid_hr = valid_hr[(valid_hr >= 40) & (valid_hr <= 120)]
        
        if valid_hr.empty:
            return 70  # 默认值
        
        # 使用清醒状态下的心率作为基线
        # 假设前30分钟可能包含清醒状态
        baseline_data = valid_hr.head(min(30, len(valid_hr)))
        return baseline_data.mean()
    
    @classmethod
    def analyze_sleep_stages_by_heart_rate_rules(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        基于心率规则的睡眠分期分析方法
        根据AASM标准和心率/HRV特征进行睡眠分期
        
        Args:
            data: 睡眠数据
            
        Returns:
            包含睡眠阶段的数据分析
        """
        logger.info("开始基于心率规则的睡眠分期分析")
        
        # 准备数据
        processed_data = cls._prepare_data(data)
        
        # 计算HRV特征
        processed_data = cls._calculate_hrv_features(processed_data)
        
        # 计算基线心率
        baseline_hr = cls.calculate_baseline_heart_rate(processed_data)
        logger.info(f"计算得到基线心率: {baseline_hr:.2f}")
        
        # 初始化睡眠阶段
        processed_data['stage_value'] = cls.STAGE_AWAKE
        processed_data['stage_label'] = cls.STAGE_LABELS[cls.STAGE_AWAKE]
        
        # 计算时间特征
        processed_data['hour'] = processed_data['upload_time'].dt.hour
        processed_data['is_night'] = (processed_data['hour'] >= 22) | (processed_data['hour'] < 6)
        
        # 基于心率和HRV特征的睡眠分期规则
        for i, row in processed_data.iterrows():
            hr = row.get('heart_rate', 0)
            hr_std = row.get('hr_std', 0)
            hr_avg = row.get('hr_avg', hr)
            is_night = row.get('is_night', False)
            
            if pd.isna(hr) or hr == 0:
                # 无心率数据，标记为清醒
                stage = cls.STAGE_AWAKE
            elif is_night:
                # 夜间睡眠分期
                if hr >= baseline_hr * 1.10 and hr_std > 8:
                    # REM阶段：心率较高且波动大
                    stage = cls.STAGE_REM
                elif hr <= baseline_hr * 0.85 and hr_std < 3:
                    # N3阶段：心率最低且稳定
                    stage = cls.STAGE_N3
                elif hr <= baseline_hr * 0.95 and hr_std < 5:
                    # N2阶段：心率较低且较稳定
                    stage = cls.STAGE_N2
                elif hr <= baseline_hr * 1.00 and hr_std < 7:
                    # N1阶段：心率略降且波动减弱
                    stage = cls.STAGE_N1
                else:
                    # 其他情况标记为清醒
                    stage = cls.STAGE_AWAKE
            else:
                # 白天：默认标记为清醒
                stage = cls.STAGE_AWAKE
            
            processed_data.at[i, 'stage_value'] = stage
            processed_data.at[i, 'stage_label'] = cls.STAGE_LABELS[stage]
        
        logger.info("基于心率规则的睡眠分期分析完成")
        return processed_data
    
    @classmethod
    def analyze_sleep_stages_by_hrv_patterns(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        基于HRV模式的睡眠分期分析方法
        更复杂的分析方法，考虑心率变异性的详细模式
        
        Args:
            data: 睡眠数据
            
        Returns:
            包含睡眠阶段的数据分析
        """
        logger.info("开始基于HRV模式的睡眠分期分析")
        
        # 准备数据
        processed_data = cls._prepare_data(data)
        
        # 计算HRV特征
        processed_data = cls._calculate_hrv_features(processed_data)
        
        # 计算基线心率
        baseline_hr = cls.calculate_baseline_heart_rate(processed_data)
        logger.info(f"计算得到基线心率: {baseline_hr:.2f}")
        
        # 初始化睡眠阶段
        processed_data['stage_value'] = cls.STAGE_AWAKE
        processed_data['stage_label'] = cls.STAGE_LABELS[cls.STAGE_AWAKE]
        
        # 计算时间特征
        processed_data['hour'] = processed_data['upload_time'].dt.hour
        processed_data['is_night'] = (processed_data['hour'] >= 22) | (processed_data['hour'] < 6)
        
        # 计算额外的HRV特征
        if 'hr_std' in processed_data.columns:
            # 计算HRV的低频/高频比值近似
            processed_data['hrv_lf_hf_ratio'] = processed_data['hr_std'] / (processed_data['hr_std'].rolling(window=20, center=True, min_periods=1).mean() + 0.1)
        
        # 基于HRV模式的睡眠分期规则
        for i, row in processed_data.iterrows():
            hr = row.get('heart_rate', 0)
            hr_std = row.get('hr_std', 0)
            hr_change_avg = row.get('hr_change_avg', 0)
            hrv_lf_hf = row.get('hrv_lf_hf_ratio', 1)
            is_night = row.get('is_night', False)
            
            if pd.isna(hr) or hr == 0:
                # 无心率数据，标记为清醒
                stage = cls.STAGE_AWAKE
            elif is_night:
                # 夜间睡眠分期
                if hr >= baseline_hr * 1.05 and hr_std > 6 and hrv_lf_hf > 1.2:
                    # REM阶段：心率较高，波动大，交感神经活跃
                    stage = cls.STAGE_REM
                elif hr <= baseline_hr * 0.85 and hr_std < 3 and hr_change_avg < 2:
                    # N3阶段：心率最低，非常稳定
                    stage = cls.STAGE_N3
                elif hr <= baseline_hr * 0.95 and hr_std < 5 and hr_change_avg < 4:
                    # N2阶段：心率较低，较稳定
                    stage = cls.STAGE_N2
                elif hr <= baseline_hr * 1.00 and hr_std < 7 and hr_change_avg < 6:
                    # N1阶段：心率略降，波动减弱
                    stage = cls.STAGE_N1
                else:
                    # 其他情况标记为清醒
                    stage = cls.STAGE_AWAKE
            else:
                # 白天：默认标记为清醒
                stage = cls.STAGE_AWAKE
            
            processed_data.at[i, 'stage_value'] = stage
            processed_data.at[i, 'stage_label'] = cls.STAGE_LABELS[stage]
        
        # 平滑睡眠阶段
        processed_data = cls._smooth_sleep_stages(processed_data)
        
        logger.info("基于HRV模式的睡眠分期分析完成")
        return processed_data
    
    @staticmethod
    def _smooth_sleep_stages(data: pd.DataFrame, window: int = 3) -> pd.DataFrame:
        """
        平滑睡眠阶段，减少碎片化
        
        Args:
            data: 包含睡眠阶段的数据
            window: 平滑窗口大小
            
        Returns:
            平滑后的数据分析
        """
        processed_data = data.copy()
        
        if 'stage_value' in processed_data.columns:
            # 使用移动平均平滑睡眠阶段
            # 注意：这里使用众数平滑，因为阶段值是分类的
            for i in range(len(processed_data)):
                start = max(0, i - window // 2)
                end = min(len(processed_data), i + window // 2 + 1)
                window_data = processed_data.iloc[start:end]['stage_value']
                if not window_data.empty:
                    # 使用众数作为平滑后的阶段值
                    most_common = window_data.mode()
                    if not most_common.empty:
                        processed_data.at[i, 'stage_value'] = most_common.iloc[0]
                        
        # 更新阶段标签
        if 'stage_value' in processed_data.columns:
            processed_data['stage_label'] = processed_data['stage_value'].apply(HeartRateSleepStageAnalyzer.get_stage_label)
        
        return processed_data
    
    @classmethod
    def generate_sleep_stage_summary(cls, data: pd.DataFrame) -> Dict:
        """
        生成睡眠阶段分析摘要
        
        Args:
            data: 包含睡眠阶段的数据
            
        Returns:
            睡眠阶段分析摘要
        """
        summary = {
            'total_records': len(data),
            'stage_distribution': {},
            'stage_duration_minutes': {},
            'average_heart_rate_by_stage': {}
        }
        
        # 计算各阶段分布
        if 'stage_value' in data.columns:
            stage_counts = data['stage_value'].value_counts()
            for stage, count in stage_counts.items():
                stage_label = cls.get_stage_label(stage)
                summary['stage_distribution'][stage_label] = count
                
                # 计算各阶段时长（假设每条记录1分钟）
                summary['stage_duration_minutes'][stage_label] = count
                
                # 计算各阶段平均心率
                stage_data = data[data['stage_value'] == stage]
                if 'heart_rate' in stage_data.columns:
                    avg_hr = stage_data['heart_rate'].mean()
                    summary['average_heart_rate_by_stage'][stage_label] = round(avg_hr, 2) if not pd.isna(avg_hr) else 0
        
        # 计算总睡眠时长
        sleep_stages = [cls.STAGE_N1, cls.STAGE_N2, cls.STAGE_N3, cls.STAGE_REM]
        total_sleep_duration = sum(
            summary['stage_duration_minutes'].get(cls.get_stage_label(stage), 0)
            for stage in sleep_stages
        )
        summary['total_sleep_duration_minutes'] = total_sleep_duration
        
        return summary


@tool
def analyze_sleep_stages_by_heart_rate(date: str) -> str:
    """
    基于心率特征的睡眠分期分析工具
    根据AASM标准和心率/HRV特征分析睡眠分期
    
    Args:
        date: 分析日期，格式为YYYY-MM-DD
        
    Returns:
        睡眠分期分析结果，包含两种分析方法的结果
    """
    try:
        logger.info(f"开始分析 {date} 的睡眠分期（基于心率特征）")
        
        # 这里应该从数据库或文件中获取睡眠数据
        # 为了演示，我们创建一个模拟数据加载函数
        def load_sleep_data(date_str):
            """加载睡眠数据"""
            # 模拟数据 - 实际应用中应该从数据库获取
            times = pd.date_range(start=f"{date_str} 22:00:00", end=f"{date_str} 23:59:59", freq="1min")
            times = times.append(pd.date_range(start=f"{date_str[:8]}{int(date_str[8:10])+1} 00:00:00", end=f"{date_str[:8]}{int(date_str[8:10])+1} 06:00:00", freq="1min"))
            
            # 生成模拟心率数据
            # 模拟睡眠周期：清醒 -> N1 -> N2 -> N3 -> N2 -> REM -> N2 -> N3 -> N2 -> REM -> 清醒
            hr_values = []
            stage_sequence = [
                (60, 75, 120),  # 清醒
                (65, 72, 60),   # N1
                (60, 68, 90),   # N2
                (55, 62, 60),   # N3
                (60, 68, 60),   # N2
                (70, 80, 45),   # REM
                (60, 68, 60),   # N2
                (55, 62, 60),   # N3
                (60, 68, 60),   # N2
                (70, 80, 45),   # REM
                (65, 80, 30)    # 清醒
            ]
            
            for min_hr, max_hr, duration in stage_sequence:
                hr_segment = np.random.uniform(min_hr, max_hr, duration)
                hr_values.extend(hr_segment)
            
            # 截取与时间对应的长度
            hr_values = hr_values[:len(times)]
            if len(hr_values) < len(times):
                # 填充剩余数据
                hr_values.extend([70] * (len(times) - len(hr_values)))
            
            # 生成模拟呼吸率
            rr_values = np.random.uniform(12, 20, len(times))
            
            # 生成模拟体动
            body_moves = np.random.uniform(0, 15, len(times))
            
            data = pd.DataFrame({
                'upload_time': times,
                'heart_rate': hr_values,
                'respiratory_rate': rr_values,
                'body_moves_ratio': body_moves
            })
            
            return data
        
        # 加载睡眠数据
        sleep_data = load_sleep_data(date)
        logger.info(f"加载到 {len(sleep_data)} 条睡眠数据")
        
        # 方法1：基于心率规则的睡眠分期分析
        analyzer = HeartRateSleepStageAnalyzer()
        result_method1 = analyzer.analyze_sleep_stages_by_heart_rate_rules(sleep_data)
        summary_method1 = analyzer.generate_sleep_stage_summary(result_method1)
        
        # 方法2：基于HRV模式的睡眠分期分析
        result_method2 = analyzer.analyze_sleep_stages_by_hrv_patterns(sleep_data)
        summary_method2 = analyzer.generate_sleep_stage_summary(result_method2)
        
        # 构建分析结果
        analysis_result = {
            'success': True,
            'date': date,
            'data_points': len(sleep_data),
            'analysis_methods': {
                'rule_based': {
                    'name': '基于心率规则的分析方法',
                    'summary': summary_method1
                },
                'hrv_based': {
                    'name': '基于HRV模式的分析方法',
                    'summary': summary_method2
                }
            },
            'sleep_stage_characteristics': {
                'awake': {
                    'heart_rate': '高，波动大',
                    'hrv': '低频成分主导'
                },
                'n1': {
                    'heart_rate': '略降，波动减弱',
                    'hrv': '高频成分开始上升'
                },
                'n2': {
                    'heart_rate': '明显降低，变异性下降',
                    'hrv': '低频/高频比值下降'
                },
                'n3': {
                    'heart_rate': '最低，保持稳定',
                    'hrv': '高频成分主导'
                },
                'rem': {
                    'heart_rate': '波动大，接近清醒',
                    'hrv': '低频成分回升'
                }
            }
        }
        
        logger.info(f"睡眠分期分析完成，日期: {date}")
        return json.dumps(analysis_result, ensure_ascii=False, default=str)
        
    except Exception as e:
        logger.error(f"分析睡眠分期失败: {str(e)}")
        error_result = {
            'success': False,
            'error': f"分析失败: {str(e)}"
        }
        return json.dumps(error_result, ensure_ascii=False)


@tool
def get_sleep_stage_analysis_help() -> str:
    """
    获取睡眠分期分析工具的帮助信息
    提供基于心率和HRV特征的睡眠分期分析工具的使用说明
    
    Returns:
        帮助信息
    """
    help_info = {
        'success': True,
        'tool_name': '睡眠分期分析工具（基于心率特征）',
        'description': '根据AASM标准和心率/HRV特征分析睡眠分期',
        'methods': [
            {
                'name': '基于心率规则的分析方法',
                'description': '基于心率水平和变异性的简单规则进行睡眠分期',
                'features': ['心率水平', '心率变异性', '时间特征']
            },
            {
                'name': '基于HRV模式的分析方法',
                'description': '基于详细的HRV模式和自主神经活动特征进行睡眠分期',
                'features': ['心率水平', '心率变异性', '心率变化率', '自主神经平衡']
            }
        ],
        'sleep_stages': [
            {
                'stage': '清醒',
                'characteristics': '心率高，波动大，HRV低频成分主导'
            },
            {
                'stage': '浅睡N1',
                'characteristics': '心率略降，波动减弱，HRV高频成分开始上升'
            },
            {
                'stage': '中睡N2',
                'characteristics': '心率明显降低，变异性下降，HRV低频/高频比值下降'
            },
            {
                'stage': '深睡N3',
                'characteristics': '心率最低，保持稳定，HRV高频成分主导'
            },
            {
                'stage': '眼动REM',
                'characteristics': '心率波动大，接近清醒，HRV低频成分回升'
            }
        ],
        'usage': '调用 analyze_sleep_stages_by_heart_rate 工具，传入日期参数（格式：YYYY-MM-DD）'
    }
    
    return json.dumps(help_info, ensure_ascii=False)
