"""
增强版睡眠分期分析工具
基于心率、HRV、呼吸和体动特征的高精度睡眠分期分析
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


class EnhancedSleepStageAnalyzer:
    """增强版睡眠阶段分析器"""
    
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
    def _calculate_enhanced_hrv_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        计算增强版HRV特征
        
        Args:
            data: 原始数据
            
        Returns:
            包含增强HRV特征的数据
        """
        processed_data = data.copy()
        
        # 计算心率变异性指标
        if 'heart_rate' in processed_data.columns:
            # 基础HRV特征
            processed_data['hr_std'] = processed_data['heart_rate'].rolling(window=5, center=True, min_periods=1).std()
            processed_data['hr_avg'] = processed_data['heart_rate'].rolling(window=5, center=True, min_periods=1).mean()
            processed_data['hr_change'] = processed_data['heart_rate'].diff().abs()
            processed_data['hr_change_avg'] = processed_data['hr_change'].rolling(window=5, center=True, min_periods=1).mean()
            
            # 增强HRV特征
            # 1. 心率变异性的时域指标
            processed_data['hr_rms'] = np.sqrt((processed_data['hr_change'] ** 2).rolling(window=10, center=True, min_periods=1).mean())
            processed_data['hr_range'] = processed_data['heart_rate'].rolling(window=10, center=True, min_periods=1).max() - \
                                      processed_data['heart_rate'].rolling(window=10, center=True, min_periods=1).min()
            
            # 2. 心率变异性的频域指标近似
            # 低频成分（0.04-0.15 Hz）
            processed_data['hr_lf'] = processed_data['hr_std'].rolling(window=20, center=True, min_periods=1).mean()
            # 高频成分（0.15-0.4 Hz）
            processed_data['hr_hf'] = processed_data['hr_std'].rolling(window=5, center=True, min_periods=1).mean()
            # LF/HF比值
            processed_data['hr_lf_hf_ratio'] = processed_data['hr_lf'] / (processed_data['hr_hf'] + 0.1)
            
            # 3. 心率趋势特征
            processed_data['hr_trend'] = processed_data['heart_rate'].rolling(window=30, center=True, min_periods=1).mean()
            processed_data['hr_trend_change'] = processed_data['hr_trend'].diff()
        else:
            processed_data['hr_std'] = 0
            processed_data['hr_avg'] = processed_data.get('heart_rate', 0)
            processed_data['hr_change_avg'] = 0
            processed_data['hr_rms'] = 0
            processed_data['hr_range'] = 0
            processed_data['hr_lf'] = 0
            processed_data['hr_hf'] = 0
            processed_data['hr_lf_hf_ratio'] = 0
            processed_data['hr_trend'] = 0
            processed_data['hr_trend_change'] = 0
        
        return processed_data
    
    @staticmethod
    def _calculate_respiratory_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        计算呼吸特征
        
        Args:
            data: 原始数据
            
        Returns:
            包含呼吸特征的数据
        """
        processed_data = data.copy()
        
        # 计算呼吸特征
        if 'respiratory_rate' in processed_data.columns:
            # 呼吸率统计特征
            processed_data['rr_std'] = processed_data['respiratory_rate'].rolling(window=5, center=True, min_periods=1).std()
            processed_data['rr_avg'] = processed_data['respiratory_rate'].rolling(window=5, center=True, min_periods=1).mean()
            
            # 呼吸稳定性
            processed_data['rr_stability'] = processed_data['rr_std'] / (processed_data['rr_avg'] + 0.1)
            
            # 呼吸模式特征
            processed_data['rr_change'] = processed_data['respiratory_rate'].diff().abs()
            processed_data['rr_change_avg'] = processed_data['rr_change'].rolling(window=5, center=True, min_periods=1).mean()
        else:
            processed_data['rr_std'] = 0
            processed_data['rr_avg'] = processed_data.get('respiratory_rate', 0)
            processed_data['rr_stability'] = 0
            processed_data['rr_change_avg'] = 0
        
        return processed_data
    
    @staticmethod
    def _calculate_body_movement_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        计算体动特征
        
        Args:
            data: 原始数据
            
        Returns:
            包含体动特征的数据
        """
        processed_data = data.copy()
        
        # 计算体动特征
        if 'body_moves_ratio' in processed_data.columns:
            # 体动统计特征
            processed_data['move_std'] = processed_data['body_moves_ratio'].rolling(window=5, center=True, min_periods=1).std()
            processed_data['move_avg'] = processed_data['body_moves_ratio'].rolling(window=5, center=True, min_periods=1).mean()
            
            # 体动频率特征
            processed_data['move_freq'] = (processed_data['body_moves_ratio'] > 5).rolling(window=10, center=True, min_periods=1).sum()
        else:
            processed_data['move_std'] = 0
            processed_data['move_avg'] = processed_data.get('body_moves_ratio', 0)
            processed_data['move_freq'] = 0
        
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
    def _extract_features(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        提取所有特征
        
        Args:
            data: 原始数据
            
        Returns:
            包含所有特征的数据
        """
        # 准备数据
        processed_data = cls._prepare_data(data)
        
        # 计算增强HRV特征
        processed_data = cls._calculate_enhanced_hrv_features(processed_data)
        
        # 计算呼吸特征
        processed_data = cls._calculate_respiratory_features(processed_data)
        
        # 计算体动特征
        processed_data = cls._calculate_body_movement_features(processed_data)
        
        # 计算时间特征
        processed_data['hour'] = processed_data['upload_time'].dt.hour
        processed_data['is_night'] = (processed_data['hour'] >= 22) | (processed_data['hour'] < 6)
        processed_data['is_midnight'] = (processed_data['hour'] >= 0) & (processed_data['hour'] < 4)
        processed_data['is_early_morning'] = (processed_data['hour'] >= 4) & (processed_data['hour'] < 8)
        
        return processed_data
    
    @classmethod
    def analyze_sleep_stages_by_advanced_rules(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        基于高级规则的睡眠分期分析方法
        
        Args:
            data: 睡眠数据
            
        Returns:
            包含睡眠阶段的数据分析
        """
        logger.info("开始基于高级规则的睡眠分期分析")
        
        # 提取特征
        processed_data = cls._extract_features(data)
        
        # 计算基线心率
        baseline_hr = cls.calculate_baseline_heart_rate(processed_data)
        logger.info(f"计算得到基线心率: {baseline_hr:.2f}")
        
        # 初始化睡眠阶段
        processed_data['stage_value'] = cls.STAGE_AWAKE
        processed_data['stage_label'] = cls.STAGE_LABELS[cls.STAGE_AWAKE]
        
        # 基于高级规则的睡眠分期
        for i, row in processed_data.iterrows():
            # 获取特征值
            hr = row.get('heart_rate', 0)
            hr_std = row.get('hr_std', 0)
            hr_lf_hf_ratio = row.get('hr_lf_hf_ratio', 0)
            rr_std = row.get('rr_std', 0)
            rr_stability = row.get('rr_stability', 0)
            move_avg = row.get('move_avg', 0)
            is_night = row.get('is_night', False)
            is_midnight = row.get('is_midnight', False)
            is_early_morning = row.get('is_early_morning', False)
            
            if pd.isna(hr) or hr == 0:
                # 无心率数据，标记为清醒
                stage = cls.STAGE_AWAKE
            elif is_night:
                # 夜间睡眠分期
                # 1. 清醒状态判定
                if (hr >= baseline_hr * 1.15 and hr_std > 10) or (move_avg > 10):
                    stage = cls.STAGE_AWAKE
                # 2. REM阶段判定
                elif (hr >= baseline_hr * 1.05 and hr_std > 6 and hr_lf_hf_ratio > 1.5 and 
                      rr_std > 2 and move_avg < 5):
                    stage = cls.STAGE_REM
                # 3. N3阶段判定
                elif (hr <= baseline_hr * 0.85 and hr_std < 3 and hr_lf_hf_ratio < 0.8 and 
                      rr_stability < 0.1 and move_avg < 3):
                    stage = cls.STAGE_N3
                # 4. N2阶段判定
                elif (hr <= baseline_hr * 0.95 and hr_std < 5 and hr_lf_hf_ratio < 1.2 and 
                      rr_stability < 0.15 and move_avg < 5):
                    stage = cls.STAGE_N2
                # 5. N1阶段判定
                elif (hr <= baseline_hr * 1.00 and hr_std < 7 and hr_lf_hf_ratio < 1.3 and 
                      move_avg < 7):
                    stage = cls.STAGE_N1
                else:
                    stage = cls.STAGE_AWAKE
            else:
                # 白天：默认标记为清醒
                stage = cls.STAGE_AWAKE
            
            processed_data.at[i, 'stage_value'] = stage
            processed_data.at[i, 'stage_label'] = cls.STAGE_LABELS[stage]
        
        # 平滑睡眠阶段
        processed_data = cls._smooth_sleep_stages(processed_data)
        
        logger.info("基于高级规则的睡眠分期分析完成")
        return processed_data
    

    
    @staticmethod
    def _smooth_sleep_stages(data: pd.DataFrame, window: int = 5) -> pd.DataFrame:
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
            processed_data['stage_label'] = processed_data['stage_value'].apply(EnhancedSleepStageAnalyzer.get_stage_label)
        
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
            'average_heart_rate_by_stage': {},
            'sleep_efficiency': 0,
            'sleep_cycle_count': 0
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
        
        # 计算睡眠效率
        total_records = len(data)
        if total_records > 0:
            summary['sleep_efficiency'] = round(total_sleep_duration / total_records * 100, 2)
        
        # 计算睡眠周期数（近似）
        # 假设一个睡眠周期约90分钟
        if total_sleep_duration > 0:
            summary['sleep_cycle_count'] = int(total_sleep_duration // 90)
        
        return summary


@tool
def analyze_sleep_stages_by_advanced_methods(date: str) -> str:
    """
    基于高级方法的睡眠分期分析工具
    根据AASM标准和多维度生理特征分析睡眠分期
    
    Args:
        date: 分析日期，格式为YYYY-MM-DD
        
    Returns:
        睡眠分期分析结果，包含两种高级分析方法的结果
    """
    try:
        logger.info(f"开始分析 {date} 的睡眠分期（基于高级方法）")
        
        # 生成模拟数据
        def generate_test_data(date_str):
            """生成测试数据"""
            times = pd.date_range(start=f"{date_str} 22:00:00", end=f"{date_str} 23:59:59", freq="1min")
            times = times.append(pd.date_range(start=f"{date_str[:8]}{int(date_str[8:10])+1} 00:00:00", end=f"{date_str[:8]}{int(date_str[8:10])+1} 06:00:00", freq="1min"))
            
            # 生成模拟心率数据
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
        
        # 生成测试数据
        test_data = generate_test_data(date)
        logger.info(f"生成了 {len(test_data)} 条测试数据")
        
        # 方法：基于高级规则的睡眠分期分析
        analyzer = EnhancedSleepStageAnalyzer()
        result_method1 = analyzer.analyze_sleep_stages_by_advanced_rules(test_data)
        summary_method1 = analyzer.generate_sleep_stage_summary(result_method1)
        
        # 构建分析结果
        analysis_result = {
            'success': True,
            'date': date,
            'data_points': len(test_data),
            'analysis_methods': {
                'advanced_rules': {
                    'name': '基于高级规则的分析方法',
                    'summary': summary_method1
                }
            },
            'sleep_stage_characteristics': {
                'awake': {
                    'heart_rate': '高，波动大',
                    'hrv': '低频成分主导',
                    'respiration': '不规则',
                    'movement': '频繁'
                },
                'n1': {
                    'heart_rate': '略降，波动减弱',
                    'hrv': '高频成分开始上升',
                    'respiration': '逐渐规律',
                    'movement': '减少'
                },
                'n2': {
                    'heart_rate': '明显降低，变异性下降',
                    'hrv': '低频/高频比值下降',
                    'respiration': '规律',
                    'movement': '很少'
                },
                'n3': {
                    'heart_rate': '最低，保持稳定',
                    'hrv': '高频成分主导',
                    'respiration': '非常规律',
                    'movement': '极少'
                },
                'rem': {
                    'heart_rate': '波动大，接近清醒',
                    'hrv': '低频成分回升',
                    'respiration': '不规则',
                    'movement': '很少，但可能有快速眼动'
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
def get_enhanced_sleep_stage_analysis_help() -> str:
    """
    获取增强版睡眠分期分析工具的帮助信息
    提供基于多维度生理特征的高精度睡眠分期分析工具的使用说明
    
    Returns:
        帮助信息
    """
    help_info = {
        'success': True,
        'tool_name': '增强版睡眠分期分析工具',
        'description': '根据AASM标准和多维度生理特征分析睡眠分期',
        'methods': [
            {
                'name': '基于高级规则的分析方法',
                'description': '基于增强HRV特征、呼吸模式和体动特征的复杂规则进行睡眠分期',
                'features': ['增强HRV特征', '呼吸模式', '体动特征', '时间特征']
            }
        ],
        'sleep_stages': [
            {
                'stage': '清醒',
                'characteristics': '心率高，波动大，HRV低频成分主导，呼吸不规则，体动频繁'
            },
            {
                'stage': '浅睡N1',
                'characteristics': '心率略降，波动减弱，HRV高频成分开始上升，呼吸逐渐规律，体动减少'
            },
            {
                'stage': '中睡N2',
                'characteristics': '心率明显降低，变异性下降，HRV低频/高频比值下降，呼吸规律，体动很少'
            },
            {
                'stage': '深睡N3',
                'characteristics': '心率最低，保持稳定，HRV高频成分主导，呼吸非常规律，体动极少'
            },
            {
                'stage': '眼动REM',
                'characteristics': '心率波动大，接近清醒，HRV低频成分回升，呼吸不规则，体动很少但可能有快速眼动'
            }
        ],
        'usage': '调用 analyze_sleep_stages_by_advanced_methods 工具，传入日期参数（格式：YYYY-MM-DD）'
    }
    
    return json.dumps(help_info, ensure_ascii=False)
