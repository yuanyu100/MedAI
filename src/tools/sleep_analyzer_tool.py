"""
睡眠分析工具模块
提供详细的睡眠数据处理和分析功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import json
import logging
from typing import Dict, List, Tuple, Optional
import warnings
from langchain_community.tools import tool



# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class SleepStageAnalyzer:
    """睡眠阶段分析器"""
    
    # 睡眠阶段常量
    STAGE_DEEP = 1
    STAGE_LIGHT = 2
    STAGE_REM = 3
    STAGE_AWAKE = 4
    
    # 睡眠阶段标签映射
    STAGE_LABELS = {
        STAGE_DEEP: "深睡",
        STAGE_LIGHT: "浅睡", 
        STAGE_REM: "眼动",
        STAGE_AWAKE: "清醒"
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
    def _prepare_sleep_data(sleep_data: pd.DataFrame) -> pd.DataFrame:
        """
        准备睡眠数据，包括排序和类型转换
        
        Args:
            sleep_data: 原始睡眠数据
            
        Returns:
            处理后的睡眠数据
        """
        # 复制数据避免修改原数据
        processed_data = sleep_data.copy()
        
        # 确保数据按时间排序
        processed_data = processed_data.sort_values('upload_time').reset_index(drop=True)
        
        # 将生理指标字段转换为数值类型
        physio_fields = ['breath_amp_avg', 'heart_amp_avg', 'breath_freq_std', 
                        'heart_freq_std', 'breath_amp_diff', 'heart_amp_diff']
        for field in physio_fields:
            if field in processed_data.columns:
                processed_data[field] = pd.to_numeric(processed_data[field], errors='coerce')
        
        return processed_data
    
    @staticmethod
    def _calculate_time_based_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        计算基于时间的特征
        
        Args:
            data: 睡眠数据
            
        Returns:
            添加了时间特征的数据
        """
        # 时间段判断
        data['hour'] = data['upload_time'].dt.hour
        data['is_morning'] = (data['hour'] >= 6) & (data['hour'] < 12)
        data['is_afternoon'] = (data['hour'] >= 12) & (data['hour'] < 18)
        data['is_evening'] = (data['hour'] >= 18) & (data['hour'] < 22)
        data['is_daytime'] = (data['hour'] >= 6) & (data['hour'] < 22)
        
        return data
    
    @staticmethod
    def _calculate_physiological_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        计算生理特征指标
        
        Args:
            data: 睡眠数据
            
        Returns:
            添加了生理特征的数据
        """
        # 幅度指标
        if 'breath_amp_avg' in data.columns and 'heart_amp_avg' in data.columns:
            breath_amp_quantile = data['breath_amp_avg'].dropna()
            heart_amp_quantile = data['heart_amp_avg'].dropna()
            breath_thresh = breath_amp_quantile.quantile(0.3) if not breath_amp_quantile.empty else 0
            heart_thresh = heart_amp_quantile.quantile(0.3) if not heart_amp_quantile.empty else 0
            
            data['amp_indicator'] = (
                (data['breath_amp_avg'].fillna(0) > breath_thresh) |
                (data['heart_amp_avg'].fillna(0) > heart_thresh)
            )
        else:
            data['amp_indicator'] = False
        
        # 频率稳定性
        if 'breath_freq_std' in data.columns and 'heart_freq_std' in data.columns:
            breath_freq_quantile = data['breath_freq_std'].dropna()
            heart_freq_quantile = data['heart_freq_std'].dropna()
            breath_freq_thresh = breath_freq_quantile.quantile(0.7) if not breath_freq_quantile.empty else float('inf')
            heart_freq_thresh = heart_freq_quantile.quantile(0.7) if not heart_freq_quantile.empty else float('inf')
            
            data['freq_stability'] = (
                (data['breath_freq_std'].fillna(float('inf')) < breath_freq_thresh) &
                (data['heart_freq_std'].fillna(float('inf')) < heart_freq_thresh)
            )
        else:
            data['freq_stability'] = True
        
        # 幅度差值
        if 'breath_amp_diff' in data.columns and 'heart_amp_diff' in data.columns:
            breath_diff_quantile = data['breath_amp_diff'].dropna()
            heart_diff_quantile = data['heart_amp_diff'].dropna()
            breath_diff_thresh = breath_diff_quantile.quantile(0.3) if not breath_diff_quantile.empty else 0
            heart_diff_thresh = heart_diff_quantile.quantile(0.3) if not heart_diff_quantile.empty else 0
            
            data['amp_diff_indicator'] = (
                (data['breath_amp_diff'].fillna(0) > breath_diff_thresh) |
                (data['heart_amp_diff'].fillna(0) > heart_diff_thresh)
            )
        else:
            data['amp_diff_indicator'] = False
        
        # 呼吸稳定性
        window_size = 5
        rolling_stats = data['respiratory_rate'].rolling(window=window_size, center=True, min_periods=1)
        rr_std = rolling_stats.std()
        rr_mean = rolling_stats.mean()
        data['respiratory_stability'] = (
            (rr_std / rr_mean * 100)
            .where(rr_mean != 0, 0)
        ).fillna(0)
        
        return data
    
    @staticmethod
    def _calculate_sleep_stages(data: pd.DataFrame, baseline_heart_rate: float) -> pd.DataFrame:
        """
        计算睡眠阶段
        
        Args:
            data: 包含特征的睡眠数据
            baseline_heart_rate: 基线心率
            
        Returns:
            包含睡眠阶段的数据分析
        """
        print(f"\n=== 调试信息: 开始计算睡眠阶段 ===")
        print(f"基线心率: {baseline_heart_rate}")
        print(f"数据总行数: {len(data)}")
        
        # 对生理指标进行移动平均平滑
        print("\n=== 调试信息: 开始数据平滑 ===")
        data['heart_rate_smoothed'] = data['heart_rate'].rolling(window=3, center=True, min_periods=1).mean()
        data['respiratory_rate_smoothed'] = data['respiratory_rate'].rolling(window=3, center=True, min_periods=1).mean()
        data['body_moves_ratio_smoothed'] = data['body_moves_ratio'].rolling(window=3, center=True, min_periods=1).mean()
        print(f"平滑前心率范围: {data['heart_rate'].min():.2f} - {data['heart_rate'].max():.2f}")
        print(f"平滑后心率范围: {data['heart_rate_smoothed'].min():.2f} - {data['heart_rate_smoothed'].max():.2f}")
        
        # 清醒判定
        print("\n=== 调试信息: 开始清醒判定 ===")
        data['is_awake'] = (
            (data['is_morning'] & (
                (data['heart_rate_smoothed'] >= baseline_heart_rate * 0.90) |  # 从0.85调整到0.90
                (data['body_moves_ratio_smoothed'] > 2) |  # 从1.5调整到2
                (data['amp_indicator'])
            )) |
            (data['is_afternoon'] & (
                (data['heart_rate_smoothed'] >= baseline_heart_rate * 0.92) |  # 从0.88调整到0.92
                (data['body_moves_ratio_smoothed'] > 3) |  # 从2调整到3
                (data['amp_indicator'])
            )) |
            (data['is_evening'] & (
                (data['heart_rate_smoothed'] >= baseline_heart_rate * 0.95) |  # 从0.90调整到0.95
                (data['body_moves_ratio_smoothed'] > 3.5) |  # 从2.5调整到3.5
                (data['amp_indicator'])
            )) |
            ((~data['is_daytime']) & (
                (data['heart_rate_smoothed'] >= baseline_heart_rate * 1.15) |  # 从1.10调整到1.15，更宽松
                ((data['heart_rate_smoothed'] >= baseline_heart_rate * 1.00) &  # 从0.95调整到1.00
                 (data['body_moves_ratio_smoothed'] > 8)) |  # 从5调整到8，更宽松
                (data['amp_indicator'] & (data['body_moves_ratio_smoothed'] > 10))  # 增加体动阈值条件
            ))
        )
        
        print(f"清醒状态占比: {data['is_awake'].sum()/len(data)*100:.2f}%")
        
        # 深睡判定
        print("\n=== 调试信息: 开始深睡判定 ===")
        overall_resp_mean = data['respiratory_rate_smoothed'].mean()
        data['is_deep'] = (
            (data['heart_rate_smoothed'] <= baseline_heart_rate * 0.90) &  # 使用平滑后的数据
            (data['respiratory_rate_smoothed'] <= overall_resp_mean * 0.95) &  # 使用平滑后的数据
            (data['respiratory_stability'] < 8) &  # 从5调整到8，更宽松
            (data['body_moves_ratio_smoothed'] <= 5) &  # 使用平滑后的数据，从3调整到5，更宽松
            (data['freq_stability']) &
            (~data['is_awake'])
        )
        print(f"深睡状态占比: {data['is_deep'].sum()/len(data)*100:.2f}%")
        
        # 计算深睡基线
        print("\n=== 调试信息: 计算深睡基线 ===")
        deep_sleep_data = data[data['is_deep']]
        actual_deep_hr = deep_sleep_data['heart_rate_smoothed'].mean() if not deep_sleep_data.empty else baseline_heart_rate * 0.75
        print(f"深睡基线心率: {actual_deep_hr:.2f}")
        print(f"深睡数据点数: {len(deep_sleep_data)}")
        
        # REM判定
        print("\n=== 调试信息: 开始REM判定 ===")
        REM_HR_MULTIPLIER_LOW = 1.10
        REM_HR_MULTIPLIER_HIGH = 1.40
        REM_RESP_STABILITY_LOW = 10
        REM_BODY_MOVES_MAX = 3
        
        data['is_rem'] = (
            (~data['is_awake']) &
            (~data['is_deep']) &
            (data['heart_rate_smoothed'] >= actual_deep_hr * REM_HR_MULTIPLIER_LOW) &  # 使用平滑后的数据
            (data['heart_rate_smoothed'] <= actual_deep_hr * REM_HR_MULTIPLIER_HIGH) &  # 使用平滑后的数据
            (data['respiratory_stability'] > REM_RESP_STABILITY_LOW) &
            (data['body_moves_ratio_smoothed'] <= REM_BODY_MOVES_MAX) &  # 使用平滑后的数据
            (data['respiratory_rate_smoothed'] >= 8) &  # 使用平滑后的数据
            (data['respiratory_rate_smoothed'] <= 20) &  # 使用平滑后的数据
            (data['amp_diff_indicator']) &
            (~data['freq_stability'])
        )
        print(f"REM状态占比: {data['is_rem'].sum()/len(data)*100:.2f}%")
        
        # 浅睡判定
        print("\n=== 调试信息: 开始浅睡判定 ===")
        data['is_light'] = (
            (~data['is_awake']) &
            (~data['is_deep']) &
            (~data['is_rem']) &
            (data['body_moves_ratio_smoothed'] <= 10) &  # 使用平滑后的数据
            (data['heart_rate_smoothed'] <= baseline_heart_rate * 0.95)  # 使用平滑后的数据
        )
        print(f"浅睡状态占比: {data['is_light'].sum()/len(data)*100:.2f}%")
        
        # 验证各状态占比总和
        total_ratio = (data['is_awake'].sum() + data['is_deep'].sum() + data['is_rem'].sum() + data['is_light'].sum()) / len(data) * 100
        print(f"\n=== 调试信息: 状态占比汇总 ===")
        print(f"总占比: {total_ratio:.2f}%")
        print(f"=== 调试信息: 睡眠阶段计算完成 ===")
        
        return data
    
    @staticmethod
    def _validate_continuous_stages(data: pd.DataFrame) -> pd.DataFrame:
        """
        验证连续睡眠阶段
        
        Args:
            data: 包含初步睡眠阶段的数据
            
        Returns:
            包含连续验证的睡眠数据
        """
        print("\n=== 调试信息: 开始连续验证 ===")
        # 增加连续验证窗口
        CONTINUOUS_MINUTES = 3  # 从2调整到3
        print(f"连续验证窗口大小: {CONTINUOUS_MINUTES}分钟")
        
        for stage in ['is_awake', 'is_deep', 'is_rem', 'is_light']:
            data[f'{stage}_continuous'] = (
                data[stage].rolling(window=CONTINUOUS_MINUTES, min_periods=1).sum() == CONTINUOUS_MINUTES
            )
        
        # 统计连续验证后的各状态占比
        print("\n=== 调试信息: 连续验证后的状态占比 ===")
        print(f"连续清醒状态占比: {data['is_awake_continuous'].sum()/len(data)*100:.2f}%")
        print(f"连续深睡状态占比: {data['is_deep_continuous'].sum()/len(data)*100:.2f}%")
        print(f"连续REM状态占比: {data['is_rem_continuous'].sum()/len(data)*100:.2f}%")
        print(f"连续浅睡状态占比: {data['is_light_continuous'].sum()/len(data)*100:.2f}%")
        
        # 初始化阶段值和标签
        data['stage_value'] = SleepStageAnalyzer.STAGE_LIGHT  # 默认为浅睡
        data['stage_label'] = SleepStageAnalyzer.STAGE_LABELS[SleepStageAnalyzer.STAGE_LIGHT]
        
        print("=== 调试信息: 连续验证完成 ===")
        return data
    
    @classmethod
    def calculate_optimized_sleep_stages(cls, sleep_data: pd.DataFrame, baseline_heart_rate: float) -> pd.DataFrame:
        """
        优化的睡眠阶段判定函数
        
        Args:
            sleep_data: 睡眠数据DataFrame
            baseline_heart_rate: 基线心率
            
        Returns:
            包含睡眠阶段判定的DataFrame
        """
        print("\n=== 调试信息: 开始优化睡眠阶段判定 ===")
        print(f"输入数据行数: {len(sleep_data)}")
        print(f"输入基线心率: {baseline_heart_rate}")
        
        try:
            # 1. 准备数据
            print("\n1. 准备数据")
            processed_data = cls._prepare_sleep_data(sleep_data)
            print(f"准备后数据行数: {len(processed_data)}")
            
            # 2. 计算时间特征
            print("\n2. 计算时间特征")
            processed_data = cls._calculate_time_based_features(processed_data)
            
            # 3. 计算生理特征
            print("\n3. 计算生理特征")
            processed_data = cls._calculate_physiological_features(processed_data)
            
            # 4. 计算睡眠阶段
            print("\n4. 计算睡眠阶段")
            processed_data = cls._calculate_sleep_stages(processed_data, baseline_heart_rate)
            
            # 5. 验证连续阶段
            print("\n5. 验证连续阶段")
            processed_data = cls._validate_continuous_stages(processed_data)
            
            print("\n=== 调试信息: 优化睡眠阶段判定完成 ===")
            return processed_data
            
        except Exception as e:
            logger.error(f"计算睡眠阶段时出错: {str(e)}")
            print(f"\n=== 调试信息: 计算睡眠阶段时出错 ===")
            print(f"错误信息: {str(e)}")
            # 返回原始数据，添加错误标记
            error_data = sleep_data.copy()
            error_data['stage_value'] = 0
            error_data['stage_label'] = "错误"
            return error_data
    
    @staticmethod
    def smooth_sleep_stages(stages_sequence: List[Dict], min_duration_threshold: int = 5) -> List[Dict]:  # 从3调整到5
        """
        平滑睡眠阶段序列，减少碎片化
        
        Args:
            stages_sequence: 睡眠阶段序列
            min_duration_threshold: 最小持续时间阈值
            
        Returns:
            平滑后的睡眠阶段序列
        """
        print("\n=== 调试信息: 开始平滑睡眠阶段 ===")
        print(f"输入阶段序列长度: {len(stages_sequence)}")
        print(f"最小持续时间阈值: {min_duration_threshold}分钟")
        
        if not stages_sequence:
            print("输入阶段序列为空，直接返回")
            return stages_sequence
        
        try:
            # 合并相邻的相同阶段
            print("\n1. 合并相邻的相同阶段")
            merged_same_stages = []
            i = 0
            while i < len(stages_sequence):
                current = stages_sequence[i].copy()
                j = i + 1
                
                while j < len(stages_sequence) and stages_sequence[j]['stage_value'] == current['stage_value']:
                    # 确保时间间隔是整数类型，避免浮点数精度问题
                    try:
                        current_interval = int(round(float(current['time_interval'])))
                        next_interval = int(round(float(stages_sequence[j]['time_interval'])))
                        current['time_interval'] = current_interval + next_interval
                    except:
                        current['time_interval'] += stages_sequence[j]['time_interval']
                    j += 1
                
                merged_same_stages.append(current)
                i = j
            
            print(f"合并后阶段序列长度: {len(merged_same_stages)}")
            
            # 移除或合并短持续时间的阶段
            if not merged_same_stages:
                print("合并后阶段序列为空，直接返回")
                return merged_same_stages
            
            print("\n2. 移除或合并短持续时间的阶段")
            result = [merged_same_stages[0]]
            
            i = 1
            while i < len(merged_same_stages):
                current = merged_same_stages[i]
                
                if current['time_interval'] < min_duration_threshold:
                    # 当前阶段太短，合并到前一个阶段
                    try:
                        # 确保时间间隔是整数类型，避免浮点数精度问题
                        current_interval = int(round(float(current['time_interval'])))
                        prev_interval = int(round(float(result[-1]['time_interval'])))
                        result[-1]['time_interval'] = prev_interval + current_interval
                        print(f"合并短阶段: {current['stage_label']} ({current_interval}分钟) 到前一个阶段")
                    except:
                        result[-1]['time_interval'] += current['time_interval']
                        print(f"合并短阶段: {current['stage_label']} ({current['time_interval']:.1f}分钟) 到前一个阶段")
                else:
                    # 当前阶段足够长，添加到结果中
                    print(f"添加阶段: {current['stage_label']} ({current['time_interval']:.1f}分钟)")
                    result.append(current)
                
                i += 1
            
            print(f"\n平滑后阶段序列长度: {len(result)}")
            print("=== 调试信息: 平滑睡眠阶段完成 ===")
            
            return result
            
        except Exception as e:
            logger.error(f"平滑睡眠阶段时出错: {str(e)}")
            print(f"平滑睡眠阶段时出错: {str(e)}")
            return stages_sequence
    
    @staticmethod
    def _calculate_trusleep_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        计算TruSleep算法所需的特征
        
        Args:
            data: 睡眠数据
            
        Returns:
            包含TruSleep特征的数据
        """
        print("\n=== 调试信息: 开始计算TruSleep特征 ===")
        
        # 复制数据避免修改原数据
        processed_data = data.copy()
        
        # 1. 基础生理指标平滑
        print("\n1. 基础生理指标平滑")
        processed_data['heart_rate_smoothed'] = processed_data['heart_rate'].rolling(window=5, center=True, min_periods=1).mean()
        processed_data['respiratory_rate_smoothed'] = processed_data['respiratory_rate'].rolling(window=5, center=True, min_periods=1).mean()
        processed_data['body_moves_ratio_smoothed'] = processed_data['body_moves_ratio'].rolling(window=5, center=True, min_periods=1).mean()
        
        # 2. 呼吸和心跳幅度处理
        print("\n2. 呼吸和心跳幅度处理")
        # 转换呼吸幅度均值为数值
        if 'breath_amp_average' in processed_data.columns:
            processed_data['breath_amp_average'] = pd.to_numeric(processed_data['breath_amp_average'], errors='coerce')
            processed_data['breath_amp_smoothed'] = processed_data['breath_amp_average'].rolling(window=5, center=True, min_periods=1).mean()
        else:
            processed_data['breath_amp_smoothed'] = 0
        
        # 转换心跳幅度均值为数值
        if 'heart_amp_average' in processed_data.columns:
            processed_data['heart_amp_average'] = pd.to_numeric(processed_data['heart_amp_average'], errors='coerce')
            processed_data['heart_amp_smoothed'] = processed_data['heart_amp_average'].rolling(window=5, center=True, min_periods=1).mean()
        else:
            processed_data['heart_amp_smoothed'] = 0
        
        # 3. 频率标准差处理
        print("\n3. 频率标准差处理")
        # 转换呼吸频率标准差为数值
        if 'breath_freq_std' in processed_data.columns:
            processed_data['breath_freq_std'] = pd.to_numeric(processed_data['breath_freq_std'], errors='coerce')
            processed_data['breath_freq_std_smoothed'] = processed_data['breath_freq_std'].rolling(window=5, center=True, min_periods=1).mean()
        else:
            processed_data['breath_freq_std_smoothed'] = 0
        
        # 转换心跳频率标准差为数值
        if 'heart_freq_std' in processed_data.columns:
            processed_data['heart_freq_std'] = pd.to_numeric(processed_data['heart_freq_std'], errors='coerce')
            processed_data['heart_freq_std_smoothed'] = processed_data['heart_freq_std'].rolling(window=5, center=True, min_periods=1).mean()
        else:
            processed_data['heart_freq_std_smoothed'] = 0
        
        # 4. 幅度差值处理
        print("\n4. 幅度差值处理")
        # 转换呼吸幅度差值为数值
        if 'breath_amp_diff' in processed_data.columns:
            processed_data['breath_amp_diff'] = pd.to_numeric(processed_data['breath_amp_diff'], errors='coerce')
            processed_data['breath_amp_diff_smoothed'] = processed_data['breath_amp_diff'].rolling(window=5, center=True, min_periods=1).mean()
        else:
            processed_data['breath_amp_diff_smoothed'] = 0
        
        # 转换心跳幅度差值为数值
        if 'heart_amp_diff' in processed_data.columns:
            processed_data['heart_amp_diff'] = pd.to_numeric(processed_data['heart_amp_diff'], errors='coerce')
            processed_data['heart_amp_diff_smoothed'] = processed_data['heart_amp_diff'].rolling(window=5, center=True, min_periods=1).mean()
        else:
            processed_data['heart_amp_diff_smoothed'] = 0
        
        # 5. 体动处理
        print("\n5. 体动处理")
        if 'has_move' in processed_data.columns:
            processed_data['has_move'] = pd.to_numeric(processed_data['has_move'], errors='coerce')
            # 计算体动频率
            processed_data['move_freq'] = processed_data['has_move'].rolling(window=10, center=True, min_periods=1).sum()
        else:
            processed_data['move_freq'] = processed_data['body_moves_ratio_smoothed'] / 10
        
        # 6. 心率变异性指标
        print("\n6. 心率变异性指标")
        # 计算心率滚动标准差作为心率变异性的近似
        processed_data['hr_variability'] = processed_data['heart_rate_smoothed'].rolling(window=10, center=True, min_periods=3).std()
        
        # 7. 呼吸模式指标
        print("\n7. 呼吸模式指标")
        # 计算呼吸频率滚动标准差
        processed_data['resp_variability'] = processed_data['respiratory_rate_smoothed'].rolling(window=10, center=True, min_periods=3).std()
        
        print("=== 调试信息: TruSleep特征计算完成 ===")
        return processed_data
    
    @staticmethod
    def calculate_trusleep_stages(data: pd.DataFrame, baseline_heart_rate: float) -> pd.DataFrame:
        """
        实现TruSleep算法进行睡眠分期
        
        Args:
            data: 睡眠数据
            baseline_heart_rate: 基线心率
            
        Returns:
            包含睡眠阶段判定的DataFrame
        """
        print("\n=== 调试信息: 开始TruSleep睡眠分期 ===")
        print(f"基线心率: {baseline_heart_rate}")
        print(f"数据总行数: {len(data)}")
        
        try:
            # 1. 计算TruSleep特征
            print("\n1. 计算TruSleep特征")
            processed_data = SleepStageAnalyzer._calculate_trusleep_features(data)
            
            # 2. 时间特征
            print("\n2. 计算时间特征")
            processed_data['hour'] = processed_data['upload_time'].dt.hour
            processed_data['is_morning'] = (processed_data['hour'] >= 6) & (processed_data['hour'] < 12)
            processed_data['is_afternoon'] = (processed_data['hour'] >= 12) & (processed_data['hour'] < 18)
            processed_data['is_evening'] = (processed_data['hour'] >= 18) & (processed_data['hour'] < 22)
            processed_data['is_night'] = (processed_data['hour'] >= 22) | (processed_data['hour'] < 6)
            
            # 3. 睡眠阶段判定
            print("\n3. 开始睡眠阶段判定")
            
            # 3.1 清醒期判定
            print("\n3.1 清醒期判定")
            processed_data['is_awake'] = (
                # 夜间清醒判定
                (processed_data['is_night'] & (
                    (processed_data['heart_rate_smoothed'] >= baseline_heart_rate * 1.15) |
                    (processed_data['move_freq'] > 3) |
                    (processed_data['body_moves_ratio_smoothed'] > 15) |
                    (processed_data['hr_variability'] > 15)
                )) |
                # 白天清醒判定
                ((~processed_data['is_night']) & (
                    (processed_data['heart_rate_smoothed'] >= baseline_heart_rate * 0.95) |
                    (processed_data['move_freq'] > 1) |
                    (processed_data['body_moves_ratio_smoothed'] > 10)
                ))
            )
            print(f"清醒状态占比: {processed_data['is_awake'].sum()/len(processed_data)*100:.2f}%")
            
            # 3.2 深睡期判定
            print("\n3.2 深睡期判定")
            # 计算呼吸和心率的稳定性阈值
            breath_stability_thresh = processed_data['resp_variability'].quantile(0.3) if len(processed_data) > 0 else 5
            hr_stability_thresh = processed_data['hr_variability'].quantile(0.3) if len(processed_data) > 0 else 5
            
            processed_data['is_deep'] = (
                (~processed_data['is_awake']) &
                (processed_data['heart_rate_smoothed'] <= baseline_heart_rate * 0.90) &
                (processed_data['heart_rate_smoothed'] >= baseline_heart_rate * 0.60) &
                (processed_data['respiratory_rate_smoothed'] <= 18) &
                (processed_data['respiratory_rate_smoothed'] >= 10) &
                (processed_data['body_moves_ratio_smoothed'] <= 5) &
                (processed_data['move_freq'] <= 0.5) &
                (processed_data['hr_variability'] < hr_stability_thresh) &
                (processed_data['resp_variability'] < breath_stability_thresh)
            )
            print(f"深睡状态占比: {processed_data['is_deep'].sum()/len(processed_data)*100:.2f}%")
            
            # 3.3 计算深睡基线
            print("\n3.3 计算深睡基线")
            deep_sleep_data = processed_data[processed_data['is_deep']]
            actual_deep_hr = deep_sleep_data['heart_rate_smoothed'].mean() if not deep_sleep_data.empty else baseline_heart_rate * 0.75
            print(f"深睡基线心率: {actual_deep_hr:.2f}")
            print(f"基线心率: {baseline_heart_rate:.2f}")
            
            # 3.4 REM期判定
            print("\n3.4 REM期判定")
            # 调整REM期判定条件，使其更加宽松
            processed_data['is_rem'] = (
                (~processed_data['is_awake']) &
                (~processed_data['is_deep']) &
                (processed_data['heart_rate_smoothed'] >= actual_deep_hr * 1.05) &  # 从1.10调整为1.05
                (processed_data['heart_rate_smoothed'] <= actual_deep_hr * 1.45) &  # 从1.40调整为1.45
                (processed_data['respiratory_rate_smoothed'] >= 11) &  # 从12调整为11
                (processed_data['respiratory_rate_smoothed'] <= 21) &  # 从20调整为21
                (processed_data['body_moves_ratio_smoothed'] <= 10) &  # 从8调整为10
                (processed_data['move_freq'] <= 1.5) &  # 从1调整为1.5
                (processed_data['hr_variability'] > hr_stability_thresh * 0.8) &  # 从1.0调整为0.8
                (processed_data['resp_variability'] > breath_stability_thresh * 0.8)  # 从1.0调整为0.8
            )
            print(f"REM状态占比: {processed_data['is_rem'].sum()/len(processed_data)*100:.2f}%")
            
            # 3.5 浅睡期判定
            print("\n3.5 浅睡期判定")
            processed_data['is_light'] = (
                (~processed_data['is_awake']) &
                (~processed_data['is_deep']) &
                (~processed_data['is_rem'])
            )
            print(f"浅睡状态占比: {processed_data['is_light'].sum()/len(processed_data)*100:.2f}%")
            
            # 4. 连续验证
            print("\n4. 连续验证")
            CONTINUOUS_MINUTES = 4  # TruSleep使用更长的连续验证窗口
            for stage in ['is_awake', 'is_deep', 'is_rem', 'is_light']:
                processed_data[f'{stage}_continuous'] = (
                    processed_data[stage].rolling(window=CONTINUOUS_MINUTES, min_periods=1).sum() == CONTINUOUS_MINUTES
                )
            
            # 5. 初始化阶段值和标签
            print("\n5. 初始化阶段值和标签")
            processed_data['stage_value'] = SleepStageAnalyzer.STAGE_LIGHT  # 默认为浅睡
            processed_data['stage_label'] = SleepStageAnalyzer.STAGE_LABELS[SleepStageAnalyzer.STAGE_LIGHT]
            
            # 按优先级分配阶段
            awake_mask = processed_data['is_awake_continuous']
            processed_data.loc[awake_mask, ['stage_value', 'stage_label']] = [4, "清醒"]
            
            deep_mask = (~awake_mask) & processed_data['is_deep_continuous']
            processed_data.loc[deep_mask, ['stage_value', 'stage_label']] = [1, "深睡"]
            
            rem_mask = (~awake_mask) & (~deep_mask) & processed_data['is_rem_continuous']
            processed_data.loc[rem_mask, ['stage_value', 'stage_label']] = [3, "眼动"]
            
            light_mask = (~awake_mask) & (~deep_mask) & (~rem_mask)
            processed_data.loc[light_mask, ['stage_value', 'stage_label']] = [2, "浅睡"]
            
            print("=== 调试信息: TruSleep睡眠分期完成 ===")
            return processed_data
            
        except Exception as e:
            logger.error(f"TruSleep睡眠分期时出错: {str(e)}")
            print(f"TruSleep睡眠分期时出错: {str(e)}")
            # 返回原始数据，添加错误标记
            error_data = data.copy()
            error_data['stage_value'] = 0
            error_data['stage_label'] = "错误"
            return error_data


class SleepTimeAnalyzer:
    """睡眠时间分析器"""
    
    # 合理起床时间范围
    REASONABLE_WAKEUP_START = time(6, 0)
    REASONABLE_WAKEUP_END = time(10, 0)
    
    # 最小有效数据段长度
    MIN_VALID_SEGMENT_COUNT = 5
    
    @staticmethod
    def _prepare_night_data(night_data: pd.DataFrame) -> pd.DataFrame:
        """
        准备夜间数据
        
        Args:
            night_data: 原始夜间数据
            
        Returns:
            处理后的夜间数据
        """
        # 按upload_time排序
        sorted_data = night_data.sort_values('upload_time').reset_index(drop=True)
        
        # 标记离床/有效数据状态
        # 考虑NaN值：NaN表示数据无效，视为离床
        sorted_data['is_off_bed'] = (sorted_data['heart_rate'] == 0) | (sorted_data['heart_rate'].isna())
        
        return sorted_data
    
    @staticmethod
    def _identify_valid_segments(data: pd.DataFrame) -> List[Dict]:
        """
        识别有效数据段
        
        Args:
            data: 处理后的夜间数据
            
        Returns:
            有效数据段列表
        """
        valid_segments = []
        current_segment_start = None
        valid_count = 0
        
        for i, row in data.iterrows():
            if not row['is_off_bed']:
                if current_segment_start is None:
                    current_segment_start = row['upload_time']
                valid_count += 1
            else:
                if current_segment_start is not None and valid_count >= SleepTimeAnalyzer.MIN_VALID_SEGMENT_COUNT:
                    valid_segments.append({
                        'start_time': current_segment_start,
                        'end_time': data.iloc[i-1]['upload_time'],
                        'count': valid_count,
                        'duration': data.iloc[i-1]['upload_time'] - current_segment_start
                    })
                current_segment_start = None
                valid_count = 0
        
        # 处理最后一个可能的有效段
        if current_segment_start is not None and valid_count >= SleepTimeAnalyzer.MIN_VALID_SEGMENT_COUNT:
            valid_segments.append({
                'start_time': current_segment_start,
                'end_time': data.iloc[-1]['upload_time'],
                'count': valid_count,
                'duration': data.iloc[-1]['upload_time'] - current_segment_start
            })
        
        return valid_segments
    
    @staticmethod
    def _validate_time_order(bedtime: datetime, wakeup_time: datetime, valid_segments: List[Dict], sorted_data: pd.DataFrame) -> Tuple[datetime, datetime]:
        """
        验证时间顺序
        
        Args:
            bedtime: 初步就寝时间
            wakeup_time: 初步起床时间
            valid_segments: 有效数据段列表
            sorted_data: 排序后的夜间数据
            
        Returns:
            修正后的(就寝时间, 起床时间)
        """
        if bedtime >= wakeup_time:
            logger.warning(f"时间顺序错误: 就寝时间({bedtime}) >= 起床时间({wakeup_time})")
            if valid_segments:
                # 使用最长有效段修正
                longest_segment = max(valid_segments, key=lambda x: x['duration'])
                bedtime = longest_segment['start_time']
                wakeup_time = longest_segment['end_time']
                logger.info(f"使用最长有效段修正: {bedtime} - {wakeup_time}")
            else:
                # 兜底方案
                bedtime = sorted_data['upload_time'].min()
                wakeup_time = sorted_data['upload_time'].max()
                logger.info(f"兜底方案: {bedtime} - {wakeup_time}")
        
        return bedtime, wakeup_time
    
    @staticmethod
    def _validate_wakeup_time(wakeup_time: datetime, target_date: datetime, valid_segments: List[Dict]) -> datetime:
        """
        验证起床时间合理性
        
        Args:
            wakeup_time: 初步起床时间
            target_date: 目标日期
            valid_segments: 有效数据段列表
            
        Returns:
            修正后的起床时间
        """
        # 检查起床时间是否在合理范围内
        if wakeup_time.date() == target_date.date() and \
           (wakeup_time.time() < SleepTimeAnalyzer.REASONABLE_WAKEUP_START or 
            wakeup_time.time() > SleepTimeAnalyzer.REASONABLE_WAKEUP_END):
            logger.info("初步起床时间不在合理范围，重新查找目标日期内的合理起床时间")
            
            # 从后往前查找合理的起床时间
            for seg in reversed(valid_segments):
                if seg['end_time'].date() == target_date.date() and \
                   SleepTimeAnalyzer.REASONABLE_WAKEUP_START <= seg['end_time'].time() <= SleepTimeAnalyzer.REASONABLE_WAKEUP_END:
                    wakeup_time = seg['end_time']
                    logger.info(f"修正起床时间: {wakeup_time}")
                    break
        
        return wakeup_time
    
    @classmethod
    def calculate_bedtime_wakeup_times(cls, night_data: pd.DataFrame, target_date: datetime, 
                                     prev_date: datetime) -> Tuple[datetime, datetime]:
        """
        计算就寝时间和起床时间
        
        Args:
            night_data: 夜间数据
            target_date: 目标日期
            prev_date: 前一天日期
            
        Returns:
            (就寝时间, 起床时间)
        """
        try:
            logger.info(f"开始计算就寝和起床时间，夜间数据量: {len(night_data)}")
            
            # 1. 准备夜间数据
            sorted_data = cls._prepare_night_data(night_data)
            
            logger.info(f"夜间数据时间范围: {sorted_data['upload_time'].min()} 到 {sorted_data['upload_time'].max()}")
            logger.info(f"离床数据点数量: {(sorted_data['is_off_bed'] == True).sum()}, "
                        f"在床数据点数量: {(sorted_data['is_off_bed'] == False).sum()}")
            
            # 2. 识别有效数据段
            valid_segments = cls._identify_valid_segments(sorted_data)
            
            logger.info(f"找到 {len(valid_segments)} 个有效数据段")
            for i, seg in enumerate(valid_segments):
                logger.debug(f"段 {i+1}: {seg['start_time']} - {seg['end_time']}, "
                            f"数据条数: {seg['count']}, 时长: {seg['duration']}")
            
            # 3. 初步确定上下床时间
            if valid_segments:
                bedtime = valid_segments[0]['start_time']
                wakeup_time = valid_segments[-1]['end_time']
                logger.info(f"初步就寝时间: {bedtime}")
                logger.info(f"初步起床时间: {wakeup_time}")
            else:
                logger.warning("未找到有效数据段，使用数据边界")
                bedtime = sorted_data['upload_time'].min()
                wakeup_time = sorted_data['upload_time'].max()
                return bedtime, wakeup_time
            
            # 4. 修正规则1：时间顺序校验
            bedtime, wakeup_time = cls._validate_time_order(bedtime, wakeup_time, valid_segments, sorted_data)
            
            # 5. 修正规则2：起床时间合理性校验
            wakeup_time = cls._validate_wakeup_time(wakeup_time, target_date, valid_segments)
            
            logger.info(f"最终确定就寝时间: {bedtime}, 起床时间: {wakeup_time}")
            return bedtime, wakeup_time
            
        except Exception as e:
            logger.error(f"计算就寝和起床时间时出错: {str(e)}")
            # 出错时返回数据边界
            if not night_data.empty:
                min_time = night_data['upload_time'].min()
                max_time = night_data['upload_time'].max()
                return min_time, max_time
            else:
                # 无数据时返回当前时间
                now = datetime.now()
                return now, now + timedelta(hours=8)


class SleepMetricsCalculator:
    """睡眠指标计算器"""
    
    # 常量定义
    IDEAL_SLEEP_HOURS_MIN = 7
    IDEAL_SLEEP_HOURS_MAX = 9
    MIN_HEART_RATE = 40
    MAX_HEART_RATE = 100
    MIN_RESPIRATORY_RATE = 8
    MAX_RESPIRATORY_RATE = 25
    STABLE_HR_STD_THRESHOLD = 8
    STABLE_ARRHYTHMIA_THRESHOLD = 30
    BED_EXIT_THRESHOLD_MINUTES = 1
    STABLE_RECORDS_NEEDED = 5
    DEFAULT_SLEEP_PREP_TIME = 10
    MIN_SLEEP_PREP_TIME = 5
    DEFAULT_SINGLE_RECORD_SLEEP_DURATION = 15
    
    # 睡眠评分权重
    SCORE_WEIGHTS = {
        'time_score': 30,        # 睡眠时长达标率
        'deep_sleep_score': 25,   # 深睡占比
        'efficiency_score': 20,   # 入睡效率
        'interference_score': 15, # 夜间干扰
        'stability_score': 10     # 体征稳定性
    }
    
    @classmethod
    def calculate_sleep_score(cls, sleep_data: Dict) -> int:
        """
        计算睡眠评分
        
        Args:
            sleep_data: 睡眠数据字典
            
        Returns:
            睡眠评分(0-100)
        """
        try:
            # 提取睡眠指标
            sleep_duration_minutes = sleep_data.get('sleep_duration_minutes', 0)
            sleep_prep_time_minutes = sleep_data.get('sleep_prep_time_minutes', 0)
            bed_exit_count = sleep_data.get('bed_exit_count', 0)
            sleep_phases = sleep_data.get('sleep_phases', {})
            deep_sleep_duration = sleep_phases.get('deep_sleep_minutes', 0)
            awake_duration = sleep_phases.get('awake_minutes', 0)
            
            # 计算各项评分
            time_score = cls._calculate_time_score(sleep_duration_minutes)
            deep_sleep_score = cls._calculate_deep_sleep_score(sleep_duration_minutes, deep_sleep_duration)
            efficiency_score = cls._calculate_efficiency_score(sleep_prep_time_minutes)
            interference_score = cls._calculate_interference_score(bed_exit_count, awake_duration)
            stability_score = cls._calculate_stability_score(sleep_duration_minutes, deep_sleep_duration, bed_exit_count)
            
            # 计算总评分
            sleep_score = min(100, round(time_score + deep_sleep_score + efficiency_score + interference_score + stability_score))
            return sleep_score
            
        except Exception as e:
            logger.error(f"计算睡眠评分时出错: {str(e)}")
            return 0
    
    @staticmethod
    def _calculate_time_score(sleep_duration_minutes: float) -> float:
        """
        计算睡眠时长评分
        
        Args:
            sleep_duration_minutes: 睡眠时长（分钟）
            
        Returns:
            睡眠时长评分
        """
        sleep_hours = sleep_duration_minutes / 60
        if SleepMetricsCalculator.IDEAL_SLEEP_HOURS_MIN <= sleep_hours <= SleepMetricsCalculator.IDEAL_SLEEP_HOURS_MAX:
            return SleepMetricsCalculator.SCORE_WEIGHTS['time_score']
        else:
            if sleep_hours < SleepMetricsCalculator.IDEAL_SLEEP_HOURS_MIN:
                deficit = SleepMetricsCalculator.IDEAL_SLEEP_HOURS_MIN - sleep_hours
                return max(0, SleepMetricsCalculator.SCORE_WEIGHTS['time_score'] - (deficit * 10))
            else:
                excess = sleep_hours - SleepMetricsCalculator.IDEAL_SLEEP_HOURS_MAX
                return max(0, SleepMetricsCalculator.SCORE_WEIGHTS['time_score'] - (excess * 5))
    
    @staticmethod
    def _calculate_deep_sleep_score(sleep_duration_minutes: float, deep_sleep_duration: float) -> float:
        """
        计算深睡占比评分
        
        Args:
            sleep_duration_minutes: 总睡眠时长
            deep_sleep_duration: 深睡时长
            
        Returns:
            深睡占比评分
        """
        if sleep_duration_minutes <= 0:
            return 0
        
        deep_sleep_ratio = (deep_sleep_duration / sleep_duration_minutes) * 100
        
        if deep_sleep_ratio >= 25:
            return SleepMetricsCalculator.SCORE_WEIGHTS['deep_sleep_score']
        elif 20 <= deep_sleep_ratio < 25:
            return 20
        elif 15 <= deep_sleep_ratio < 20:
            return 15
        elif 10 <= deep_sleep_ratio < 15:
            return 10
        elif deep_sleep_ratio >= 5:
            return 5
        else:
            return 0
    
    @staticmethod
    def _calculate_efficiency_score(sleep_prep_time_minutes: float) -> float:
        """
        计算入睡效率评分
        
        Args:
            sleep_prep_time_minutes: 睡眠准备时间
            
        Returns:
            入睡效率评分
        """
        if sleep_prep_time_minutes <= 30:
            return SleepMetricsCalculator.SCORE_WEIGHTS['efficiency_score']
        elif 31 <= sleep_prep_time_minutes <= 60:
            return 15
        elif 61 <= sleep_prep_time_minutes <= 90:
            return 10
        else:
            return 5
    
    @staticmethod
    def _calculate_interference_score(bed_exit_count: int, awake_duration: float) -> float:
        """
        计算夜间干扰评分
        
        Args:
            bed_exit_count: 离床次数
            awake_duration: 清醒时长
            
        Returns:
            夜间干扰评分
        """
        if bed_exit_count == 0 and awake_duration <= 10:
            return SleepMetricsCalculator.SCORE_WEIGHTS['interference_score']
        elif bed_exit_count == 1 or (11 <= awake_duration <= 30):
            return 10
        elif bed_exit_count == 2 or (31 <= awake_duration <= 60):
            return 5
        else:
            return 0
    
    @staticmethod
    def _calculate_stability_score(sleep_duration_minutes: float, deep_sleep_duration: float, bed_exit_count: int) -> float:
        """
        计算体征稳定性评分
        
        Args:
            sleep_duration_minutes: 总睡眠时长
            deep_sleep_duration: 深睡时长
            bed_exit_count: 离床次数
            
        Returns:
            体征稳定性评分
        """
        if sleep_duration_minutes <= 0:
            return 3
        
        deep_sleep_ratio = (deep_sleep_duration / sleep_duration_minutes) * 100
        
        if deep_sleep_ratio >= 20 and bed_exit_count <= 1:
            return SleepMetricsCalculator.SCORE_WEIGHTS['stability_score']
        elif deep_sleep_ratio >= 10 and bed_exit_count <= 2:
            return 7
        else:
            return 3
    
    @staticmethod
    def _prepare_data(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
        """
        准备数据，包括类型转换和排序
        
        Args:
            df: 原始数据
            numeric_columns: 需要转换为数值的列
            
        Returns:
            处理后的数据
        """
        # 复制数据避免修改原数据
        processed_df = df.copy()
        
        # 检查数据是否为空
        if processed_df.empty:
            return processed_df
        
        # 转换数值列
        for col in numeric_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
        # 排序数据
        processed_df = processed_df.sort_values('upload_time').reset_index(drop=True)
        
        return processed_df
    
    @classmethod
    def _calculate_basic_sleep_metrics(cls, night_data: pd.DataFrame, df: pd.DataFrame, target_date: datetime, prev_date: datetime) -> Dict:
        """
        计算基本睡眠指标
        
        Args:
            night_data: 夜间数据
            df: 完整数据
            target_date: 目标日期
            prev_date: 前一天日期
            
        Returns:
            基本睡眠指标
        """
        try:
            # 计算就寝时间和起床时间
            bedtime, wakeup_time = SleepTimeAnalyzer.calculate_bedtime_wakeup_times(
                night_data, target_date, prev_date
            )
            
            # 计算卧床时间
            time_in_bed = wakeup_time - bedtime
            time_in_bed_minutes = time_in_bed.total_seconds() / 60
            
            # 计算睡眠时长和离床次数
            sleep_period_data = df[(df['upload_time'] >= bedtime) & (df['upload_time'] <= wakeup_time)].copy()
            sleep_duration_minutes = cls._calculate_sleep_duration(sleep_period_data)
            bed_exit_count = cls._calculate_bed_exit_count(sleep_period_data)
            sleep_prep_time = cls._calculate_sleep_prep_time(sleep_period_data, bedtime)
            
            # 计算平均生理指标
            avg_metrics = cls._calculate_average_metrics(sleep_period_data)
            
            return {
                'bedtime': bedtime,
                'wakeup_time': wakeup_time,
                'time_in_bed_minutes': time_in_bed_minutes,
                'sleep_duration_minutes': sleep_duration_minutes,
                'bed_exit_count': bed_exit_count,
                'sleep_prep_time_minutes': sleep_prep_time,
                'avg_metrics': avg_metrics
            }
        except Exception as e:
            logger.error(f"计算基本睡眠指标时出错: {str(e)}")
            return {
                'bedtime': datetime.now(),
                'wakeup_time': datetime.now() + timedelta(hours=8),
                'time_in_bed_minutes': 480,
                'sleep_duration_minutes': 0,
                'bed_exit_count': 0,
                'sleep_prep_time_minutes': cls.DEFAULT_SLEEP_PREP_TIME,
                'avg_metrics': cls._calculate_average_metrics(pd.DataFrame())
            }
    
    @staticmethod
    def _calculate_sleep_duration(sleep_period_data: pd.DataFrame) -> float:
        """
        计算睡眠时长
        
        Args:
            sleep_period_data: 睡眠时段数据
            
        Returns:
            睡眠时长（分钟）
        """
        if sleep_period_data.empty:
            return 0
        
        # 检查是否有睡眠阶段数据
        if 'stage_value' in sleep_period_data.columns:
            # 使用睡眠分期结果来计算睡眠时长
            # 睡眠阶段值：1=深睡, 2=浅睡, 3=眼动, 4=清醒
            sleep_period_data['is_sleeping'] = sleep_period_data['stage_value'].isin([1, 2, 3])
        else:
            # 如果没有睡眠阶段数据，使用生理指标判断
            sleep_period_data['is_sleeping'] = (
                (sleep_period_data['heart_rate'] >= SleepMetricsCalculator.MIN_HEART_RATE) &
                (sleep_period_data['heart_rate'] <= 75) &  # 从70调整到75，更宽松
                (sleep_period_data['respiratory_rate'] >= 10) &  # 从12调整到10，更宽松
                (sleep_period_data['respiratory_rate'] <= 20) &  # 从18调整到20，更宽松
                (sleep_period_data['body_moves_ratio'] <= 15)  # 从10调整到15，更宽松
            )
        
        # 计算睡眠时长
        if len(sleep_period_data) == 1:
            return SleepMetricsCalculator.DEFAULT_SINGLE_RECORD_SLEEP_DURATION if sleep_period_data.iloc[0]['is_sleeping'] else 0
        
        # 对于多条记录，计算连续睡眠时段
        sleep_period_data = sleep_period_data.sort_values('upload_time').reset_index(drop=True)
        sleep_period_data['sleep_change'] = sleep_period_data['is_sleeping'].ne(sleep_period_data['is_sleeping'].shift())
        sleep_period_data['segment_id'] = sleep_period_data['sleep_change'].cumsum()
        
        sleep_segments = sleep_period_data[sleep_period_data['is_sleeping']].groupby('segment_id').agg({
            'upload_time': ['min', 'max']
        }).reset_index()
        
        if sleep_segments.empty:
            return 0
        
        sleep_segments['duration'] = (sleep_segments['upload_time']['max'] - sleep_segments['upload_time']['min']).dt.total_seconds() / 60
        return sleep_segments['duration'].sum()
    
    @staticmethod
    def _calculate_bed_exit_count(sleep_period_data: pd.DataFrame) -> int:
        """
        计算离床次数
        
        Args:
            sleep_period_data: 睡眠时段数据
            
        Returns:
            离床次数
        """
        if sleep_period_data.empty or len(sleep_period_data) <= 1:
            return 0
        
        # 标记离床状态
        if 'is_person' in sleep_period_data.columns:
            sleep_period_data['is_off_bed'] = (sleep_period_data['is_person'] == 0)
        else:
            sleep_period_data['is_off_bed'] = (sleep_period_data['heart_rate'] == 0)
        
        # 识别离床时段
        sleep_period_data['off_bed_change'] = sleep_period_data['is_off_bed'].ne(sleep_period_data['is_off_bed'].shift())
        sleep_period_data['off_bed_segment_id'] = sleep_period_data['off_bed_change'].cumsum()
        
        off_bed_segments = sleep_period_data[sleep_period_data['is_off_bed']].groupby('off_bed_segment_id').agg({
            'upload_time': ['min', 'max']
        }).reset_index()
        
        if off_bed_segments.empty:
            return 0
        
        # 计算有效离床次数
        off_bed_segments['duration'] = (off_bed_segments['upload_time']['max'] - off_bed_segments['upload_time']['min']).dt.total_seconds() / 60
        return (off_bed_segments['duration'] >= SleepMetricsCalculator.BED_EXIT_THRESHOLD_MINUTES).sum()
    
    @staticmethod
    def _calculate_sleep_prep_time(sleep_period_data: pd.DataFrame, bedtime: datetime) -> float:
        """
        计算睡眠准备时间
        
        Args:
            sleep_period_data: 睡眠时段数据
            bedtime: 就寝时间
            
        Returns:
            睡眠准备时间（分钟）
        """
        print(f"\n=== 开始计算睡眠准备时间 ===")
        print(f"就寝时间: {bedtime}")
        print(f"睡眠时段数据量: {len(sleep_period_data)} 条")
        if not sleep_period_data.empty:
            print(f"数据时间范围: {sleep_period_data['upload_time'].min()} 到 {sleep_period_data['upload_time'].max()}")
        
        if sleep_period_data.empty:
            print(f"数据为空，返回默认值: {SleepMetricsCalculator.DEFAULT_SLEEP_PREP_TIME} 分钟")
            return SleepMetricsCalculator.DEFAULT_SLEEP_PREP_TIME
        
        # 计算心率标准差和心律失常均值
        sleep_period_data['hr_std_5'] = sleep_period_data['heart_rate'].rolling(
            window=SleepMetricsCalculator.STABLE_RECORDS_NEEDED, 
            center=True, 
            min_periods=1
        ).std()
        
        if 'arrhythmia_ratio' in sleep_period_data.columns:
            sleep_period_data['arrhythmia_avg_5'] = sleep_period_data['arrhythmia_ratio'].rolling(
                window=SleepMetricsCalculator.STABLE_RECORDS_NEEDED, 
                center=True, 
                min_periods=1
            ).mean()
        else:
            sleep_period_data['arrhythmia_avg_5'] = 100
        
        # 计算呼吸率标准差
        if 'respiratory_rate' in sleep_period_data.columns:
            sleep_period_data['rr_std_5'] = sleep_period_data['respiratory_rate'].rolling(
                window=SleepMetricsCalculator.STABLE_RECORDS_NEEDED, 
                center=True, 
                min_periods=1
            ).std()
        else:
            sleep_period_data['rr_std_5'] = 100
        
        # 计算体动比率均值
        if 'body_moves_ratio' in sleep_period_data.columns:
            sleep_period_data['body_moves_avg_5'] = sleep_period_data['body_moves_ratio'].rolling(
                window=SleepMetricsCalculator.STABLE_RECORDS_NEEDED, 
                center=True, 
                min_periods=1
            ).mean()
        else:
            sleep_period_data['body_moves_avg_5'] = 100
        
        # 睡眠启动点识别参数
        SLEEP_START_WINDOW = 10  # 主判定窗口延长至10分钟
        SLEEP_START_VERIFY_WINDOW = 8  # 二次验证窗口8分钟
        AWAKE_BASELINE_WINDOW_START = 15  # 清醒基线从上床后15分钟开始
        AWAKE_BASELINE_WINDOW_END = 45   # 至45分钟结束
        
        SLEEP_HR_RATIO = 0.75  # 心率阈值降至清醒基线的75%
        MAX_SLEEP_START_BODY_MOVE = 0.2  # 体动比例≤0.2%
        MIN_SLEEP_START_RESP_RATE = 12
        MAX_SLEEP_START_RESP_RATE = 15  # 呼吸率上限降至15次/分
        STABLE_HR_STD_THRESHOLD = 2  # 心率标准差≤2
        STABLE_RESP_STD_THRESHOLD = 1.5  # 呼吸标准差≤1.5
        STABLE_BODY_MOVE_THRESHOLD = 1  # 稳定期体动≤1%
        
        # 计算清醒状态的心率基线（取就寝后15-45分钟的平均心率作为清醒基线）
        bedtime_window = sleep_period_data[
            (sleep_period_data['upload_time'] >= bedtime + timedelta(minutes=AWAKE_BASELINE_WINDOW_START)) &
            (sleep_period_data['upload_time'] <= bedtime + timedelta(minutes=AWAKE_BASELINE_WINDOW_END))
        ]
        
        # 计算清醒基线时，考虑体动率，确保只使用真正清醒时的数据
        if not bedtime_window.empty:
            # 过滤掉体动率过低的数据，确保只使用真正清醒时的数据
            awake_data = bedtime_window[bedtime_window.get('body_moves_ratio', 0) >= 1]
            if not awake_data.empty:
                awake_heart_rate_baseline = awake_data['heart_rate'].mean()
            else:
                # 如果没有足够的体动数据，使用整个窗口的平均心率
                awake_heart_rate_baseline = bedtime_window['heart_rate'].mean()
        else:
            # 如果没有足够的数据，使用整体平均心率
            awake_heart_rate_baseline = sleep_period_data['heart_rate'].mean()
        
        # 确保心率基线合理，设置上下限
        awake_heart_rate_baseline = max(60, min(100, awake_heart_rate_baseline))
        
        # 输出心率基线计算结果
        print(f"\n=== 心率基线计算结果 ===")
        print(f"清醒基线窗口: {AWAKE_BASELINE_WINDOW_START}-{AWAKE_BASELINE_WINDOW_END}分钟")
        print(f"基线窗口数据量: {len(bedtime_window)} 条")
        print(f"计算得到的心率基线: {awake_heart_rate_baseline:.2f} 次/分")
        print(f"心率基线范围: 60-100 次/分")
        
        # 计算心率和呼吸率的标准差
        sleep_period_data['hr_std_5'] = sleep_period_data['heart_rate'].rolling(
            window=5, center=True, min_periods=1
        ).std()
        
        if 'respiratory_rate' in sleep_period_data.columns:
            sleep_period_data['resp_std_5'] = sleep_period_data['respiratory_rate'].rolling(
                window=5, center=True, min_periods=1
            ).std()
        else:
            sleep_period_data['resp_std_5'] = 0
        
        # 第一步：主窗口判定
        sleep_start_mask = (
            (sleep_period_data['heart_rate'] <= awake_heart_rate_baseline * SLEEP_HR_RATIO) &
            (sleep_period_data.get('body_moves_ratio', 0) <= MAX_SLEEP_START_BODY_MOVE) &
            (sleep_period_data.get('respiratory_rate', 0) >= MIN_SLEEP_START_RESP_RATE) &
            (sleep_period_data.get('respiratory_rate', 0) <= MAX_SLEEP_START_RESP_RATE) &
            (sleep_period_data['hr_std_5'] <= STABLE_HR_STD_THRESHOLD) &
            (sleep_period_data.get('resp_std_5', 0) <= STABLE_RESP_STD_THRESHOLD)
        )
        sleep_period_data['is_sleep_start_candidate'] = sleep_start_mask
        sleep_period_data['sleep_start_main_window'] = sleep_period_data['is_sleep_start_candidate']\
            .rolling(window=SLEEP_START_WINDOW, min_periods=SLEEP_START_WINDOW).sum() == SLEEP_START_WINDOW
        
        # 第二步：二次验证窗口（确保不反弹）
        sleep_period_data['sleep_start_verify_window'] = sleep_period_data['sleep_start_main_window']\
            .rolling(window=SLEEP_START_VERIFY_WINDOW, min_periods=1).max()  # 主窗口达标后，后续SLEEP_START_VERIFY_WINDOW分钟需维持
        
        # 最终睡眠启动点：同时满足主窗口+验证窗口
        final_sleep_start_indices = sleep_period_data[
            (sleep_period_data['sleep_start_main_window']) &
            (sleep_period_data['sleep_start_verify_window'])
        ].index
        
        # 输出睡眠启动点检测结果
        print(f"\n=== 睡眠启动点检测结果 ===")
        print(f"主判定窗口: {SLEEP_START_WINDOW}分钟")
        print(f"二次验证窗口: {SLEEP_START_VERIFY_WINDOW}分钟")
        print(f"检测到的睡眠启动点数量: {len(final_sleep_start_indices)}")
        if not final_sleep_start_indices.empty:
            first_sleep_start_idx = final_sleep_start_indices[0]
            sleep_start_time = sleep_period_data.loc[first_sleep_start_idx, 'upload_time'] - timedelta(minutes=SLEEP_START_WINDOW-1)
            sleep_prep_time = (sleep_start_time - bedtime).total_seconds() / 60
            print(f"第一个睡眠启动点时间: {sleep_start_time}")
            print(f"计算得到的睡眠准备时间: {sleep_prep_time:.2f} 分钟")
        
        # 基于睡眠周期规律的辅助判断
        # 如果没有找到睡眠启动点，尝试根据睡眠周期规律推断
        if final_sleep_start_indices.empty:
            # 计算从就寝时间开始的时间点，按照90分钟的睡眠周期规律
            # 第一个REM阶段通常出现在入睡后70-90分钟
            for cycle_minutes in [70, 80, 90, 100, 110, 120, 180, 270]:
                # 计算可能的REM阶段开始时间
                rem_start_time = bedtime + timedelta(minutes=cycle_minutes)
                # 检查该时间点前后的数据是否符合REM睡眠特征
                rem_window_data = sleep_period_data[
                    (sleep_period_data['upload_time'] >= rem_start_time - timedelta(minutes=10)) &
                    (sleep_period_data['upload_time'] <= rem_start_time + timedelta(minutes=20))
                ]
                
                if not rem_window_data.empty:
                    # 检查是否符合REM睡眠特征：心率较高、呼吸不稳定、体动少
                    avg_hr = rem_window_data['heart_rate'].mean()
                    avg_rr = rem_window_data['respiratory_rate'].mean()
                    avg_body_move = rem_window_data['body_moves_ratio'].mean()
                    hr_std = rem_window_data['heart_rate'].std()
                    rr_std = rem_window_data['respiratory_rate'].std()
                    
                    # REM睡眠特征：心率较高（接近清醒水平）、呼吸不稳定（标准差大）、体动少
                    if ((avg_hr > awake_heart_rate_baseline * 0.9) and 
                        (hr_std > 6) and 
                        (rr_std > 3) and 
                        (avg_body_move <= 1.5)):
                        # 找到可能的REM阶段，推断入睡时间为REM开始前90分钟
                        inferred_sleep_start_time = rem_start_time - timedelta(minutes=90)
                        # 检查推断的入睡时间是否合理
                        if inferred_sleep_start_time >= bedtime:
                            # 计算睡眠准备时间
                            sleep_prep_time = (inferred_sleep_start_time - bedtime).total_seconds() / 60
                            
                            # 检查推断的睡眠准备时间是否合理
                            # 分析从就寝时间到推断入睡时间的体动和心率
                            prep_period_data = sleep_period_data[
                                (sleep_period_data['upload_time'] >= bedtime) &
                                (sleep_period_data['upload_time'] < inferred_sleep_start_time)
                            ]
                            
                            if not prep_period_data.empty:
                                avg_prep_body_move = prep_period_data['body_moves_ratio'].mean()
                                avg_prep_heart_rate = prep_period_data['heart_rate'].mean()
                                
                                # 检查整个睡眠准备期的心率变化
                                # 查找心率持续较高的时间段
                                high_hr_periods = []
                                current_period_start = None
                                
                                for i in range(len(sleep_period_data)):
                                    current_time = sleep_period_data.loc[i, 'upload_time']
                                    if current_time >= inferred_sleep_start_time:
                                        break
                                    current_hr = sleep_period_data.loc[i, 'heart_rate']
                                    
                                    # 如果心率超过68次/分钟，认为可能处于清醒状态
                                    if current_hr > 68:
                                        if current_period_start is None:
                                            current_period_start = current_time
                                    else:
                                        if current_period_start is not None:
                                            # 计算高心率持续时间
                                            duration = (current_time - current_period_start).total_seconds() / 60
                                            if duration >= 30:  # 持续30分钟以上
                                                high_hr_periods.append((current_period_start, current_time))
                                            current_period_start = None
                                
                                # 如果有高心率持续时间段，使用最后一个高心率期结束时间作为入睡时间
                                if high_hr_periods:
                                    last_high_hr_end = high_hr_periods[-1][1]
                                    sleep_prep_time = (last_high_hr_end - bedtime).total_seconds() / 60
                                # 综合考虑体动率和心率，当体动频繁或心率较高时，认为用户可能仍然清醒
                                elif avg_prep_body_move > 3 or avg_prep_heart_rate > 68:
                                    # 查找心率稳定下降且体动较少的时间点
                                    low_hr_found = False
                                    for i in range(len(sleep_period_data)):
                                        current_time = sleep_period_data.loc[i, 'upload_time']
                                        if current_time < bedtime + timedelta(minutes=120):
                                            continue
                                        if current_time >= inferred_sleep_start_time:
                                            break
                                        # 检查当前时间点及之后4分钟的心率和体动情况
                                        window_data = sleep_period_data[
                                            (sleep_period_data['upload_time'] >= current_time) &
                                            (sleep_period_data['upload_time'] <= current_time + timedelta(minutes=4))
                                        ]
                                        if len(window_data) >= 5:
                                            if (window_data['heart_rate'].mean() <= 65 and 
                                                window_data['body_moves_ratio'].mean() <= 1):
                                                # 找到心率稳定下降且体动较少的时间点
                                                new_sleep_start_time = current_time
                                                sleep_prep_time = (new_sleep_start_time - bedtime).total_seconds() / 60
                                                low_hr_found = True
                                                break
                                    # 如果没有找到合适的时间点，使用推断的睡眠准备时间
                                    if not low_hr_found:
                                        # 确保睡眠准备时间至少为90分钟
                                        sleep_prep_time = max(sleep_prep_time, 90)
                            # 输出REM睡眠特征检测结果
                            print(f"\n=== REM睡眠特征检测结果 ===")
                            print(f"检测到的REM阶段开始时间: {rem_start_time}")
                            print(f"推断的睡眠启动时间: {inferred_sleep_start_time}")
                            print(f"计算得到的睡眠准备时间: {sleep_prep_time:.2f} 分钟")
                            print(f"REM睡眠特征: 心率={avg_hr:.2f}, 心率标准差={hr_std:.2f}, 呼吸标准差={rr_std:.2f}, 体动率={avg_body_move:.2f}")
                            
                            # 返回计算结果
                            return max(sleep_prep_time, SleepMetricsCalculator.MIN_SLEEP_PREP_TIME)
        
        if not final_sleep_start_indices.empty:
            # 使用睡眠启动点计算入睡准备时间
            first_sleep_start_idx = final_sleep_start_indices[0]
            # 睡眠启动点是主窗口满足条件的结束时间，因此需要减去(SLEEP_START_WINDOW-1)分钟得到开始时间
            sleep_start_time = sleep_period_data.loc[first_sleep_start_idx, 'upload_time'] - timedelta(minutes=SLEEP_START_WINDOW-1)
            sleep_prep_time = (sleep_start_time - bedtime).total_seconds() / 60
            
            # 额外检查：如果在睡眠准备期内体动频繁，延长睡眠准备时间
            # 分析更长时间段的体动模式，确保准确检测长时间清醒状态
            first_three_hours_data = sleep_period_data[
                (sleep_period_data['upload_time'] >= bedtime) &
                (sleep_period_data['upload_time'] <= bedtime + timedelta(hours=3))
            ]
            
            if not first_three_hours_data.empty:
                # 分析每小时的体动率和心率
                hourly_data = []
                for hour in range(3):
                    start_time = bedtime + timedelta(hours=hour)
                    end_time = start_time + timedelta(hours=1)
                    hour_data = first_three_hours_data[
                        (first_three_hours_data['upload_time'] >= start_time) &
                        (first_three_hours_data['upload_time'] < end_time)
                    ]
                    if not hour_data.empty:
                        hourly_data.append({
                            'hour': hour + 1,
                            'avg_body_move': hour_data['body_moves_ratio'].mean(),
                            'avg_heart_rate': hour_data['heart_rate'].mean(),
                            'data_points': len(hour_data)
                        })
                
                # 综合分析体动模式
                high_activity_hours = 0
                for hour_info in hourly_data:
                    if hour_info['avg_body_move'] > 3 or hour_info['avg_heart_rate'] > 68:
                        high_activity_hours += 1
                
                # 如果前3小时中有2个或更多小时体动频繁或心率较高，认为用户可能仍然清醒
                if high_activity_hours >= 2:
                    # 查找心率稳定下降且体动较少的时间点
                    # 从120分钟开始查找连续5分钟心率≤65且体动率≤1%的时间点
                    low_hr_found = False
                    for i in range(len(sleep_period_data)):
                        current_time = sleep_period_data.loc[i, 'upload_time']
                        if current_time < bedtime + timedelta(minutes=120):
                            continue
                        
                        # 检查当前时间点及之后4分钟的心率和体动情况
                        window_data = sleep_period_data[
                            (sleep_period_data['upload_time'] >= current_time) &
                            (sleep_period_data['upload_time'] <= current_time + timedelta(minutes=4))
                        ]
                        
                        if len(window_data) >= 5:
                            if (window_data['heart_rate'].mean() <= 65 and 
                                window_data['body_moves_ratio'].mean() <= 1):
                                # 找到连续5分钟心率≤65且体动率≤1%的时间点
                                new_sleep_start_time = current_time
                                sleep_prep_time = (new_sleep_start_time - bedtime).total_seconds() / 60
                                low_hr_found = True
                                break
                    
                    # 如果没有找到合适的时间点，使用120分钟作为睡眠准备时间
                    if not low_hr_found:
                        sleep_prep_time = 120
                # 如果前3小时中有1个小时体动频繁或心率较高，延长睡眠准备时间至90分钟
                elif high_activity_hours == 1:
                    sleep_prep_time = max(sleep_prep_time, 90)
                
                # 输出体动率分析结果
                print(f"\n=== 体动率分析结果 ===")
                for hour_info in hourly_data:
                    print(f"第{hour_info['hour']}小时: 体动率={hour_info['avg_body_move']:.2f}%, 心率={hour_info['avg_heart_rate']:.2f}, 数据点={hour_info['data_points']}")
                print(f"高活动小时数: {high_activity_hours}")
                print(f"调整后的睡眠准备时间: {sleep_prep_time:.2f} 分钟")
        else:
            # 如果没有找到睡眠启动点，使用原来的稳定睡眠点判定
            # 找到第一个稳定睡眠位置，考虑更多生理指标
            # 构建稳定睡眠点的条件，根据数据中是否存在 arrhythmia_ratio 字段来调整条件
            if 'arrhythmia_ratio' in sleep_period_data.columns:
                stable_mask = (
                    (sleep_period_data['hr_std_5'] <= STABLE_HR_STD_THRESHOLD) & 
                    (sleep_period_data['arrhythmia_avg_5'] <= SleepMetricsCalculator.STABLE_ARRHYTHMIA_THRESHOLD) &
                    (sleep_period_data.get('resp_std_5', 0) <= STABLE_RESP_STD_THRESHOLD) &  # 呼吸标准差 <= 2
                    (sleep_period_data.get('body_moves_avg_5', 0) <= STABLE_BODY_MOVE_THRESHOLD) &  # 体动比率 <= 2%
                    (sleep_period_data['heart_rate'] <= awake_heart_rate_baseline * 0.85)  # 心率降至清醒基线的85%以下
                )
            else:
                # 如果没有 arrhythmia_ratio 字段，跳过这个条件
                stable_mask = (
                    (sleep_period_data['hr_std_5'] <= STABLE_HR_STD_THRESHOLD) &
                    (sleep_period_data.get('resp_std_5', 0) <= STABLE_RESP_STD_THRESHOLD) &  # 呼吸标准差 <= 2
                    (sleep_period_data.get('body_moves_avg_5', 0) <= STABLE_BODY_MOVE_THRESHOLD) &  # 体动比率 <= 2%
                    (sleep_period_data['heart_rate'] <= awake_heart_rate_baseline * 0.85)  # 心率降至清醒基线的85%以下
                )
            stable_indices = sleep_period_data[stable_mask].index
            
            # 分析体动率和心率模式，判断用户是否真的入睡
            print(f"\n=== 体动率和心率模式分析 ===")
            # 计算整个睡眠时段的平均体动率和心率
            avg_body_move_total = sleep_period_data['body_moves_ratio'].mean()
            avg_heart_rate_total = sleep_period_data['heart_rate'].mean()
            print(f"整个睡眠时段平均体动率: {avg_body_move_total:.4f}%")
            print(f"整个睡眠时段平均心率: {avg_heart_rate_total:.2f} 次/分")
            print(f"清醒基线心率: {awake_heart_rate_baseline:.2f} 次/分")
            
            # 检查是否存在明显的体动或心率变化
            body_move_std = sleep_period_data['body_moves_ratio'].std()
            heart_rate_std = sleep_period_data['heart_rate'].std()
            print(f"体动率标准差: {body_move_std:.4f}")
            print(f"心率标准差: {heart_rate_std:.2f}")
            
            # 如果体动率非常低但心率仍然较高，认为用户可能仍然清醒
            if avg_body_move_total < 0.1 and avg_heart_rate_total > awake_heart_rate_baseline * 0.9:
                print("警告: 体动率极低但心率较高，可能仍然处于清醒状态")
                # 查找心率开始下降的时间点
                sleep_prep_time = 0
                
                # 分析每小时的心率变化，找到第一个心率明显下降的小时
                hourly_hr_data = []
                for hour in range(8):  # 检查前8小时
                    start_time = bedtime + timedelta(hours=hour)
                    end_time = start_time + timedelta(hours=1)
                    hour_data = sleep_period_data[
                        (sleep_period_data['upload_time'] >= start_time) &
                        (sleep_period_data['upload_time'] < end_time)
                    ]
                    if not hour_data.empty:
                        hourly_hr = hour_data['heart_rate'].mean()
                        hourly_hr_data.append({
                            'hour': hour,
                            'avg_hr': hourly_hr,
                            'start_time': start_time,
                            'end_time': end_time
                        })
                
                # 输出每小时心率数据
                print("\n=== 每小时心率分析 ===")
                for hour_info in hourly_hr_data:
                    print(f"第{hour_info['hour']}小时: 平均心率={hour_info['avg_hr']:.2f} 次/分, 时间范围: {hour_info['start_time']} - {hour_info['end_time']}")
                
                # 查找第一个心率低于清醒基线90%的小时
                for hour_info in hourly_hr_data:
                    if hour_info['avg_hr'] <= awake_heart_rate_baseline * 0.9:
                        # 找到心率开始下降的时间点
                        sleep_start_time = hour_info['start_time']
                        sleep_prep_time = (sleep_start_time - bedtime).total_seconds() / 60
                        print(f"\n找到心率下降时间点: {sleep_start_time}")
                        print(f"计算得到的睡眠准备时间: {sleep_prep_time:.2f} 分钟")
                        break
                
                # 如果没有找到心率下降的时间点，基于心率最高值计算
                if sleep_prep_time == 0:
                    print("\n未找到心率下降时间点，基于心率最高值计算睡眠准备时间")
                    if hourly_hr_data:
                        # 找到心率最高值的时间点
                        max_hr_info = max(hourly_hr_data, key=lambda x: x['avg_hr'])
                        print(f"找到心率最高值: {max_hr_info['avg_hr']:.2f} 次/分, 时间范围: {max_hr_info['start_time']} - {max_hr_info['end_time']}")
                        
                        # 从最高值之后查找心率开始下降的时间点
                        max_hr_index = next((i for i, info in enumerate(hourly_hr_data) if info['hour'] == max_hr_info['hour']), -1)
                        if max_hr_index != -1 and max_hr_index < len(hourly_hr_data) - 1:
                            # 从最高值之后开始查找
                            for i in range(max_hr_index + 1, len(hourly_hr_data)):
                                current_hr = hourly_hr_data[i]['avg_hr']
                                # 如果心率比最高值下降了5%以上，认为开始入睡
                                if current_hr <= max_hr_info['avg_hr'] * 0.95:
                                    sleep_start_time = hourly_hr_data[i]['start_time']
                                    sleep_prep_time = (sleep_start_time - bedtime).total_seconds() / 60
                                    print(f"找到心率开始下降的时间点: {sleep_start_time}")
                                    print(f"基于心率最高值计算的睡眠准备时间: {sleep_prep_time:.2f} 分钟")
                                    break
                        
                        # 如果仍然没有找到，使用心率最高值之后2小时作为睡眠开始时间
                        if sleep_prep_time == 0:
                            sleep_start_time = max_hr_info['end_time'] + timedelta(hours=1)
                            sleep_prep_time = (sleep_start_time - bedtime).total_seconds() / 60
                            print(f"未找到明显的心率下降，使用心率最高值之后2小时作为睡眠开始时间")
                            print(f"计算得到的睡眠准备时间: {sleep_prep_time:.2f} 分钟")
                    else:
                        # 如果没有每小时心率数据，使用默认值，但不超过实际卧床时间
                        print("无每小时心率数据，使用默认睡眠准备时间")
                        # 计算实际卧床时间
                        actual_bed_time = (sleep_period_data['upload_time'].max() - sleep_period_data['upload_time'].min()).total_seconds() / 60
                        # 使用默认值但不超过实际卧床时间
                        sleep_prep_time = min(180, actual_bed_time)  # 默认3小时，但不超过实际卧床时间
            elif stable_indices.empty:
                # 如果没有找到稳定点，检查前60分钟的体动情况
                first_hour_data = sleep_period_data[
                    (sleep_period_data['upload_time'] >= bedtime) &
                    (sleep_period_data['upload_time'] <= bedtime + timedelta(minutes=60))
                ]
                
                if not first_hour_data.empty:
                    avg_body_move_first_hour = first_hour_data['body_moves_ratio'].mean()
                    # 如果前60分钟平均体动率超过5%，认为体动频繁，使用60分钟作为睡眠准备时间
                    if avg_body_move_first_hour > 5:
                        return 60
                # 如果没有找到稳定点，设置最大睡眠准备时间为120分钟
                return min(SleepMetricsCalculator.DEFAULT_SLEEP_PREP_TIME, 120)
            else:
                # 找到第一个稳定睡眠点
                first_stable_idx = stable_indices[0]
                stable_sleep_start = sleep_period_data.loc[first_stable_idx, 'upload_time']
                sleep_prep_time = (stable_sleep_start - bedtime).total_seconds() / 60
                print(f"找到稳定睡眠点: {stable_sleep_start}")
                print(f"计算得到的睡眠准备时间: {sleep_prep_time:.2f} 分钟")
        
        # 移除最大睡眠准备时间限制，让系统根据实际情况判断
        # 但仍然确保睡眠准备时间不小于最小限制
        final_sleep_prep_time = max(sleep_prep_time, SleepMetricsCalculator.MIN_SLEEP_PREP_TIME)
        
        # 输出最终计算结果
        print(f"\n=== 最终睡眠准备时间计算结果 ===")
        print(f"计算得到的睡眠准备时间: {sleep_prep_time:.2f} 分钟")
        print(f"最小睡眠准备时间: {SleepMetricsCalculator.MIN_SLEEP_PREP_TIME} 分钟")
        print(f"最终返回的睡眠准备时间: {final_sleep_prep_time:.2f} 分钟")
        print(f"=== 睡眠准备时间计算完成 ===")
        
        return final_sleep_prep_time
    
    @staticmethod
    def _calculate_average_metrics(sleep_period_data: pd.DataFrame) -> Dict:
        """
        计算平均生理指标
        
        Args:
            sleep_period_data: 睡眠时段数据
            
        Returns:
            平均生理指标
        """
        if sleep_period_data.empty:
            return {
                "avg_heart_rate": 0,
                "avg_respiratory_rate": 0,
                "avg_body_moves_ratio": 0,
                "avg_heartbeat_interval": 0,
                "avg_rms_heartbeat_interval": 0
            }
        
        # 计算平均指标
        avg_heart_rate = sleep_period_data['heart_rate'].mean()
        avg_respiratory_rate = sleep_period_data['respiratory_rate'].mean()
        avg_body_moves = sleep_period_data['body_moves_ratio'].mean() if 'body_moves_ratio' in sleep_period_data.columns else 0
        
        return {
            "avg_heart_rate": round(float(avg_heart_rate), 2) if pd.notna(avg_heart_rate) else 0,
            "avg_respiratory_rate": round(float(avg_respiratory_rate), 2) if pd.notna(avg_respiratory_rate) else 0,
            "avg_body_moves_ratio": round(float(avg_body_moves), 2) if pd.notna(avg_body_moves) else 0,
            "avg_heartbeat_interval": 0,
            "avg_rms_heartbeat_interval": 0
        }
    
    @classmethod
    def _calculate_sleep_stages(cls, sleep_period_data: pd.DataFrame, use_trusleep: bool = True, sleep_staging_method: str = "ensemble") -> Dict:
        """
        计算睡眠阶段
        
        Args:
            sleep_period_data: 睡眠时段数据
            use_trusleep: 是否使用 TruSleep 算法
            sleep_staging_method: 睡眠分期方法，可选值: "rule" (基于规则推理), "ensemble" (基于集成学习)
            
        Returns:
            睡眠阶段数据
        """
        # 初始化返回值
        result = {
            'deep_sleep_duration': 0,
            'light_sleep_duration': 0,
            'rem_sleep_duration': 0,
            'awake_duration': 0,
            'sleep_stage_segments': []
        }
        
        if sleep_period_data.empty:
            return result
        
        try:
            # 筛选有效数据
            print(f"\n=== 调试信息: _calculate_sleep_stages 方法 ===")
            print(f"输入数据行数: {len(sleep_period_data)}")
            print(f"输入时间范围: {sleep_period_data['upload_time'].min()} 到 {sleep_period_data['upload_time'].max()}")
            print(f"睡眠分期方法: {sleep_staging_method}")
            
            # 打印数据的基本统计信息
            if not sleep_period_data.empty:
                print(f"心率范围: {sleep_period_data['heart_rate'].min():.2f} - {sleep_period_data['heart_rate'].max():.2f}")
                print(f"呼吸率范围: {sleep_period_data['respiratory_rate'].min():.2f} - {sleep_period_data['respiratory_rate'].max():.2f}")
                print(f"体动率范围: {sleep_period_data['body_moves_ratio'].min():.4f} - {sleep_period_data['body_moves_ratio'].max():.4f}")
            
            # 筛选有效数据
            sleep_data = sleep_period_data[
                (sleep_period_data['heart_rate'] >= SleepMetricsCalculator.MIN_HEART_RATE) &
                (sleep_period_data['heart_rate'] <= SleepMetricsCalculator.MAX_HEART_RATE) &
                (sleep_period_data['respiratory_rate'] >= SleepMetricsCalculator.MIN_RESPIRATORY_RATE) &
                (sleep_period_data['respiratory_rate'] <= SleepMetricsCalculator.MAX_RESPIRATORY_RATE) &
                (sleep_period_data['heart_rate'].notna()) &
                (sleep_period_data['respiratory_rate'].notna())
            ].copy()
            
            print(f"筛选后数据行数: {len(sleep_data)}")
            
            if sleep_data.empty:
                print("警告: 筛选后的数据为空，返回空结果")
                return result
            
            # 计算基线心率
            if 'is_awake' in sleep_data.columns:
                awake_data = sleep_data[sleep_data['is_awake']]
            else:
                # 简单估算清醒数据
                awake_data = sleep_data[
                    (sleep_data['heart_rate'] >= 70) |
                    (sleep_data['body_moves_ratio'] > 20)
                ]
            
            baseline_heart_rate = awake_data['heart_rate'].mean() if not awake_data.empty else 70
            
            # 计算睡眠阶段
            print(f"\n=== 调试信息: 使用 {sleep_staging_method} 睡眠分期分析方法 ===")
            
            # 导入所需的分析器
            from .rule_based_sleep_stage_analyzer import RuleBasedSleepStageAnalyzer
            
            # 使用基于规则推理的方法（默认方法，不使用机器学习）
            print(f"\n=== 调试信息: 调用 RuleBasedSleepStageAnalyzer ===")
            print(f"输入数据行数: {len(sleep_data)}")
            print(f"输入时间范围: {sleep_data['upload_time'].min()} 到 {sleep_data['upload_time'].max()}")
            
            rule_analyzer = RuleBasedSleepStageAnalyzer()
            analyzed_data = rule_analyzer.analyze_sleep_stages_by_rules(sleep_data)
            
            print(f"\n=== 调试信息: RuleBasedSleepStageAnalyzer 结果 ===")
            print(f"分析后数据行数: {len(analyzed_data)}")
            if not analyzed_data.empty:
                print(f"分析后阶段分布:")
                stage_counts = analyzed_data['stage_value'].value_counts()
                for stage, count in stage_counts.items():
                    print(f"阶段 {stage} ({rule_analyzer.get_stage_label(stage)}): {count} 条")
            
            # 规则推理分析器: 0=清醒, 1=N1, 2=N2, 3=N3, 4=REM
            # 当前系统: 1=深睡, 2=浅睡, 3=眼动, 4=清醒
            stage_mapping = {
                0: 4,  # 清醒 → 清醒
                1: 2,  # N1 → 浅睡
                2: 2,  # N2 → 浅睡
                3: 1,  # N3 → 深睡
                4: 3   # REM → 眼动
            }
            print(f"\n=== 调试信息: 阶段映射 ===")
            print(stage_mapping)
            
            # 打印分析器返回的原始阶段值，查看映射前的阶段分布
            print(f"\n=== 调试信息: 映射前的阶段分布 ===")
            print(f"分析器返回的阶段值: {analyzed_data['stage_value'].unique()}")
            
            # 确保 analyzed_data 的索引和 sleep_data 的索引匹配
            # 首先创建一个字典，将 upload_time 映射到 stage_value，确保值都是整数类型
            stage_map = {}
            for time, value in zip(analyzed_data['upload_time'], analyzed_data['stage_value']):
                try:
                    # 确保值都是整数类型，避免浮点数精度问题
                    if isinstance(value, (int, float)):
                        stage_map[time] = int(round(value))
                    elif not pd.isna(value):
                        stage_map[time] = int(round(float(value)))
                    else:
                        stage_map[time] = 0
                except:
                    stage_map[time] = 0
            
            # 然后使用这个字典来填充 sleep_data['stage_value']
            sleep_data['stage_value'] = sleep_data['upload_time'].map(stage_map)
            
            # 应用阶段映射
            sleep_data['stage_value'] = sleep_data['stage_value'].map(stage_mapping).fillna(4)
            # 确保所有值都是整数类型，避免浮点数精度问题
            def safe_int_conversion(x):
                try:
                    if isinstance(x, (int, float)):
                        return int(round(x))
                    elif not pd.isna(x):
                        return int(round(float(x)))
                    else:
                        return 4
                except:
                    return 4
            sleep_data['stage_value'] = sleep_data['stage_value'].apply(safe_int_conversion)
            
            # 打印映射后的阶段分布
            print(f"\n=== 调试信息: 映射后的阶段分布 ===")
            print(f"映射后的阶段值: {sleep_data['stage_value'].unique()}")
            print(f"映射后的阶段值数量: {len(sleep_data['stage_value'].unique())}")
            
            sleep_data['stage_label'] = sleep_data['stage_value'].apply(SleepStageAnalyzer.get_stage_label)
            
            # 计算时间间隔
            time_diffs = sleep_data['upload_time'].diff().dt.total_seconds() / 60
            # 确保时间间隔是整数，避免浮点数精度问题
            time_intervals = time_diffs.fillna(0).apply(lambda x: max(1, int(round(x))))
            if len(time_intervals) > 1:
                try:
                    time_intervals.iloc[0] = time_intervals.iloc[1]
                except:
                    time_intervals.iloc[0] = 1
            else:
                time_intervals.iloc[0] = 1
            
            # 生成阶段序列
            stages_sequence = []
            for idx, row in sleep_data.iterrows():
                try:
                    # 确保 stage_value 是整数
                    if isinstance(row['stage_value'], (int, float)):
                        stage_value = int(round(row['stage_value']))
                    elif not pd.isna(row['stage_value']):
                        stage_value = int(row['stage_value'])
                    else:
                        stage_value = 4
                except:
                    stage_value = 4  # 默认值为清醒
                
                # 确保 time_interval 是数值类型
                try:
                    time_interval = time_intervals.iloc[idx] if idx < len(time_intervals) else 1
                    if not isinstance(time_interval, (int, float)):
                        time_interval = float(time_interval)
                except:
                    time_interval = 1
                
                stages_sequence.append({
                    'stage_value': stage_value,
                    'stage_label': row['stage_label'],
                    'time': row['upload_time'],
                    'time_interval': time_interval
                })
            
            # 打印原始阶段序列，查看平滑前的阶段分布
            print(f"\n=== 调试信息: 平滑前的阶段序列 ===")
            stage_counts_original = {}
            for stage in stages_sequence:
                stage_value = stage['stage_value']
                if stage_value not in stage_counts_original:
                    stage_counts_original[stage_value] = 0
                stage_counts_original[stage_value] += 1
            
            for stage_value, count in stage_counts_original.items():
                print(f"阶段 {stage_value} ({SleepStageAnalyzer.get_stage_label(stage_value)}): {count} 条")
            
            # 平滑处理
            smoothed_stages = SleepStageAnalyzer.smooth_sleep_stages(stages_sequence, min_duration_threshold=3)
            
            # 打印平滑后的阶段序列
            print(f"\n=== 调试信息: 平滑后的阶段序列 ===")
            for i, stage in enumerate(smoothed_stages):
                print(f"阶段 {i}: {stage['stage_label']} ({stage['time_interval']:.1f}分钟)")
            
            # 计算各阶段时长和生成阶段片段
            cls._process_sleep_stages(smoothed_stages, result)
            
            # 手动计算睡眠阶段时长，确保结果正确
            print(f"\n=== 调试信息: 手动计算睡眠阶段时长 ===")
            total_deep = 0
            total_light = 0
            total_rem = 0
            total_awake = 0
            
            for stage in smoothed_stages:
                try:
                    duration = float(stage['time_interval'])
                    if stage['stage_value'] == 1:
                        total_deep += duration
                    elif stage['stage_value'] == 2:
                        total_light += duration
                    elif stage['stage_value'] == 3:
                        total_rem += duration
                    elif stage['stage_value'] == 4:
                        total_awake += duration
                except:
                    pass
            
            print(f"手动计算的深睡时长: {total_deep:.2f} 分钟")
            print(f"手动计算的浅睡时长: {total_light:.2f} 分钟")
            print(f"手动计算的REM时长: {total_rem:.2f} 分钟")
            print(f"手动计算的清醒时长: {total_awake:.2f} 分钟")
            
            # 更新结果
            result['deep_sleep_duration'] = total_deep
            result['light_sleep_duration'] = total_light
            result['rem_sleep_duration'] = total_rem
            result['awake_duration'] = total_awake
            
        except Exception as e:
            logger.error(f"计算睡眠阶段时出错: {str(e)}")
            print(f"计算睡眠阶段时出错: {str(e)}")
        
        return result
    
    @staticmethod
    def _process_sleep_stages(smoothed_stages: List[Dict], result: Dict) -> None:
        """
        处理平滑后的睡眠阶段数据
        
        Args:
            smoothed_stages: 平滑后的睡眠阶段序列
            result: 结果字典，将被更新
        """
        current_stage = None
        current_stage_duration = 0
        current_stage_start_time = None
        previous_stage_end_time = None
        
        for i, stage_info in enumerate(smoothed_stages):
            stage_value = stage_info['stage_value']
            time_interval = stage_info['time_interval']
            stage_time = stage_info.get('time')
            
            # 确保 stage_value 是整数类型
            if not isinstance(stage_value, int):
                try:
                    stage_value = int(round(float(stage_value)))
                except:
                    stage_value = 4
            
            # 确保 time_interval 是整数类型，避免浮点数精度问题
            if not isinstance(time_interval, int):
                try:
                    time_interval = int(round(float(time_interval)))
                except:
                    time_interval = 1
            
            if current_stage != stage_value:
                if current_stage is not None:
                    # 计算当前阶段的结束时间
                    if current_stage_start_time:
                        from datetime import datetime, timedelta
                        try:
                            if isinstance(current_stage_start_time, str):
                                start_time = datetime.fromisoformat(current_stage_start_time.replace('Z', '+00:00'))
                            else:
                                start_time = current_stage_start_time
                            
                            # 计算当前阶段的结束时间
                            # 如果不是最后一个阶段，使用下一阶段的开始时间作为当前阶段的结束时间
                            if i < len(smoothed_stages) - 1:
                                next_stage_info = smoothed_stages[i+1]
                                next_stage_time = next_stage_info.get('time')
                                if next_stage_time:
                                    if isinstance(next_stage_time, str):
                                        end_time = datetime.fromisoformat(next_stage_time.replace('Z', '+00:00'))
                                    else:
                                        end_time = next_stage_time
                                else:
                                    end_time = start_time + timedelta(minutes=current_stage_duration)
                            else:
                                end_time = start_time + timedelta(minutes=current_stage_duration)
                            
                            start_time_str = start_time.isoformat().replace('T', ' ')
                            end_time_str = end_time.isoformat().replace('T', ' ')
                            previous_stage_end_time = end_time
                        except:
                            start_time_str = None
                            end_time_str = None
                            previous_stage_end_time = None
                    else:
                        start_time_str = None
                        end_time_str = None
                        previous_stage_end_time = None
                    
                    # 添加当前阶段到结果
                    result['sleep_stage_segments'].append({
                        "label": SleepStageAnalyzer.get_stage_label(current_stage),
                        "value": str(int(round(current_stage_duration))),
                        "start_time": start_time_str,
                        "end_time": end_time_str
                    })
                    
                    # 累加各阶段时长
                    if current_stage == 1:
                        result['deep_sleep_duration'] += current_stage_duration
                    elif current_stage == 2:
                        result['light_sleep_duration'] += current_stage_duration
                    elif current_stage == 3:
                        result['rem_sleep_duration'] += current_stage_duration
                    elif current_stage == 4:
                        result['awake_duration'] += current_stage_duration
                
                current_stage = stage_value
                current_stage_duration = time_interval
                # 使用前一阶段的结束时间作为当前阶段的开始时间
                if previous_stage_end_time:
                    current_stage_start_time = previous_stage_end_time
                else:
                    current_stage_start_time = stage_time
            else:
                current_stage_duration += time_interval
        
        # 处理最后一个阶段
        if current_stage is not None:
            # 计算最后一个阶段的结束时间
            if current_stage_start_time:
                from datetime import datetime, timedelta
                try:
                    if isinstance(current_stage_start_time, str):
                        start_time = datetime.fromisoformat(current_stage_start_time.replace('Z', '+00:00'))
                    else:
                        start_time = current_stage_start_time
                    end_time = start_time + timedelta(minutes=current_stage_duration)
                    start_time_str = start_time.isoformat().replace('T', ' ')
                    end_time_str = end_time.isoformat().replace('T', ' ')
                except:
                    start_time_str = None
                    end_time_str = None
            else:
                start_time_str = None
                end_time_str = None
            
            result['sleep_stage_segments'].append({
                "label": SleepStageAnalyzer.get_stage_label(current_stage),
                "value": str(int(round(current_stage_duration))),
                "start_time": start_time_str,
                "end_time": end_time_str
            })
            
            if current_stage == 1:
                result['deep_sleep_duration'] += current_stage_duration
            elif current_stage == 2:
                result['light_sleep_duration'] += current_stage_duration
            elif current_stage == 3:
                result['rem_sleep_duration'] += current_stage_duration
            elif current_stage == 4:
                result['awake_duration'] += current_stage_duration
    

    

    


    @classmethod
    def _calculate_sleep_phase_ratios(cls, sleep_duration_minutes: float, deep_sleep_duration: float, 
                                    light_sleep_duration: float, rem_sleep_duration: float, 
                                    awake_duration: float, time_in_bed_minutes: float) -> Dict:
        """
        计算睡眠阶段占比
        
        Args:
            sleep_duration_minutes: 总睡眠时长
            deep_sleep_duration: 深睡时长
            light_sleep_duration: 浅睡时长
            rem_sleep_duration: REM时长
            awake_duration: 清醒时长
            time_in_bed_minutes: 卧床时间
            
        Returns:
            睡眠阶段占比
        """
        # 重新计算总睡眠时长
        total_sleep_time = deep_sleep_duration + light_sleep_duration + rem_sleep_duration
        
        # 确保各项睡眠阶段总和不超过总卧床时间
        total_sleep_phases = total_sleep_time + awake_duration
        time_in_bed_minutes_total = time_in_bed_minutes
        
        # 打印调试信息
        print(f"\n=== 调试信息: _calculate_sleep_phase_ratios ===")
        print(f"输入的睡眠时长: {sleep_duration_minutes:.2f} 分钟")
        print(f"深睡时长: {deep_sleep_duration:.2f} 分钟")
        print(f"浅睡时长: {light_sleep_duration:.2f} 分钟")
        print(f"REM时长: {rem_sleep_duration:.2f} 分钟")
        print(f"清醒时长: {awake_duration:.2f} 分钟")
        print(f"卧床时间: {time_in_bed_minutes_total:.2f} 分钟")
        print(f"计算的总睡眠时长: {total_sleep_time:.2f} 分钟")
        print(f"计算的总睡眠阶段时长: {total_sleep_phases:.2f} 分钟")
        
        if total_sleep_phases > time_in_bed_minutes_total:
            if total_sleep_phases > 0:
                adjustment_factor = time_in_bed_minutes_total / total_sleep_phases
                deep_sleep_duration *= adjustment_factor
                light_sleep_duration *= adjustment_factor
                rem_sleep_duration *= adjustment_factor
                awake_duration *= adjustment_factor
                # 睡眠时长应该是调整后的深睡+浅睡+REM时长
                sleep_duration_minutes = deep_sleep_duration + light_sleep_duration + rem_sleep_duration
                print(f"调整因子: {adjustment_factor:.2f}")
                print(f"调整后的睡眠时长: {sleep_duration_minutes:.2f} 分钟")
        else:
            # 直接使用原始的睡眠时长
            sleep_duration_minutes = total_sleep_time
            print(f"使用原始睡眠时长: {sleep_duration_minutes:.2f} 分钟")
        
        # 计算各阶段占比
        if sleep_duration_minutes > 0:
            deep_sleep_ratio = (deep_sleep_duration / sleep_duration_minutes) * 100
            light_sleep_ratio = (light_sleep_duration / sleep_duration_minutes) * 100
            rem_sleep_ratio = (rem_sleep_duration / sleep_duration_minutes) * 100
            # 清醒时长占比应该基于卧床时间，而不是睡眠时长
            awake_ratio = (awake_duration / time_in_bed_minutes_total) * 100 if time_in_bed_minutes_total > 0 else 0
            print(f"深睡占比: {deep_sleep_ratio:.2f}%")
            print(f"浅睡占比: {light_sleep_ratio:.2f}%")
            print(f"REM占比: {rem_sleep_ratio:.2f}%")
            print(f"清醒占比: {awake_ratio:.2f}%")
        else:
            if time_in_bed_minutes > 0:
                awake_ratio = 100
                deep_sleep_ratio = light_sleep_ratio = rem_sleep_ratio = 0
                print("警告: 睡眠时长为0，设置清醒占比为100%")
            else:
                deep_sleep_ratio = light_sleep_ratio = rem_sleep_ratio = awake_ratio = 0
                print("警告: 睡眠时长和卧床时间都为0")
        
        # 确保各比例不超过100%
        deep_sleep_ratio = min(deep_sleep_ratio, 100)
        light_sleep_ratio = min(light_sleep_ratio, 100)
        rem_sleep_ratio = min(rem_sleep_ratio, 100)
        awake_ratio = min(awake_ratio, 100)
        
        return {
            'sleep_duration_minutes': sleep_duration_minutes,
            'deep_sleep_duration': deep_sleep_duration,
            'light_sleep_duration': light_sleep_duration,
            'rem_sleep_duration': rem_sleep_duration,
            'awake_duration': awake_duration,
            'deep_sleep_ratio': deep_sleep_ratio,
            'light_sleep_ratio': light_sleep_ratio,
            'rem_sleep_ratio': rem_sleep_ratio,
            'awake_ratio': awake_ratio
        }
    
    @staticmethod
    def _generate_summary(sleep_score: int) -> str:
        """
        生成睡眠质量总结
        
        Args:
            sleep_score: 睡眠评分
            
        Returns:
            睡眠质量总结
        """
        if sleep_score >= 80:
            return "睡眠质量优秀"
        elif sleep_score >= 60:
            return "睡眠质量良好"
        elif sleep_score >= 40:
            return "睡眠质量一般"
        else:
            return "睡眠质量较差"
    
    @classmethod
    def calculate_sleep_metrics(cls, df: pd.DataFrame, date_str: str, sleep_staging_method: str = "ensemble") -> Dict:
        """
        分析睡眠指标

        Args:
            df: 包含当日及前一天数据的DataFrame
            date_str: 目标日期字符串
            sleep_staging_method: 睡眠分期方法，可选值: "rule" (基于规则推理), "ensemble" (基于集成学习)

        Returns:
            包含睡眠分析结果的字典
        """
        print(f"\n=== 调试信息: 开始执行 calculate_sleep_metrics 方法 ===")
        print(f"目标日期: {date_str}")
        print(f"睡眠分期方法: {sleep_staging_method}")
        print(f"原始数据量: {len(df)}")
        
        try:
            logger.info(f"开始分析 {date_str} 的睡眠指标，原始数据量: {len(df)}")
            
            # 1. 准备数据
            numeric_columns = [
                'heart_rate', 'respiratory_rate', 'avg_heartbeat_interval', 
                'rms_heartbeat_interval', 'std_heartbeat_interval', 'arrhythmia_ratio', 'body_moves_ratio',
                'has_move', 'breath_amp_average', 'heart_amp_average', 'breath_freq_std',
                'heart_freq_std', 'breath_amp_diff', 'heart_amp_diff'
            ]
            
            df = cls._prepare_data(df, numeric_columns)
            
            if df.empty:
                return {
                    "error": "没有有效的生理指标数据",
                    "date": date_str
                }
            
            target_date = datetime.strptime(date_str, '%Y-%m-%d')
            prev_date = target_date - timedelta(days=1)
            
            # 2. 合并两天的数据
            target_day_data = df[df['upload_time'].dt.date == target_date.date()].copy()
            prev_day_data = df[df['upload_time'].dt.date == prev_date.date()].copy()
            combined_data = pd.concat([prev_day_data, target_day_data]).sort_values('upload_time')
            
            if combined_data.empty:
                return {
                    "error": "目标日期及前一天没有数据",
                    "date": date_str
                }
            
            logger.info(f"合并后数据量: {len(combined_data)}")
            
            # 3. 标记离床时段
            if 'is_person' in combined_data.columns:
                combined_data['is_off_bed'] = (combined_data['is_person'] == 0)
            else:
                combined_data['is_off_bed'] = (combined_data['heart_rate'] == 0) | (combined_data['heart_rate'].isna())
            
            # 4. 筛选夜间数据
            night_start = pd.Timestamp.combine(prev_date.date(), pd.Timestamp('20:00').time())
            night_end = pd.Timestamp.combine(target_date.date(), pd.Timestamp('10:00').time())
            
            night_data = combined_data[(combined_data['upload_time'] >= night_start) & 
                                      (combined_data['upload_time'] <= night_end)].copy()
            
            if night_data.empty:
                return {
                    "error": "夜间时段没有有效数据",
                    "date": date_str
                }
            
            logger.info(f"夜间数据量: {len(night_data)}")
            
            # 5. 计算基本睡眠指标
            basic_metrics = cls._calculate_basic_sleep_metrics(night_data, df, target_date, prev_date)
            
            # 6. 计算睡眠阶段（从就寝时间开始，以便找到实际的入睡时间）
            sleep_phases_data = cls._calculate_sleep_stages(
                df[(df['upload_time'] >= basic_metrics['bedtime']) & 
                   (df['upload_time'] <= basic_metrics['wakeup_time'])].copy(),
                use_trusleep=True,
                sleep_staging_method=sleep_staging_method
            )
            
            # 7. 根据睡眠阶段结果确定实际的入睡时间
            # 入睡时间定义为第一次进入稳定非清醒阶段的时间
            actual_sleep_start_time = basic_metrics['bedtime']
            
            # 打印获取的数据库数据，以便更好地理解数据情况
            print(f"\n=== 数据库数据概览 ===")
            print(f"数据总量: {len(df)} 条")
            print(f"夜间数据量: {len(night_data)} 条")
            print(f"数据时间范围: {df['upload_time'].min()} 到 {df['upload_time'].max()}")
            
            # 分析前几个小时的体动率
            print("\n=== 体动率分析 ===")
            for i in range(6):
                start_time = basic_metrics['bedtime'] + timedelta(hours=i)
                end_time = start_time + timedelta(hours=1)
                hour_data = df[(df['upload_time'] >= start_time) & (df['upload_time'] < end_time)]
                if not hour_data.empty:
                    avg_body_move = hour_data['body_moves_ratio'].mean()
                    avg_heart_rate = hour_data['heart_rate'].mean()
                    avg_resp_rate = hour_data['respiratory_rate'].mean()
                    print(f"第{i+1}小时 ({start_time.hour}:00-{end_time.hour}:00): 体动率={avg_body_move:.2f}%, 心率={avg_heart_rate:.2f}, 呼吸率={avg_resp_rate:.2f}")
            
            # 对于规则方法，直接使用 _calculate_sleep_prep_time 的结果
            if sleep_staging_method == "rule":
                # 重新计算睡眠准备时间，使用专门的方法
                sleep_period_data = df[(df['upload_time'] >= basic_metrics['bedtime']) & 
                                       (df['upload_time'] <= basic_metrics['wakeup_time'])].copy()
                print(f"\n=== 计算睡眠准备时间 ===")
                print(f"使用rule方法，分析数据: {len(sleep_period_data)} 条")
                sleep_prep_time = SleepMetricsCalculator._calculate_sleep_prep_time(sleep_period_data, basic_metrics['bedtime'])
                print(f"计算得到的睡眠准备时间: {sleep_prep_time:.2f} 分钟")
            else:
                # 对于其他方法，使用原来的逻辑
                sleep_prep_time = 0
                if sleep_phases_data['sleep_stage_segments']:
                    # 查找第一个持续时间较长的非清醒阶段
                    found_sleep_start = False
                    for i, segment in enumerate(sleep_phases_data['sleep_stage_segments']):
                        # 跳过前60分钟内的短睡眠阶段
                        if segment.get('start_time'):
                            try:
                                segment_start_time = datetime.fromisoformat(segment['start_time'].replace(' ', 'T'))
                                time_from_bedtime = (segment_start_time - basic_metrics['bedtime']).total_seconds() / 60
                                
                                # 对于前60分钟内的阶段，需要持续时间至少10分钟才算真正入睡
                                if time_from_bedtime < 60:
                                    if segment['label'] != "清醒" and int(segment['value']) >= 10:
                                        actual_sleep_start_time = segment_start_time
                                        sleep_prep_time = time_from_bedtime
                                        found_sleep_start = True
                                        break
                                else:
                                    # 60分钟后，只要是非清醒阶段就算真正入睡
                                    if segment['label'] != "清醒":
                                        actual_sleep_start_time = segment_start_time
                                        sleep_prep_time = time_from_bedtime
                                        found_sleep_start = True
                                        break
                            except:
                                pass
                    
                    # 如果没有找到合适的睡眠开始点，使用原来的逻辑
                    if not found_sleep_start:
                        for segment in sleep_phases_data['sleep_stage_segments']:
                            if segment['label'] != "清醒":
                                if segment.get('start_time'):
                                    try:
                                        actual_sleep_start_time = datetime.fromisoformat(segment['start_time'].replace(' ', 'T'))
                                        sleep_prep_time = (actual_sleep_start_time - basic_metrics['bedtime']).total_seconds() / 60
                                        break
                                    except:
                                        pass
                else:
                    # 如果没有睡眠阶段数据，使用原来的睡眠准备时间
                    sleep_prep_time = basic_metrics['sleep_prep_time_minutes']
            
            # 确保睡眠准备时间合理
            # 计算实际卧床时间
            actual_bed_time = (sleep_period_data['upload_time'].max() - sleep_period_data['upload_time'].min()).total_seconds() / 60
            # 确保睡眠准备时间不为负数且不超过实际卧床时间
            sleep_prep_time = max(0, min(sleep_prep_time, actual_bed_time))
            # 相应调整实际入睡时间
            actual_sleep_start_time = basic_metrics['bedtime'] + timedelta(minutes=sleep_prep_time)
            
            # 8. 更新基本指标中的睡眠准备时间和入睡时间
            basic_metrics['sleep_prep_time_minutes'] = sleep_prep_time
            # 同时更新实际入睡时间
            basic_metrics['actual_sleep_start_time'] = actual_sleep_start_time
            
            # 10. 添加从就寝时间到实际入睡时间的清醒阶段
            if sleep_prep_time > 0:
                # 确保 sleep_phases_data 包含必要的键
                if 'sleep_stage_segments' not in sleep_phases_data:
                    sleep_phases_data['sleep_stage_segments'] = []
                if 'awake_duration' not in sleep_phases_data:
                    sleep_phases_data['awake_duration'] = 0
                
                # 打印添加睡眠准备时间阶段前的睡眠阶段数量
                print(f"添加睡眠准备时间阶段前的睡眠阶段数量: {len(sleep_phases_data['sleep_stage_segments'])}")
                
                # 在sleep_stage_segments开头添加清醒阶段
                sleep_phases_data['sleep_stage_segments'].insert(0, {
                    "label": "清醒",
                    "value": str(int(sleep_prep_time)),
                    "start_time": basic_metrics['bedtime'].isoformat().replace('T', ' '),
                    "end_time": actual_sleep_start_time.isoformat().replace('T', ' ')
                })
                # 同时更新清醒时长
                sleep_phases_data['awake_duration'] += sleep_prep_time
                
                # 打印添加睡眠准备时间阶段后的睡眠阶段数量
                print(f"添加睡眠准备时间阶段后的睡眠阶段数量: {len(sleep_phases_data['sleep_stage_segments'])}")
                print(f"成功添加睡眠准备时间阶段: {sleep_prep_time:.2f} 分钟")
                print(f"就寝时间: {basic_metrics['bedtime']}")
                print(f"实际入睡时间: {actual_sleep_start_time}")
            
            # 9. 重新计算从实际入睡时间开始的睡眠阶段
            print(f"\n=== 调试信息: 第二次调用 _calculate_sleep_stages 方法 ===")
            print(f"实际入睡时间: {actual_sleep_start_time}")
            print(f"起床时间: {basic_metrics['wakeup_time']}")
            
            # 提取从实际入睡时间开始的数据，确保睡眠分期从准备入睡时间之后开始
            sleep_period_data = df[(df['upload_time'] >= actual_sleep_start_time) & 
                                  (df['upload_time'] <= basic_metrics['wakeup_time'])].copy()
            
            print(f"从实际入睡时间开始的数据行数: {len(sleep_period_data)}")
            if not sleep_period_data.empty:
                print(f"数据时间范围: {sleep_period_data['upload_time'].min()} 到 {sleep_period_data['upload_time'].max()}")
                print(f"平均心率: {sleep_period_data['heart_rate'].mean():.2f}")
                print(f"平均呼吸率: {sleep_period_data['respiratory_rate'].mean():.2f}")
                print(f"平均体动率: {sleep_period_data['body_moves_ratio'].mean():.4f}")
            else:
                print("警告: 从实际入睡时间开始的数据为空！")
                # 如果从实际入睡时间开始的数据为空，使用从就寝时间开始的数据
                sleep_period_data = df[(df['upload_time'] >= basic_metrics['bedtime']) & 
                                      (df['upload_time'] <= basic_metrics['wakeup_time'])].copy()
                print(f"使用从就寝时间开始的数据，数据行数: {len(sleep_period_data)}")
            
            sleep_phases_data = cls._calculate_sleep_stages(
                sleep_period_data,
                use_trusleep=True,
                sleep_staging_method=sleep_staging_method
            )
            
            print(f"\n=== 调试信息: _calculate_sleep_stages 方法返回结果 ===")
            print(f"深睡时长: {sleep_phases_data['deep_sleep_duration']} 分钟")
            print(f"浅睡时长: {sleep_phases_data['light_sleep_duration']} 分钟")
            print(f"REM时长: {sleep_phases_data['rem_sleep_duration']} 分钟")
            print(f"清醒时长: {sleep_phases_data['awake_duration']} 分钟")
            print(f"睡眠阶段数量: {len(sleep_phases_data['sleep_stage_segments'])}")
            for i, segment in enumerate(sleep_phases_data['sleep_stage_segments']):
                print(f"阶段{i+1}: {segment['label']} - {segment['value']} 分钟")
            
            # 再次添加从就寝时间到实际入睡时间的清醒阶段，确保它被添加到最终结果中
            if sleep_prep_time > 0:
                # 确保 sleep_phases_data 包含必要的键
                if 'sleep_stage_segments' not in sleep_phases_data:
                    sleep_phases_data['sleep_stage_segments'] = []
                if 'awake_duration' not in sleep_phases_data:
                    sleep_phases_data['awake_duration'] = 0
                
                # 直接在sleep_stage_segments开头添加清醒阶段，不检查是否存在
                # 这样可以确保即使重新计算睡眠阶段后，睡眠准备时间阶段仍然被添加
                sleep_phases_data['sleep_stage_segments'].insert(0, {
                    "label": "清醒",
                    "value": str(int(sleep_prep_time)),
                    "start_time": basic_metrics['bedtime'].isoformat().replace('T', ' '),
                    "end_time": actual_sleep_start_time.isoformat().replace('T', ' ')
                })
                # 同时更新清醒时长
                sleep_phases_data['awake_duration'] += sleep_prep_time
                
                # 打印添加睡眠准备时间阶段后的睡眠阶段数量
                print(f"再次添加睡眠准备时间阶段后的睡眠阶段数量: {len(sleep_phases_data['sleep_stage_segments'])}")
                print(f"再次成功添加睡眠准备时间阶段: {sleep_prep_time:.2f} 分钟")
                print(f"就寝时间: {basic_metrics['bedtime']}")
                print(f"实际入睡时间: {actual_sleep_start_time}")
                
                # 对于rule方法，打印睡眠阶段分析结果
                if sleep_staging_method == "rule":
                    # 计算从实际入睡时间到最后阶段结束的总时长
                    if len(sleep_phases_data['sleep_stage_segments']) > 1:
                        total_sleep_duration = sum(int(segment['value']) for segment in sleep_phases_data['sleep_stage_segments'][1:])  # 排除第一个清醒阶段
                    else:
                        total_sleep_duration = 0
                    # 确保总时长合理
                    if total_sleep_duration > 0:
                        print(f"\n=== Rule方法睡眠阶段分析结果 ===")
                        print(f"睡眠准备时间: {sleep_prep_time:.2f} 分钟")
                        print(f"实际入睡时间: {actual_sleep_start_time}")
                        print(f"睡眠阶段数量: {len(sleep_phases_data['sleep_stage_segments'])}")
                        print(f"总睡眠时长: {total_sleep_duration} 分钟")
                        for i, segment in enumerate(sleep_phases_data['sleep_stage_segments']):
                            print(f"阶段{i+1}: {segment['label']} - {segment['value']} 分钟")
                    else:
                        print(f"\n=== Rule方法睡眠阶段分析结果 ===")
                        print(f"睡眠准备时间: {sleep_prep_time:.2f} 分钟")
                        print(f"实际入睡时间: {actual_sleep_start_time}")
                        print(f"睡眠阶段数量: {len(sleep_phases_data['sleep_stage_segments'])}")
                        print("未检测到睡眠阶段")
            
            # 11. 计算睡眠阶段占比
            # 计算实际的总睡眠时长（深睡+浅睡+REM）
            actual_sleep_duration = sleep_phases_data['deep_sleep_duration'] + sleep_phases_data['light_sleep_duration'] + sleep_phases_data['rem_sleep_duration']
            phase_ratios = cls._calculate_sleep_phase_ratios(
                actual_sleep_duration,
                sleep_phases_data['deep_sleep_duration'],
                sleep_phases_data['light_sleep_duration'],
                sleep_phases_data['rem_sleep_duration'],
                sleep_phases_data['awake_duration'],
                basic_metrics['time_in_bed_minutes']
            )
            
            # 处理睡眠阶段截断，确保所有阶段都不超过wakeup_time
            wakeup_time = basic_metrics['wakeup_time']
            truncated_segments = []
            total_awake = 0
            total_deep = 0
            total_light = 0
            total_rem = 0
            
            for segment in sleep_phases_data['sleep_stage_segments']:
                # 解析开始和结束时间
                start_time_str = segment.get('start_time')
                end_time_str = segment.get('end_time')
                
                if start_time_str and end_time_str:
                    try:
                        start_time = datetime.fromisoformat(start_time_str.replace(' ', 'T'))
                        end_time = datetime.fromisoformat(end_time_str.replace(' ', 'T'))
                        
                        # 检查结束时间是否超过wakeup_time
                        if end_time > wakeup_time:
                            # 截断到wakeup_time
                            truncated_end_time = wakeup_time
                            # 计算截断后的持续时间
                            original_duration = int(segment.get('value', 0))
                            truncated_duration = (truncated_end_time - start_time).total_seconds() / 60
                            
                            # 更新阶段信息
                            truncated_segment = segment.copy()
                            truncated_segment['end_time'] = truncated_end_time.strftime('%Y-%m-%d %H:%M:%S')
                            truncated_segment['value'] = str(int(round(truncated_duration)))
                            
                            truncated_segments.append(truncated_segment)
                            
                            # 更新各阶段时长
                            label = segment.get('label')
                            if label == "清醒":
                                total_awake += truncated_duration
                            elif label == "深睡":
                                total_deep += truncated_duration
                            elif label == "浅睡":
                                total_light += truncated_duration
                            elif label == "眼动":
                                total_rem += truncated_duration
                        else:
                            # 结束时间未超过wakeup_time，直接添加
                            truncated_segments.append(segment)
                            
                            # 更新各阶段时长
                            label = segment.get('label')
                            duration = int(segment.get('value', 0))
                            if label == "清醒":
                                total_awake += duration
                            elif label == "深睡":
                                total_deep += duration
                            elif label == "浅睡":
                                total_light += duration
                            elif label == "眼动":
                                total_rem += duration
                    except:
                        # 解析时间出错，直接添加
                        truncated_segments.append(segment)
                else:
                    # 缺少时间信息，直接添加
                    truncated_segments.append(segment)
            
            # 更新睡眠阶段数据
            sleep_phases_data['sleep_stage_segments'] = truncated_segments
            sleep_phases_data['awake_duration'] = total_awake
            sleep_phases_data['deep_sleep_duration'] = total_deep
            sleep_phases_data['light_sleep_duration'] = total_light
            sleep_phases_data['rem_sleep_duration'] = total_rem
            
            # 重新计算睡眠阶段占比
            actual_sleep_duration = total_deep + total_light + total_rem
            phase_ratios = cls._calculate_sleep_phase_ratios(
                actual_sleep_duration,
                total_deep,
                total_light,
                total_rem,
                total_awake,
                basic_metrics['time_in_bed_minutes']
            )
            
            # 12. 构建睡眠数据字典
            sleep_data_dict = {
                "date": date_str,
                "bedtime": basic_metrics['bedtime'].strftime('%Y-%m-%d %H:%M:%S'),
                "wakeup_time": basic_metrics['wakeup_time'].strftime('%Y-%m-%d %H:%M:%S'),
                "time_in_bed_minutes": round(basic_metrics['time_in_bed_minutes'], 2),
                "sleep_duration_minutes": round(phase_ratios['sleep_duration_minutes'], 2),
                "bed_exit_count": int(basic_metrics['bed_exit_count']),
                "sleep_prep_time_minutes": round(basic_metrics['sleep_prep_time_minutes'], 2),
                "sleep_phases": {
                    "deep_sleep_minutes": round(phase_ratios['deep_sleep_duration'], 2),
                    "light_sleep_minutes": round(phase_ratios['light_sleep_duration'], 2),
                    "rem_sleep_minutes": round(phase_ratios['rem_sleep_duration'], 2),
                    "awake_minutes": round(phase_ratios['awake_duration'], 2),
                    "deep_sleep_percentage": round(phase_ratios['deep_sleep_ratio'], 2),
                    "light_sleep_percentage": round(phase_ratios['light_sleep_ratio'], 2),
                    "rem_sleep_percentage": round(phase_ratios['rem_sleep_ratio'], 2),
                    "awake_percentage": round(phase_ratios['awake_ratio'], 2)
                },
                "sleep_stage_segments": truncated_segments,
                "average_metrics": basic_metrics['avg_metrics']
            }
            
            # 13. 计算睡眠评分
            sleep_data_dict['sleep_score'] = cls.calculate_sleep_score(sleep_data_dict)
            
            # 14. 添加总结
            sleep_data_dict['summary'] = cls._generate_summary(sleep_data_dict['sleep_score'])
            
            return sleep_data_dict
            
        except Exception as e:
            logger.error(f"分析睡眠指标时出错: {str(e)}")
            return {
                "error": f"分析睡眠指标时出错: {str(e)}",
                "date": date_str
            }


def convert_numpy_types(obj):
    """
    递归转换numpy/pandas类型为原生Python类型
    
    Args:
        obj: 要转换的对象
        
    Returns:
        转换后的对象
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.api.types.is_datetime64_any_dtype(type(obj)) or isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat().replace('T', ' ') if hasattr(obj, 'isoformat') else str(obj)
    else:
        return obj


def analyze_single_day_sleep_data(date_str: str, table_name: str = "vital_signs", sleep_staging_method: str = "ensemble") -> str:
    """
    分析单日睡眠数据
    
    Args:
        date_str: 日期字符串，格式如 '2024-12-20'
        table_name: 数据库表名，默认为 "vital_signs"
        sleep_staging_method: 睡眠分期方法，可选值: "rule" (基于规则推理), "ensemble" (基于集成学习)
        
    Returns:
        JSON格式的睡眠分析结果
    """
    try:
        # 使用数据库管理器
        from src.db.database import get_db_manager
        db_manager = get_db_manager()
        
        # 获取睡眠数据
        df = db_manager.get_sleep_data_for_date_range_and_time(
            table_name,
            date_str,
            start_hour=20,  # 晚上8点
            end_hour=10     # 早上10点
        )
        
        logger.info(f"查询到 {len(df)} 条数据")
        
        if df.empty:
            logger.warning(f"数据库中没有找到 {date_str} 期间的睡眠数据")
            # 返回格式一致但数据为0的结果
            from src.utils.response_handler import SleepAnalysisResponse
            response = SleepAnalysisResponse(
                success=True,
                date=date_str,
                bedtime=f"{date_str} 00:00:00",
                wakeup_time=f"{date_str} 00:00:00",
                time_in_bed_minutes=0,
                sleep_duration_minutes=0,
                sleep_score=0,
                bed_exit_count=0,
                sleep_prep_time_minutes=0,
                sleep_phases={
                    "deep_sleep_minutes": 0,
                    "light_sleep_minutes": 0,
                    "rem_sleep_minutes": 0,
                    "awake_minutes": 0,
                    "deep_sleep_percentage": 0,
                    "light_sleep_percentage": 0,
                    "rem_sleep_percentage": 0,
                    "awake_percentage": 0
                },
                sleep_stage_segments=[],
                average_metrics={
                    "avg_heart_rate": 0,
                    "avg_respiratory_rate": 0,
                    "avg_body_moves_ratio": 0,
                    "avg_heartbeat_interval": 0,
                    "avg_rms_heartbeat_interval": 0
                },
                summary="暂无数据",
                message=f"在{date_str}期间没有找到睡眠数据"
            )
            return response.to_json()
        
        # 分析睡眠指标
        result = SleepMetricsCalculator.calculate_sleep_metrics(df, date_str, sleep_staging_method)
        
        # 转换numpy类型
        result = convert_numpy_types(result)
        
        logger.info(f"睡眠分析完成，结果: {result.get('bedtime', 'N/A')} 到 {result.get('wakeup_time', 'N/A')}")
        
        # 使用SleepAnalysisResponse类包装结果
        from src.utils.response_handler import SleepAnalysisResponse
        response = SleepAnalysisResponse(
            success=True,
            date=result.get('date', date_str),
            bedtime=result.get('bedtime', f"{date_str} 00:00:00"),
            wakeup_time=result.get('wakeup_time', f"{date_str} 00:00:00"),
            time_in_bed_minutes=result.get('time_in_bed_minutes', 0),
            sleep_duration_minutes=result.get('sleep_duration_minutes', 0),
            sleep_score=result.get('sleep_score', 0),
            bed_exit_count=result.get('bed_exit_count', 0),
            sleep_prep_time_minutes=result.get('sleep_prep_time_minutes', 0),
            sleep_phases=result.get('sleep_phases', {
                "deep_sleep_minutes": 0,
                "light_sleep_minutes": 0,
                "rem_sleep_minutes": 0,
                "awake_minutes": 0,
                "deep_sleep_percentage": 0,
                "light_sleep_percentage": 0,
                "rem_sleep_percentage": 0,
                "awake_percentage": 0
            }),
            sleep_stage_segments=result.get('sleep_stage_segments', []),
            average_metrics=result.get('average_metrics', {
                "avg_heart_rate": 0,
                "avg_respiratory_rate": 0,
                "avg_body_moves_ratio": 0,
                "avg_heartbeat_interval": 0,
                "avg_rms_heartbeat_interval": 0
            }),
            summary=result.get('summary', '分析完成')
        )
        return response.to_json()
        
    except Exception as e:
        import traceback
        error_msg = f"单日睡眠分析失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        
        from src.utils.response_handler import ApiResponse
        # 返回错误格式但保持一致的结构
        response = ApiResponse.error(
            error=str(e),
            message="睡眠分析失败",
            data={
                "date": date_str,
                "bedtime": f"{date_str} 00:00:00",
                "wakeup_time": f"{date_str} 00:00:00",
                "time_in_bed_minutes": 0,
                "sleep_duration_minutes": 0,
                "sleep_score": 0,
                "bed_exit_count": 0,
                "sleep_prep_time_minutes": 0,
                "sleep_phases": {
                    "deep_sleep_minutes": 0,
                    "light_sleep_minutes": 0,
                    "rem_sleep_minutes": 0,
                    "awake_minutes": 0,
                    "deep_sleep_percentage": 0,
                    "light_sleep_percentage": 0,
                    "rem_sleep_percentage": 0,
                    "awake_percentage": 0
                },
                "sleep_stage_segments": [],
                "average_metrics": {
                    "avg_heart_rate": 0,
                    "avg_respiratory_rate": 0,
                    "avg_body_moves_ratio": 0,
                    "avg_heartbeat_interval": 0,
                    "avg_rms_heartbeat_interval": 0
                },
                "summary": "分析失败",
                "error": str(e)
            }
        )
        return response.to_json()


def analyze_single_day_sleep_data_with_device(date_str: str, device_sn: str, table_name: str = "vital_signs", sleep_staging_method: str = "rule") -> str:
    """
    分析单日睡眠数据（带设备参数）
    
    Args:
        date_str: 日期字符串，格式如 '2024-12-20'
        device_sn: 设备序列号
        table_name: 数据库表名，默认为 "vital_signs"
        sleep_staging_method: 睡眠分期方法，可选值: "rule" (基于规则推理), "ensemble" (基于集成学习)
        
    Returns:
        JSON格式的睡眠分析结果
    """
    try:
        # 使用数据库管理器
        from src.db.database import get_db_manager
        db_manager = get_db_manager()
        
        # 获取指定设备的睡眠数据
        df = db_manager.get_sleep_data_for_date_range_and_time_with_device(
            table_name,
            date_str,
            device_sn,
            start_hour=20,  # 晚上8点
            end_hour=10     # 早上10点
        )
        
        logger.info(f"查询到 {len(df)} 条数据")
        
        if df.empty:
            logger.warning(f"数据库中没有找到 {date_str} 期间 {device_sn} 的睡眠数据")
            # 返回格式一致但数据为0的结果
            from src.utils.response_handler import SleepAnalysisResponse
            response = SleepAnalysisResponse(
                success=True,
                date=date_str,
                device_sn=device_sn,
                bedtime=f"{date_str} 00:00:00",
                wakeup_time=f"{date_str} 00:00:00",
                time_in_bed_minutes=0,
                sleep_duration_minutes=0,
                sleep_score=0,
                bed_exit_count=0,
                sleep_prep_time_minutes=0,
                sleep_phases={
                    "deep_sleep_minutes": 0,
                    "light_sleep_minutes": 0,
                    "rem_sleep_minutes": 0,
                    "awake_minutes": 0,
                    "deep_sleep_percentage": 0,
                    "light_sleep_percentage": 0,
                    "rem_sleep_percentage": 0,
                    "awake_percentage": 0
                },
                sleep_stage_segments=[],
                average_metrics={
                    "avg_heart_rate": 0,
                    "avg_respiratory_rate": 0,
                    "avg_body_moves_ratio": 0,
                    "avg_heartbeat_interval": 0,
                    "avg_rms_heartbeat_interval": 0
                },
                summary="暂无数据",
                message=f"在{date_str}期间没有找到{device_sn}的睡眠数据"
            )
            return response.to_json()
        
        # 分析睡眠指标
        result = SleepMetricsCalculator.calculate_sleep_metrics(df, date_str, sleep_staging_method)
        result['device_sn'] = device_sn
        
        # 转换numpy类型
        result = convert_numpy_types(result)
        
        logger.info(f"睡眠分析完成，结果: {result.get('bedtime', 'N/A')} 到 {result.get('wakeup_time', 'N/A')}")
        
        # 使用SleepAnalysisResponse类包装结果
        from src.utils.response_handler import SleepAnalysisResponse
        response = SleepAnalysisResponse(
            success=True,
            date=result.get('date', date_str),
            device_sn=device_sn,
            bedtime=result.get('bedtime', f"{date_str} 00:00:00"),
            wakeup_time=result.get('wakeup_time', f"{date_str} 00:00:00"),
            time_in_bed_minutes=result.get('time_in_bed_minutes', 0),
            sleep_duration_minutes=result.get('sleep_duration_minutes', 0),
            sleep_score=result.get('sleep_score', 0),
            bed_exit_count=result.get('bed_exit_count', 0),
            sleep_prep_time_minutes=result.get('sleep_prep_time_minutes', 0),
            sleep_phases=result.get('sleep_phases', {
                "deep_sleep_minutes": 0,
                "light_sleep_minutes": 0,
                "rem_sleep_minutes": 0,
                "awake_minutes": 0,
                "deep_sleep_percentage": 0,
                "light_sleep_percentage": 0,
                "rem_sleep_percentage": 0,
                "awake_percentage": 0
            }),
            sleep_stage_segments=result.get('sleep_stage_segments', []),
            average_metrics=result.get('average_metrics', {
                "avg_heart_rate": 0,
                "avg_respiratory_rate": 0,
                "avg_body_moves_ratio": 0,
                "avg_heartbeat_interval": 0,
                "avg_rms_heartbeat_interval": 0
            }),
            summary=result.get('summary', '分析完成')
        )
        return response.to_json()
        
    except Exception as e:
        import traceback
        error_msg = f"单日睡眠分析失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        
        from src.utils.response_handler import ApiResponse
        # 返回错误格式但保持一致的结构
        response = ApiResponse.error(
            error=str(e),
            message="睡眠分析失败",
            data={
                "date": date_str,
                "device_sn": device_sn,
                "bedtime": f"{date_str} 00:00:00",
                "wakeup_time": f"{date_str} 00:00:00",
                "time_in_bed_minutes": 0,
                "sleep_duration_minutes": 0,
                "sleep_score": 0,
                "bed_exit_count": 0,
                "sleep_prep_time_minutes": 0,
                "sleep_phases": {
                    "deep_sleep_minutes": 0,
                    "light_sleep_minutes": 0,
                    "rem_sleep_minutes": 0,
                    "awake_minutes": 0,
                    "deep_sleep_percentage": 0,
                    "light_sleep_percentage": 0,
                    "rem_sleep_percentage": 0,
                    "awake_percentage": 0
                },
                "sleep_stage_segments": [],
                "average_metrics": {
                    "avg_heart_rate": 0,
                    "avg_respiratory_rate": 0,
                    "avg_body_moves_ratio": 0,
                    "avg_heartbeat_interval": 0,
                    "avg_rms_heartbeat_interval": 0
                },
                "summary": "分析失败",
                "error": str(e)
            }
        )
        return response.to_json()


@tool
def analyze_sleep_by_date(date: str, runtime: object = None, table_name: str = "vital_signs", sleep_staging_method: str = "ensemble") -> str:
    """
    根据指定日期分析睡眠数据
    
    Args:
        date: 日期字符串，格式如 '2024-12-20'
        runtime: ToolRuntime 运行时上下文
        table_name: 数据库表名，默认为 "vital_signs"
        sleep_staging_method: 睡眠分期方法，可选值: "rule" (基于规则推理), "ensemble" (基于集成学习)
        
    Returns:
        JSON格式的睡眠分析结果
    """
    return analyze_single_day_sleep_data(date, table_name, sleep_staging_method)
