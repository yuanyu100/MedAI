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
from langchain.tools import tool, ToolRuntime

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
        # 清醒判定
        data['is_awake'] = (
            (data['is_morning'] & (
                (data['heart_rate'] >= baseline_heart_rate * 0.85) |
                (data['body_moves_ratio'] > 1.5) |
                (data['amp_indicator'])
            )) |
            (data['is_afternoon'] & (
                (data['heart_rate'] >= baseline_heart_rate * 0.88) |
                (data['body_moves_ratio'] > 2) |
                (data['amp_indicator'])
            )) |
            (data['is_evening'] & (
                (data['heart_rate'] >= baseline_heart_rate * 0.90) |
                (data['body_moves_ratio'] > 2.5) |
                (data['amp_indicator'])
            )) |
            ((~data['is_daytime']) & (
                (data['heart_rate'] >= baseline_heart_rate * 1.10) |
                ((data['heart_rate'] >= baseline_heart_rate * 0.95) &
                 (data['body_moves_ratio'] > 5)) |
                (data['amp_indicator'])
            ))
        )
        
        # 深睡判定
        overall_resp_mean = data['respiratory_rate'].mean()
        data['is_deep'] = (
            (data['heart_rate'] <= baseline_heart_rate * 0.90) &  # 从0.85调整到0.90，更宽松
            (data['respiratory_rate'] <= overall_resp_mean * 0.95) &  # 从0.9调整到0.95，更宽松
            (data['respiratory_stability'] < 8) &  # 从5调整到8，更宽松
            (data['body_moves_ratio'] <= 5) &  # 从3调整到5，更宽松
            (data['freq_stability']) &
            (~data['is_awake'])
        )
        
        # 计算深睡基线
        deep_sleep_data = data[data['is_deep']]
        actual_deep_hr = deep_sleep_data['heart_rate'].mean() if not deep_sleep_data.empty else baseline_heart_rate * 0.75
        
        # REM判定
        REM_HR_MULTIPLIER_LOW = 1.10
        REM_HR_MULTIPLIER_HIGH = 1.40
        REM_RESP_STABILITY_LOW = 10
        REM_BODY_MOVES_MAX = 3
        
        data['is_rem'] = (
            (~data['is_awake']) &
            (~data['is_deep']) &
            (data['heart_rate'] >= actual_deep_hr * REM_HR_MULTIPLIER_LOW) &
            (data['heart_rate'] <= actual_deep_hr * REM_HR_MULTIPLIER_HIGH) &
            (data['respiratory_stability'] > REM_RESP_STABILITY_LOW) &
            (data['body_moves_ratio'] <= REM_BODY_MOVES_MAX) &
            (data['respiratory_rate'] >= 8) &
            (data['respiratory_rate'] <= 20) &
            (data['amp_diff_indicator']) &
            (~data['freq_stability'])
        )
        
        # 浅睡判定
        data['is_light'] = (
            (~data['is_awake']) &
            (~data['is_deep']) &
            (~data['is_rem']) &
            (data['body_moves_ratio'] <= 10) &
            (data['heart_rate'] <= baseline_heart_rate * 0.95)
        )
        
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
        # 连续验证
        CONTINUOUS_MINUTES = 2
        for stage in ['is_awake', 'is_deep', 'is_rem', 'is_light']:
            data[f'{stage}_continuous'] = (
                data[stage].rolling(window=CONTINUOUS_MINUTES, min_periods=1).sum() == CONTINUOUS_MINUTES
            )
        
        # 初始化阶段值和标签
        data['stage_value'] = SleepStageAnalyzer.STAGE_LIGHT  # 默认为浅睡
        data['stage_label'] = SleepStageAnalyzer.STAGE_LABELS[SleepStageAnalyzer.STAGE_LIGHT]
        
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
        try:
            # 1. 准备数据
            processed_data = cls._prepare_sleep_data(sleep_data)
            
            # 2. 计算时间特征
            processed_data = cls._calculate_time_based_features(processed_data)
            
            # 3. 计算生理特征
            processed_data = cls._calculate_physiological_features(processed_data)
            
            # 4. 计算睡眠阶段
            processed_data = cls._calculate_sleep_stages(processed_data, baseline_heart_rate)
            
            # 5. 验证连续阶段
            processed_data = cls._validate_continuous_stages(processed_data)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"计算睡眠阶段时出错: {str(e)}")
            # 返回原始数据，添加错误标记
            error_data = sleep_data.copy()
            error_data['stage_value'] = 0
            error_data['stage_label'] = "错误"
            return error_data
    
    @staticmethod
    def smooth_sleep_stages(stages_sequence: List[Dict], min_duration_threshold: int = 3) -> List[Dict]:
        """
        平滑睡眠阶段序列，减少碎片化
        
        Args:
            stages_sequence: 睡眠阶段序列
            min_duration_threshold: 最小持续时间阈值
            
        Returns:
            平滑后的睡眠阶段序列
        """
        if not stages_sequence:
            return stages_sequence
        
        try:
            # 合并相邻的相同阶段
            merged_same_stages = []
            i = 0
            while i < len(stages_sequence):
                current = stages_sequence[i].copy()
                j = i + 1
                
                while j < len(stages_sequence) and stages_sequence[j]['stage_value'] == current['stage_value']:
                    current['time_interval'] += stages_sequence[j]['time_interval']
                    j += 1
                
                merged_same_stages.append(current)
                i = j
            
            # 移除或合并短持续时间的阶段
            if not merged_same_stages:
                return merged_same_stages
            
            result = [merged_same_stages[0]]
            
            i = 1
            while i < len(merged_same_stages):
                current = merged_same_stages[i]
                
                if current['time_interval'] < min_duration_threshold:
                    # 当前阶段太短，合并到前一个阶段
                    result[-1]['time_interval'] += current['time_interval']
                else:
                    # 当前阶段足够长，添加到结果中
                    result.append(current)
                
                i += 1
            
            return result
            
        except Exception as e:
            logger.error(f"平滑睡眠阶段时出错: {str(e)}")
            return stages_sequence


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
        
        # 判断睡眠状态
        sleep_period_data['is_sleeping'] = (
            (sleep_period_data['heart_rate'] >= SleepMetricsCalculator.MIN_HEART_RATE) &
            (sleep_period_data['heart_rate'] <= 70) &
            (sleep_period_data['respiratory_rate'] >= 12) &
            (sleep_period_data['respiratory_rate'] <= 18) &
            (sleep_period_data['body_moves_ratio'] <= 10)
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
        if sleep_period_data.empty:
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
        
        # 找到第一个稳定睡眠位置
        stable_mask = (
            (sleep_period_data['hr_std_5'] <= SleepMetricsCalculator.STABLE_HR_STD_THRESHOLD) & 
            (sleep_period_data['arrhythmia_avg_5'] <= SleepMetricsCalculator.STABLE_ARRHYTHMIA_THRESHOLD)
        )
        stable_indices = sleep_period_data[stable_mask].index
        
        if stable_indices.empty:
            return SleepMetricsCalculator.DEFAULT_SLEEP_PREP_TIME
        
        first_stable_idx = stable_indices[0]
        stable_sleep_start = sleep_period_data.loc[first_stable_idx, 'upload_time']
        sleep_prep_time = (stable_sleep_start - bedtime).total_seconds() / 60
        return max(sleep_prep_time, SleepMetricsCalculator.MIN_SLEEP_PREP_TIME)
    
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
    def _calculate_sleep_stages(cls, sleep_period_data: pd.DataFrame) -> Dict:
        """
        计算睡眠阶段
        
        Args:
            sleep_period_data: 睡眠时段数据
            
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
            sleep_data = sleep_period_data[
                (sleep_period_data['heart_rate'] >= SleepMetricsCalculator.MIN_HEART_RATE) &
                (sleep_period_data['heart_rate'] <= SleepMetricsCalculator.MAX_HEART_RATE) &
                (sleep_period_data['respiratory_rate'] >= SleepMetricsCalculator.MIN_RESPIRATORY_RATE) &
                (sleep_period_data['respiratory_rate'] <= SleepMetricsCalculator.MAX_RESPIRATORY_RATE) &
                (sleep_period_data['heart_rate'].notna()) &
                (sleep_period_data['respiratory_rate'].notna())
            ].copy()
            
            if sleep_data.empty:
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
            sleep_data = SleepStageAnalyzer.calculate_optimized_sleep_stages(
                sleep_data, baseline_heart_rate
            )
            
            # 按优先级分配阶段
            awake_mask = sleep_data['is_awake_continuous']
            sleep_data.loc[awake_mask, ['stage_value', 'stage_label']] = [4, "清醒"]
            
            deep_mask = (~awake_mask) & sleep_data['is_deep_continuous']
            sleep_data.loc[deep_mask, ['stage_value', 'stage_label']] = [1, "深睡"]
            
            rem_mask = (~awake_mask) & (~deep_mask) & sleep_data['is_rem_continuous']
            sleep_data.loc[rem_mask, ['stage_value', 'stage_label']] = [3, "眼动"]
            
            light_mask = (~awake_mask) & (~deep_mask) & (~rem_mask)
            sleep_data.loc[light_mask, ['stage_value', 'stage_label']] = [2, "浅睡"]
            
            # 计算时间间隔
            time_diffs = sleep_data['upload_time'].diff().dt.total_seconds() / 60
            time_intervals = time_diffs.fillna(0).clip(lower=1)
            if len(time_intervals) > 1:
                time_intervals.iloc[0] = time_intervals.iloc[1]
            else:
                time_intervals.iloc[0] = 1
            
            # 生成阶段序列
            stages_sequence = [
                {
                    'stage_value': int(row['stage_value']),
                    'stage_label': row['stage_label'],
                    'time': row['upload_time'],
                    'time_interval': time_intervals.iloc[idx] if idx < len(time_intervals) else 1
                }
                for idx, row in sleep_data.iterrows()
            ]
            
            # 平滑处理
            smoothed_stages = SleepStageAnalyzer.smooth_sleep_stages(stages_sequence, min_duration_threshold=3)
            
            # 计算各阶段时长和生成阶段片段
            cls._process_sleep_stages(smoothed_stages, result)
            
        except Exception as e:
            logger.error(f"计算睡眠阶段时出错: {str(e)}")
        
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
                        "value": str(int(current_stage_duration)),
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
                "value": str(int(current_stage_duration)),
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
        
        if total_sleep_phases > time_in_bed_minutes_total:
            if total_sleep_phases > 0:
                adjustment_factor = time_in_bed_minutes_total / total_sleep_phases
                deep_sleep_duration *= adjustment_factor
                light_sleep_duration *= adjustment_factor
                rem_sleep_duration *= adjustment_factor
                awake_duration *= adjustment_factor
                # 睡眠时长应该是调整后的深睡+浅睡+REM时长
                sleep_duration_minutes = deep_sleep_duration + light_sleep_duration + rem_sleep_duration
        else:
            # 直接使用原始的睡眠时长
            sleep_duration_minutes = total_sleep_time
        
        # 计算各阶段占比
        if sleep_duration_minutes > 0:
            deep_sleep_ratio = (deep_sleep_duration / sleep_duration_minutes) * 100
            light_sleep_ratio = (light_sleep_duration / sleep_duration_minutes) * 100
            rem_sleep_ratio = (rem_sleep_duration / sleep_duration_minutes) * 100
            # 清醒时长占比应该基于卧床时间，而不是睡眠时长
            awake_ratio = (awake_duration / time_in_bed_minutes_total) * 100 if time_in_bed_minutes_total > 0 else 0
        else:
            if time_in_bed_minutes > 0:
                awake_ratio = 100
                deep_sleep_ratio = light_sleep_ratio = rem_sleep_ratio = 0
            else:
                deep_sleep_ratio = light_sleep_ratio = rem_sleep_ratio = awake_ratio = 0
        
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
    def calculate_sleep_metrics(cls, df: pd.DataFrame, date_str: str) -> Dict:
        """
        分析睡眠指标
        
        Args:
            df: 包含当日及前一天数据的DataFrame
            date_str: 目标日期字符串
            
        Returns:
            包含睡眠分析结果的字典
        """
        try:
            logger.info(f"开始分析 {date_str} 的睡眠指标，原始数据量: {len(df)}")
            
            # 1. 准备数据
            numeric_columns = [
                'heart_rate', 'respiratory_rate', 'avg_heartbeat_interval', 
                'rms_heartbeat_interval', 'std_heartbeat_interval', 'arrhythmia_ratio', 'body_moves_ratio'
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
            
            # 计算入睡时间（就寝时间 + 睡眠准备时间）
            sleep_prep_time = basic_metrics['sleep_prep_time_minutes']
            sleep_start_time = basic_metrics['bedtime'] + timedelta(minutes=sleep_prep_time)
            
            # 6. 计算睡眠阶段（从入睡时间开始）
            sleep_phases_data = cls._calculate_sleep_stages(
                df[(df['upload_time'] >= sleep_start_time) & 
                   (df['upload_time'] <= basic_metrics['wakeup_time'])].copy()
            )
            
            # 添加从就寝时间到入睡时间的清醒阶段
            if sleep_prep_time > 0:
                # 确定清醒阶段的结束时间
                # 如果有睡眠阶段数据，使用第一个睡眠阶段的开始时间作为清醒阶段的结束时间
                if sleep_phases_data['sleep_stage_segments']:
                    first_stage = sleep_phases_data['sleep_stage_segments'][0]
                    if first_stage.get('start_time'):
                        awake_end_time_str = first_stage['start_time']
                    else:
                        awake_end_time_str = sleep_start_time.isoformat().replace('T', ' ')
                else:
                    awake_end_time_str = sleep_start_time.isoformat().replace('T', ' ')
                
                # 在sleep_stage_segments开头添加清醒阶段
                sleep_phases_data['sleep_stage_segments'].insert(0, {
                    "label": "清醒",
                    "value": str(int(sleep_prep_time)),
                    "start_time": basic_metrics['bedtime'].isoformat().replace('T', ' '),
                    "end_time": awake_end_time_str
                })
                # 同时更新清醒时长
                sleep_phases_data['awake_duration'] += sleep_prep_time
            
            # 7. 计算睡眠阶段占比
            phase_ratios = cls._calculate_sleep_phase_ratios(
                basic_metrics['sleep_duration_minutes'],
                sleep_phases_data['deep_sleep_duration'],
                sleep_phases_data['light_sleep_duration'],
                sleep_phases_data['rem_sleep_duration'],
                sleep_phases_data['awake_duration'],
                basic_metrics['time_in_bed_minutes']
            )
            
            # 8. 构建睡眠数据字典
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
                "sleep_stage_segments": sleep_phases_data['sleep_stage_segments'],
                "average_metrics": basic_metrics['avg_metrics']
            }
            
            # 9. 计算睡眠评分
            sleep_data_dict['sleep_score'] = cls.calculate_sleep_score(sleep_data_dict)
            
            # 10. 添加总结
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


def analyze_single_day_sleep_data(date_str: str, table_name: str = "vital_signs") -> str:
    """
    分析单日睡眠数据
    
    Args:
        date_str: 日期字符串，格式如 '2024-12-20'
        table_name: 数据库表名，默认为 "vital_signs"
        
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
        result = SleepMetricsCalculator.calculate_sleep_metrics(df, date_str)
        
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


def analyze_single_day_sleep_data_with_device(date_str: str, device_sn: str, table_name: str = "vital_signs") -> str:
    """
    分析单日睡眠数据（带设备参数）
    
    Args:
        date_str: 日期字符串，格式如 '2024-12-20'
        device_sn: 设备序列号
        table_name: 数据库表名，默认为 "vital_signs"
        
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
        result = SleepMetricsCalculator.calculate_sleep_metrics(df, date_str)
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
def analyze_sleep_by_date(date: str, runtime: ToolRuntime = None, table_name: str = "vital_signs") -> str:
    """
    根据指定日期分析睡眠数据
    
    Args:
        date: 日期字符串，格式如 '2024-12-20'
        runtime: ToolRuntime 运行时上下文
        table_name: 数据库表名，默认为 "vital_signs"
        
    Returns:
        JSON格式的睡眠分析结果
    """
    return analyze_single_day_sleep_data(date, table_name)
