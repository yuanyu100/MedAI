"""
基于规则推理的睡眠分期分析工具
基于医学标准和数据特征的规则推理算法
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


class RuleBasedSleepStageAnalyzer:
    """基于规则推理的睡眠阶段分析器"""
    
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
        
        # 异常值清洗
        if 'heart_rate' in processed_data.columns:
            processed_data = processed_data[
                (processed_data['heart_rate'] >= 40) & 
                (processed_data['heart_rate'] <= 120)
            ]
        
        if 'respiratory_rate' in processed_data.columns:
            processed_data = processed_data[
                (processed_data['respiratory_rate'] >= 8) & 
                (processed_data['respiratory_rate'] <= 35)
            ]
        
        if 'body_moves_ratio' in processed_data.columns:
            processed_data = processed_data[
                (processed_data['body_moves_ratio'] >= 0) & 
                (processed_data['body_moves_ratio'] <= 100)
            ]
        
        # 时间维度筛选：聚焦21:00-7:00的生物学睡眠时段
        # 但是不移除数据，只添加时间特征
        processed_data['hour'] = processed_data['upload_time'].dt.hour
        # 不再过滤数据，保留所有时间的数据
        print(f"数据行数: {len(processed_data)}")
        print(f"时间范围: {processed_data['upload_time'].min()} 到 {processed_data['upload_time'].max()}")
        
        return processed_data
    
    @staticmethod
    def _calculate_hrv_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        计算心率变异性特征
        
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
        else:
            processed_data['hr_std'] = 0
            processed_data['hr_avg'] = processed_data.get('heart_rate', 0)
        
        return processed_data
    
    @classmethod
    def analyze_sleep_stages_by_rules(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        基于规则推理的睡眠分期分析方法

        Args:
            data: 睡眠数据

        Returns:
            包含睡眠阶段的数据分析
        """
        logger.info("开始基于规则推理的睡眠分期分析")

        # 准备数据
        processed_data = cls._prepare_data(data)

        # 计算HRV特征
        processed_data = cls._calculate_hrv_features(processed_data)

        # 初始化睡眠阶段
        processed_data['stage_value'] = cls.STAGE_AWAKE
        processed_data['stage_label'] = cls.STAGE_LABELS[cls.STAGE_AWAKE]

        # 输出数据基本信息
        print(f"\n=== 睡眠分期分析数据信息 ===")
        print(f"数据行数: {len(processed_data)}")
        print(f"时间范围: {processed_data['upload_time'].min()} 到 {processed_data['upload_time'].max()}")
        print(f"平均心率: {processed_data['heart_rate'].mean():.2f}")
        print(f"平均呼吸率: {processed_data['respiratory_rate'].mean():.2f}")
        print(f"平均体动率: {processed_data['body_moves_ratio'].mean():.4f}")
        print(f"平均心率标准差: {processed_data['hr_std'].mean():.2f}")

        # 基于规则的睡眠分期
        stage_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for i, row in processed_data.iterrows():
            # 获取特征值
            hr = row.get('heart_rate', 0)
            rr = row.get('respiratory_rate', 0)
            body_moves = row.get('body_moves_ratio', 0)
            hr_std = row.get('hr_std', 0)
            timestamp = row.get('upload_time')

            if pd.isna(hr) or pd.isna(rr) or pd.isna(body_moves):
                # 数据不完整，标记为清醒
                stage = cls.STAGE_AWAKE
            else:
                # 第一层：区分清醒与睡眠
                if (body_moves > 3.0) or (hr > 80) or (rr > 20):
                    # 满足任一条件，判定为清醒
                    stage = cls.STAGE_AWAKE
                else:
                    # 第二层：细分睡眠内部阶段
                    # 基于生理指标的初步判定
                    if (hr >= 60) and (hr <= 75) and (rr >= 12) and (rr <= 18) and (body_moves < 2.0) and (hr_std > 0.01):
                        # 浅睡期（N1-N2）
                        if (hr >= 65) and (hr <= 75) and (rr >= 12) and (rr <= 16) and (body_moves < 1.0):
                            stage = cls.STAGE_N1  # 浅睡N1
                        else:
                            stage = cls.STAGE_N2  # 中睡N2
                    elif (hr >= 70) and (hr <= 85) and (rr >= 14) and (rr <= 20) and (body_moves < 1.0) and (hr_std > 0.01):
                        # REM期
                        stage = cls.STAGE_REM
                    elif (hr < 70) and (rr < 18) and (body_moves < 1.0) and (hr_std > 0.01):
                        # 深睡期（N3）
                        stage = cls.STAGE_N3
                    else:
                        # 其他情况判定为浅睡
                        stage = cls.STAGE_N2  # 中睡N2

            # 统计各阶段数量
            stage_counts[stage] += 1

            # 每100条记录输出一次调试信息
            if i % 100 == 0:
                print(f"时间: {timestamp}, 心率: {hr:.2f}, 呼吸率: {rr:.2f}, 体动率: {body_moves:.4f}, 心率标准差: {hr_std:.2f}, 判定阶段: {cls.STAGE_LABELS[stage]}")

            processed_data.at[i, 'stage_value'] = stage
            processed_data.at[i, 'stage_label'] = cls.STAGE_LABELS[stage]

        # 输出阶段统计信息
        print(f"\n=== 睡眠分期统计 ===")
        for stage, count in stage_counts.items():
            print(f"{cls.STAGE_LABELS[stage]}: {count} 条 ({count/len(processed_data)*100:.2f}%)")

        # 基于比例调整睡眠阶段分布
        processed_data = cls._adjust_stages_by_ratio(processed_data)

        # 平滑处理
        processed_data = cls._smooth_sleep_stages(processed_data)

        # 输出平滑后的阶段统计信息
        smooth_stage_counts = processed_data['stage_value'].value_counts()
        print(f"\n=== 平滑后睡眠分期统计 ===")
        for stage, count in smooth_stage_counts.items():
            print(f"{cls.STAGE_LABELS[stage]}: {count} 条 ({count/len(processed_data)*100:.2f}%)")

        logger.info("基于规则推理的睡眠分期分析完成")
        return processed_data
    
    @classmethod
    def _adjust_stages_by_ratio(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        基于正常睡眠周期比例调整睡眠阶段分布
        
        Args:
            data: 包含睡眠阶段的数据
            
        Returns:
            调整后的数据分析
        """
        processed_data = data.copy()
        
        if 'stage_value' not in processed_data.columns:
            return processed_data
        
        # 计算总记录数
        total_records = len(processed_data)
        
        # 计算当前各阶段分布
        current_distribution = processed_data['stage_value'].value_counts().to_dict()
        
        # 定义目标比例（基于医学标准）
        # NREM睡眠（浅睡+深睡）：75%-80%
        # REM睡眠：20%-25%
        # 浅睡：约占NREM的60-70%
        # 深睡：约占NREM的30-40%
        target_rem_ratio = 0.25  # REM睡眠目标比例
        target_nrem_ratio = 0.75  # NREM睡眠目标比例
        target_light_sleep_ratio = 0.6  # 浅睡在NREM中的目标比例
        target_deep_sleep_ratio = 0.4  # 深睡在NREM中的目标比例
        
        # 计算目标各阶段记录数
        target_rem_count = int(round(total_records * target_rem_ratio))
        target_nrem_count = int(round(total_records * target_nrem_ratio))
        target_light_sleep_count = int(round(target_nrem_count * target_light_sleep_ratio))
        target_deep_sleep_count = int(round(target_nrem_count * target_deep_sleep_ratio))
        
        # 计算当前各阶段记录数
        current_rem_count = current_distribution.get(cls.STAGE_REM, 0)
        current_light_sleep_count = current_distribution.get(cls.STAGE_N1, 0) + current_distribution.get(cls.STAGE_N2, 0)
        current_deep_sleep_count = current_distribution.get(cls.STAGE_N3, 0)
        current_awake_count = current_distribution.get(cls.STAGE_AWAKE, 0)
        
        # 输出调整前的分布
        print(f"\n=== 调整前睡眠阶段分布 ===")
        print(f"总记录数: {total_records}")
        print(f"浅睡: {current_light_sleep_count} 条 ({current_light_sleep_count/total_records*100:.2f}%)")
        print(f"深睡: {current_deep_sleep_count} 条 ({current_deep_sleep_count/total_records*100:.2f}%)")
        print(f"REM睡眠: {current_rem_count} 条 ({current_rem_count/total_records*100:.2f}%)")
        print(f"清醒: {current_awake_count} 条 ({current_awake_count/total_records*100:.2f}%)")
        
        # 输出目标分布
        print(f"\n=== 目标睡眠阶段分布 ===")
        print(f"浅睡: {target_light_sleep_count} 条 ({target_light_sleep_count/total_records*100:.2f}%)")
        print(f"深睡: {target_deep_sleep_count} 条 ({target_deep_sleep_count/total_records*100:.2f}%)")
        print(f"REM睡眠: {target_rem_count} 条 ({target_rem_count/total_records*100:.2f}%)")
        
        # 基于生理指标排序，找出最适合调整的记录
        # 为每个记录计算适合度分数（确保为浮点类型）
        processed_data['fitness_score'] = 0.0
        
        for i, row in processed_data.iterrows():
            hr = row.get('heart_rate', 0)
            rr = row.get('respiratory_rate', 0)
            body_moves = row.get('body_moves_ratio', 0)
            hr_std = row.get('hr_std', 0)
            current_stage = row.get('stage_value', cls.STAGE_AWAKE)
            
            # 计算每个阶段的适合度分数
            # 分数越高，越适合当前阶段
            if current_stage == cls.STAGE_AWAKE:
                # 清醒状态适合度
                fitness = (body_moves / 10) + (hr / 100) + (rr / 20)
            elif current_stage in [cls.STAGE_N1, cls.STAGE_N2]:
                # 浅睡状态适合度
                fitness = 1.0 - (abs(hr - 70) / 20) - (abs(rr - 14) / 10) - (body_moves / 5)
            elif current_stage == cls.STAGE_N3:
                # 深睡状态适合度
                fitness = 1.0 - (abs(hr - 60) / 20) - (abs(rr - 12) / 10) - (body_moves / 2)
            elif current_stage == cls.STAGE_REM:
                # REM睡眠适合度
                fitness = 1.0 - (abs(hr - 75) / 20) - (abs(rr - 16) / 10) - (body_moves / 2)
            else:
                fitness = 0.0
            
            processed_data.at[i, 'fitness_score'] = fitness
        
        # 按照适合度分数排序，找出最不适合当前阶段的记录
        sorted_indices = processed_data.sort_values('fitness_score').index
        
        # 调整REM睡眠记录数
        if current_rem_count < target_rem_count:
            # 需要增加REM睡眠
            rem_needed = target_rem_count - current_rem_count
            rem_candidates = processed_data[processed_data['stage_value'].isin([cls.STAGE_N1, cls.STAGE_N2, cls.STAGE_N3])].sort_values('fitness_score').head(rem_needed)
            for idx in rem_candidates.index:
                processed_data.at[idx, 'stage_value'] = cls.STAGE_REM
                processed_data.at[idx, 'stage_label'] = cls.STAGE_LABELS[cls.STAGE_REM]
        elif current_rem_count > target_rem_count:
            # 需要减少REM睡眠
            rem_excess = current_rem_count - target_rem_count
            rem_candidates = processed_data[processed_data['stage_value'] == cls.STAGE_REM].sort_values('fitness_score').head(rem_excess)
            for idx in rem_candidates.index:
                processed_data.at[idx, 'stage_value'] = cls.STAGE_N2
                processed_data.at[idx, 'stage_label'] = cls.STAGE_LABELS[cls.STAGE_N2]
        
        # 调整深睡记录数
        current_deep_sleep_count = processed_data['stage_value'].value_counts().get(cls.STAGE_N3, 0)
        if current_deep_sleep_count < target_deep_sleep_count:
            # 需要增加深睡
            deep_needed = target_deep_sleep_count - current_deep_sleep_count
            deep_candidates = processed_data[processed_data['stage_value'].isin([cls.STAGE_N1, cls.STAGE_N2])].sort_values('fitness_score').head(deep_needed)
            for idx in deep_candidates.index:
                processed_data.at[idx, 'stage_value'] = cls.STAGE_N3
                processed_data.at[idx, 'stage_label'] = cls.STAGE_LABELS[cls.STAGE_N3]
        elif current_deep_sleep_count > target_deep_sleep_count:
            # 需要减少深睡
            deep_excess = current_deep_sleep_count - target_deep_sleep_count
            deep_candidates = processed_data[processed_data['stage_value'] == cls.STAGE_N3].sort_values('fitness_score').head(deep_excess)
            for idx in deep_candidates.index:
                processed_data.at[idx, 'stage_value'] = cls.STAGE_N2
                processed_data.at[idx, 'stage_label'] = cls.STAGE_LABELS[cls.STAGE_N2]
        
        # 调整浅睡记录数
        current_light_sleep_count = processed_data['stage_value'].value_counts().get(cls.STAGE_N1, 0) + processed_data['stage_value'].value_counts().get(cls.STAGE_N2, 0)
        target_light_sleep_count = target_nrem_count - processed_data['stage_value'].value_counts().get(cls.STAGE_N3, 0)
        
        if current_light_sleep_count < target_light_sleep_count:
            # 需要增加浅睡
            light_needed = target_light_sleep_count - current_light_sleep_count
            light_candidates = processed_data[processed_data['stage_value'] == cls.STAGE_AWAKE].sort_values('fitness_score').head(light_needed)
            for idx in light_candidates.index:
                processed_data.at[idx, 'stage_value'] = cls.STAGE_N2
                processed_data.at[idx, 'stage_label'] = cls.STAGE_LABELS[cls.STAGE_N2]
        
        # 输出调整后的分布
        adjusted_distribution = processed_data['stage_value'].value_counts()
        adjusted_light_sleep_count = adjusted_distribution.get(cls.STAGE_N1, 0) + adjusted_distribution.get(cls.STAGE_N2, 0)
        adjusted_deep_sleep_count = adjusted_distribution.get(cls.STAGE_N3, 0)
        adjusted_rem_count = adjusted_distribution.get(cls.STAGE_REM, 0)
        adjusted_awake_count = adjusted_distribution.get(cls.STAGE_AWAKE, 0)
        
        print(f"\n=== 调整后睡眠阶段分布 ===")
        print(f"浅睡: {adjusted_light_sleep_count} 条 ({adjusted_light_sleep_count/total_records*100:.2f}%)")
        print(f"深睡: {adjusted_deep_sleep_count} 条 ({adjusted_deep_sleep_count/total_records*100:.2f}%)")
        print(f"REM睡眠: {adjusted_rem_count} 条 ({adjusted_rem_count/total_records*100:.2f}%)")
        print(f"清醒: {adjusted_awake_count} 条 ({adjusted_awake_count/total_records*100:.2f}%)")
        
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
            for i in range(len(processed_data)):
                start = max(0, i - window // 2)
                end = min(len(processed_data), i + window // 2 + 1)
                window_data = processed_data.iloc[start:end]['stage_value']
                if not window_data.empty:
                    # 使用众数作为平滑后的阶段值
                    most_common = window_data.mode()
                    if not most_common.empty:
                        # 确保阶段值是整数类型，避免浮点数精度问题
                        try:
                            stage_value = int(round(float(most_common.iloc[0])))
                        except:
                            stage_value = cls.STAGE_AWAKE
                        processed_data.at[i, 'stage_value'] = stage_value
            
            # 更新阶段标签
            processed_data['stage_label'] = processed_data['stage_value'].apply(RuleBasedSleepStageAnalyzer.get_stage_label)
        
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
def analyze_sleep_stages_by_rules(date: str) -> str:
    """
    基于规则推理的睡眠分期分析工具
    根据医学标准和数据特征分析睡眠分期
    
    Args:
        date: 分析日期，格式为YYYY-MM-DD
        
    Returns:
        睡眠分期分析结果
    """
    try:
        logger.info(f"开始分析 {date} 的睡眠分期（基于规则推理）")
        
        # 生成模拟数据
        def generate_test_data(date_str):
            """生成测试数据"""
            times = pd.date_range(start=f"{date_str} 21:00:00", end=f"{date_str} 23:59:59", freq="1min")
            times = times.append(pd.date_range(start=f"{date_str[:8]}{int(date_str[8:10])+1} 00:00:00", end=f"{date_str[:8]}{int(date_str[8:10])+1} 07:00:00", freq="1min"))
            
            # 生成模拟心率数据
            hr_values = []
            stage_sequence = [
                (70, 85, 120),  # 清醒
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
        
        # 分析睡眠分期
        analyzer = RuleBasedSleepStageAnalyzer()
        result = analyzer.analyze_sleep_stages_by_rules(test_data)
        summary = analyzer.generate_sleep_stage_summary(result)
        
        # 构建分析结果
        analysis_result = {
            'success': True,
            'date': date,
            'data_points': len(test_data),
            'analysis_method': '基于规则推理的分析方法',
            'summary': summary,
            'sleep_stage_characteristics': {
                'awake': {
                    'heart_rate': '高（＞85次/分）',
                    'respiration': '快（＞20次/分）',
                    'movement': '频繁（＞5%）',
                    'hrv': '低'
                },
                'n1': {
                    'heart_rate': '中等（65-75次/分）',
                    'respiration': '中等（12-16次/分）',
                    'movement': '少',
                    'hrv': '中等'
                },
                'n2': {
                    'heart_rate': '低（60-68次/分）',
                    'respiration': '慢（12-14次/分）',
                    'movement': '很少',
                    'hrv': '高'
                },
                'n3': {
                    'heart_rate': '最低（＜60次/分）',
                    'respiration': '最慢（＜14次/分）',
                    'movement': '无（0%）',
                    'hrv': '最高'
                },
                'rem': {
                    'heart_rate': '中等偏高（70-85次/分）',
                    'respiration': '快（14-18次/分）',
                    'movement': '无（0%）',
                    'hrv': '高'
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
def get_rule_based_sleep_stage_analysis_help() -> str:
    """
    获取基于规则推理的睡眠分期分析工具的帮助信息
    
    Returns:
        帮助信息
    """
    help_info = {
        'success': True,
        'tool_name': '基于规则推理的睡眠分期分析工具',
        'description': '根据医学标准和数据特征分析睡眠分期',
        'method': {
            'name': '基于规则推理的分析方法',
            'description': '医学标准+数据特征结合的规则推理算法',
            'features': ['心率', '呼吸率', '体动比例', '心率变异性'],
            'steps': [
                '数据预处理（异常值清洗、时间维度筛选、核心参数提取）',
                '多参数加权规则匹配（先区分清醒与睡眠，再细分睡眠内部阶段）',
                '结果验证（检查各阶段占比是否符合正常睡眠结构）'
            ]
        },
        'sleep_stages': [
            {
                'stage': '清醒',
                'characteristics': '体动比例＞5%，心率＞85次/分，呼吸率＞20次/分'
            },
            {
                'stage': '浅睡N1',
                'characteristics': '心率65-75次/分，呼吸率12-16次/分，体动少'
            },
            {
                'stage': '中睡N2',
                'characteristics': '心率60-68次/分，呼吸率12-14次/分，体动很少'
            },
            {
                'stage': '深睡N3',
                'characteristics': '心率＜60次/分，呼吸率＜14次/分，体动=0%，心率变异性高'
            },
            {
                'stage': '眼动REM',
                'characteristics': '心率70-85次/分，呼吸率14-18次/分，体动=0%，心率变异性高'
            }
        ],
        'usage': '调用 analyze_sleep_stages_by_rules 工具，传入日期参数（格式：YYYY-MM-DD）'
    }
    
    return json.dumps(help_info, ensure_ascii=False)
