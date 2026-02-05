# """
# 高级睡眠分期分析工具
# 基于多维度生理特征和先进机器学习算法的高精度睡眠分期分析
# """

# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import json
# import logging
# from typing import Dict, List, Tuple, Optional
# from langchain_community.tools import tool
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# import xgboost as xgb
# from scipy import signal
# from scipy.stats import entropy
# from sklearn.neural_network import MLPClassifier

# # 配置日志
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class AdvancedSleepStageAnalyzer:
#     """高级睡眠阶段分析器"""
    
#     # 睡眠阶段常量
#     STAGE_AWAKE = 0
#     STAGE_N1 = 1
#     STAGE_N2 = 2
#     STAGE_N3 = 3
#     STAGE_REM = 4
    
#     # 睡眠阶段标签映射
#     STAGE_LABELS = {
#         STAGE_AWAKE: "清醒",
#         STAGE_N1: "浅睡N1", 
#         STAGE_N2: "中睡N2",
#         STAGE_N3: "深睡N3",
#         STAGE_REM: "眼动REM"
#     }
    
#     @classmethod
#     def get_stage_label(cls, stage_value: int) -> str:
#         """
#         根据阶段值返回对应的标签
        
#         Args:
#             stage_value: 睡眠阶段值
            
#         Returns:
#             睡眠阶段标签
#         """
#         return cls.STAGE_LABELS.get(stage_value, "未知")
    
#     @staticmethod
#     def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
#         """
#         准备数据，包括排序和类型转换
        
#         Args:
#             data: 原始数据
            
#         Returns:
#             处理后的数据
#         """
#         # 复制数据避免修改原数据
#         processed_data = data.copy()
        
#         # 确保数据按时间排序
#         processed_data = processed_data.sort_values('upload_time').reset_index(drop=True)
        
#         # 将生理指标字段转换为数值类型
#         numeric_fields = ['heart_rate', 'respiratory_rate', 'body_moves_ratio']
#         for field in numeric_fields:
#             if field in processed_data.columns:
#                 processed_data[field] = pd.to_numeric(processed_data[field], errors='coerce')
        
#         # 异常值检测和处理
#         for field in numeric_fields:
#             if field in processed_data.columns:
#                 # 使用IQR方法检测异常值
#                 Q1 = processed_data[field].quantile(0.25)
#                 Q3 = processed_data[field].quantile(0.75)
#                 IQR = Q3 - Q1
#                 lower_bound = Q1 - 1.5 * IQR
#                 upper_bound = Q3 + 1.5 * IQR
                
#                 # 替换异常值为NaN
#                 processed_data[field] = processed_data[field].apply(
#                     lambda x: x if (x >= lower_bound and x <= upper_bound) else np.nan
#                 )
                
#                 # 插值处理缺失值
#                 processed_data[field] = processed_data[field].interpolate(method='linear', limit_direction='both')
        
#         return processed_data
    
#     @staticmethod
#     def _calculate_advanced_hrv_features(data: pd.DataFrame) -> pd.DataFrame:
#         """
#         计算高级HRV特征
        
#         Args:
#             data: 原始数据
            
#         Returns:
#             包含高级HRV特征的数据
#         """
#         processed_data = data.copy()
        
#         # 计算心率变异性指标
#         if 'heart_rate' in processed_data.columns:
#             # 基础HRV特征
#             processed_data['hr_std'] = processed_data['heart_rate'].rolling(window=5, center=True, min_periods=1).std()
#             processed_data['hr_avg'] = processed_data['heart_rate'].rolling(window=5, center=True, min_periods=1).mean()
#             processed_data['hr_change'] = processed_data['heart_rate'].diff().abs()
#             processed_data['hr_change_avg'] = processed_data['hr_change'].rolling(window=5, center=True, min_periods=1).mean()
            
#             # 增强HRV特征
#             # 1. 心率变异性的时域指标
#             processed_data['hr_rms'] = np.sqrt((processed_data['hr_change'] ** 2).rolling(window=10, center=True, min_periods=1).mean())
#             processed_data['hr_range'] = processed_data['heart_rate'].rolling(window=10, center=True, min_periods=1).max() - \
#                                       processed_data['heart_rate'].rolling(window=10, center=True, min_periods=1).min()
            
#             # 2. 心率变异性的频域指标近似
#             # 低频成分（0.04-0.15 Hz）
#             processed_data['hr_lf'] = processed_data['hr_std'].rolling(window=20, center=True, min_periods=1).mean()
#             # 高频成分（0.15-0.4 Hz）
#             processed_data['hr_hf'] = processed_data['hr_std'].rolling(window=5, center=True, min_periods=1).mean()
#             # LF/HF比值
#             processed_data['hr_lf_hf_ratio'] = processed_data['hr_lf'] / (processed_data['hr_hf'] + 0.1)
            
#             # 3. 心率趋势特征
#             processed_data['hr_trend'] = processed_data['heart_rate'].rolling(window=30, center=True, min_periods=1).mean()
#             processed_data['hr_trend_change'] = processed_data['hr_trend'].diff()
            
#             # 4. 非线性HRV特征
#             # 近似熵
#             def approximate_entropy(x, m=2, r=0.2):
#                 if len(x) < m + 2:
#                     return 0
#                 n = len(x)
#                 x = np.array(x)
#                 phi = []
#                 for k in range(m, m + 2):
#                     if n - k + 1 < 1:
#                         return 0
#                     xm = np.array([x[i:i + k] for i in range(n - k + 1)])
#                     C = []
#                     for i in range(len(xm)):
#                         d = np.max(np.abs(xm - xm[i]), axis=1)
#                         C.append(np.sum(d <= r * np.std(x)) / (n - k + 1))
#                     # 过滤掉无效值
#                     valid_C = [c for c in C if c > 0]
#                     if not valid_C:
#                         return 0
#                     phi.append(np.mean(np.log(valid_C)))
#                 if len(phi) < 2:
#                     return 0
#                 return np.abs(phi[0] - phi[1])
            
#             # 计算近似熵
#             window_size = 20
#             ae_values = []
#             for i in range(len(processed_data)):
#                 start = max(0, i - window_size // 2)
#                 end = min(len(processed_data), i + window_size // 2 + 1)
#                 window_data = processed_data.iloc[start:end]['heart_rate'].dropna().values
#                 if len(window_data) >= 10:
#                     ae = approximate_entropy(window_data)
#                 else:
#                     ae = 0
#                 ae_values.append(ae)
#             processed_data['hr_approx_entropy'] = ae_values
            
#             # 5. 心率复杂度特征
#             processed_data['hr_median'] = processed_data['heart_rate'].rolling(window=10, center=True, min_periods=1).median()
#             processed_data['hr_iqr'] = processed_data['heart_rate'].rolling(window=10, center=True, min_periods=1).quantile(0.75) - \
#                                     processed_data['heart_rate'].rolling(window=10, center=True, min_periods=1).quantile(0.25)
#         else:
#             # 初始化默认值
#             default_features = ['hr_std', 'hr_avg', 'hr_change_avg', 'hr_rms', 'hr_range',
#                                'hr_lf', 'hr_hf', 'hr_lf_hf_ratio', 'hr_trend', 'hr_trend_change',
#                                'hr_approx_entropy', 'hr_median', 'hr_iqr']
#             for feature in default_features:
#                 processed_data[feature] = 0
        
#         return processed_data
    
#     @staticmethod
#     def _calculate_respiratory_features(data: pd.DataFrame) -> pd.DataFrame:
#         """
#         计算呼吸特征
        
#         Args:
#             data: 原始数据
            
#         Returns:
#             包含呼吸特征的数据
#         """
#         processed_data = data.copy()
        
#         # 计算呼吸特征
#         if 'respiratory_rate' in processed_data.columns:
#             # 呼吸率统计特征
#             processed_data['rr_std'] = processed_data['respiratory_rate'].rolling(window=5, center=True, min_periods=1).std()
#             processed_data['rr_avg'] = processed_data['respiratory_rate'].rolling(window=5, center=True, min_periods=1).mean()
            
#             # 呼吸稳定性
#             processed_data['rr_stability'] = processed_data['rr_std'] / (processed_data['rr_avg'] + 0.1)
            
#             # 呼吸模式特征
#             processed_data['rr_change'] = processed_data['respiratory_rate'].diff().abs()
#             processed_data['rr_change_avg'] = processed_data['rr_change'].rolling(window=5, center=True, min_periods=1).mean()
            
#             # 呼吸深度变化
#             processed_data['rr_range'] = processed_data['respiratory_rate'].rolling(window=10, center=True, min_periods=1).max() - \
#                                       processed_data['respiratory_rate'].rolling(window=10, center=True, min_periods=1).min()
#         else:
#             processed_data['rr_std'] = 0
#             processed_data['rr_avg'] = processed_data.get('respiratory_rate', 0)
#             processed_data['rr_stability'] = 0
#             processed_data['rr_change_avg'] = 0
#             processed_data['rr_range'] = 0
        
#         return processed_data
    
#     @staticmethod
#     def _calculate_body_movement_features(data: pd.DataFrame) -> pd.DataFrame:
#         """
#         计算体动特征
        
#         Args:
#             data: 原始数据
            
#         Returns:
#             包含体动特征的数据
#         """
#         processed_data = data.copy()
        
#         # 计算体动特征
#         if 'body_moves_ratio' in processed_data.columns:
#             # 体动统计特征
#             processed_data['move_std'] = processed_data['body_moves_ratio'].rolling(window=5, center=True, min_periods=1).std()
#             processed_data['move_avg'] = processed_data['body_moves_ratio'].rolling(window=5, center=True, min_periods=1).mean()
            
#             # 体动频率特征
#             processed_data['move_freq'] = (processed_data['body_moves_ratio'] > 5).rolling(window=10, center=True, min_periods=1).sum()
            
#             # 体动强度特征
#             processed_data['move_max'] = processed_data['body_moves_ratio'].rolling(window=10, center=True, min_periods=1).max()
#             processed_data['move_intensity'] = processed_data['move_max'] * processed_data['move_freq']
#         else:
#             processed_data['move_std'] = 0
#             processed_data['move_avg'] = processed_data.get('body_moves_ratio', 0)
#             processed_data['move_freq'] = 0
#             processed_data['move_max'] = 0
#             processed_data['move_intensity'] = 0
        
#         return processed_data
    
#     @staticmethod
#     def _calculate_cardiopulmonary_coupling_features(data: pd.DataFrame) -> pd.DataFrame:
#         """
#         计算心肺耦合特征
        
#         Args:
#             data: 原始数据
            
#         Returns:
#             包含心肺耦合特征的数据
#         """
#         processed_data = data.copy()
        
#         # 计算心肺耦合特征
#         if 'heart_rate' in processed_data.columns and 'respiratory_rate' in processed_data.columns:
#             # 心率和呼吸率的相关性
#             window_size = 30
#             corr_values = []
#             for i in range(len(processed_data)):
#                 start = max(0, i - window_size // 2)
#                 end = min(len(processed_data), i + window_size // 2 + 1)
#                 hr_data = processed_data.iloc[start:end]['heart_rate'].dropna().values
#                 rr_data = processed_data.iloc[start:end]['respiratory_rate'].dropna().values
#                 if len(hr_data) >= 10 and len(rr_data) >= 10:
#                     # 确保长度一致
#                     min_len = min(len(hr_data), len(rr_data))
#                     hr_data = hr_data[:min_len]
#                     rr_data = rr_data[:min_len]
#                     corr = np.corrcoef(hr_data, rr_data)[0, 1]
#                     if np.isnan(corr):
#                         corr = 0
#                 else:
#                     corr = 0
#                 corr_values.append(corr)
#             processed_data['cp_correlation'] = corr_values
            
#             # 心率和呼吸率的同步性
#             processed_data['hr_rr_ratio'] = processed_data['heart_rate'] / (processed_data['respiratory_rate'] + 0.1)
#         else:
#             processed_data['cp_correlation'] = 0
#             processed_data['hr_rr_ratio'] = 0
        
#         return processed_data
    
#     @classmethod
#     def calculate_baseline_heart_rate(cls, data: pd.DataFrame) -> float:
#         """
#         计算基线心率
        
#         Args:
#             data: 睡眠数据
            
#         Returns:
#             基线心率
#         """
#         if 'heart_rate' not in data.columns:
#             return 70  # 默认值
        
#         # 过滤有效心率数据
#         valid_hr = data['heart_rate'].dropna()
#         valid_hr = valid_hr[(valid_hr >= 40) & (valid_hr <= 120)]
        
#         if valid_hr.empty:
#             return 70  # 默认值
        
#         # 改进的基线心率计算方法
#         # 1. 计算整个数据集的心率分布
#         hr_mean = valid_hr.mean()
#         hr_std = valid_hr.std()
        
#         # 2. 找出心率较高的时间段，这些时间段更可能是清醒状态
#         # 取心率高于平均值1个标准差的部分作为清醒状态的候选
#         awake_candidates = valid_hr[valid_hr > hr_mean + hr_std]
        
#         if not awake_candidates.empty:
#             # 如果有足够的候选数据，使用它们的平均值作为基线心率
#             return awake_candidates.mean()
#         else:
#             # 如果没有足够的候选数据，使用整个数据集的上四分位数作为基线心率
#             return valid_hr.quantile(0.75)
        
#         # 3. 作为最后的备选，使用传统方法
#         # baseline_data = valid_hr.head(min(30, len(valid_hr)))
#         # return baseline_data.mean()
    
#     @classmethod
#     def _extract_features(cls, data: pd.DataFrame) -> pd.DataFrame:
#         """
#         提取所有特征
        
#         Args:
#             data: 原始数据
            
#         Returns:
#             包含所有特征的数据
#         """
#         # 准备数据
#         processed_data = cls._prepare_data(data)
        
#         # 计算高级HRV特征
#         processed_data = cls._calculate_advanced_hrv_features(processed_data)
        
#         # 计算呼吸特征
#         processed_data = cls._calculate_respiratory_features(processed_data)
        
#         # 计算体动特征
#         processed_data = cls._calculate_body_movement_features(processed_data)
        
#         # 计算心肺耦合特征
#         processed_data = cls._calculate_cardiopulmonary_coupling_features(processed_data)
        
#         # 计算时间特征
#         processed_data['hour'] = processed_data['upload_time'].dt.hour
#         processed_data['is_night'] = (processed_data['hour'] >= 22) | (processed_data['hour'] < 6)
#         processed_data['is_midnight'] = (processed_data['hour'] >= 0) & (processed_data['hour'] < 4)
#         processed_data['is_early_morning'] = (processed_data['hour'] >= 4) & (processed_data['hour'] < 8)
        
#         # 计算时间序列特征
#         processed_data['time_index'] = np.arange(len(processed_data))
#         processed_data['sin_time'] = np.sin(2 * np.pi * processed_data['time_index'] / 1440)  # 24小时周期
#         processed_data['cos_time'] = np.cos(2 * np.pi * processed_data['time_index'] / 1440)  # 24小时周期
        
#         return processed_data
    
#     @classmethod
#     def analyze_sleep_stages_by_advanced_rules(cls, data: pd.DataFrame) -> pd.DataFrame:
#         """
#         基于高级规则的睡眠分期分析方法
        
#         Args:
#             data: 睡眠数据
            
#         Returns:
#             包含睡眠阶段的数据分析
#         """
#         logger.info("开始基于高级规则的睡眠分期分析")
        
#         # 提取特征
#         processed_data = cls._extract_features(data)
        
#         # 计算基线心率
#         baseline_hr = cls.calculate_baseline_heart_rate(processed_data)
#         logger.info(f"计算得到基线心率: {baseline_hr:.2f}")
        
#         # 初始化睡眠阶段
#         processed_data['stage_value'] = cls.STAGE_AWAKE
#         processed_data['stage_label'] = cls.STAGE_LABELS[cls.STAGE_AWAKE]
        
#         # 基于高级规则的睡眠分期
#         for i, row in processed_data.iterrows():
#             # 获取特征值
#             hr = row.get('heart_rate', 0)
#             hr_std = row.get('hr_std', 0)
#             hr_lf_hf_ratio = row.get('hr_lf_hf_ratio', 0)
#             hr_approx_entropy = row.get('hr_approx_entropy', 0)
#             rr_std = row.get('rr_std', 0)
#             rr_stability = row.get('rr_stability', 0)
#             move_avg = row.get('move_avg', 0)
#             move_intensity = row.get('move_intensity', 0)
#             cp_correlation = row.get('cp_correlation', 0)
#             is_night = row.get('is_night', False)
#             is_midnight = row.get('is_midnight', False)
#             is_early_morning = row.get('is_early_morning', False)
            
#             if pd.isna(hr) or hr == 0:
#                 # 无心率数据，标记为清醒
#                 stage = cls.STAGE_AWAKE
#             elif is_night:
#                 # 夜间睡眠分期
#                 # 1. 清醒状态判定 - 调整阈值，避免过多误判
#                 if (hr >= baseline_hr * 1.20 and hr_std > 12) or (move_intensity > 60):
#                     stage = cls.STAGE_AWAKE
#                 # 2. REM阶段判定
#                 elif (hr >= baseline_hr * 1.05 and hr_std > 6 and hr_lf_hf_ratio > 1.4 and 
#                       rr_std > 2 and move_avg < 6 and hr_approx_entropy > 0.05):
#                     stage = cls.STAGE_REM
#                 # 3. N3阶段判定
#                 elif (hr <= baseline_hr * 0.90 and hr_std < 4 and hr_lf_hf_ratio < 0.9 and 
#                       rr_stability < 0.15 and move_avg < 4 and cp_correlation > 0.2):
#                     stage = cls.STAGE_N3
#                 # 4. N2阶段判定
#                 elif (hr <= baseline_hr * 1.00 and hr_std < 6 and hr_lf_hf_ratio < 1.3 and 
#                       rr_stability < 0.20 and move_avg < 6 and cp_correlation > 0.1):
#                     stage = cls.STAGE_N2
#                 # 5. N1阶段判定
#                 elif (hr <= baseline_hr * 1.05 and hr_std < 8 and hr_lf_hf_ratio < 1.4 and 
#                       move_avg < 8):
#                     stage = cls.STAGE_N1
#                 else:
#                     # 对于边缘情况，默认判定为N2，而不是清醒
#                     stage = cls.STAGE_N2
#             else:
#                 # 白天：默认标记为清醒
#                 stage = cls.STAGE_AWAKE
            
#             processed_data.at[i, 'stage_value'] = stage
#             processed_data.at[i, 'stage_label'] = cls.STAGE_LABELS[stage]
        
#         # 应用睡眠阶段转换规则
#         processed_data = cls._apply_sleep_stage_transition_rules(processed_data)
        
#         # 平滑睡眠阶段
#         processed_data = cls._smooth_sleep_stages(processed_data)
        
#         logger.info("基于高级规则的睡眠分期分析完成")
#         return processed_data
    
#     @classmethod
#     def analyze_sleep_stages_by_ensemble_learning(cls, data: pd.DataFrame, models=None) -> pd.DataFrame:
#         """
#         基于集成学习的睡眠分期分析方法
        
#         Args:
#             data: 睡眠数据
#             models: 预训练模型集合（可选）
            
#         Returns:
#             包含睡眠阶段的数据分析
#         """
#         logger.info("开始基于集成学习的睡眠分期分析")
        
#         # 提取特征
#         processed_data = cls._extract_features(data)
        
#         # 选择特征
#         feature_columns = [
#             'heart_rate', 'hr_std', 'hr_avg', 'hr_change_avg', 'hr_rms', 'hr_range',
#             'hr_lf', 'hr_hf', 'hr_lf_hf_ratio', 'hr_trend', 'hr_trend_change',
#             'hr_approx_entropy', 'hr_median', 'hr_iqr',
#             'respiratory_rate', 'rr_std', 'rr_avg', 'rr_stability', 'rr_change_avg', 'rr_range',
#             'body_moves_ratio', 'move_std', 'move_avg', 'move_freq', 'move_max', 'move_intensity',
#             'cp_correlation', 'hr_rr_ratio',
#             'hour', 'is_night', 'is_midnight', 'is_early_morning',
#             'sin_time', 'cos_time'
#         ]
        
#         # 过滤有效特征
#         valid_features = [col for col in feature_columns if col in processed_data.columns]
        
#         # 准备特征矩阵
#         X = processed_data[valid_features].fillna(0)
        
#         # 检查是否有预训练模型
#         if models is None:
#             # 使用基于规则的结果作为标签进行训练
#             rule_based_result = cls.analyze_sleep_stages_by_advanced_rules(data)
#             y = rule_based_result['stage_value']
            
#             # 训练多个分类器
#             rf_clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
#             xgb_clf = xgb.XGBClassifier(n_estimators=200, random_state=42, n_jobs=-1)
#             mlp_clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=200, random_state=42)
            
#             # 创建集成分类器
#             ensemble_clf = VotingClassifier(
#                 estimators=[
#                     ('rf', rf_clf),
#                     ('xgb', xgb_clf),
#                     ('mlp', mlp_clf)
#                 ],
#                 voting='hard'
#             )
            
#             # 训练集成分类器
#             if len(X) > 100:
#                 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#                 ensemble_clf.fit(X_train, y_train)
#                 y_pred = ensemble_clf.predict(X_test)
#                 accuracy = accuracy_score(y_test, y_pred)
#                 kappa = cohen_kappa_score(y_test, y_pred)
#                 logger.info(f"集成学习模型准确率: {accuracy:.2f}")
#                 logger.info(f"Cohen's Kappa系数: {kappa:.2f}")
#                 logger.info("分类报告:\n" + classification_report(y_test, y_pred))
#             else:
#                 ensemble_clf.fit(X, y)
#         else:
#             ensemble_clf = models
        
#         # 预测睡眠阶段
#         y_pred = ensemble_clf.predict(X)
#         processed_data['stage_value'] = y_pred
#         processed_data['stage_label'] = processed_data['stage_value'].apply(cls.get_stage_label)
        
#         # 应用睡眠阶段转换规则
#         processed_data = cls._apply_sleep_stage_transition_rules(processed_data)
        
#         # 平滑睡眠阶段
#         processed_data = cls._smooth_sleep_stages(processed_data)
        
#         logger.info("基于集成学习的睡眠分期分析完成")
#         return processed_data
    
#     @classmethod
#     def _apply_sleep_stage_transition_rules(cls, data: pd.DataFrame) -> pd.DataFrame:
#         """
#         应用睡眠阶段转换规则
        
#         Args:
#             data: 包含睡眠阶段的数据
            
#         Returns:
#             应用转换规则后的数据
#         """
#         processed_data = data.copy()
        
#         # 应用睡眠阶段转换规则
#         if 'stage_value' in processed_data.columns:
#             # 1. 确保睡眠阶段的持续时间合理
#             min_duration = 3  # 最小持续时间（分钟）
#             current_stage = processed_data.iloc[0]['stage_value']
#             start_idx = 0
            
#             for i in range(1, len(processed_data)):
#                 if processed_data.iloc[i]['stage_value'] != current_stage:
#                     # 检查当前阶段的持续时间
#                     duration = i - start_idx
#                     if duration < min_duration and start_idx > 0:
#                         # 如果持续时间太短，使用前一个阶段
#                         processed_data.loc[start_idx:i-1, 'stage_value'] = processed_data.iloc[start_idx-1]['stage_value']
#                     # 更新当前阶段和起始索引
#                     current_stage = processed_data.iloc[i]['stage_value']
#                     start_idx = i
            
#             # 2. 确保睡眠周期的合理性
#             # 睡眠周期通常是：清醒 -> N1 -> N2 -> N3 -> N2 -> REM -> N2 -> ...
#             # 这里实现一个简单的规则来确保合理的转换
#             for i in range(1, len(processed_data)):
#                 prev_stage = processed_data.iloc[i-1]['stage_value']
#                 current_stage = processed_data.iloc[i]['stage_value']
                
#                 # 不合理的转换规则
#                 if prev_stage == cls.STAGE_N3 and current_stage == cls.STAGE_N1:
#                     # N3应该先回到N2，再到N1
#                     processed_data.at[i, 'stage_value'] = cls.STAGE_N2
#                 elif prev_stage == cls.STAGE_REM and current_stage == cls.STAGE_N3:
#                     # REM后通常回到N2，再到N3
#                     processed_data.at[i, 'stage_value'] = cls.STAGE_N2
#                 elif prev_stage == cls.STAGE_AWAKE and current_stage == cls.STAGE_N3:
#                     # 清醒后通常先到N1，再到N2，再到N3
#                     processed_data.at[i, 'stage_value'] = cls.STAGE_N1
            
#             # 更新阶段标签
#             processed_data['stage_label'] = processed_data['stage_value'].apply(cls.get_stage_label)
        
#         return processed_data
    
#     @staticmethod
#     def _smooth_sleep_stages(data: pd.DataFrame, window: int = 5) -> pd.DataFrame:
#         """
#         平滑睡眠阶段，减少碎片化
        
#         Args:
#             data: 包含睡眠阶段的数据
#             window: 平滑窗口大小
            
#         Returns:
#             平滑后的数据分析
#         """
#         processed_data = data.copy()
        
#         if 'stage_value' in processed_data.columns:
#             # 使用移动平均平滑睡眠阶段
#             # 注意：这里使用众数平滑，因为阶段值是分类的
#             for i in range(len(processed_data)):
#                 start = max(0, i - window // 2)
#                 end = min(len(processed_data), i + window // 2 + 1)
#                 window_data = processed_data.iloc[start:end]['stage_value']
#                 if not window_data.empty:
#                     # 使用众数作为平滑后的阶段值
#                     most_common = window_data.mode()
#                     if not most_common.empty:
#                         processed_data.at[i, 'stage_value'] = most_common.iloc[0]
            
#             # 更新阶段标签
#             processed_data['stage_label'] = processed_data['stage_value'].apply(AdvancedSleepStageAnalyzer.get_stage_label)
        
#         return processed_data
    
#     @classmethod
#     def generate_sleep_stage_summary(cls, data: pd.DataFrame) -> Dict:
#         """
#         生成睡眠阶段分析摘要
        
#         Args:
#             data: 包含睡眠阶段的数据
            
#         Returns:
#             睡眠阶段分析摘要
#         """
#         summary = {
#             'total_records': len(data),
#             'stage_distribution': {},
#             'stage_duration_minutes': {},
#             'average_heart_rate_by_stage': {},
#             'sleep_efficiency': 0,
#             'sleep_cycle_count': 0,
#             'sleep_quality_score': 0,
#             'sleep_onset_latency': 0,
#             'wake_after_sleep_onset': 0,
#             'total_sleep_time': 0
#         }
        
#         # 计算各阶段分布
#         if 'stage_value' in data.columns:
#             stage_counts = data['stage_value'].value_counts()
#             for stage, count in stage_counts.items():
#                 stage_label = cls.get_stage_label(stage)
#                 summary['stage_distribution'][stage_label] = count
                
#                 # 计算各阶段时长（假设每条记录1分钟）
#                 summary['stage_duration_minutes'][stage_label] = count
                
#                 # 计算各阶段平均心率
#                 stage_data = data[data['stage_value'] == stage]
#                 if 'heart_rate' in stage_data.columns:
#                     avg_hr = stage_data['heart_rate'].mean()
#                     summary['average_heart_rate_by_stage'][stage_label] = round(avg_hr, 2) if not pd.isna(avg_hr) else 0
        
#         # 计算总睡眠时长
#         sleep_stages = [cls.STAGE_N1, cls.STAGE_N2, cls.STAGE_N3, cls.STAGE_REM]
#         total_sleep_duration = sum(
#             summary['stage_duration_minutes'].get(cls.get_stage_label(stage), 0)
#             for stage in sleep_stages
#         )
#         summary['total_sleep_time'] = total_sleep_duration
        
#         # 计算睡眠效率
#         total_records = len(data)
#         if total_records > 0:
#             summary['sleep_efficiency'] = round(total_sleep_duration / total_records * 100, 2)
        
#         # 计算睡眠周期数（近似）
#         # 假设一个睡眠周期约90分钟
#         if total_sleep_duration > 0:
#             summary['sleep_cycle_count'] = int(total_sleep_duration // 90)
        
#         # 计算睡眠质量评分
#         # 基于各阶段的比例和睡眠效率
#         deep_sleep_duration = summary['stage_duration_minutes'].get(cls.get_stage_label(cls.STAGE_N3), 0)
#         rem_sleep_duration = summary['stage_duration_minutes'].get(cls.get_stage_label(cls.STAGE_REM), 0)
        
#         if total_sleep_duration > 0:
#             deep_sleep_ratio = deep_sleep_duration / total_sleep_duration
#             rem_sleep_ratio = rem_sleep_duration / total_sleep_duration
            
#             # 睡眠质量评分计算
#             sleep_quality = (
#                 summary['sleep_efficiency'] * 0.4 +  # 睡眠效率占40%
#                 deep_sleep_ratio * 100 * 0.3 +      # 深睡比例占30%
#                 rem_sleep_ratio * 100 * 0.2 +       # REM比例占20%
#                 min(summary['sleep_cycle_count'], 5) * 2  # 睡眠周期数占10%
#             )
#             summary['sleep_quality_score'] = round(min(sleep_quality, 100), 2)
        
#         # 计算睡眠潜伏期（入睡时间）
#         # 假设从记录开始到第一次进入N1阶段的时间
#         if 'stage_value' in data.columns:
#             sleep_onset_idx = data[data['stage_value'] == cls.STAGE_N1].index.min()
#             if not pd.isna(sleep_onset_idx):
#                 summary['sleep_onset_latency'] = int(sleep_onset_idx)
        
#         # 计算睡眠后觉醒时间（WASO）
#         # 假设睡眠开始后到结束前的清醒时间
#         if 'stage_value' in data.columns:
#             sleep_onset_idx = data[data['stage_value'] == cls.STAGE_N1].index.min()
#             if not pd.isna(sleep_onset_idx):
#                 waso_data = data.iloc[int(sleep_onset_idx):]
#                 waso_duration = len(waso_data[waso_data['stage_value'] == cls.STAGE_AWAKE])
#                 summary['wake_after_sleep_onset'] = waso_duration
        
#         return summary
    
#     @classmethod
#     def generate_sleep_quality_evaluation(cls, summary: Dict) -> Dict:
#         """
#         生成睡眠质量评估
        
#         Args:
#             summary: 睡眠阶段分析摘要
            
#         Returns:
#             睡眠质量评估
#         """
#         evaluation = {
#             'overall_quality': '良好',
#             'score': summary.get('sleep_quality_score', 0),
#             'components': {},
#             'suggestions': []
#         }
        
#         # 评估总体睡眠质量
#         quality_score = summary.get('sleep_quality_score', 0)
#         if quality_score >= 85:
#             evaluation['overall_quality'] = '优秀'
#         elif quality_score >= 70:
#             evaluation['overall_quality'] = '良好'
#         elif quality_score >= 50:
#             evaluation['overall_quality'] = '一般'
#         else:
#             evaluation['overall_quality'] = '较差'
        
#         # 评估各组成部分
#         evaluation['components'] = {
#             'sleep_efficiency': {
#                 'value': summary.get('sleep_efficiency', 0),
#                 'assessment': '良好' if summary.get('sleep_efficiency', 0) >= 85 else '一般' if summary.get('sleep_efficiency', 0) >= 70 else '较差'
#             },
#             'deep_sleep': {
#                 'value': summary['stage_duration_minutes'].get('深睡N3', 0),
#                 'assessment': '良好' if summary['stage_duration_minutes'].get('深睡N3', 0) >= 60 else '一般' if summary['stage_duration_minutes'].get('深睡N3', 0) >= 30 else '较差'
#             },
#             'rem_sleep': {
#                 'value': summary['stage_duration_minutes'].get('眼动REM', 0),
#                 'assessment': '良好' if summary['stage_duration_minutes'].get('眼动REM', 0) >= 60 else '一般' if summary['stage_duration_minutes'].get('眼动REM', 0) >= 30 else '较差'
#             },
#             'sleep_cycles': {
#                 'value': summary.get('sleep_cycle_count', 0),
#                 'assessment': '良好' if summary.get('sleep_cycle_count', 0) >= 4 else '一般' if summary.get('sleep_cycle_count', 0) >= 3 else '较差'
#             }
#         }
        
#         # 生成睡眠建议
#         if summary.get('sleep_efficiency', 0) < 85:
#             evaluation['suggestions'].append('提高睡眠效率：保持规律的作息时间，睡前避免使用电子设备，创造安静舒适的睡眠环境')
        
#         if summary['stage_duration_minutes'].get('深睡N3', 0) < 60:
#             evaluation['suggestions'].append('增加深睡眠时间：睡前可以进行轻度运动，避免咖啡因和酒精，保持卧室温度适宜')
        
#         if summary['stage_duration_minutes'].get('眼动REM', 0) < 60:
#             evaluation['suggestions'].append('增加REM睡眠时间：保持规律的睡眠 schedule，减少压力，避免睡前大餐')
        
#         if summary.get('sleep_cycle_count', 0) < 4:
#             evaluation['suggestions'].append('优化睡眠周期：尝试在90分钟的倍数时间入睡和起床，避免打断睡眠周期')
        
#         if summary.get('sleep_onset_latency', 0) > 30:
#             evaluation['suggestions'].append('改善入睡困难：睡前放松活动，如阅读或冥想，避免睡前剧烈运动')
        
#         if summary.get('wake_after_sleep_onset', 0) > 30:
#             evaluation['suggestions'].append('减少睡眠中断：保持卧室安静，避免睡前饮水过多，检查是否有睡眠呼吸暂停等问题')
        
#         return evaluation


# @tool
# def analyze_sleep_stages_by_advanced_methods(date: str) -> str:
#     """
#     基于高级方法的睡眠分期分析工具
#     根据AASM标准和多维度生理特征分析睡眠分期
    
#     Args:
#         date: 分析日期，格式为YYYY-MM-DD
        
#     Returns:
#         睡眠分期分析结果，包含两种高级分析方法的结果
#     """
#     try:
#         logger.info(f"开始分析 {date} 的睡眠分期（基于高级方法）")
        
#         # 生成模拟数据
#         def generate_test_data(date_str):
#             """生成测试数据"""
#             times = pd.date_range(start=f"{date_str} 22:00:00", end=f"{date_str} 23:59:59", freq="1min")
#             times = times.append(pd.date_range(start=f"{date_str[:8]}{int(date_str[8:10])+1} 00:00:00", end=f"{date_str[:8]}{int(date_str[8:10])+1} 06:00:00", freq="1min"))
            
#             # 生成模拟心率数据
#             hr_values = []
#             stage_sequence = [
#                 (60, 75, 120),  # 清醒
#                 (65, 72, 60),   # N1
#                 (60, 68, 90),   # N2
#                 (55, 62, 60),   # N3
#                 (60, 68, 60),   # N2
#                 (70, 80, 45),   # REM
#                 (60, 68, 60),   # N2
#                 (55, 62, 60),   # N3
#                 (60, 68, 60),   # N2
#                 (70, 80, 45),   # REM
#                 (65, 80, 30)    # 清醒
#             ]
            
#             for min_hr, max_hr, duration in stage_sequence:
#                 hr_segment = np.random.uniform(min_hr, max_hr, duration)
#                 hr_values.extend(hr_segment)
            
#             # 截取与时间对应的长度
#             hr_values = hr_values[:len(times)]
#             if len(hr_values) < len(times):
#                 hr_values.extend([70] * (len(times) - len(hr_values)))
            
#             # 生成模拟呼吸率
#             rr_values = np.random.uniform(12, 20, len(times))
            
#             # 生成模拟体动
#             body_moves = np.random.uniform(0, 15, len(times))
            
#             data = pd.DataFrame({
#                 'upload_time': times,
#                 'heart_rate': hr_values,
#                 'respiratory_rate': rr_values,
#                 'body_moves_ratio': body_moves
#             })
            
#             return data
        
#         # 生成测试数据
#         test_data = generate_test_data(date)
#         logger.info(f"生成了 {len(test_data)} 条测试数据")
        
#         # 方法1：基于高级规则的睡眠分期分析
#         analyzer = AdvancedSleepStageAnalyzer()
#         result_method1 = analyzer.analyze_sleep_stages_by_advanced_rules(test_data)
#         summary_method1 = analyzer.generate_sleep_stage_summary(result_method1)
#         quality_eval1 = analyzer.generate_sleep_quality_evaluation(summary_method1)
        
#         # 方法2：基于集成学习的睡眠分期分析
#         result_method2 = analyzer.analyze_sleep_stages_by_ensemble_learning(test_data)
#         summary_method2 = analyzer.generate_sleep_stage_summary(result_method2)
#         quality_eval2 = analyzer.generate_sleep_quality_evaluation(summary_method2)
        
#         # 构建分析结果
#         analysis_result = {
#             'success': True,
#             'date': date,
#             'data_points': len(test_data),
#             'analysis_methods': {
#                 'advanced_rules': {
#                     'name': '基于高级规则的分析方法',
#                     'summary': summary_method1,
#                     'quality_evaluation': quality_eval1
#                 },
#                 'ensemble_learning': {
#                     'name': '基于集成学习的分析方法',
#                     'summary': summary_method2,
#                     'quality_evaluation': quality_eval2
#                 }
#             },
#             'sleep_stage_characteristics': {
#                 'awake': {
#                     'heart_rate': '高，波动大',
#                     'hrv': '低频成分主导，复杂度高',
#                     'respiration': '不规则',
#                     'movement': '频繁，强度高',
#                     'cp_coupling': '弱'
#                 },
#                 'n1': {
#                     'heart_rate': '略降，波动减弱',
#                     'hrv': '高频成分开始上升，复杂度降低',
#                     'respiration': '逐渐规律',
#                     'movement': '减少',
#                     'cp_coupling': '中等'
#                 },
#                 'n2': {
#                     'heart_rate': '明显降低，变异性下降',
#                     'hrv': '低频/高频比值下降，复杂度低',
#                     'respiration': '规律',
#                     'movement': '很少',
#                     'cp_coupling': '强'
#                 },
#                 'n3': {
#                     'heart_rate': '最低，保持稳定',
#                     'hrv': '高频成分主导，复杂度最低',
#                     'respiration': '非常规律',
#                     'movement': '极少',
#                     'cp_coupling': '很强'
#                 },
#                 'rem': {
#                     'heart_rate': '波动大，接近清醒',
#                     'hrv': '低频成分回升，复杂度高',
#                     'respiration': '不规则',
#                     'movement': '很少，但可能有快速眼动',
#                     'cp_coupling': '中等'
#                 }
#             },
#             'sleep_quality_overview': {
#                 'average_score': round((quality_eval1['score'] + quality_eval2['score']) / 2, 2),
#                 'dominant_method': 'ensemble_learning' if quality_eval2['score'] > quality_eval1['score'] else 'advanced_rules'
#             }
#         }
        
#         logger.info(f"睡眠分期分析完成，日期: {date}")
#         return json.dumps(analysis_result, ensure_ascii=False, default=str)
        
#     except Exception as e:
#         logger.error(f"分析睡眠分期失败: {str(e)}")
#         error_result = {
#             'success': False,
#             'error': f"分析失败: {str(e)}"
#         }
#         return json.dumps(error_result, ensure_ascii=False)


# @tool
# def get_advanced_sleep_stage_analysis_help() -> str:
#     """
#     获取高级睡眠分期分析工具的帮助信息
#     提供基于多维度生理特征的高精度睡眠分期分析工具的使用说明
    
#     Returns:
#         帮助信息
#     """
#     help_info = {
#         'success': True,
#         'tool_name': '高级睡眠分期分析工具',
#         'description': '根据AASM标准和多维度生理特征分析睡眠分期',
#         'methods': [
#             {
#                 'name': '基于高级规则的分析方法',
#                 'description': '基于高级HRV特征、呼吸模式、体动特征和心肺耦合特征的复杂规则进行睡眠分期',
#                 'features': ['高级HRV特征', '呼吸模式', '体动特征', '心肺耦合特征', '时间特征', '非线性动力学特征']
#             },
#             {
#                 'name': '基于集成学习的分析方法',
#                 'description': '基于随机森林、XGBoost和神经网络的集成分类器进行睡眠分期',
#                 'features': ['高级HRV特征', '呼吸模式', '体动特征', '心肺耦合特征', '时间特征', '非线性动力学特征', '自主神经平衡']
#             }
#         ],
#         'sleep_stages': [
#             {
#                 'stage': '清醒',
#                 'characteristics': '心率高，波动大，HRV低频成分主导，呼吸不规则，体动频繁'
#             },
#             {
#                 'stage': '浅睡N1',
#                 'characteristics': '心率略降，波动减弱，HRV高频成分开始上升，呼吸逐渐规律，体动减少'
#             },
#             {
#                 'stage': '中睡N2',
#                 'characteristics': '心率明显降低，变异性下降，HRV低频/高频比值下降，呼吸规律，体动很少'
#             },
#             {
#                 'stage': '深睡N3',
#                 'characteristics': '心率最低，保持稳定，HRV高频成分主导，呼吸非常规律，体动极少'
#             },
#             {
#                 'stage': '眼动REM',
#                 'characteristics': '心率波动大，接近清醒，HRV低频成分回升，呼吸不规则，体动很少但可能有快速眼动'
#             }
#         ],
#         'evaluation_metrics': [
#             '睡眠质量评分',
#             '睡眠效率',
#             '睡眠周期数',
#             '深睡时长',
#             'REM睡眠时长',
#             '睡眠潜伏期',
#             '睡眠后觉醒时间',
#             'Cohen\'s Kappa系数'
#         ],
#         'usage': '调用 analyze_sleep_stages_by_advanced_methods 工具，传入日期参数（格式：YYYY-MM-DD）'
#     }
    
#     return json.dumps(help_info, ensure_ascii=False)