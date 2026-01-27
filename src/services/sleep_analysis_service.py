"""
睡眠分析服务模块
提供高级睡眠分析功能，结合格式化时间信息与智能体分析
"""

from datetime import datetime, timedelta
import json
import logging
from typing import Dict, Optional, Any, Tuple

# 导入睡眠分析工具
from ..tools.sleep_analyzer_tool import analyze_single_day_sleep_data
from ..tools.physiological_analyzer_tool import analyze_single_day_physiological_data

# 导入时间格式化工具
from ..utils.time_formatter import create_sleep_analysis_prompt_with_time

# 导入数据库管理
from ..db.database import get_db_manager

# 导入智能体
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from improved_agent import run_improved_agent
except ImportError:
    logging.warning("智能体模块导入失败，将使用默认分析模式")
    run_improved_agent = None

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SleepAnalysisService:
    """
    睡眠分析服务类
    提供高级睡眠分析功能，结合格式化时间信息与智能体分析
    """
    
    # 常量定义
    DATE_FORMAT = '%Y-%m-%d'
    TIME_FORMAT = '%H:%M'
    DEFAULT_SLEEP_PREP_TIME = 24
    
    def __init__(self):
        """
        初始化睡眠分析服务
        """
        self.db_manager = get_db_manager()
    
    def validate_date_format(self, date_str: str) -> bool:
        """
        验证日期格式
        
        Args:
            date_str: 日期字符串
            
        Returns:
            bool: 日期格式是否有效
        """
        try:
            datetime.strptime(date_str, self.DATE_FORMAT)
            return True
        except ValueError:
            return False
    
    def get_stored_sleep_data(self, date: str) -> Dict[str, Any]:
        """
        从数据库获取已存储的睡眠分析数据
        
        Args:
            date: 日期字符串，格式如 '2024-12-20'
            
        Returns:
            包含存储数据的字典，如果无数据则返回空字典
        """
        try:
            logger.info(f"尝试从数据库获取 {date} 的睡眠数据")
            stored_data_raw = self.db_manager.get_calculated_sleep_data(date)
            
            if stored_data_raw.empty:
                logger.info(f"数据库中没有找到 {date} 的睡眠数据")
                return {'success': True, 'data': None}
            
            # 转换数据为字典格式
            stored_record = stored_data_raw.to_dict('records')[0]
            stored_data = {
                'success': True,
                'data': stored_record
            }
            logger.info(f"从数据库获取到 {date} 的睡眠数据")
            return stored_data
        except Exception as e:
            logger.error(f"获取存储的睡眠数据失败: {str(e)}")
            return {'success': False, 'error': f"数据库操作失败: {str(e)}"}
    
    def analyze_sleep_data(self, date: str) -> Dict[str, Any]:
        """
        分析睡眠数据
        
        Args:
            date: 日期字符串，格式如 '2024-12-20'
            
        Returns:
            包含睡眠分析结果的字典
        """
        try:
            logger.info(f"开始分析 {date} 的睡眠数据")
            
            # 首先尝试获取睡眠数据
            sleep_data_json = analyze_single_day_sleep_data(date)
            
            # 解析睡眠数据
            sleep_data = json.loads(sleep_data_json)
            
            # 检查是否有错误
            if not sleep_data.get('success', True):
                error_msg = sleep_data.get('error', '未知错误')
                logger.warning(f"睡眠数据获取失败: {error_msg}")
                
                # 尝试获取生理数据作为备选
                return self._get_physiological_data_as_fallback(date, error_msg)
            else:
                # 提取睡眠数据
                sleep_data_content = sleep_data.get('data', {})
                logger.info(f"成功获取 {date} 的睡眠数据")
                
                # 存储睡眠数据到数据库
                self._store_sleep_data(sleep_data_content, date)
                
                return {
                    'success': True,
                    'data': sleep_data_content,
                    'is_physiological_data': False
                }
        
        except json.JSONDecodeError as e:
            logger.error(f"数据解析失败: {str(e)}")
            return {
                'success': False,
                'error': f"数据解析失败：无法解析睡眠数据",
                'data': None
            }
        except Exception as e:
            logger.error(f"分析睡眠数据失败: {str(e)}")
            return {
                'success': False,
                'error': f"分析睡眠数据失败：{str(e)}",
                'data': None
            }
    
    def _get_physiological_data_as_fallback(self, date: str, original_error: str) -> Dict[str, Any]:
        """
        获取生理数据作为备选
        
        Args:
            date: 日期字符串
            original_error: 原始错误信息
            
        Returns:
            包含生理数据的字典
        """
        try:
            logger.info(f"尝试获取 {date} 的生理数据作为备选")
            physio_data_json = analyze_single_day_physiological_data(date)
            physio_data = json.loads(physio_data_json)
            
            if not physio_data.get('success', True):
                error_msg = physio_data.get('error', '未知错误')
                logger.error(f"生理数据获取也失败: {error_msg}")
                return {
                    'success': False,
                    'error': f"数据获取失败：睡眠数据和生理数据都无法获取",
                    'data': None
                }
            else:
                logger.info("使用生理数据作为备选")
                return {
                    'success': True,
                    'data': physio_data.get('data', {}),
                    'is_physiological_data': True
                }
        except Exception as e:
            logger.error(f"获取生理数据失败: {str(e)}")
            return {
                'success': False,
                'error': f"数据获取失败：{original_error}",
                'data': None
            }
    
    def _store_sleep_data(self, sleep_data: Dict[str, Any], date: str) -> None:
        """
        存储睡眠数据到数据库
        
        Args:
            sleep_data: 睡眠数据
            date: 日期字符串
        """
        try:
            self.db_manager.store_calculated_sleep_data(sleep_data)
            logger.info(f"{date} 的睡眠数据已存储到数据库")
        except Exception as e:
            logger.warning(f"存储睡眠数据失败: {str(e)}")
    
    def _fetch_sleep_data(self, date: str, force_refresh: bool = False) -> Tuple[Optional[Dict[str, Any]], bool, Optional[str]]:
        """
        获取睡眠数据
        
        Args:
            date: 日期字符串
            force_refresh: 是否强制刷新
            
        Returns:
            Tuple[Optional[Dict[str, Any]], bool, Optional[str]]: (睡眠数据, 是否为生理数据, 错误信息)
        """
        # 获取睡眠数据
        sleep_data = None
        is_physiological_data = False
        error_msg = None
        
        if not force_refresh:
            # 尝试从数据库获取已存储的分析结果
            stored_data = self.get_stored_sleep_data(date)
            if stored_data.get('success') and stored_data.get('data'):
                sleep_data = stored_data['data']
                logger.info("使用数据库中已存储的睡眠数据")
            else:
                logger.info("数据库中无存储数据，将重新分析")
                analysis_result = self.analyze_sleep_data(date)
                if analysis_result.get('success') and analysis_result.get('data'):
                    sleep_data = analysis_result['data']
                    is_physiological_data = analysis_result.get('is_physiological_data', False)
                else:
                    error_msg = analysis_result.get('error', '未知错误')
                    logger.error(f"获取睡眠数据失败: {error_msg}")
        else:
            # 强制刷新，重新分析
            logger.info("强制刷新，重新分析睡眠数据")
            analysis_result = self.analyze_sleep_data(date)
            if analysis_result.get('success') and analysis_result.get('data'):
                sleep_data = analysis_result['data']
                is_physiological_data = analysis_result.get('is_physiological_data', False)
            else:
                error_msg = analysis_result.get('error', '未知错误')
                logger.error(f"获取睡眠数据失败: {error_msg}")
        
        return sleep_data, is_physiological_data, error_msg
    
    def run_sleep_analysis_with_formatted_time(self, date: str, force_refresh: bool = False) -> str:
        """
        运行睡眠分析，使用格式化的时间信息作为提示
        
        Args:
            date: 日期字符串，格式如 '2024-12-20'
            force_refresh: 是否强制刷新，跳过缓存
            
        Returns:
            str: 智能体分析结果
        """
        logger.info(f"开始运行睡眠分析，日期: {date}, 强制刷新: {force_refresh}")
        
        # 检查日期格式
        if not self.validate_date_format(date):
            error_msg = f"无效的日期格式: {date}，请使用 'YYYY-MM-DD' 格式"
            logger.error(error_msg)
            return f"错误: {error_msg}"
        
        # 获取睡眠数据
        sleep_data, is_physiological_data, error_msg = self._fetch_sleep_data(date, force_refresh)
        
        if error_msg:
            return f"数据获取失败：{error_msg}"
        
        if sleep_data is None:
            return f"数据获取失败：无法获取 {date} 的睡眠数据"
        
        # 创建分析提示
        prompt = self._create_analysis_prompt(date, sleep_data, is_physiological_data)
        
        # 调用智能体进行分析
        if run_improved_agent:
            try:
                logger.info("调用智能体进行深度分析")
                result = run_improved_agent(date, thread_id=f"sleep_analysis_{date}", force_refresh=force_refresh)
                logger.info("智能体分析完成")
                return result
            except Exception as e:
                logger.error(f"智能体分析失败: {str(e)}")
                # 智能体分析失败时，返回基本分析结果
                return self._get_basic_analysis_result(date, sleep_data, is_physiological_data)
        else:
            # 智能体不可用时，返回基本分析结果
            logger.info("智能体不可用，返回基本分析结果")
            return self._get_basic_analysis_result(date, sleep_data, is_physiological_data)
    
    def _create_analysis_prompt(self, date: str, sleep_data: Dict[str, Any], is_physiological_data: bool) -> str:
        """
        创建分析提示
        
        Args:
            date: 日期字符串
            sleep_data: 睡眠数据
            is_physiological_data: 是否为生理数据
            
        Returns:
            str: 分析提示
        """
        if is_physiological_data:
            # 使用生理数据构建基本提示
            prompt = f"请分析 {date} 的生理指标数据：\n\n{str(sleep_data)}"
            logger.info("使用生理数据构建分析提示")
        else:
            # 使用睡眠数据创建包含格式化时间的提示
            prompt = create_sleep_analysis_prompt_with_time(date, sleep_data)
            logger.info("使用睡眠数据和格式化时间构建分析提示")
        return prompt
    
    def get_formatted_sleep_time_summary(self, date: str) -> str:
        """
        获取格式化的睡眠时间摘要
        
        Args:
            date: 日期字符串，格式如 '2024-12-20'
            
        Returns:
            str: 格式化的时间摘要
        """
        logger.info(f"获取 {date} 的睡眠时间摘要")
        
        # 检查日期格式
        if not self.validate_date_format(date):
            error_msg = f"无效的日期格式: {date}，请使用 'YYYY-MM-DD' 格式"
            logger.error(error_msg)
            return f"错误: {error_msg}"
        
        # 获取睡眠数据
        sleep_data = None
        
        # 尝试从数据库获取已存储的分析结果
        stored_data = self.get_stored_sleep_data(date)
        if stored_data.get('success') and stored_data.get('data'):
            sleep_data = stored_data['data']
            logger.info("使用数据库中已存储的睡眠数据")
        else:
            logger.info("数据库中无存储数据，将重新分析")
            # 获取睡眠数据
            try:
                sleep_data_json = analyze_single_day_sleep_data(date)
                sleep_data_result = json.loads(sleep_data_json)
                
                if not sleep_data_result.get('success', True):
                    error_msg = sleep_data_result.get('error', '未知错误')
                    logger.warning(f"睡眠数据获取失败: {error_msg}")
                    return f"无法获取 {date} 的睡眠时间数据"
                
                sleep_data = sleep_data_result.get('data', {})
                
                # 存储睡眠数据到数据库
                self._store_sleep_data(sleep_data, date)
                    
            except json.JSONDecodeError as e:
                logger.error(f"数据解析失败: {str(e)}")
                return f"数据解析失败：无法解析睡眠数据"
            except Exception as e:
                logger.error(f"获取睡眠数据失败: {str(e)}")
                return f"无法获取 {date} 的睡眠时间数据"
        
        # 从数据中提取时间信息
        bedtime_str = sleep_data.get('bedtime', '')
        wakeup_time_str = sleep_data.get('wakeup_time', '')
        sleep_prep_time_minutes = sleep_data.get('sleep_prep_time_minutes', self.DEFAULT_SLEEP_PREP_TIME)
        
        if bedtime_str and wakeup_time_str:
            try:
                # 解析时间字符串
                bedtime = self._parse_time_string(bedtime_str)
                wakeup_time = self._parse_time_string(wakeup_time_str)
                
                if bedtime and wakeup_time:
                    # 计算入睡时间
                    sleep_start_time = bedtime + timedelta(minutes=sleep_prep_time_minutes)
                    
                    # 格式化时间
                    bedtime_formatted = bedtime.strftime(self.TIME_FORMAT)
                    sleep_start_formatted = sleep_start_time.strftime(self.TIME_FORMAT)
                    wakeup_formatted = wakeup_time.strftime(self.TIME_FORMAT)
                    
                    return f"我 {bedtime_formatted} 上床，{sleep_start_formatted} 入睡，{wakeup_formatted} 醒来"
                else:
                    logger.warning(f"时间解析失败")
                    return f"无法从数据中提取 {date} 的时间信息"
            except Exception as e:
                logger.error(f"时间解析失败: {str(e)}")
                return f"无法从数据中提取 {date} 的时间信息"
        else:
            logger.warning(f"睡眠数据中缺少时间信息")
            return f"无法从数据中提取 {date} 的时间信息"
    
    def _parse_time_string(self, time_str: str) -> Optional[datetime]:
        """
        解析时间字符串
        
        Args:
            time_str: 时间字符串
            
        Returns:
            Optional[datetime]: 解析后的时间对象
        """
        try:
            if time_str.endswith('Z'):
                return datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            else:
                return datetime.fromisoformat(time_str)
        except Exception as e:
            logger.error(f"时间解析失败: {str(e)}")
            return None
    
    def _get_basic_analysis_result(self, date: str, sleep_data: Dict[str, Any], is_physiological_data: bool) -> str:
        """
        获取基本分析结果
        
        Args:
            date: 日期字符串
            sleep_data: 睡眠数据字典
            is_physiological_data: 是否为生理数据
            
        Returns:
            str: 基本分析结果
        """
        if is_physiological_data:
            # 生理数据基本分析
            return self._analyze_physiological_data(date, sleep_data)
        else:
            # 睡眠数据基本分析
            return self._analyze_sleep_data(date, sleep_data)
    
    def _analyze_physiological_data(self, date: str, physio_data: Dict[str, Any]) -> str:
        """
        分析生理数据
        
        Args:
            date: 日期字符串
            physio_data: 生理数据
            
        Returns:
            str: 生理数据分析结果
        """
        heart_rate_metrics = physio_data.get('heart_rate_metrics', {})
        respiratory_metrics = physio_data.get('respiratory_metrics', {})
        
        avg_hr = heart_rate_metrics.get('avg_heart_rate', 0)
        avg_rr = respiratory_metrics.get('avg_respiratory_rate', 0)
        
        result = f"{date} 生理指标分析：\n"
        result += f"- 平均心率: {avg_hr} 次/分钟\n"
        result += f"- 平均呼吸率: {avg_rr} 次/分钟\n"
        result += f"- 心率范围: {heart_rate_metrics.get('min_heart_rate', 0)} - {heart_rate_metrics.get('max_heart_rate', 0)} 次/分钟\n"
        result += f"- 呼吸率范围: {respiratory_metrics.get('min_respiratory_rate', 0)} - {respiratory_metrics.get('max_respiratory_rate', 0)} 次/分钟\n"
        
        apnea_count = respiratory_metrics.get('apnea_count', 0)
        if apnea_count > 0:
            result += f"- 呼吸暂停次数: {apnea_count}\n"
            result += f"- 平均呼吸暂停时长: {respiratory_metrics.get('avg_apnea_duration', 0)} 秒\n"
        
        return result
    
    def _analyze_sleep_data(self, date: str, sleep_data: Dict[str, Any]) -> str:
        """
        分析睡眠数据
        
        Args:
            date: 日期字符串
            sleep_data: 睡眠数据
            
        Returns:
            str: 睡眠数据分析结果
        """
        bedtime = sleep_data.get('bedtime', '未知')
        wakeup_time = sleep_data.get('wakeup_time', '未知')
        sleep_duration = sleep_data.get('sleep_duration_minutes', 0)
        sleep_score = sleep_data.get('sleep_score', 0)
        
        # 计算睡眠时长（小时和分钟）
        sleep_hours = int(sleep_duration // 60)
        sleep_minutes = int(sleep_duration % 60)
        
        # 睡眠阶段数据
        sleep_phases = sleep_data.get('sleep_phases', {})
        deep_sleep = sleep_phases.get('deep_sleep_minutes', 0)
        deep_sleep_pct = sleep_phases.get('deep_sleep_percentage', 0)
        
        result = f"{date} 睡眠分析结果：\n"
        result += f"- 就寝时间: {bedtime}\n"
        result += f"- 起床时间: {wakeup_time}\n"
        result += f"- 睡眠时长: {sleep_hours} 小时 {sleep_minutes} 分钟\n"
        result += f"- 睡眠评分: {sleep_score} 分\n"
        result += f"- 深睡时长: {deep_sleep} 分钟 ({deep_sleep_pct}%)\n"
        
        bed_exit_count = sleep_data.get('bed_exit_count', 0)
        if bed_exit_count > 0:
            result += f"- 夜间离床次数: {bed_exit_count} 次\n"
        
        # 添加睡眠质量评价
        quality = self._get_sleep_quality_rating(sleep_score)
        result += f"- 睡眠质量评价: {quality}\n"
        
        # 添加睡眠总结
        summary = sleep_data.get('summary', '')
        if summary:
            result += f"- 睡眠总结: {summary}\n"
        
        return result
    
    def _get_sleep_quality_rating(self, sleep_score: int) -> str:
        """
        根据睡眠评分获取睡眠质量评价
        
        Args:
            sleep_score: 睡眠评分
            
        Returns:
            str: 睡眠质量评价
        """
        if sleep_score >= 90:
            return "优秀"
        elif sleep_score >= 80:
            return "良好"
        elif sleep_score >= 70:
            return "一般"
        elif sleep_score >= 60:
            return "较差"
        else:
            return "差"


# 全局睡眠分析服务实例
_sleep_analysis_service = None


def get_sleep_analysis_service() -> SleepAnalysisService:
    """
    获取睡眠分析服务实例
    
    Returns:
        SleepAnalysisService: 睡眠分析服务实例
    """
    global _sleep_analysis_service
    if _sleep_analysis_service is None:
        _sleep_analysis_service = SleepAnalysisService()
    return _sleep_analysis_service


def run_sleep_analysis_with_formatted_time(date: str, force_refresh: bool = False) -> str:
    """
    运行睡眠分析，使用格式化的时间信息作为提示
    
    Args:
        date: 日期字符串，格式如 '2024-12-20'
        force_refresh: 是否强制刷新，跳过缓存
        
    Returns:
        str: 智能体分析结果
    """
    service = get_sleep_analysis_service()
    return service.run_sleep_analysis_with_formatted_time(date, force_refresh)


def get_formatted_sleep_time_summary(date: str) -> str:
    """
    获取格式化的睡眠时间摘要
    
    Args:
        date: 日期字符串，格式如 '2024-12-20'
        
    Returns:
        str: 格式化的时间摘要
    """
    service = get_sleep_analysis_service()
    return service.get_formatted_sleep_time_summary(date)


if __name__ == "__main__":
    # 示例使用
    test_date = "2024-12-20"  # 使用一个测试日期
    
    print("获取格式化时间摘要:")
    time_summary = get_formatted_sleep_time_summary(test_date)
    print(time_summary)
    
    print("\n运行完整的睡眠分析:")
    analysis_result = run_sleep_analysis_with_formatted_time(test_date, force_refresh=True)
    print(analysis_result)
