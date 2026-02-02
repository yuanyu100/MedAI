from typing import Any, Optional, Dict, Union
import json
from datetime import datetime, time
from langchain.tools import tool


class ApiResponse:
    """
    统一的API响应类，用于封装返回格式
    """
    
    def __init__(self, success: bool = True, data: Any = None, message: str = "", error: str = None):
        self.success = success
        self.data = data
        self.message = message
        self.error = error
#        self.timestamp = datetime.now().isoformat()  # 注释掉时间戳字段
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "success": self.success
        }
        
        if self.data is not None:
            result["data"] = self.data
        
        if self.message:
            result["message"] = self.message
            
        if self.error:
            result["error"] = self.error
            result["success"] = False  # 如果有错误，设置success为False
            
        return result
    
    def to_json(self, ensure_ascii: bool = False, indent: int = 2) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=indent)
    
    @classmethod
    def success(cls, data: Any = None, message: str = ""):
        """创建成功的响应"""
        return cls(success=True, data=data, message=message)
    
    @classmethod
    def error(cls, error: str, message: str = "", data: Any = None):
        """创建错误的响应"""
        return cls(success=False, error=error, message=message, data=data)
    
    @classmethod
    def no_data(cls, message: str = "暂无数据", data: Any = None):
        """创建无数据的响应"""
        if data is None:
            # 默认返回格式一致但数据为0的对象
            data = cls.get_default_empty_data()
        return cls(success=True, data=data, message=message)
    
    @staticmethod
    def get_default_empty_data() -> Dict:
        """获取默认的空数据格式"""
        return {
            "date": "",
            "bedtime": "",
            "wakeup_time": "",
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
            "summary": "暂无数据"
        }
    
    @staticmethod
    def get_default_physiological_empty_data() -> Dict:
        """获取默认的生理指标空数据格式"""
        return {
            "date": "",
            "heart_rate_metrics": {
                "avg_heart_rate": 0,
                "min_heart_rate": 0,
                "max_heart_rate": 0,
                "heart_rate_variability": 0,
                "heart_rate_stability": 0
            },
            "respiratory_metrics": {
                "avg_respiratory_rate": 0,
                "min_respiratory_rate": 0,
                "max_respiratory_rate": 0,
                "respiratory_stability": 0,
                "apnea_events_per_hour": 0,
                "apnea_count": 0,
                "avg_apnea_duration": 0,
                "max_apnea_duration": 0
            },
            "sleep_metrics": {
                "avg_body_moves_ratio": 0,
                "body_movement_frequency": 0,
                "sleep_efficiency": 0
            },
            "summary": "暂无数据"
        }


class SimpleApiResponse:
    """
    简化的API响应类，不包含timestamp字段
    """
    
    def __init__(self, success: bool = True, data: Any = None, message: str = "", error: str = None):
        self.success = success
        self.data = data
        self.message = message
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "success": self.success
        }
        
        if self.data is not None:
            result["data"] = self.data
        
        if self.message:
            result["message"] = self.message
            
        if self.error:
            result["error"] = self.error
            result["success"] = False  # 如果有错误，设置success为False
            
        return result
    
    def to_json(self, ensure_ascii: bool = False, indent: int = 2) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=indent)
    
    @classmethod
    def success(cls, data: Any = None, message: str = ""):
        """创建成功的响应"""
        return cls(success=True, data=data, message=message)
    
    @classmethod
    def error(cls, error: str, message: str = "", data: Any = None):
        """创建错误的响应"""
        return cls(success=False, error=error, message=message, data=data)


class SleepAnalysisResponse(ApiResponse):
    """睡眠分析专用响应类"""
    
    def __init__(self, 
                 success: bool = True, 
                 date: str = "",
                 bedtime: str = "",
                 wakeup_time: str = "",
                 time_in_bed_minutes: int = 0,
                 sleep_duration_minutes: int = 0,
                 sleep_score: int = 0,
                 bed_exit_count: int = 0,
                 sleep_prep_time_minutes: int = 0,
                 sleep_phases: Dict = None,
                 sleep_stage_segments: list = None,
                 average_metrics: Dict = None,
                 summary: str = "",
                 device_sn: str = None,
                 message: str = "",
                 error: str = None):
        
        if sleep_phases is None:
            sleep_phases = {
                "deep_sleep_minutes": 0,
                "light_sleep_minutes": 0,
                "rem_sleep_minutes": 0,
                "awake_minutes": 0,
                "deep_sleep_percentage": 0,
                "light_sleep_percentage": 0,
                "rem_sleep_percentage": 0,
                "awake_percentage": 0
            }
        
        if sleep_stage_segments is None:
            sleep_stage_segments = []
        
        if average_metrics is None:
            average_metrics = {
                "avg_heart_rate": 0,
                "avg_respiratory_rate": 0,
                "avg_body_moves_ratio": 0,
                "avg_heartbeat_interval": 0,
                "avg_rms_heartbeat_interval": 0
            }
        
        data = {
            "date": date,
            "bedtime": bedtime,
            "wakeup_time": wakeup_time,
            "time_in_bed_minutes": time_in_bed_minutes,
            "sleep_duration_minutes": sleep_duration_minutes,
            "sleep_score": sleep_score,
            "bed_exit_count": bed_exit_count,
            "sleep_prep_time_minutes": sleep_prep_time_minutes,
            "sleep_phases": sleep_phases,
            "sleep_stage_segments": sleep_stage_segments,
            "average_metrics": average_metrics,
            "summary": summary
        }
        
        if device_sn:
            data["device_sn"] = device_sn
        
        super().__init__(success=success, data=data, message=message, error=error)


class PhysiologicalAnalysisResponse(ApiResponse):
    """生理指标分析专用响应类"""
    
    def __init__(self, 
                 success: bool = True, 
                 date: str = "",
                 heart_rate_metrics: Dict = None,
                 respiratory_metrics: Dict = None,
                 sleep_metrics: Dict = None,
                 summary: str = "",
                 device_sn: str = None,
                 message: str = "",
                 error: str = None):
        
        if heart_rate_metrics is None:
            heart_rate_metrics = {
                "avg_heart_rate": 0,
                "min_heart_rate": 0,
                "max_heart_rate": 0,
                "heart_rate_variability": 0,
                "heart_rate_stability": 0
            }
        
        if respiratory_metrics is None:
            respiratory_metrics = {
                "avg_respiratory_rate": 0,
                "min_respiratory_rate": 0,
                "max_respiratory_rate": 0,
                "respiratory_stability": 0,
                "apnea_events_per_hour": 0,
                "apnea_count": 0,
                "avg_apnea_duration": 0,
                "max_apnea_duration": 0
            }
        
        if sleep_metrics is None:
            sleep_metrics = {
                "avg_body_moves_ratio": 0,
                "body_movement_frequency": 0,
                "sleep_efficiency": 0
            }
        
        data = {
            "date": date,
            "heart_rate_metrics": heart_rate_metrics,
            "respiratory_metrics": respiratory_metrics,
            "sleep_metrics": sleep_metrics,
            "summary": summary
        }
        
        if device_sn:
            data["device_sn"] = device_sn
        
        super().__init__(success=success, data=data, message=message, error=error)


class SleepDataCheckResponse(ApiResponse):
    """睡眠数据检查专用响应类"""
    
    def __init__(self, 
                 success: bool = True, 
                 date: str = "",
                 check_period: Dict = None,
                 has_sleep_data: bool = False,
                 has_heart_rate_data: bool = False,
                 has_respiratory_rate_data: bool = False,
                 total_records: int = 0,
                 heart_rate_records: int = 0,
                 respiratory_rate_records: int = 0,
                 first_record_time: str = None,
                 last_record_time: str = None,
                 device_sn: str = None,
                 message: str = "",
                 error: str = None):
        
        if check_period is None:
            check_period = {
                "start_time": "",
                "end_time": ""
            }
        
        data = {
            "date": date,
            "check_period": check_period,
            "has_sleep_data": has_sleep_data,
            "has_heart_rate_data": has_heart_rate_data,
            "has_respiratory_rate_data": has_respiratory_rate_data,
            "total_records": total_records,
            "heart_rate_records": heart_rate_records,
            "respiratory_rate_records": respiratory_rate_records,
            "first_record_time": first_record_time,
            "last_record_time": last_record_time,
            "message": message
        }
        
        if device_sn:
            data["device_sn"] = device_sn
        
        super().__init__(success=success, data=data, message=message, error=error)