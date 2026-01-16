import logging
import time
from typing import Optional, Union

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

# 数据库连接超时时间（秒），每次尝试 15 秒，共尝试 2 次
DB_CONNECTION_TIMEOUT = 15
DB_MAX_RETRIES = 2


class MemoryManager:
    """Memory Manager 单例类"""

    _instance: Optional['MemoryManager'] = None
    _checkpointer: Optional[Union['MySQLSaver', MemorySaver]] = None
    _setup_done: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_db_url_safe(self) -> Optional[str]:
        """安全获取 db_url，失败时返回 None"""
        try:
            from ..database.db import get_db_url
            db_url = get_db_url()
            if db_url and db_url.strip():
                return db_url
            logger.warning("db_url is empty, will fallback to MemorySaver")
            return None
        except Exception as e:
            logger.warning(f"Failed to get db_url: {e}, will fallback to MemorySaver")
            return None

    def _create_fallback_checkpointer(self) -> MemorySaver:
        """创建内存兜底 checkpointer"""
        self._checkpointer = MemorySaver()
        logger.warning("Using MemorySaver as fallback checkpointer (data will not persist across restarts)")
        return self._checkpointer

    def get_checkpointer(self) -> BaseCheckpointSaver:
        """获取 checkpointer，优先使用 MySQLSaver，失败时退化为 MemorySaver"""
        if self._checkpointer is not None:
            return self._checkpointer

        # 1. 尝试获取 db_url
        db_url = self._get_db_url_safe()
        if not db_url:
            return self._create_fallback_checkpointer()

        # 2. 尝试连接MySQL数据库
        try:
            # 尝试导入MySQL相关模块 - 直接从pymysql子模块导入
            from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
            import pymysql
            
            # 检查是否为MySQL URL
            if db_url.startswith("mysql"):
                # 验证MySQL连接是否可用
                try:
                    with PyMySQLSaver.from_conn_string(db_url) as temp_saver:
                        temp_saver.setup()
                        logger.info("MySQL connection validated successfully")
                    
                    # 由于PyMySQLSaver设计为上下文管理器，我们不能直接保存实例
                    # 但我们可以确认连接是有效的，并记录这一状态
                    logger.info("MySQL connection is valid, will use MySQL when needed")
                except Exception as setup_error:
                    logger.warning(f"MySQL setup failed: {setup_error}, falling back to MemorySaver")
                    return self._create_fallback_checkpointer()
                
                # 创建并返回MySQLSaver实例（在实际使用时创建）
                class MySQLSaverWrapper(BaseCheckpointSaver):
                    def __init__(self, conn_string):
                        self.conn_string = conn_string
                        
                    def setup(self):
                        with PyMySQLSaver.from_conn_string(self.conn_string) as saver:
                            saver.setup()
                    
                    def put(self, thread_id, checkpoint, metadata, config_ids):
                        with PyMySQLSaver.from_conn_string(self.conn_string) as saver:
                            return saver.put(thread_id, checkpoint, metadata, config_ids)
                                
                    def get_tuple(self, config):
                        with PyMySQLSaver.from_conn_string(self.conn_string) as saver:
                            return saver.get_tuple(config)
                                
                    def list(self, config, **kwargs):
                        with PyMySQLSaver.from_conn_string(self.conn_string) as saver:
                            return saver.list(config, **kwargs)
                
                self._checkpointer = MySQLSaverWrapper(db_url)
                return self._checkpointer
            else:
                # 如果不是MySQL URL，回退到内存
                return self._create_fallback_checkpointer()
                
        except ImportError as e:
            logger.warning(f"MySQL support not available: {e}, falling back to MemorySaver")
            return self._create_fallback_checkpointer()
        except Exception as e:
            logger.warning(f"Failed to create MySQLSaver: {e}, will fallback to MemorySaver")
            return self._create_fallback_checkpointer()


_memory_manager: Optional[MemoryManager] = None


def get_memory_saver() -> BaseCheckpointSaver:
    """获取 checkpointer，优先使用 MySQLSaver，db_url 不可用或连接失败时退化为 MemorySaver"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager.get_checkpointer()