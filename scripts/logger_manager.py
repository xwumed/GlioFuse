import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

class LoggerManager:
    """统一的日志管理器"""
    
    def __init__(self, experiment_name: str = "default", log_dir: Optional[str] = None):
        self.experiment_name = experiment_name
        
        # 设置日志目录
        if log_dir is None:
            self.log_dir = os.path.join('..', 'results', 'logs', experiment_name)
        else:
            self.log_dir = log_dir
            
        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 设置日志文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.main_log_file = os.path.join(self.log_dir, f'{experiment_name}_{timestamp}.log')
        
        # 配置根日志器
        self._setup_root_logger()
        
    def _setup_root_logger(self):
        """配置根日志器"""
        # 获取根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # 清除现有的处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 创建格式器
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # 文件处理器（带轮转）
        file_handler = RotatingFileHandler(
            self.main_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
    def get_logger(self, name: str) -> logging.Logger:
        """获取指定名称的日志器"""
        return logging.getLogger(name)
    
    def create_experiment_logger(self, model_name: str, fold: Optional[int] = None) -> logging.Logger:
        """为特定实验创建专用日志器"""
        if fold is not None:
            logger_name = f"{self.experiment_name}.{model_name}.fold_{fold}"
            log_file = os.path.join(self.log_dir, f"{model_name}_fold_{fold}.log")
        else:
            logger_name = f"{self.experiment_name}.{model_name}"
            log_file = os.path.join(self.log_dir, f"{model_name}.log")
        
        logger = logging.getLogger(logger_name)
        
        # 如果已经配置过，直接返回
        if logger.handlers:
            return logger
            
        logger.setLevel(logging.DEBUG)
        
        # 创建文件处理器
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def get_log_dir(self) -> str:
        """获取日志目录路径"""
        return self.log_dir