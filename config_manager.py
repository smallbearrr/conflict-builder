"""
配置文件读取模块
支持从config/config.cfg读取API密钥和其他配置信息
"""

import os
import configparser
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = "config/config.cfg"):
        self.config_path = Path(config_path)
        self.config = configparser.ConfigParser()
        self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        if not self.config_path.exists():
            logger.warning(f"配置文件不存在: {self.config_path}")
            return
        
        try:
            self.config.read(self.config_path, encoding='utf-8')
            logger.info(f"成功加载配置文件: {self.config_path}")
        except Exception as e:
            logger.error(f"读取配置文件失败: {e}")
    
    def get_api_config(self):
        """获取API配置"""
        config = {
            'api_key': None,
            'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'model': 'qwen-max-latest'
        }
        
        # 从配置文件读取
        if self.config.has_section('API_KEYS'):
            config['api_key'] = self.config.get('API_KEYS', 'ali_api_key', fallback=None)
            config['base_url'] = self.config.get('API_KEYS', 'ali_base_url', 
                                               fallback='https://dashscope.aliyuncs.com/compatible-mode/v1')
        
        # 如果配置文件中没有，尝试从环境变量读取
        if not config['api_key']:
            config['api_key'] = os.getenv('DASHSCOPE_API_KEY')
        
        return config
    
    def get_rag_config(self):
        """获取RAG配置"""
        config = {
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'chunk_size': 500,
            'chunk_overlap': 100,
            'vector_store_path': './chroma_db'
        }
        
        if self.config.has_section('RAG_CONFIG'):
            config['embedding_model'] = self.config.get('RAG_CONFIG', 'embedding_model', 
                                                       fallback='sentence-transformers/all-MiniLM-L6-v2')
            config['chunk_size'] = self.config.getint('RAG_CONFIG', 'chunk_size', fallback=500)
            config['chunk_overlap'] = self.config.getint('RAG_CONFIG', 'chunk_overlap', fallback=100)
            config['vector_store_path'] = self.config.get('RAG_CONFIG', 'vector_store_path', 
                                                         fallback='./chroma_db')
        
        return config
    
    def get_dataset_config(self):
        """获取数据集配置"""
        config = {
            'dataset_path': 'dataset/locomo10.json',
            'num_samples': None
        }
        
        if self.config.has_section('DATASET'):
            config['dataset_path'] = self.config.get('DATASET', 'dataset_path', 
                                                    fallback='dataset/locomo10.json')
            config['num_samples'] = self.config.getint('DATASET', 'num_samples', fallback=None)
        
        return config


# 全局配置管理器实例
config_manager = ConfigManager()