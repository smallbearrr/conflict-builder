"""
记忆系统抽象基类
定义 store 和 query 两个标准接口
"""
from abc import ABC, abstractmethod
from typing import Optional


class BaseMemoryAPI(ABC):
    """记忆系统抽象基类"""
    
    @abstractmethod
    def store(self, content: str, time: str = None) -> bool:
        """
        储存内容到记忆系统
        
        Args:
            content: 要储存的文本内容
            time: 可选的时间标记
            
        Returns:
            bool: 是否成功储存
        """
        pass
    
    @abstractmethod
    def query(self, question: str, system_prompt: str = None, temperature: float = None) -> str:
        """
        查询记忆并生成回答
        
        Args:
            question: 用户问题
            system_prompt: 系统提示词（可选）
            temperature: 温度参数
            
        Returns:
            str: 模型返回的回答
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空记忆"""
        pass
    
    def get_memory_count(self) -> int:
        """获取记忆条目数量（可选实现）"""
        return 0
