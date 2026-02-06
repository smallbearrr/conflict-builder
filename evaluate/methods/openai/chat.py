"""
OpenAI Chat API 接口实现
使用对话历史列表作为记忆机制
"""
import os
import sys
from pathlib import Path
from openai import OpenAI

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
from config_manager import ConfigManager
from evaluate.methods.base import BaseMemoryAPI


class ChatAPI(BaseMemoryAPI):
    """OpenAI Chat API 封装类 - 使用对话历史列表作为记忆"""
    
    def __init__(self):
        """初始化 OpenAI 客户端"""
        config_path = project_root / "config" / "config.cfg"
        config_manager = ConfigManager(config_path=str(config_path))
        
        # 从 [openai] section 读取配置
        config = config_manager.config
        api_key = config.get('openai', 'openai_api_key', fallback=None)
        base_url = config.get('openai', 'openai_base_url', fallback='https://api.openai.com/v1')
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # 从配置读取模型参数
        self.model = config.get('openai', 'openai_model', fallback='openai/gpt-4o')
        self.default_temperature = config.getfloat('openai', 'openai_temperature', fallback=0.1)
        self.default_max_tokens = config.getint('openai', 'openai_max_tokens', fallback=4096)
        
        # 对话历史记录（用于记忆功能）
        self.conversation_history = []
        self._system_prompt = None
    
    def store(self, content: str, time: str = None) -> bool:
        """
        储存内容到记忆（添加为用户消息并让模型确认）
        
        Args:
            content: 要储存的文本内容
            time: 可选的时间标记
            
        Returns:
            bool: 是否成功储存
        """
        # 构造储存消息
        store_content = content
        if time:
            store_content = f"[{time}] {content}"
        
        # 添加到历史作为用户提供的信息
        self.conversation_history.append({
            "role": "user", 
            "content": f"请记住以下信息：{store_content}"
        })
        # 添加助手确认
        self.conversation_history.append({
            "role": "assistant", 
            "content": "好的，我已记住这条信息。"
        })
        return True
    
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
        if temperature is None:
            temperature = self.default_temperature
        
        if system_prompt:
            self._system_prompt = system_prompt
        
        # 添加用户问题到历史
        self.conversation_history.append({"role": "user", "content": question})
        
        # 构建完整消息列表
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.extend(self.conversation_history)
            
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=self.default_max_tokens
        )
        
        assistant_message = response.choices[0].message.content
        
        # 保存助手回复到历史
        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message
    
    def clear(self) -> None:
        """清空记忆"""
        self.conversation_history = []
        self._system_prompt = None
    
    def get_memory_count(self) -> int:
        """获取记忆条目数量"""
        # 每个 store 产生 2 条记录（用户+助手），每个 query 也产生 2 条
        return len(self.conversation_history) // 2
    
    def get_history(self):
        """获取对话历史"""
        return self.conversation_history.copy()


# 创建全局实例
_chat_api = None

def get_chat_api() -> ChatAPI:
    """获取 ChatAPI 单例实例"""
    global _chat_api
    if _chat_api is None:
        _chat_api = ChatAPI()
    return _chat_api


def store(content: str, time: str = None) -> bool:
    """
    储存内容到记忆系统
    
    Args:
        content: 要储存的文本内容
        time: 可选的时间标记
        
    Returns:
        bool: 是否成功储存
    """
    api = get_chat_api()
    return api.store(content, time)


def query(question: str, system_prompt: str = None, temperature: float = None) -> str:
    """
    查询记忆并生成回答
    
    Args:
        question: 用户问题
        system_prompt: 系统提示词（可选）
        temperature: 温度参数
        
    Returns:
        str: 模型返回的回答
    """
    api = get_chat_api()
    return api.query(question, system_prompt, temperature)
