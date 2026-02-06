"""
通用冲突检测器
删除了RAG相关部分，使用methods子文件夹下的chat.py接口进行冲突检测
"""
import json
import os
import sys
import re
import importlib.util
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import argparse

# 获取项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 默认路径配置
DATA_DIR = project_root / "evaluate" / "data"
RESULTS_DIR = project_root / "evaluate" / "results"
METHODS_DIR = project_root / "evaluate" / "methods"


def setup_logger(log_dir: str = "logs", method_name: str = "default") -> logging.Logger:
    """配置日志记录器"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{method_name}_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("通用冲突检测器启动")
    logger.info("=" * 80)
    return logger


# 冲突检测提示词
SYSTEM_PROMPT = """你是一个专业的文本一致性分析专家。你需要记住用户输入的所有文本内容，并在接收到新内容时，检索你记忆中的历史内容，判断是否存在冲突。

冲突类型定义:
1. **数值冲突**: 相同指标的数值不一致(如时间、比例、数量等)
2. **语义冲突**: 相同概念的描述出现矛盾(如推荐与不推荐、肯定与否定)
3. **逻辑冲突**: 前后逻辑关系矛盾(如因果关系颠倒、条件不一致)

输出要求:
- 必须严格按照以下JSON Schema格式输出
- 只返回JSON,不要任何额外解释
- 如果没有冲突,严禁进行编造
- 返回的语句必须逐字逐句从**原文提取**,**不得修改任何文字或标点**,**不得添加任何内容**

JSON Schema:
{
  "type": "object",
  "properties": {
    "conflicts": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "type": {"type": "integer", "enum": [1, 2, 3], "description": "1=数值冲突, 2=语义冲突, 3=逻辑冲突"},
          "old_sentence": {"type": "string", "description": "从记忆中检索到的原句"},
          "new_sentence": {"type": "string", "description": "当前输入中的冲突句子"},
          "description": {"type": "string", "description": "冲突描述"}
        },
        "required": ["type", "old_sentence", "new_sentence", "description"]
      }
    }
  },
  "required": ["conflicts"]
}
"""

# 第一个chunk的提示词 - 仅记忆
FIRST_CHUNK_PROMPT = """请记住以下内容(第{chunk_id}块):

{text}

请确认你已经记住了这段内容。回复:
```json
{{"status": "remembered", "chunk_id": {chunk_id}}}
```
"""

# 后续chunk的提示词 - 记忆并检测冲突
MEMORY_AND_DETECT_PROMPT = """请记住以下新内容(第{chunk_id}块)，并检查它是否与你之前记住的内容存在冲突:

{text}

请执行以下步骤:
1. 将上述内容存入你的记忆
2. 检索你之前记住的所有内容
3. 判断当前内容是否与之前的内容存在冲突(数值冲突、语义冲突或逻辑冲突)

严格按照JSON Schema格式输出:
{{
  "status": "remembered",
  "chunk_id": {chunk_id},
  "conflicts": []  // 冲突数组，每个元素包含 type(1/2/3), old_sentence, new_sentence, description
}}
"""


def load_chat_module(method_name: str):
    """
    动态加载指定方法的chat模块
    
    Args:
        method_name: 方法名称(如 'openai', 'amem', 'memgpt')
        
    Returns:
        加载的chat模块
    """
    chat_file = METHODS_DIR / method_name / "chat.py"
    
    if not chat_file.exists():
        raise FileNotFoundError(f"找不到方法 '{method_name}' 的 chat.py 文件: {chat_file}")
    
    spec = importlib.util.spec_from_file_location(f"{method_name}_chat", chat_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module


class ConflictDetector:
    """通用冲突检测器"""
    
    def __init__(self, method_name: str, chunks_file: str = None, output_file: str = None):
        """
        初始化冲突检测器
        
        Args:
            method_name: 使用的方法名称(对应methods目录下的子文件夹)
            chunks_file: chunks.json文件路径(默认为 evaluate/data/chunks.json)
            output_file: 输出的冲突检测结果文件路径(默认为 evaluate/results/{method_name}_conflicts.json)
        """
        self.method_name = method_name
        self.chunks_file = chunks_file or str(DATA_DIR / "chunks.json")
        self.output_file = output_file or str(RESULTS_DIR / f"{method_name}_conflicts.json")
        
        self.logger = setup_logger(str(project_root / "logs"), method_name)
        
        self.logger.info(f"初始化冲突检测器")
        self.logger.info(f"  方法: {method_name}")
        self.logger.info(f"  输入文件: {self.chunks_file}")
        self.logger.info(f"  输出文件: {self.output_file}")
        
        # 加载chat模块
        self.logger.info(f"加载 {method_name} 的chat模块...")
        self.chat_module = load_chat_module(method_name)
        self.logger.info("chat模块加载完成")
        
        # 获取chat API实例（用于保持会话状态）
        if hasattr(self.chat_module, 'get_chat_api'):
            self.chat_api = self.chat_module.get_chat_api()
        else:
            self.chat_api = None
        
        # 检查是否支持分离的 store/query 接口
        self.has_memory_api = hasattr(self.chat_module, 'store') and hasattr(self.chat_module, 'query')
    
    def load_chunks(self) -> List[Dict[str, Any]]:
        """加载chunks.json"""
        self.logger.info(f"加载chunks文件: {self.chunks_file}")
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        self.logger.info(f"已加载 {len(chunks)} 个chunks")
        return chunks
    
    def send_first_chunk(self, chunk_id: int, text: str) -> bool:
        """
        发送第一个chunk，让LLM记住内容
        
        Args:
            chunk_id: chunk的ID
            text: chunk的文本内容
            
        Returns:
            bool: 是否成功记住
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"发送第一个chunk (ID: {chunk_id}) 让LLM记忆")
        self.logger.info(f"{'='*60}")
        
        if self.has_memory_api:
            # 使用 store 接口直接储存原始文本
            self.logger.info(f"使用 store 接口储存原始文本")
            self.logger.info(f"{'~'*40} 储存内容开始 {'~'*40}")
            self.logger.info(f"{text}")
            self.logger.info(f"{'~'*40} 储存内容结束 {'~'*40}")
            success = self.chat_module.store(text)
            self.logger.info(f"储存结果: {'成功' if success else '失败'}")
            return success
        else:
            # 使用 chat 接口
            prompt = FIRST_CHUNK_PROMPT.format(chunk_id=chunk_id, text=text)
            self.logger.info(f"{'~'*40} SYSTEM_PROMPT 开始 {'~'*40}")
            self.logger.info(f"{SYSTEM_PROMPT}")
            self.logger.info(f"{'~'*40} SYSTEM_PROMPT 结束 {'~'*40}")
            self.logger.info(f"{'~'*40} USER_PROMPT 开始 {'~'*40}")
            self.logger.info(f"{prompt}")
            self.logger.info(f"{'~'*40} USER_PROMPT 结束 {'~'*40}")
            result = self.chat_module.chat(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.1,
                max_tokens=4096
            )
            self.logger.info(f"{'~'*40} LLM返回内容 开始 {'~'*40}")
            self.logger.info(f"{result}")
            self.logger.info(f"{'~'*40} LLM返回内容 结束 {'~'*40}")
            return True
    
    def send_chunk_and_detect(self, chunk_id: int, text: str) -> List[Dict[str, Any]]:
        """
        先查询检测冲突，再储存chunk
        
        Args:
            chunk_id: chunk的ID
            text: chunk的文本内容
            
        Returns:
            List[Dict]: 检测到的冲突列表
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"发送chunk (ID: {chunk_id}) 进行冲突检测和记忆")
        self.logger.info(f"{'='*60}")
        
        if self.has_memory_api:
            # 先用 query 检测冲突
            query_prompt = f"""请检查以下新内容是否与你记忆中的历史内容存在冲突:

新内容(第{chunk_id}块):
{text}

按照系统提示中的JSON Schema格式输出结果。"""
            
            self.logger.info(f"使用 query 接口检测冲突")
            self.logger.info(f"{'~'*40} SYSTEM_PROMPT 开始 {'~'*40}")
            self.logger.info(f"{SYSTEM_PROMPT}")
            self.logger.info(f"{'~'*40} SYSTEM_PROMPT 结束 {'~'*40}")
            self.logger.info(f"{'~'*40} QUERY_PROMPT 开始 {'~'*40}")
            self.logger.info(f"{query_prompt}")
            self.logger.info(f"{'~'*40} QUERY_PROMPT 结束 {'~'*40}")
            result = self.chat_module.query(query_prompt, system_prompt=SYSTEM_PROMPT)
            self.logger.info(f"{'~'*40} LLM返回内容 开始 {'~'*40}")
            self.logger.info(f"{result}")
            self.logger.info(f"{'~'*40} LLM返回内容 结束 {'~'*40}")
            
            # 再储存原始文本
            self.logger.info(f"使用 store 接口储存原始文本")
            self.logger.info(f"{'~'*40} 储存内容开始 {'~'*40}")
            self.logger.info(f"{text}")
            self.logger.info(f"{'~'*40} 储存内容结束 {'~'*40}")
            self.chat_module.store(text)
        else:
            # 使用 chat 接口
            prompt = MEMORY_AND_DETECT_PROMPT.format(chunk_id=chunk_id, text=text)
            self.logger.info(f"{'~'*40} SYSTEM_PROMPT 开始 {'~'*40}")
            self.logger.info(f"{SYSTEM_PROMPT}")
            self.logger.info(f"{'~'*40} SYSTEM_PROMPT 结束 {'~'*40}")
            self.logger.info(f"{'~'*40} USER_PROMPT 开始 {'~'*40}")
            self.logger.info(f"{prompt}")
            self.logger.info(f"{'~'*40} USER_PROMPT 结束 {'~'*40}")
            result = self.chat_module.chat(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.1,
                max_tokens=4096
            )
            self.logger.info(f"{'~'*40} LLM返回内容 开始 {'~'*40}")
            self.logger.info(f"{result}")
            self.logger.info(f"{'~'*40} LLM返回内容 结束 {'~'*40}")
        
        # 解析JSON响应
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            conflicts = data.get("conflicts", [])
            self.logger.info(f"\n检测到 {len(conflicts)} 个冲突")
            return conflicts
        else:
            self.logger.warning("未能解析JSON响应")
            return []
    
    def process_chunks(self) -> List[Dict[str, Any]]:
        """
        处理所有chunks，逐个发送让LLM记忆并检测冲突
        
        Returns:
            List[Dict]: 检测到的所有冲突
        """
        chunks = self.load_chunks()
        all_conflicts = []
        
        print(f"\n开始处理 {len(chunks)} 个chunks...")
        self.logger.info(f"\n开始处理 {len(chunks)} 个chunks")
        
        for idx, chunk in enumerate(chunks):
            chunk_id = chunk['chunk_id']
            text = chunk.get('processed_text', chunk.get('original_text', ''))
            
            if not text:
                self.logger.warning(f"Chunk {chunk_id} 文本为空,跳过")
                continue
            
            print(f"\n处理 Chunk {chunk_id} ({idx+1}/{len(chunks)})...")
            self.logger.info(f"\n{'#'*80}")
            self.logger.info(f"处理 Chunk {chunk_id}")
            self.logger.info(f"{'#'*80}")
            
            if idx == 0:
                # 第一个chunk只需要记忆
                success = self.send_first_chunk(chunk_id, text)
                if success:
                    print(f"  已记忆第一个chunk")
                else:
                    print(f"  记忆失败")
            else:
                # 后续chunk需要记忆并检测冲突
                conflicts = self.send_chunk_and_detect(chunk_id, text)
                
                # 记录冲突
                for conflict in conflicts:
                    conflict_record = {
                        "current_chunk_id": chunk_id,
                        "conflict_type": conflict.get("type"),
                        "old_sentence": conflict.get("old_sentence", ""),
                        "new_sentence": conflict.get("new_sentence", ""),
                        "description": conflict.get("description", ""),
                        "method": self.method_name
                    }
                    all_conflicts.append(conflict_record)
                    self.logger.info(f"\n发现冲突:")
                    self.logger.info(f"  类型: {conflict_record['conflict_type']}")
                    self.logger.info(f"  旧句子: {conflict_record['old_sentence']}")
                    self.logger.info(f"  新句子: {conflict_record['new_sentence']}")
                
                if not conflicts:
                    self.logger.info("未发现冲突")
                    print(f"  未发现冲突")
                else:
                    print(f"  发现 {len(conflicts)} 个冲突")
        
        print(f"\n处理完成!共检测到 {len(all_conflicts)} 个冲突")
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"处理完成!共检测到 {len(all_conflicts)} 个冲突")
        self.logger.info(f"{'='*80}")
        
        return all_conflicts
    
    def save_results(self, conflicts: List[Dict[str, Any]]):
        """保存检测结果"""
        self.logger.info(f"保存检测结果到: {self.output_file}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(self.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(conflicts, f, ensure_ascii=False, indent=4)
        
        print(f"\n检测结果已保存到: {self.output_file}")
        self.logger.info(f"检测结果已保存")
        
        # 生成统计信息
        stats = {
            "method": self.method_name,
            "total_conflicts": len(conflicts),
            "type_distribution": {}
        }
        
        for conflict in conflicts:
            conflict_type = conflict.get('conflict_type')
            if conflict_type:
                stats['type_distribution'][conflict_type] = \
                    stats['type_distribution'].get(conflict_type, 0) + 1
        
        stats_file = self.output_file.replace('.json', '_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)
        
        print(f"\n统计信息:")
        print(f"  方法: {stats['method']}")
        print(f"  总冲突数: {stats['total_conflicts']}")
        print(f"  类型分布: {stats['type_distribution']}")
        print(f"  统计文件: {stats_file}")
        
        self.logger.info(f"\n统计信息:")
        self.logger.info(f"  方法: {stats['method']}")
        self.logger.info(f"  总冲突数: {stats['total_conflicts']}")
        self.logger.info(f"  类型分布: {stats['type_distribution']}")
        self.logger.info(f"  统计文件: {stats_file}")


def list_available_methods() -> List[str]:
    """列出所有可用的方法"""
    methods = []
    if METHODS_DIR.exists():
        for item in METHODS_DIR.iterdir():
            if item.is_dir() and (item / "chat.py").exists():
                methods.append(item.name)
    return methods


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='通用冲突检测器')
    parser.add_argument('--method', '-m', type=str, default='openai',
                        help='使用的方法名称 (默认: openai)')
    parser.add_argument('--chunks', '-c', type=str, default=None,
                        help='chunks.json文件路径 (默认: evaluate/data/chunks.json)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出文件路径 (默认: evaluate/results/{method}_conflicts.json)')
    parser.add_argument('--list-methods', '-l', action='store_true',
                        help='列出所有可用的方法')
    
    args = parser.parse_args()
    
    # 列出可用方法
    if args.list_methods:
        methods = list_available_methods()
        print("可用的方法:")
        for m in methods:
            print(f"  - {m}")
        return
    
    print("通用冲突检测器")
    print("=" * 60)
    
    # 检查方法是否存在
    available_methods = list_available_methods()
    if args.method not in available_methods:
        print(f"错误: 方法 '{args.method}' 不存在")
        print(f"可用的方法: {', '.join(available_methods)}")
        return
    
    # 检查输入文件
    chunks_file = args.chunks or str(DATA_DIR / "chunks.json")
    if not os.path.exists(chunks_file):
        print(f"错误: chunks文件不存在 - {chunks_file}")
        return
    
    # 创建检测器
    detector = ConflictDetector(
        method_name=args.method,
        chunks_file=chunks_file,
        output_file=args.output
    )
    
    # 执行冲突检测
    conflicts = detector.process_chunks()
    
    # 保存结果
    detector.save_results(conflicts)
    
    print("\n冲突检测完成!")


if __name__ == "__main__":
    main()
