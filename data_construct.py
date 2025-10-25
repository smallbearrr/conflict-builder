"""
使用阿里Qwen-Max自动构建文本冲突数据集
基于长文档分块，LLM自动生成前后矛盾的语句
"""

import json
import re
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import configparser
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from datetime import datetime

# 导入配置管理器和提示词
from config_manager import ConfigManager
from prompts import (
    EXTRACT_STATEMENTS_PROMPT,
    EXTRACT_STATEMENTS_SYSTEM,
    get_conflict_prompt,
    GENERATE_CONFLICT_SYSTEM,
    CATEGORY_DESCRIPTIONS
)

# 配置日志记录
def setup_logger(log_dir: str = "logs"):
    """
    配置日志记录器
    
    Args:
        log_dir: 日志文件目录
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名(带时间戳)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"data_construct_{timestamp}.log")
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("文本冲突数据集自动构建工具 - 日志记录启动")
    logger.info("=" * 80)
    
    return logger

# 初始化日志记录器
logger = setup_logger()

# 初始化配置管理器(使用正确的配置文件路径)
project_root = Path(__file__).parent
config_manager = ConfigManager(config_path=str(project_root / "config" / "config.cfg"))

# 初始化嵌入模型(用于计算相似度)
print("加载嵌入模型...")
logger.info("开始加载嵌入模型: sentence-transformers/all-MiniLM-L6-v2")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("嵌入模型加载完成")
logger.info("嵌入模型加载完成")

# 计算余弦相似度
def calculate_cosine_similarity(text1: str, text2: str) -> float:
    """
    计算两个文本之间的余弦相似度
    
    Args:
        text1: 第一个文本
        text2: 第二个文本
    
    Returns:
        float: 余弦相似度(0-1之间)
    """
    logger.debug(f"计算相似度:\n  文本1: {text1[:100]}...\n  文本2: {text2[:100]}...")
    embeddings = embedding_model.encode([text1, text2])
    similarity = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    logger.debug(f"相似度计算结果: {similarity:.4f}")
    return float(similarity)

# 初始化Qwen客户端
def init_qwen_client():
    """初始化Qwen API客户端"""
    logger.info("初始化Qwen API客户端")
    api_config = config_manager.get_api_config()
    
    if not api_config['api_key']:
        logger.error("API密钥未配置")
        raise ValueError("API密钥未配置！请检查 config/config.cfg 文件中的 ali_api_key 配置")
    
    logger.info(f"API Base URL: {api_config['base_url']}")
    client = OpenAI(
        api_key=api_config['api_key'],
        base_url=api_config['base_url']
    )
    logger.info("Qwen API客户端初始化成功")
    return client

# 读取并分块文档
def load_and_chunk_document(file_path: str, num_chunks: int = 10) -> List[Dict[str, Any]]:
    """
    加载文档并分成指定数量的块,确保不切断句子
    
    Args:
        file_path: 文档路径
        num_chunks: 分块数量
    
    Returns:
        List[Dict]: 包含chunk_id, text的列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 按句子分割(中文句号、问号、感叹号、英文句号等)
    import re
    sentences = re.split(r'([。!?;;\n]+)', text)
    # 将分隔符与前面的句子合并
    full_sentences = []
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            full_sentences.append(sentences[i] + sentences[i+1])
    if len(sentences) % 2 == 1:  # 如果最后有剩余
        full_sentences.append(sentences[-1])
    
    # 过滤空句子
    full_sentences = [s.strip() for s in full_sentences if s.strip()]
    
    total_sentences = len(full_sentences)
    sentences_per_chunk = total_sentences // num_chunks
    
    chunks = []
    
    for i in range(num_chunks):
        start_idx = i * sentences_per_chunk
        end_idx = (i + 1) * sentences_per_chunk if i < num_chunks - 1 else total_sentences
        
        chunk_content = ''.join(full_sentences[start_idx:end_idx])
        
        chunks.append({
            "chunk_id": i,
            "original_text": chunk_content.strip(),
            "processed_text": ""  # 初始为空,后续会插入冲突陈述
        })
    
    print(f"文档已分成 {len(chunks)} 个块(按句子边界分割)")
    logger.info(f"文档分块完成: 共{len(chunks)}个块, 总句子数{total_sentences}")
    for chunk in chunks:
        logger.info(f"  Chunk {chunk['chunk_id']}: {len(chunk['original_text'])}字符")
    return chunks

# 保存分块结果
def save_chunks(chunks: List[Dict[str, Any]], output_path: str):
    """保存分块结果到JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
    print(f"分块结果已保存到: {output_path}")

# 加载分块结果
def load_chunks(chunks_path: str) -> List[Dict[str, Any]]:
    """从JSON文件加载分块结果"""
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"已加载 {len(chunks)} 个文本块")
    return chunks

# 将冲突陈述插入到chunks中
def insert_conflicts_into_chunks(chunks: List[Dict[str, Any]], dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将冲突陈述插入到对应chunk的随机位置
    
    Args:
        chunks: 原始chunk列表
        dataset: 冲突数据集
    
    Returns:
        List[Dict]: 更新后的chunk列表(包含processed_text字段)
    """
    import random
    random.seed(42)
    
    # 初始化processed_text为original_text
    for chunk in chunks:
        if 'original_text' in chunk:
            chunk['processed_text'] = chunk['original_text']
        else:
            chunk['processed_text'] = ""
    
    print(f"\n将 {len(dataset)} 个冲突陈述插入到对应chunks...")
    
    # 按chunk_id分组冲突陈述
    conflicts_by_chunk = {}
    for conflict in dataset:
        target_chunk_id = conflict['conflicting_statement']['chunk']
        if target_chunk_id not in conflicts_by_chunk:
            conflicts_by_chunk[target_chunk_id] = []
        conflicts_by_chunk[target_chunk_id].append(conflict['conflicting_statement']['statement'])
    
    # 为每个chunk插入冲突陈述
    for chunk_id, conflict_statements in conflicts_by_chunk.items():
        if chunk_id >= len(chunks):
            print(f"  ⚠️ Chunk {chunk_id} 超出范围,跳过")
            continue
        
        chunk = chunks[chunk_id]
        text = chunk['processed_text']
        
        if not text:
            print(f"  ⚠️ Chunk {chunk_id} 文本为空,跳过")
            continue
        
        # 按句子分割
        sentences = re.split(r'([。!?;;]+)', text)
        full_sentences = []
        for i in range(0, len(sentences)-1, 2):
            if i+1 < len(sentences):
                full_sentences.append(sentences[i] + sentences[i+1])
        if len(sentences) % 2 == 1:
            full_sentences.append(sentences[-1])
        
        full_sentences = [s for s in full_sentences if s.strip()]
        
        if len(full_sentences) < 2:
            print(f"  ⚠️ Chunk {chunk_id} 句子数不足,直接追加")
            for stmt in conflict_statements:
                chunk['processed_text'] += stmt
            continue
        
        # 为每个冲突陈述随机选择插入位置
        for stmt in conflict_statements:
            # 随机选择两个相邻句子之间的位置(1到len-1之间)
            insert_pos = random.randint(1, len(full_sentences))
            full_sentences.insert(insert_pos, stmt)
            print(f"  Chunk {chunk_id}: 在位置 {insert_pos} 插入冲突陈述")
        
        # 重新组合文本
        chunk['processed_text'] = ''.join(full_sentences)
    
    print(f"冲突陈述插入完成")
    return chunks

# 使用LLM提取关键陈述
def extract_key_statements(client: OpenAI, chunk_text: str, chunk_id: int, num_statements: int = 3) -> List[Dict[str, Any]]:
    """
    使用LLM从文本块中提取关键陈述
    
    Args:
        client: OpenAI客户端
        chunk_text: 文本块内容
        chunk_id: 块ID
        num_statements: 要提取的陈述数量
    
    Returns:
        List[Dict]: 包含statement和position的列表
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"提取关键陈述 - Chunk {chunk_id}")
    logger.info(f"{'='*60}")
    
    prompt = EXTRACT_STATEMENTS_PROMPT.format(
        num_statements=num_statements,
        chunk_text=chunk_text[:2000]
    )
    
    logger.info(f"系统提示词:\n{EXTRACT_STATEMENTS_SYSTEM}")
    logger.info(f"\n用户提示词:\n{prompt}")

    try:
        logger.info(f"\n发送API请求: model=qwen-max-latest, temperature=0.1, max_tokens=1000")
        response = client.chat.completions.create(
            model="qwen-max-latest",
            messages=[
                {"role": "system", "content": EXTRACT_STATEMENTS_SYSTEM},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # 降低温度以提高准确性
            max_tokens=1000
        )
        
        result = response.choices[0].message.content
        logger.info(f"\nLLM返回内容:\n{result}")
        
        # 提取JSON
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            statements = data.get("statements", [])
            
            # 添加chunk_id
            for stmt in statements:
                stmt["chunk"] = chunk_id
            
            logger.info(f"\n成功提取 {len(statements)} 条陈述:")
            for i, stmt in enumerate(statements, 1):
                logger.info(f"  {i}. {stmt['statement']}")
            
            return statements
        else:
            logger.warning(f"Chunk {chunk_id} 未能解析JSON响应")
            print(f"⚠️ Chunk {chunk_id} 未能解析JSON响应")
            return []
            
    except Exception as e:
        logger.error(f"Chunk {chunk_id} 提取陈述失败: {e}", exc_info=True)
        print(f"❌ Chunk {chunk_id} 提取陈述失败: {e}")
        return []

# 生成冲突陈述
def generate_conflicting_statement(client: OpenAI, original_stmt: str, category: int, 
                                   target_chunk_id: int, chunk_distance: int) -> Dict[str, Any]:
    """
    使用LLM生成与原陈述冲突的语句
    
    Args:
        client: OpenAI客户端
        original_stmt: 原始陈述
        category: 冲突类型 (1=数值冲突, 2=语义冲突, 3=逻辑冲突)
        target_chunk_id: 目标块ID
        chunk_distance: 块间距离
    
    Returns:
        Dict: 冲突陈述信息
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"生成冲突陈述 - 类型{category} ({CATEGORY_DESCRIPTIONS.get(category, '未知')})")
    logger.info(f"{'='*60}")
    logger.info(f"原始陈述: {original_stmt}")
    logger.info(f"目标Chunk: {target_chunk_id}, 块距离: {chunk_distance}")
    
    # 根据类型获取对应的提示词
    prompt = get_conflict_prompt(category, original_stmt)
    
    logger.info(f"\n系统提示词:\n{GENERATE_CONFLICT_SYSTEM}")
    logger.info(f"\n用户提示词:\n{prompt}")

    try:
        logger.info(f"\n发送API请求: model=qwen-max-latest, temperature=0.7, max_tokens=500")
        response = client.chat.completions.create(
            model="qwen-max-latest",
            messages=[
                {"role": "system", "content": GENERATE_CONFLICT_SYSTEM},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        result = response.choices[0].message.content
        logger.info(f"\nLLM返回内容:\n{result}")
        
        # 提取JSON
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            
            conflict_info = {
                "statement": data.get("conflicting_statement", ""),
                "chunk": target_chunk_id,
                "position": -1,  # LLM生成的,无具体位置
                "contradiction_level": data.get("contradiction_level", 0)
            }
            
            logger.info(f"\n生成的冲突陈述: {conflict_info['statement']}")
            logger.info(f"矛盾层级: {conflict_info['contradiction_level']}")
            
            return conflict_info
        else:
            logger.warning("未能解析冲突陈述的JSON响应")
            print(f"⚠️ 未能解析冲突陈述的JSON响应")
            return None
            
    except Exception as e:
        logger.error(f"生成冲突陈述失败: {e}", exc_info=True)
        print(f"❌ 生成冲突陈述失败: {e}")
        return None

# 构建数据集
def build_conflict_dataset(client: OpenAI, chunks: List[Dict[str, Any]], 
                          num_conflicts: int = 10, 
                          min_chunk_distance: int = 3) -> List[Dict[str, Any]]:
    """
    构建完整的冲突数据集
    
    Args:
        client: OpenAI客户端
        chunks: 文档块列表
        num_conflicts: 要生成的冲突数量
        min_chunk_distance: 最小块间距离
    
    Returns:
        List[Dict]: 冲突数据集
    """
    import random
    import math
    random.seed(42)
    
    dataset = []
    
    # 计算每个chunk应该提取的陈述数量(向上取整)
    num_valid_chunks = len(chunks) - 1
    statements_per_chunk = math.ceil(num_conflicts / num_valid_chunks)
    
    print(f"\n从 {len(chunks)} 个块中提取关键陈述...")
    print(f"  有效chunk数: {num_valid_chunks} (排除最后一个)")
    print(f"  每个chunk提取: {statements_per_chunk} 条陈述")
    logger.info(f"\n开始构建冲突数据集: {len(chunks)}个块, 目标{num_conflicts}个冲突")
    logger.info(f"有效chunk数: {num_valid_chunks}, 每个chunk提取{statements_per_chunk}条陈述")
    
    # 第一步：从所有块中提取关键陈述(排除最后一个chunk,因为没有后续chunk可插入冲突)
    all_statements = []
    for chunk in chunks[:-1]:  # 排除最后一个chunk
        print(f"  处理 Chunk {chunk['chunk_id']}...")
        logger.info(f"\n处理 Chunk {chunk['chunk_id']}")
        # 从original_text字段读取原始文本
        chunk_text = chunk.get('original_text', chunk.get('text', ''))
        statements = extract_key_statements(client, chunk_text, chunk['chunk_id'], num_statements=statements_per_chunk)
        all_statements.extend(statements)
    
    print(f"共提取 {len(all_statements)} 条关键陈述(从前{len(chunks)-1}个chunk)")
    logger.info(f"\n总计提取 {len(all_statements)} 条关键陈述(排除最后一个chunk)")
    
    # 第二步：随机选择陈述并生成冲突
    print(f"\n生成 {num_conflicts} 个冲突对...")
    logger.info(f"\n开始生成 {num_conflicts} 个冲突对")
    
    selected_statements = random.sample(all_statements, min(num_conflicts, len(all_statements)))
    
    for idx, original_stmt_info in enumerate(selected_statements, 1):
        print(f"\n  [{idx}/{num_conflicts}] 处理陈述: {original_stmt_info['statement'][:50]}...")
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"冲突对 {idx}/{num_conflicts}")
        logger.info(f"{'#'*80}")
        
        original_chunk = original_stmt_info['chunk']
        
        # 首先尝试选择距离足够远且在原始chunk之后的目标块
        available_chunks = [c for c in range(original_chunk + min_chunk_distance, len(chunks))]
        
        # 如果没有满足min_chunk_distance的块,则从所有后续块中选择
        if not available_chunks:
            available_chunks = [c for c in range(original_chunk + 1, len(chunks))]
            if not available_chunks:
                print(f"    ⚠️ 没有后续块可用，跳过")
                logger.warning(f"Chunk {original_chunk} 没有后续块可用，跳过该冲突对")
                continue
            print(f"    ℹ️ 无法满足最小距离{min_chunk_distance},从剩余{len(available_chunks)}个后续块中选择")
            logger.info(f"无法满足最小距离{min_chunk_distance},从{len(available_chunks)}个后续块中选择")
        
        target_chunk = random.choice(available_chunks)
        chunk_distance = target_chunk - original_chunk
        
        # 随机选择冲突类型
        category = random.choice([1, 2, 3])
        logger.info(f"选择目标Chunk: {target_chunk}, 块距离: {chunk_distance}")
        logger.info(f"冲突类型: {category} ({CATEGORY_DESCRIPTIONS.get(category, '未知')})")
        
        # 生成冲突陈述
        conflicting_info = generate_conflicting_statement(
            client, 
            original_stmt_info['statement'],
            category,
            target_chunk,
            chunk_distance
        )
        
        if conflicting_info:
            # 计算实际的余弦相似度
            logger.info(f"\n计算余弦相似度")
            actual_similarity = calculate_cosine_similarity(
                original_stmt_info['statement'],
                conflicting_info['statement']
            )
            
            conflict_record = {
                "category": category,
                "original_statement": {
                    "statement": original_stmt_info['statement'],
                    "chunk": original_chunk,
                    "position": original_stmt_info.get('position', 0)
                },
                "conflicting_statement": conflicting_info,
                "chunk_distance": chunk_distance,
                "similarity": round(actual_similarity, 4),  # 使用实际计算的余弦相似度
                "contradiction_level": conflicting_info['contradiction_level'],
                "source": "auto"
            }
            
            dataset.append(conflict_record)
            print(f"    生成冲突对 (类型{category}, 距离{chunk_distance}块, 相似度{actual_similarity:.4f})")
            logger.info(f"\n冲突对生成成功:")
            logger.info(f"  类型: {category}")
            logger.info(f"  块距离: {chunk_distance}")
            logger.info(f"  余弦相似度: {actual_similarity:.4f}")
            logger.info(f"  矛盾层级: {conflicting_info['contradiction_level']}")
        else:
            print(f"    ❌ 冲突生成失败")
            logger.error(f"冲突对 {idx} 生成失败")
    
    print(f"\n成功生成 {len(dataset)} 个冲突对")
    logger.info(f"\n冲突数据集构建完成: 成功生成 {len(dataset)} 个冲突对")
    return dataset

# 保存数据集
def save_dataset(dataset: List[Dict[str, Any]], output_path: str):
    """保存数据集到JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    print(f"数据集已保存到: {output_path}")

# 生成统计报告
def generate_statistics(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """生成数据集统计信息"""
    stats = {
        "total_conflicts": len(dataset),
        "category_distribution": {},
        "contradiction_level_distribution": {},
        "avg_chunk_distance": 0,
        "avg_similarity": 0,
        "source_distribution": {}
    }
    
    for record in dataset:
        # 类别分布
        cat = record['category']
        stats['category_distribution'][cat] = stats['category_distribution'].get(cat, 0) + 1
        
        # 矛盾层级分布
        level = record['contradiction_level']
        stats['contradiction_level_distribution'][level] = \
            stats['contradiction_level_distribution'].get(level, 0) + 1
        
        # 来源分布
        source = record['source']
        stats['source_distribution'][source] = stats['source_distribution'].get(source, 0) + 1
        
        # 累加用于计算平均值
        stats['avg_chunk_distance'] += record['chunk_distance']
        stats['avg_similarity'] += record['similarity']
    
    # 计算平均值
    if dataset:
        stats['avg_chunk_distance'] /= len(dataset)
        stats['avg_similarity'] /= len(dataset)
    
    return stats

# 主函数
def main():
    """主函数"""
    print("文本冲突数据集自动构建工具")
    print("=" * 60)
    
    # 配置参数
    input_file = "dataset/data.txt"
    chunks_file = "dataset/chunks.json"
    output_file = "dataset/error_data1.json"
    num_chunks = 10
    num_conflicts = 50
    min_chunk_distance = 3
    
    logger.info("\n程序参数配置:")
    logger.info(f"  输入文件: {input_file}")
    logger.info(f"  分块文件: {chunks_file}")
    logger.info(f"  输出文件: {output_file}")
    logger.info(f"  分块数量: {num_chunks}")
    logger.info(f"  冲突数量: {num_conflicts}")
    logger.info(f"  最小块距离: {min_chunk_distance}")
    
    # 检查输入文件
    if not os.path.exists(input_file):
        print(f"输入文件不存在: {input_file}")
        logger.error(f"输入文件不存在: {input_file}")
        return
    
    # 初始化Qwen客户端
    print("\n初始化Qwen API客户端...")
    client = init_qwen_client()
    print("客户端初始化成功")
    
    # 步骤1: 加载并分块文档,保存到chunks.json
    print(f"\n步骤1: 加载文档并分块")
    print(f"  输入文件: {input_file}")
    logger.info("\n" + "="*80)
    logger.info("步骤1: 加载文档并分块")
    logger.info("="*80)
    chunks = load_and_chunk_document(input_file, num_chunks)
    save_chunks(chunks, chunks_file)
    logger.info(f"分块结果已保存: {chunks_file}")
    
    # print(f"\n📊 分块统计:")
    # for chunk in chunks:
    #     print(f"  Chunk {chunk['chunk_id']}: {len(chunk['text'])} 字符")
    
    # 步骤2: 从chunks.json读取分块,构建冲突数据集
    print(f"\n步骤2: 从分块文件读取并构建冲突数据集")
    logger.info("\n" + "="*80)
    logger.info("步骤2: 构建冲突数据集")
    logger.info("="*80)
    chunks = load_chunks(chunks_file)
    
    dataset = build_conflict_dataset(
        client, 
        chunks, 
        num_conflicts=num_conflicts,
        min_chunk_distance=min_chunk_distance
    )
    
    # 保存数据集
    print(f"\n步骤3: 保存冲突数据集...")
    logger.info("\n" + "="*80)
    logger.info("步骤3: 保存冲突数据集")
    logger.info("="*80)
    save_dataset(dataset, output_file)
    logger.info(f"冲突数据集已保存: {output_file}")
    
    # 步骤4: 将冲突陈述插入到chunks并更新chunks.json
    print(f"\n步骤4: 将冲突陈述插入到chunks")
    logger.info("\n" + "="*80)
    logger.info("步骤4: 插入冲突陈述到chunks")
    logger.info("="*80)
    chunks = insert_conflicts_into_chunks(chunks, dataset)
    save_chunks(chunks, chunks_file)
    logger.info(f"更新后的分块文件已保存: {chunks_file}")
    
    # 生成统计报告
    print(f"\n数据集统计:")
    logger.info("\n" + "="*80)
    logger.info("数据集统计信息")
    logger.info("="*80)
    stats = generate_statistics(dataset)
    print(f"  总冲突数: {stats['total_conflicts']}")
    print(f"  类别分布: {stats['category_distribution']}")
    print(f"  矛盾层级: {stats['contradiction_level_distribution']}")
    print(f"  平均块距离: {stats['avg_chunk_distance']:.2f}")
    print(f"  平均相似度: {stats['avg_similarity']:.2f}")
    print(f"  来源分布: {stats['source_distribution']}")
    
    logger.info(f"总冲突数: {stats['total_conflicts']}")
    logger.info(f"类别分布: {stats['category_distribution']}")
    logger.info(f"矛盾层级分布: {stats['contradiction_level_distribution']}")
    logger.info(f"平均块距离: {stats['avg_chunk_distance']:.2f}")
    logger.info(f"平均相似度: {stats['avg_similarity']:.2f}")
    logger.info(f"来源分布: {stats['source_distribution']}")
    
    # 保存统计信息
    stats_file = output_file.replace('.json', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)
    print(f"\n统计信息已保存到: {stats_file}")
    logger.info(f"统计信息已保存: {stats_file}")
    
    print("\n数据集构建完成！")
    print(f"  - 分块文件(含插入的冲突): {chunks_file}")
    print(f"  - 冲突数据集: {output_file}")
    print(f"  - 统计信息: {stats_file}")
    
    logger.info("\n" + "="*80)
    logger.info("数据集构建完成")
    logger.info("="*80)
    logger.info(f"分块文件: {chunks_file}")
    logger.info(f"冲突数据集: {output_file}")
    logger.info(f"统计信息: {stats_file}")
    logger.info(f"日志文件: logs/data_construct_*.log")

if __name__ == "__main__":
    main()
