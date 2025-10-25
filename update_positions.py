"""
更新error_data.json中的position字段
根据陈述文本精准定位其在对应chunk中的句子位置
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

def split_into_sentences(text: str) -> List[str]:
    """
    将文本按句子分割
    
    Args:
        text: 输入文本
    
    Returns:
        List[str]: 句子列表
    """
    # 按句子分割(中文句号、问号、感叹号、分号等)
    sentences = re.split(r'([。!?;;]+)', text)
    
    # 将分隔符与前面的句子合并
    full_sentences = []
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            full_sentences.append(sentences[i] + sentences[i+1])
    if len(sentences) % 2 == 1:  # 如果最后有剩余
        full_sentences.append(sentences[-1])
    
    # 过滤空句子并去除首尾空格
    full_sentences = [s.strip() for s in full_sentences if s.strip()]
    
    return full_sentences

def find_sentence_position(statement: str, chunk_text: str) -> Optional[int]:
    """
    在chunk文本中查找陈述所在的句子位置(支持模糊匹配)
    
    Args:
        statement: 要查找的陈述
        chunk_text: chunk文本内容
    
    Returns:
        Optional[int]: 句子位置(从0开始),如果未找到返回None
    """
    sentences = split_into_sentences(chunk_text)
    
    # 清理陈述文本用于比对
    clean_statement = statement.strip()
    
    # 方法1: 精确匹配 - 完全相等
    for idx, sentence in enumerate(sentences):
        if clean_statement == sentence:
            return idx
    
    # 方法2: 包含匹配 - 陈述包含在句子中,或句子包含在陈述中
    for idx, sentence in enumerate(sentences):
        if clean_statement in sentence or sentence in clean_statement:
            return idx
    
    # 方法3: 模糊匹配 - 基于相似度(去除标点和空格后比较)
    def normalize_text(text: str) -> str:
        """规范化文本:去除标点符号和空格"""
        # 去除常见标点符号和空格
        import string
        translator = str.maketrans('', '', string.punctuation + '，。、；：！？''""（）【】《》—…·\t\n ')
        return text.translate(translator)
    
    normalized_statement = normalize_text(clean_statement)
    
    for idx, sentence in enumerate(sentences):
        normalized_sentence = normalize_text(sentence)
        
        # 如果规范化后完全相等
        if normalized_statement == normalized_sentence:
            return idx
        
        # 如果规范化后互相包含
        if normalized_statement in normalized_sentence or normalized_sentence in normalized_statement:
            return idx
    
    # 方法4: 基于编辑距离的模糊匹配(相似度阈值80%)
    def similarity_ratio(s1: str, s2: str) -> float:
        """计算两个字符串的相似度(基于最长公共子序列)"""
        if not s1 or not s2:
            return 0.0
        
        # 使用简单的字符重叠率
        set1 = set(s1)
        set2 = set(s2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    best_match_idx = None
    best_similarity = 0.0
    threshold = 0.8  # 80%相似度阈值
    
    for idx, sentence in enumerate(sentences):
        # 计算规范化后的相似度
        sim = similarity_ratio(normalized_statement, normalize_text(sentence))
        if sim > best_similarity and sim >= threshold:
            best_similarity = sim
            best_match_idx = idx
    
    if best_match_idx is not None:
        return best_match_idx
    
    return None

def update_positions_in_dataset(chunks_file: str, error_data_file: str, output_file: str = None):
    """
    更新error_data.json中的position字段
    
    Args:
        chunks_file: chunks.json文件路径
        error_data_file: error_data.json文件路径
        output_file: 输出文件路径(如果为None,则覆盖原文件)
    """
    print("开始更新position字段...")
    print("=" * 60)
    
    # 加载chunks
    print(f"\n加载chunks文件: {chunks_file}")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"已加载 {len(chunks)} 个chunks")
    
    # 加载error_data
    print(f"\n加载error_data文件: {error_data_file}")
    with open(error_data_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"已加载 {len(dataset)} 条冲突记录")
    
    # 更新position字段
    print(f"\n开始定位句子位置...")
    
    updated_count = 0
    not_found_count = 0
    
    for idx, record in enumerate(dataset, 1):
        print(f"\n[{idx}/{len(dataset)}] 处理冲突记录...")
        
        # 处理original_statement的position
        orig_chunk_id = record['original_statement']['chunk']
        orig_statement = record['original_statement']['statement']
        
        if orig_chunk_id < len(chunks):
            # 只在original_text中查找
            chunk_text = chunks[orig_chunk_id].get('original_text', '')
            if not chunk_text:
                print(f"  ⚠️ original_statement: Chunk {orig_chunk_id} 的 original_text 为空")
                not_found_count += 1
            else:
                position = find_sentence_position(orig_statement, chunk_text)
                
                if position is not None:
                    record['original_statement']['position'] = position
                    print(f"  original_statement: Chunk {orig_chunk_id}, 句子位置 {position}")
                    updated_count += 1
                else:
                    print(f"  ⚠️ original_statement: 未在Chunk {orig_chunk_id}的original_text中找到匹配")
                    print(f"     陈述: {orig_statement[:80]}...")
                    # 显示该chunk的前几个句子供调试
                    sentences = split_into_sentences(chunk_text)
                    print(f"     Chunk中共有 {len(sentences)} 个句子")
                    if sentences:
                        print(f"     第一个句子: {sentences[0][:80]}...")
                    not_found_count += 1
        else:
            print(f"  ❌ original_statement: Chunk {orig_chunk_id} 超出范围")
            not_found_count += 1
        
        # 处理conflicting_statement的position
        conf_chunk_id = record['conflicting_statement']['chunk']
        conf_statement = record['conflicting_statement']['statement']
        
        if conf_chunk_id < len(chunks):
            # 只在processed_text中查找
            chunk_text = chunks[conf_chunk_id].get('processed_text', '')
            if not chunk_text:
                print(f"  ⚠️ conflicting_statement: Chunk {conf_chunk_id} 的 processed_text 为空")
                not_found_count += 1
            else:
                position = find_sentence_position(conf_statement, chunk_text)
                
                if position is not None:
                    record['conflicting_statement']['position'] = position
                    print(f"  conflicting_statement: Chunk {conf_chunk_id}, 句子位置 {position}")
                    updated_count += 1
                else:
                    print(f"  ⚠️ conflicting_statement: 未在Chunk {conf_chunk_id}的processed_text中找到匹配")
                    print(f"     陈述: {conf_statement[:80]}...")
                    # 显示该chunk的前几个句子供调试
                    sentences = split_into_sentences(chunk_text)
                    print(f"     Chunk中共有 {len(sentences)} 个句子")
                    if sentences:
                        print(f"     第一个句子: {sentences[0][:80]}...")
                    not_found_count += 1
        else:
            print(f"  ❌ conflicting_statement: Chunk {conf_chunk_id} 超出范围")
            not_found_count += 1
    
    # 保存更新后的数据集
    output_path = output_file if output_file else error_data_file
    print(f"\n保存更新后的数据集到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    
    # 统计信息
    print(f"\n更新统计:")
    print(f"  总记录数: {len(dataset)}")
    print(f"  成功更新的position数: {updated_count}")
    print(f"  未找到的position数: {not_found_count}")
    print(f"  更新成功率: {updated_count / (updated_count + not_found_count) * 100:.1f}%")
    
    print("\nPosition字段更新完成！")

def main():
    """主函数"""
    # 配置文件路径
    chunks_file = "dataset/chunks.json"
    error_data_file = "dataset/error_data1.json"
    output_file = None  # None表示覆盖原文件,也可以指定新文件名
    
    # 检查文件是否存在
    if not Path(chunks_file).exists():
        print(f"Chunks文件不存在: {chunks_file}")
        return
    
    if not Path(error_data_file).exists():
        print(f"Error data文件不存在: {error_data_file}")
        return
    
    # 执行更新
    update_positions_in_dataset(chunks_file, error_data_file, output_file)

if __name__ == "__main__":
    main()
