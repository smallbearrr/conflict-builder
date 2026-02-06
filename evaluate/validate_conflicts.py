"""
冲突检测结果验证器
用于判断检测出的冲突是否存在于标准答案(error_data.json)中
"""
import json
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI

# 获取项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config_manager import ConfigManager

# 默认路径配置
DATA_DIR = project_root / "evaluate" / "data"
RESULTS_DIR = project_root / "evaluate" / "results"


class ConflictValidator:
    """冲突检测结果验证器"""
    
    def __init__(self, error_data_file: str = None, results_file: str = None):
        """
        初始化验证器
        
        Args:
            error_data_file: 标准答案文件路径 (error_data.json)
            results_file: 检测结果文件路径 (xxx_conflicts.json)
        """
        self.error_data_file = error_data_file or str(DATA_DIR / "error_data.json")
        self.results_file = results_file
        
        # 加载配置
        config_path = project_root / "config" / "config.cfg"
        config_manager = ConfigManager(config_path=str(config_path))
        config = config_manager.config
        
        # 初始化 OpenAI 客户端
        self.api_key = config.get('openai', 'openai_api_key', fallback=None)
        self.base_url = config.get('openai', 'openai_base_url', fallback=None)
        self.model = config.get('openai', 'openai_model', fallback='openai/gpt-4o')
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        ) if self.base_url else OpenAI(api_key=self.api_key)
        
        # 加载标准答案
        self.error_data = self.load_error_data()
    
    def load_error_data(self) -> List[Dict]:
        """加载标准答案"""
        with open(self.error_data_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_results(self, results_file: str = None) -> List[Dict]:
        """加载检测结果"""
        file_path = results_file or self.results_file
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_ground_truth_by_chunk(self, chunk_id: int) -> List[Dict]:
        """
        根据 chunk_id 获取标准答案中所有 conflicting_statement.chunk 相同的项
        
        Args:
            chunk_id: 当前检测冲突的 chunk ID
            
        Returns:
            List[Dict]: 匹配的标准答案列表
        """
        matches = []
        for item in self.error_data:
            if item.get('conflicting_statement', {}).get('chunk') == chunk_id:
                matches.append({
                    'original_statement': item['original_statement']['statement'],
                    'conflicting_statement': item['conflicting_statement']['statement'],
                    'category': item.get('category'),
                    'original_chunk': item['original_statement']['chunk'],
                    'conflicting_chunk': item['conflicting_statement']['chunk']
                })
        return matches
    
    def validate_single_conflict(self, detected: Dict, ground_truths: List[Dict]) -> Dict:
        """
        使用 LLM 验证单个检测到的冲突是否在标准答案中存在
        
        Args:
            detected: 检测到的冲突
            ground_truths: 该 chunk 对应的所有标准答案
            
        Returns:
            Dict: 验证结果
        """
        if not ground_truths:
            return {
                'detected': detected,
                'match': False,
                'reason': '该chunk在标准答案中没有冲突记录'
            }
        
        # 构造提示词
        ground_truth_text = "\n".join([
            f"标答{i+1}:\n  原句: {gt['original_statement']}\n  冲突句: {gt['conflicting_statement']}"
            for i, gt in enumerate(ground_truths)
        ])
        
        prompt = f"""请判断检测到的冲突是否与标准答案中的某一对匹配。

检测到的冲突:
  原句(old_sentence): {detected.get('old_sentence', '')}
  冲突句(new_sentence): {detected.get('new_sentence', '')}

标准答案(共{len(ground_truths)}对):
{ground_truth_text}

判断规则:
1. 原句和冲突句都需要能够匹配上才算匹配成功
2. 匹配不要求完全一致，但核心内容和含义必须相同
3. 如果检测到的原句对应标答的original_statement，冲突句对应标答的conflicting_statement，则匹配成功

请严格按照以下JSON格式输出:
{{
  "match": true/false,
  "matched_index": 匹配的标答编号(1开始，如果不匹配则为null),
  "reason": "匹配/不匹配的原因说明"
}}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个专业的文本匹配专家，负责判断检测到的冲突是否与标准答案匹配。只返回JSON，不要任何额外解释。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        result_text = response.choices[0].message.content
        
        # 解析 JSON
        try:
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    'detected': detected,
                    'ground_truths_count': len(ground_truths),
                    'match': result.get('match', False),
                    'matched_index': result.get('matched_index'),
                    'reason': result.get('reason', '')
                }
        except json.JSONDecodeError:
            pass
        
        return {
            'detected': detected,
            'ground_truths_count': len(ground_truths),
            'match': False,
            'reason': f'LLM返回解析失败: {result_text}'
        }
    
    def validate_all(self, results_file: str = None) -> Dict:
        """
        验证所有检测结果
        
        Args:
            results_file: 检测结果文件路径
            
        Returns:
            Dict: 验证统计结果
        """
        results = self.load_results(results_file)
        
        # 计算标准答案中的总冲突数
        total_ground_truth = len(self.error_data)
        
        print(f"\n开始验证 {len(results)} 个检测到的冲突...")
        print(f"标准答案中共有 {total_ground_truth} 个冲突")
        print("=" * 60)
        
        validations = []
        true_positives = 0
        false_positives = 0
        matched_ground_truths = set()  # 记录已匹配的标答索引
        
        for idx, detected in enumerate(results):
            chunk_id = detected.get('current_chunk_id')
            print(f"\n验证第 {idx+1}/{len(results)} 个冲突 (chunk_id={chunk_id})...")
            
            # 获取该 chunk 的标准答案
            ground_truths = self.get_ground_truth_by_chunk(chunk_id)
            print(f"  标准答案中有 {len(ground_truths)} 个冲突对")
            
            # 验证
            validation = self.validate_single_conflict(detected, ground_truths)
            validations.append(validation)
            
            if validation['match']:
                true_positives += 1
                # 记录匹配的标答（用于计算召回率）
                matched_idx = validation.get('matched_index')
                if matched_idx:
                    # 找到对应的标答在 error_data 中的索引
                    for i, item in enumerate(self.error_data):
                        if item.get('conflicting_statement', {}).get('chunk') == chunk_id:
                            if item['conflicting_statement']['statement'] == ground_truths[matched_idx - 1]['conflicting_statement']:
                                matched_ground_truths.add(i)
                                break
                print(f"  ✓ 匹配成功 (标答{validation.get('matched_index')})")
            else:
                false_positives += 1
                print(f"  ✗ 匹配失败: {validation['reason']}")
        
        # 计算各项指标
        total_detected = len(results)
        false_negatives = total_ground_truth - len(matched_ground_truths)  # 未被检测到的标答
        
        # 精确率 Precision = TP / (TP + FP)
        precision = true_positives / total_detected if total_detected > 0 else 0
        
        # 召回率 Recall = TP / (TP + FN) = 匹配的标答数 / 标答总数
        recall = len(matched_ground_truths) / total_ground_truth if total_ground_truth > 0 else 0
        
        # F1 分数 = 2 * Precision * Recall / (Precision + Recall)
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # F0.5 分数 (更重视精确率)
        f0_5_score = 1.25 * precision * recall / (0.25 * precision + recall) if (0.25 * precision + recall) > 0 else 0
        
        # F2 分数 (更重视召回率)
        f2_score = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
        
        # 统计结果
        stats = {
            'total_ground_truth': total_ground_truth,
            'total_detected': total_detected,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'matched_ground_truths': len(matched_ground_truths),
            'metrics': {
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1_score, 4),
                'f0_5_score': round(f0_5_score, 4),
                'f2_score': round(f2_score, 4)
            },
            'validations': validations
        }
        
        print("\n" + "=" * 60)
        print("验证完成!")
        print(f"\n【数量统计】")
        print(f"  标准答案总数: {stats['total_ground_truth']}")
        print(f"  检测到的冲突数: {stats['total_detected']}")
        print(f"  正确检测(TP): {stats['true_positives']}")
        print(f"  错误检测(FP): {stats['false_positives']}")
        print(f"  漏检(FN): {stats['false_negatives']}")
        print(f"\n【评估指标】")
        print(f"  精确率(Precision): {stats['metrics']['precision']:.2%}")
        print(f"  召回率(Recall): {stats['metrics']['recall']:.2%}")
        print(f"  F1分数: {stats['metrics']['f1_score']:.4f}")
        print(f"  F0.5分数(重精确): {stats['metrics']['f0_5_score']:.4f}")
        print(f"  F2分数(重召回): {stats['metrics']['f2_score']:.4f}")
        
        return stats
    
    def save_validation_results(self, stats: Dict, output_file: str = None):
        """保存验证结果"""
        if output_file is None:
            # 从结果文件名生成验证结果文件名
            base_name = Path(self.results_file).stem
            output_file = str(RESULTS_DIR / f"{base_name}_validation.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)
        
        print(f"\n验证结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='冲突检测结果验证器')
    parser.add_argument('--results', '-r', type=str, required=True,
                        help='检测结果文件路径 (如 evaluate/results/amem_conflicts.json)')
    parser.add_argument('--error-data', '-e', type=str, default=None,
                        help='标准答案文件路径 (默认: evaluate/data/error_data.json)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='验证结果输出文件路径')
    
    args = parser.parse_args()
    
    print("冲突检测结果验证器")
    print("=" * 60)
    
    # 创建验证器
    validator = ConflictValidator(
        error_data_file=args.error_data,
        results_file=args.results
    )
    
    # 执行验证
    stats = validator.validate_all()
    
    # 保存结果
    validator.save_validation_results(stats, args.output)


if __name__ == "__main__":
    main()
