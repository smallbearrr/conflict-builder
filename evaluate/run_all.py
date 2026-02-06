"""
一键运行冲突检测和验证
"""
import argparse
from conflict_detector import ConflictDetector
from validate_conflicts import ConflictValidator

def main():
    parser = argparse.ArgumentParser(description='一键运行冲突检测和验证')
    parser.add_argument('--method', '-m', type=str, default='openai',
                        help='使用的方法名称 (默认: openai)')
    args = parser.parse_args()
    
    method = args.method
    
    print(f"\n{'='*60}")
    print(f"【步骤1】运行冲突检测 (方法: {method})")
    print(f"{'='*60}")
    
    # 冲突检测
    detector = ConflictDetector(method_name=method)
    conflicts = detector.process_chunks()
    detector.save_results(conflicts)
    
    print(f"\n{'='*60}")
    print(f"【步骤2】验证检测结果")
    print(f"{'='*60}")
    
    # 验证
    validator = ConflictValidator(results_file=detector.output_file)
    stats = validator.validate_all()
    validator.save_validation_results(stats)
    
    print(f"\n{'='*60}")
    print("完成!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
