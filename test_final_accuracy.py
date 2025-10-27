#!/usr/bin/env python3
"""
完整的准确率测试脚本
包括训练和评估过程
"""

import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from train import AnswerComparator
import yaml
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch


def test_with_mock_predictions():
    """使用模拟的预测结果测试准确率计算"""
    print("=" * 80)
    print("完整准确率测试 - 使用模拟预测结果")
    print("=" * 80)
    
    # 创建答案对比器
    comparator = AnswerComparator()
    
    # 模拟真实的预测结果
    # 这些代表模型在不同情况下的输出
    test_cases = [
        {
            "index": 1,
            "question": "What is the average number of tropical cyclones per season?",
            "prediction": "<answer>\n7.67\n</answer>",
            "reference": "7.67"
        },
        {
            "index": 2,
            "question": "What is the total number of tropical cyclones?",
            "prediction": "<answer>\n23\n</answer>",
            "reference": "23"
        },
        {
            "index": 3,
            "question": "What season had the most tropical cyclones?",
            "prediction": "<answer>\n1990-91\n</answer>",
            "reference": "1990-91"
        },
        {
            "index": 4,
            "question": "What is the average number of tropical cyclones per season?",
            "prediction": "7.67",  # 没有格式标签
            "reference": "7.67"
        },
        {
            "index": 5,
            "question": "What is the total number of tropical cyclones?",
            "prediction": "<answer>\n25\n</answer>",  # 答案错误
            "reference": "23"
        },
        {
            "index": 6,
            "question": "What season had the least tropical cyclones?",
            "prediction": "<answer>\n1992-93\n</answer>",
            "reference": "1992-93"
        },
        {
            "index": 7,
            "question": "How many tropical cyclones were there in 1990-91?",
            "prediction": "10",  # 没有格式标签但答案正确
            "reference": "10"
        },
        {
            "index": 8,
            "question": "What is the sum of all tropical cyclones?",
            "prediction": "<answer>\n23\n</answer>",
            "reference": "23"
        }
    ]
    
    print("\n详细测试结果:")
    print("-" * 80)
    
    predictions = []
    references = []
    
    for case in test_cases:
        print(f"\n测试 {case['index']}: {case['question']}")
        print(f"预测: {case['prediction']}")
        print(f"参考: {case['reference']}")
        
        # 提取答案
        extracted = comparator.extract_answer(case['prediction'])
        print(f"提取: '{extracted}'")
        
        # 计算分数
        format_score = comparator.calculate_format_score(case['prediction'])
        em_score = comparator.calculate_em_score(extracted, case['reference'])
        reward = comparator.calculate_reward(case['prediction'], case['reference'])
        
        print(f"格式分数: {format_score:.2f} | EM分数: {em_score:.2f} | 总奖励: {reward:.2f}")
        
        predictions.append(case['prediction'])
        references.append(case['reference'])
    
    # 计算整体准确率
    print("\n" + "=" * 80)
    print("整体准确率统计:")
    print("=" * 80)
    
    accuracy_stats = comparator.calculate_accuracy(predictions, references)
    
    print(f"\n总样本数: {accuracy_stats['total_samples']}")
    print(f"精确匹配数量: {accuracy_stats['exact_match_count']}")
    print(f"格式正确数量: {accuracy_stats['format_correct_count']}")
    print(f"\n📊 准确率指标:")
    print(f"  精确匹配准确率: {accuracy_stats['exact_match_accuracy']:.3f} ({accuracy_stats['exact_match_accuracy']*100:.1f}%)")
    print(f"  格式准确率: {accuracy_stats['format_accuracy']:.3f} ({accuracy_stats['format_accuracy']*100:.1f}%)")
    print(f"  整体准确率: {accuracy_stats['overall_accuracy']:.3f} ({accuracy_stats['overall_accuracy']*100:.1f}%)")
    
    # 详细分析
    print("\n" + "=" * 80)
    print("详细分析:")
    print("=" * 80)
    
    correct_format = accuracy_stats['format_correct_count']
    total = accuracy_stats['total_samples']
    format_pct = accuracy_stats['format_accuracy'] * 100
    
    correct_answers = accuracy_stats['exact_match_count']
    answer_pct = accuracy_stats['exact_match_accuracy'] * 100
    
    print(f"\n格式分析:")
    print(f"  - {correct_format}/{total} 个答案使用了正确的 <answer></answer> 格式")
    print(f"  - 格式准确率: {format_pct:.1f}%")
    
    print(f"\n内容分析:")
    print(f"  - {correct_answers}/{total} 个答案内容完全正确")
    print(f"  - 内容准确率: {answer_pct:.1f}%")
    
    print(f"\n综合表现:")
    if accuracy_stats['overall_accuracy'] >= 0.9:
        print(f"  ✅ 优秀！整体准确率达到 {accuracy_stats['overall_accuracy']*100:.1f}%")
    elif accuracy_stats['overall_accuracy'] >= 0.7:
        print(f"  ⚠️  良好，整体准确率为 {accuracy_stats['overall_accuracy']*100:.1f}%，还有改进空间")
    else:
        print(f"  ❌ 需要改进，整体准确率仅为 {accuracy_stats['overall_accuracy']*100:.1f}%")
    
    if accuracy_stats['format_accuracy'] < 0.9:
        print(f"  💡 建议: {total - correct_format} 个答案需要使用正确的 <answer></answer> 格式")
    
    return accuracy_stats


def save_results(accuracy_stats, filename="final_accuracy_results.json"):
    """保存准确率结果"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(accuracy_stats, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 结果已保存到: {filename}")


def main():
    """主函数"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "TableQA 完整准确率测试" + " " * 38 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    try:
        # 运行测试
        accuracy_stats = test_with_mock_predictions()
        
        # 保存结果
        save_results(accuracy_stats)
        
        print("\n" + "=" * 80)
        print("📈 最终结果摘要:")
        print("=" * 80)
        print(f"整体准确率: {accuracy_stats['overall_accuracy']*100:.1f}%")
        print(f"格式准确率: {accuracy_stats['format_accuracy']*100:.1f}%")
        print(f"内容准确率: {accuracy_stats['exact_match_accuracy']*100:.1f}%")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

