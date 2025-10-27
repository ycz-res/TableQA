#!/usr/bin/env python3
"""
演示新的答案格式和准确率计算
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from train import AnswerComparator
import re


def demo_answer_format():
    """演示答案格式和准确率计算"""
    print("=== 答案格式和准确率计算演示 ===")
    
    # 创建答案对比器
    comparator = AnswerComparator()
    
    # 模拟测试数据
    test_cases = [
        {
            "question": "What is the average number of tropical cyclones per season?",
            "generated": "<answer>\n7.67\n</answer>",
            "reference": "7.67",
            "description": "正确格式 + 正确答案"
        },
        {
            "question": "What is the total number of tropical cyclones?",
            "generated": "<answer>\n23\n</answer>",
            "reference": "23",
            "description": "正确格式 + 正确答案"
        },
        {
            "question": "What is the average number of tropical cyclones per season?",
            "generated": "7.67",
            "reference": "7.67",
            "description": "错误格式 + 正确答案"
        },
        {
            "question": "What is the average number of tropical cyclones per season?",
            "generated": "<answer>\n8.5\n</answer>",
            "reference": "7.67",
            "description": "正确格式 + 错误答案"
        },
        {
            "question": "What is the average number of tropical cyclones per season?",
            "generated": "8.5",
            "reference": "7.67",
            "description": "错误格式 + 错误答案"
        }
    ]
    
    print("测试案例:")
    print("-" * 80)
    
    predictions = []
    references = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n案例 {i}: {case['description']}")
        print(f"问题: {case['question']}")
        print(f"生成答案: {case['generated']}")
        print(f"参考答案: {case['reference']}")
        
        # 提取答案
        extracted_answer = comparator.extract_answer(case['generated'])
        print(f"提取的答案: '{extracted_answer}'")
        
        # 计算分数
        format_score = comparator.calculate_format_score(case['generated'])
        em_score = comparator.calculate_em_score(extracted_answer, case['reference'])
        reward = comparator.calculate_reward(case['generated'], case['reference'])
        
        print(f"格式分数: {format_score:.2f}")
        print(f"EM分数: {em_score:.2f}")
        print(f"总奖励: {reward:.2f}")
        
        predictions.append(case['generated'])
        references.append(case['reference'])
    
    # 计算整体准确率
    print("\n" + "=" * 80)
    print("整体准确率统计:")
    print("=" * 80)
    
    accuracy_stats = comparator.calculate_accuracy(predictions, references)
    
    print(f"总样本数: {accuracy_stats['total_samples']}")
    print(f"精确匹配数量: {accuracy_stats['exact_match_count']}")
    print(f"格式正确数量: {accuracy_stats['format_correct_count']}")
    print(f"精确匹配准确率: {accuracy_stats['exact_match_accuracy']:.3f}")
    print(f"格式准确率: {accuracy_stats['format_accuracy']:.3f}")
    print(f"整体准确率: {accuracy_stats['overall_accuracy']:.3f}")
    
    # 分析结果
    print("\n" + "=" * 80)
    print("结果分析:")
    print("=" * 80)
    
    if accuracy_stats['format_accuracy'] == 1.0:
        print("✅ 所有答案都使用了正确的<answer></answer>格式")
    else:
        print(f"❌ 只有 {accuracy_stats['format_accuracy']:.1%} 的答案使用了正确格式")
    
    if accuracy_stats['exact_match_accuracy'] == 1.0:
        print("✅ 所有答案都完全正确")
    else:
        print(f"❌ 只有 {accuracy_stats['exact_match_accuracy']:.1%} 的答案完全正确")
    
    print(f"\n综合表现: 格式准确率 {accuracy_stats['format_accuracy']:.1%}, 内容准确率 {accuracy_stats['exact_match_accuracy']:.1%}")


def demo_answer_extraction():
    """演示答案提取功能"""
    print("\n=== 答案提取功能演示 ===")
    
    comparator = AnswerComparator()
    
    test_texts = [
        "<answer>\n42\n</answer>",
        "<answer>7.67</answer>",
        "The answer is 42",
        "<answer>\nThe average is 7.67\n</answer>",
        "Some text <answer>23</answer> more text",
        "No answer tags here"
    ]
    
    print("答案提取测试:")
    print("-" * 50)
    
    for i, text in enumerate(test_texts, 1):
        extracted = comparator.extract_answer(text)
        has_format = comparator.calculate_format_score(text)
        
        print(f"{i}. 输入: {text}")
        print(f"   提取: '{extracted}'")
        print(f"   格式正确: {has_format == 1.0}")
        print()


if __name__ == "__main__":
    demo_answer_format()
    demo_answer_extraction()
    
    print("\n" + "=" * 80)
    print("演示完成!")
    print("=" * 80)
    print("总结:")
    print("1. 模型输出应该使用 <answer>答案</answer> 格式")
    print("2. 准确率计算包括格式准确率和内容准确率")
    print("3. 总奖励 = 0.1 * 格式分数 + 0.9 * EM分数")
    print("4. 整体准确率 = 精确匹配准确率")
