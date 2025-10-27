#!/usr/bin/env python3
"""
测试优化后的训练代码
验证SFT冷启动和GPRO训练流程
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from train import AnswerComparator, GPROTrainer, TableQADataset
import yaml
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer


def test_answer_comparator():
    """测试答案对比器"""
    print("=== 测试答案对比器 ===")
    
    comparator = AnswerComparator()
    
    # 测试格式分数计算
    test_cases = [
        {
            "text": "<answer>\n42\n</answer>",
            "reference": "42",
            "expected_format": 1.0,
            "expected_em": 1.0
        },
        {
            "text": "<answer>\n7.67\n</answer>",
            "reference": "7.67", 
            "expected_format": 1.0,
            "expected_em": 1.0
        },
        {
            "text": "42",
            "reference": "42",
            "expected_format": 0.0,
            "expected_em": 1.0
        },
        {
            "text": "<answer>\nwrong answer\n</answer>",
            "reference": "42",
            "expected_format": 1.0,
            "expected_em": 0.0
        }
    ]
    
    for i, case in enumerate(test_cases):
        format_score = comparator.calculate_format_score(case["text"])
        em_score = comparator.calculate_em_score(
            comparator.extract_answer(case["text"]), 
            case["reference"]
        )
        reward = comparator.calculate_reward(case["text"], case["reference"])
        
        print(f"测试案例 {i+1}:")
        print(f"  格式分数: {format_score:.2f} (期望: {case['expected_format']:.2f})")
        print(f"  EM分数: {em_score:.2f} (期望: {case['expected_em']:.2f})")
        print(f"  总奖励: {reward:.2f}")
        print()


def test_dataset_format():
    """测试数据集格式"""
    print("=== 测试数据集格式 ===")
    
    # 加载配置
    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建模拟数据
    test_data = [
        {
            "question": "What is the average number of tropical cyclones per season?",
            "table": {
                "columns": ["season", "tropical cyclones"],
                "data": [
                    ["1990-91", "10"],
                    ["1991-92", "10"],
                    ["1992-93", "3"]
                ]
            },
            "answer": "7.67"
        }
    ]
    
    # 加载tokenizer
    model_path = config.get("model", {}).get("name", "./models/pretrained/Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建数据集
    dataset = TableQADataset(test_data, tokenizer, 512)
    
    # 测试数据格式
    sample = dataset[0]
    decoded_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
    
    print("生成的训练文本:")
    print(decoded_text)
    print()
    
    # 检查是否包含正确的格式
    has_answer = '<answer>' in decoded_text and '</answer>' in decoded_text
    
    print(f"包含<answer>标签: {has_answer}")
    print()


def test_gpro_trainer():
    """测试GPRO训练器"""
    print("=== 测试GPRO训练器 ===")
    
    # 加载配置
    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 加载模型和tokenizer
    model_path = config.get("model", {}).get("name", "./models/pretrained/Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # 创建GPRO训练器
    gpro_trainer = GPROTrainer(model, tokenizer, config)
    
    # 测试数据
    test_data = [
        {
            "question": "What is the average number of tropical cyclones per season?",
            "table": {
                "columns": ["season", "tropical cyclones"],
                "data": [
                    ["1990-91", "10"],
                    ["1991-92", "10"],
                    ["1992-93", "3"]
                ]
            },
            "answer": "7.67"
        }
    ]
    
    # 测试响应生成
    print("测试响应生成...")
    response = gpro_trainer.generate_response(
        test_data[0]["question"], 
        test_data[0]["table"]
    )
    print(f"生成的响应: {response[:200]}...")
    
    # 测试奖励计算
    reward = gpro_trainer.comparator.calculate_reward(response, test_data[0]["answer"])
    print(f"奖励分数: {reward:.4f}")
    print()


def main():
    """主测试函数"""
    print("开始测试优化后的训练代码...")
    print()
    
    try:
        test_answer_comparator()
        test_dataset_format()
        test_gpro_trainer()
        
        print("✅ 所有测试通过!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
