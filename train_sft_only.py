#!/usr/bin/env python3
"""
简化的训练脚本 - 只进行SFT训练
修复格式问题
"""
import sys
sys.path.append('src')
sys.path.append('datasets/tablebench')

import yaml
import torch
from load_tablebench import TableBenchLoader
from train import train_sft_cold_start, setup_lora
from modelscope import AutoModelForCausalLM, AutoTokenizer
from eval import evaluate_pipeline, AnswerComparator
import json
import time

def load_config():
    """加载配置"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def train_sft_only():
    """只进行SFT训练"""
    print("="*80)
    print("SFT训练 - 修复格式问题")
    print("="*80)
    
    # 加载配置
    config = load_config()
    
    # 使用中等数据集
    config['data']['train_samples'] = 200
    config['data']['eval_samples'] = 50
    
    # 加载数据集
    print("\n[1/3] 加载数据集...")
    loader = TableBenchLoader('datasets/tablebench')
    
    train_data = loader.load(version='base', max_samples=config['data']['train_samples'])
    eval_data = loader.load(version='base', max_samples=config['data']['eval_samples'])
    
    print(f"✓ 训练集: {len(train_data)} 条")
    print(f"✓ 测试集: {len(eval_data)} 条")
    
    # 训练模型
    print("\n[2/3] 开始SFT训练...")
    try:
        model, tokenizer = train_sft_cold_start(train_data, eval_data, config)
        print("✓ SFT训练完成")
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        return None
    
    # 测试模型输出格式
    print("\n[3/3] 测试模型输出格式...")
    from src.pipeline import TableQAPipeline
    from src.mcp_tools import MCPToolManager
    
    mcp_manager = MCPToolManager()
    pipeline = TableQAPipeline(model, tokenizer, mcp_manager)
    
    # 测试几个样本
    test_samples = eval_data[:10]
    format_correct = 0
    content_correct = 0
    
    for i, sample in enumerate(test_samples):
        question = sample['question']
        table = sample['table']
        reference = sample['answer']
        
        print(f"\n测试样本 {i+1}: {question[:40]}...")
        
        try:
            result = pipeline.process_question_simple(question, table)
            prediction = result.get('final_answer', '')
            
            print(f"预测: {repr(prediction)}")
            print(f"参考: {repr(reference)}")
            
            # 检查格式
            if '<answer>' in prediction and '</answer>' in prediction:
                format_correct += 1
                print("✓ 格式正确")
                
                # 提取答案内容
                import re
                answer_match = re.search(r'<answer>(.*?)</answer>', prediction, re.DOTALL)
                if answer_match:
                    extracted_answer = answer_match.group(1).strip()
                    print(f"提取的答案: {repr(extracted_answer)}")
                    
                    # 简单的内容匹配（标准化后比较）
                    pred_norm = extracted_answer.lower().strip()
                    ref_norm = reference.lower().strip()
                    
                    if pred_norm == ref_norm:
                        content_correct += 1
                        print("✓ 内容正确")
                    else:
                        print("❌ 内容不匹配")
                else:
                    print("❌ 无法提取答案内容")
            else:
                print("❌ 格式错误")
                
        except Exception as e:
            print(f"❌ 生成失败: {e}")
    
    format_accuracy = format_correct / len(test_samples)
    content_accuracy = content_correct / len(test_samples)
    
    print(f"\n" + "="*60)
    print("测试结果")
    print("="*60)
    print(f"格式准确率: {format_accuracy:.2%} ({format_correct}/{len(test_samples)})")
    print(f"内容准确率: {content_accuracy:.2%} ({content_correct}/{len(test_samples)})")
    print(f"整体准确率: {content_accuracy:.2%}")
    
    return pipeline

def main():
    """主函数"""
    pipeline = train_sft_only()
    
    if pipeline:
        print("\n" + "="*80)
        print("SFT训练完成")
        print("="*80)
        
        print("\n建议:")
        if format_accuracy > 0.8:
            print("✓ 格式训练成功，可以进行完整训练")
        else:
            print("⚠ 格式训练需要改进，建议调整训练参数")
            
        print("\n下一步:")
        print("1. 运行完整评估: python3 run_full_evaluation.py")
        print("2. 或运行完整训练: python3 src/train.py")
    else:
        print("\n训练失败，请检查错误信息")

if __name__ == "__main__":
    main()
