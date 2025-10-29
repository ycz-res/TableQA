#!/usr/bin/env python3
"""
改进的训练脚本 - 修复格式问题
使用小数据集快速测试
"""
import sys
sys.path.append('src')
sys.path.append('datasets/tablebench')

import yaml
import torch
from load_tablebench import TableBenchLoader
from train import train_end_to_end_pipeline, setup_lora
from modelscope import AutoModelForCausalLM, AutoTokenizer
from eval import evaluate_pipeline, AnswerComparator
import json
import time

def load_config():
    """加载配置"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def test_format_training():
    """测试格式训练"""
    print("="*80)
    print("测试格式训练 - 使用小数据集")
    print("="*80)
    
    # 加载配置
    config = load_config()
    
    # 使用小数据集进行快速测试
    config['data']['train_samples'] = 50
    config['data']['eval_samples'] = 20
    
    # 加载数据集
    print("\n[1/4] 加载数据集...")
    loader = TableBenchLoader('datasets/tablebench')
    
    train_data = loader.load(version='base', max_samples=config['data']['train_samples'])
    eval_data = loader.load(version='base', max_samples=config['data']['eval_samples'])
    
    print(f"✓ 训练集: {len(train_data)} 条")
    print(f"✓ 测试集: {len(eval_data)} 条")
    
    # 检查数据格式
    print("\n[2/4] 检查数据格式...")
    sample = train_data[0]
    print(f"问题: {sample['question'][:50]}...")
    print(f"答案: {sample['answer']}")
    
    # 训练模型
    print("\n[3/4] 开始训练...")
    try:
        model, tokenizer = train_end_to_end_pipeline(train_data, eval_data, config)
        print("✓ 训练完成")
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        return None
    
    # 测试模型输出格式
    print("\n[4/4] 测试模型输出格式...")
    from src.pipeline import TableQAPipeline
    from src.mcp_tools import MCPToolManager
    
    mcp_manager = MCPToolManager()
    pipeline = TableQAPipeline(model, tokenizer, mcp_manager)
    
    # 测试几个样本
    test_samples = eval_data[:5]
    format_correct = 0
    
    for i, sample in enumerate(test_samples):
        question = sample['question']
        table = sample['table']
        reference = sample['answer']
        
        print(f"\n测试样本 {i+1}: {question[:30]}...")
        
        try:
            result = pipeline.process_question_simple(question, table)
            prediction = result.get('final_answer', '')
            
            print(f"预测: {repr(prediction)}")
            print(f"参考: {repr(reference)}")
            
            # 检查格式
            if '<answer>' in prediction and '</answer>' in prediction:
                format_correct += 1
                print("✓ 格式正确")
            else:
                print("❌ 格式错误")
                
        except Exception as e:
            print(f"❌ 生成失败: {e}")
    
    format_accuracy = format_correct / len(test_samples)
    print(f"\n格式准确率: {format_accuracy:.2%} ({format_correct}/{len(test_samples)})")
    
    return pipeline

def main():
    """主函数"""
    pipeline = test_format_training()
    
    if pipeline:
        print("\n" + "="*80)
        print("格式训练测试完成")
        print("="*80)
        
        # 如果格式正确率较高，可以进行完整训练
        print("\n建议:")
        print("1. 如果格式准确率 > 80%，可以增加训练数据量")
        print("2. 如果格式准确率 < 50%，需要调整训练参数")
        print("3. 可以运行完整训练: python3 src/train.py")
    else:
        print("\n训练失败，请检查错误信息")

if __name__ == "__main__":
    main()
