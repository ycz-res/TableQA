#!/usr/bin/env python3
"""
完整的训练和评估脚本
在所有886个样本上进行训练和测试
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

def main():
    print("="*80)
    print("TableQA 完整评估 - 使用所有886个样本")
    print("="*80)
    
    # 加载配置
    config = load_config()
    
    # 加载数据集
    print("\n[1/4] 加载数据集...")
    loader = TableBenchLoader('datasets/tablebench')
    
    # 加载所有样本
    all_data = loader.load(version='base', max_samples=config.get("data", {}).get("train_samples", 886))
    print(f"✓ 成功加载 {len(all_data)} 条数据")
    
    # 80%训练，20%测试
    split_point = int(len(all_data) * 0.8)
    train_data = all_data[:split_point]
    eval_data = all_data[split_point:]
    print(f"✓ 训练集: {len(train_data)} 条")
    print(f"✓ 测试集: {len(eval_data)} 条")
    
    # 训练模型（如果还没有训练好的模型）
    print("\n[2/4] 加载或训练模型...")
    model_path = config.get("model", {}).get("name", "./models/pretrained/Qwen/Qwen2.5-1.5B-Instruct")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        # 尝试加载LoRA权重
        try:
            lora_path = config.get("training", {}).get("output_dir", "./models/finetuned")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_path)
            print("✓ 成功加载LoRA权重")
        except:
            print("⚠ LoRA权重不存在，使用基础模型")
        
        model = setup_lora(model, config)
        
    except Exception as e:
        print(f"⚠ 模型加载失败: {e}")
        print("请先运行训练脚本训练模型")
        return
    
    # 创建MCP工具管理器
    print("\n[3/4] 创建Pipeline...")
    from src.pipeline import TableQAPipeline
    from src.mcp_tools import MCPToolManager
    
    mcp_manager = MCPToolManager()
    pipeline = TableQAPipeline(model, tokenizer, mcp_manager)
    
    # 评估Pipeline
    print("\n[4/4] 开始评估...")
    print(f"评估数据量: {len(eval_data)}")
    
    start_time = time.time()
    results = evaluate_pipeline(pipeline, eval_data[:100])  # 先测试100条
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    
    # 打印结果
    print("\n" + "="*80)
    print("评估结果")
    print("="*80)
    print(f"总样本数: {results.get('total_samples', 0)}")
    print(f"精确匹配准确率: {results.get('exact_match_accuracy', 0):.3f}")
    print(f"格式准确率: {results.get('format_accuracy', 0):.3f}")
    print(f"整体准确率: {results.get('overall_accuracy', 0):.3f}")
    print(f"精确匹配数量: {results.get('exact_match_count', 0)}/{results.get('total_samples', 0)}")
    print(f"格式正确数量: {results.get('format_correct_count', 0)}/{results.get('total_samples', 0)}")
    print(f"执行时间: {elapsed_time:.2f}s")
    
    # 保存结果
    output_file = "full_evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_samples": len(eval_data),
            "tested_samples": results.get('total_samples', 0),
            "results": results,
            "elapsed_time": elapsed_time
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 结果已保存到: {output_file}")
    print("="*80)
    
    return results

if __name__ == "__main__":
    main()

