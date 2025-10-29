#!/usr/bin/env python3
"""
使用问题拆分的完整训练脚本
"""
import sys
sys.path.append('src')
sys.path.append('datasets/tablebench')

import yaml
import torch
import time
from load_tablebench import TableBenchLoader
from train import train_sft_cold_start, setup_lora
from modelscope import AutoModelForCausalLM, AutoTokenizer
from src.pipeline import TableQAPipeline
from src.mcp_tools import MCPToolManager
from src.eval import AnswerComparator
import json

def load_config():
    """加载配置"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def train_with_splitting():
    """使用问题拆分进行训练"""
    print("="*80)
    print("使用问题拆分的完整训练")
    print("="*80)
    
    # 加载配置
    config = load_config()
    
    # 使用所有数据进行训练
    config['data']['train_samples'] = None  # 使用所有数据
    config['data']['eval_samples'] = None    # 使用所有数据
    
    # 加载数据集
    print("\n[1/4] 加载数据集...")
    loader = TableBenchLoader('datasets/tablebench')
    
    train_data = loader.load(version='base', max_samples=config['data']['train_samples'])
    eval_data = loader.load(version='base', max_samples=config['data']['eval_samples'])
    
    print(f"✓ 训练集: {len(train_data)} 条")
    print(f"✓ 测试集: {len(eval_data)} 条")
    
    # 训练模型
    print("\n[2/4] 开始SFT训练...")
    try:
        model, tokenizer = train_sft_cold_start(train_data, eval_data, config)
        print("✓ SFT训练完成")
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        return None
    
    # 创建Pipeline
    print("\n[3/4] 创建Pipeline...")
    mcp_manager = MCPToolManager()
    pipeline = TableQAPipeline(model, tokenizer, mcp_manager)
    print("✓ Pipeline创建完成")
    
    # 测试问题拆分功能
    print("\n[4/4] 测试问题拆分功能...")
    test_samples = eval_data[:5]
    
    for i, sample in enumerate(test_samples):
        question = sample['question']
        table = sample['table']
        reference = sample['answer']
        
        print(f"\n测试样本 {i+1}: {question[:40]}...")
        
        try:
            # 使用完整的问题拆分流程
            result = pipeline.process_question(question, table)
            prediction = result.get('final_answer', '')
            
            print(f"策略: {result.get('strategy', 'unknown')}")
            print(f"迭代次数: {result.get('iterations', 0)}")
            print(f"子任务数量: {len(result.get('subtasks', []))}")
            
            # 显示子任务
            for j, subtask in enumerate(result.get('subtasks', [])[:2]):
                print(f"  子任务{j+1}: {subtask.get('description', '')[:50]}...")
            
            print(f"预测: {repr(prediction)}")
            print(f"参考: {repr(reference)}")
            
            # 检查格式
            if '<answer>' in prediction and '</answer>' in prediction:
                print("✓ 格式正确")
            else:
                print("❌ 格式错误")
                
        except Exception as e:
            print(f"❌ 生成失败: {e}")
    
    return pipeline

def evaluate_with_splitting(pipeline):
    """使用问题拆分进行评估"""
    print("\n" + "="*80)
    print("使用问题拆分进行评估")
    print("="*80)
    
    # 加载配置
    config = load_config()
    
    # 加载评估数据
    print("\n[1/3] 加载评估数据...")
    loader = TableBenchLoader('datasets/tablebench')
    eval_data = loader.load(version='base', max_samples=config['data']['eval_samples'])
    
    print(f"✓ 评估数据: {len(eval_data)} 条")
    
    # 执行评估
    print("\n[2/3] 执行评估...")
    results = {
        "total_samples": len(eval_data),
        "successful_executions": 0,
        "failed_executions": 0,
        "execution_times": [],
        "predictions": [],
        "references": [],
        "strategy_stats": {},
        "subtask_stats": []
    }
    
    comparator = AnswerComparator()
    
    for i, sample in enumerate(eval_data):
        question = sample['question']
        table = sample['table']
        reference = sample['answer']
        
        print(f"评估样本 {i+1}/{len(eval_data)}: {question[:50]}...")
        
        start_time = time.time()
        
        try:
            # 使用完整的问题拆分流程
            result = pipeline.process_question(question, table)
            execution_time = time.time() - start_time
            
            prediction = result.get('final_answer', '')
            
            # 统计
            results["execution_times"].append(execution_time)
            results["predictions"].append(prediction)
            results["references"].append(reference)
            
            # 统计策略
            strategy = result.get('strategy', 'unknown')
            results["strategy_stats"][strategy] = results["strategy_stats"].get(strategy, 0) + 1
            
            # 统计子任务
            subtasks = result.get('subtasks', [])
            results["subtask_stats"].append({
                "sample_id": i+1,
                "strategy": strategy,
                "subtask_count": len(subtasks),
                "iterations": result.get('iterations', 0)
            })
            
            # 检查是否成功
            if prediction and len(prediction.strip()) > 0:
                results["successful_executions"] += 1
                print(f"  ✓ 成功 ({execution_time:.2f}s)")
            else:
                results["failed_executions"] += 1
                print(f"  ❌ 失败 ({execution_time:.2f}s)")
            
        except Exception as e:
            execution_time = time.time() - start_time
            results["failed_executions"] += 1
            results["execution_times"].append(execution_time)
            results["predictions"].append("")
            results["references"].append(reference)
            print(f"  ❌ 异常: {e} ({execution_time:.2f}s)")
    
    # 计算准确率
    print("\n[3/3] 计算准确率...")
    eval_metrics = comparator.calculate_accuracy(results["predictions"], results["references"])
    results.update(eval_metrics)
    
    # 计算性能指标
    total = results["total_samples"]
    success = results["successful_executions"]
    results["success_rate"] = success / total if total > 0 else 0.0
    results["avg_execution_time"] = sum(results["execution_times"]) / len(results["execution_times"]) if results["execution_times"] else 0.0
    
    return results

def main():
    """主函数"""
    # 训练
    pipeline = train_with_splitting()
    
    if pipeline:
        # 评估
        results = evaluate_with_splitting(pipeline)
        
        # 打印结果
        print("\n" + "="*80)
        print("最终评估结果")
        print("="*80)
        print(f"总样本数: {results['total_samples']}")
        print(f"成功率: {results['success_rate']:.3f}")
        print(f"精确匹配准确率: {results['exact_match_accuracy']:.3f}")
        print(f"格式准确率: {results['format_accuracy']:.3f}")
        print(f"整体准确率: {results['overall_accuracy']:.3f}")
        print(f"精确匹配数量: {results['exact_match_count']}/{results['total_samples']}")
        print(f"格式正确数量: {results['format_correct_count']}/{results['total_samples']}")
        print(f"平均执行时间: {results['avg_execution_time']:.2f}s")
        
        print(f"\n策略分布:")
        for strategy, count in results['strategy_stats'].items():
            print(f"  {strategy}: {count}")
        
        # 保存结果
        with open('splitting_training_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 结果已保存到: splitting_training_results.json")
        
        print("\n" + "="*80)
        print("训练和评估完成")
        print("="*80)
        
        # 分析结果
        if results['format_accuracy'] > 0.8:
            print("✓ 格式训练成功！")
        else:
            print("⚠ 格式训练需要改进")
            
        if results['exact_match_accuracy'] > 0.3:
            print("✓ 内容准确率较好")
        else:
            print("⚠ 内容准确率需要改进")
            
        print(f"\n建议:")
        print(f"1. 如果格式准确率 < 80%，需要调整训练参数")
        print(f"2. 如果内容准确率 < 30%，需要增加训练数据或调整模型")
        print(f"3. 可以尝试增加训练轮数或使用更大的数据集")
        
    else:
        print("\n训练失败，请检查错误信息")

if __name__ == "__main__":
    main()
