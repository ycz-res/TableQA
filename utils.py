"""
评估相关功能
"""

import json
import yaml
from typing import Dict, List, Any, TYPE_CHECKING
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from collections import Counter
import time
import re

if TYPE_CHECKING:
    from pipeline import TableQAPipeline


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration file with defaults
    
    Args:
        config_path: Config file path
        
    Returns:
        Configuration dictionary with all parameters
    """
    from pathlib import Path
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        user_config = yaml.safe_load(f) or {}
    
    # Build full config with defaults
    config = {
        "model": user_config.get("model", "Qwen/Qwen2.5-1.5B-Instruct"),
        "dataset": user_config.get("dataset", "tablebench"),
        "epochs": int(user_config.get("epochs", 3)),
        "batch_size": int(user_config.get("batch_size", 2)),
        "learning_rate": float(user_config.get("learning_rate", 2e-5)),
        "train_samples": user_config.get("train_samples"),
        "eval_samples": user_config.get("eval_samples"),
    }
    
    # Compute derived paths
    model = config["model"]
    dataset = config["dataset"]
    
    if "/" in model and not model.startswith("."):
        config["model_path"] = f"./models/pretrained/{model}"
    else:
        config["model_path"] = model
    
    config["dataset_path"] = f"./data/{dataset}" if not dataset.startswith(".") else dataset
    config["output_dir"] = "./models/finetuned"
    
    # LoRA defaults
    config["lora"] = {
        "r": 16, "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "lora_dropout": 0.1, "bias": "none"
    }
    
    # Generation defaults
    config["generation"] = {
        "max_length": 1024, "max_new_tokens": 512,
        "temperature": 0.7, "top_p": 0.9, "do_sample": True
    }
    
    # Training details
    config["training"] = {
        "warmup_steps": 20, "logging_steps": 5, "eval_steps": 10,
        "save_steps": 20, "weight_decay": 0.01, "gradient_accumulation_steps": 2
    }
    
    # GPRO
    config["gpro"] = {
        "epochs": 3, "batch_size": 4, "learning_rate": 1e-5,
        "weight_decay": 0.01, "logging_steps": 10,
        "format_weight": 0.1, "em_weight": 0.9
    }
    
    # Evaluation
    config["evaluation"] = {
        "results_file": "./results/evaluation_results.json",
        "normalize_answers": True, "case_sensitive": False, "remove_punctuation": True
    }
    
    return config


class AnswerComparator:
    """答案对比器 - 用于计算EM和Format分数"""
    
    def __init__(self):
        self.answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    
    def extract_answer(self, text: str) -> str:
        """从文本中提取答案部分"""
        # 尝试提取<answer>标签中的内容
        answer_match = self.answer_pattern.search(text)
        if answer_match:
            return answer_match.group(1).strip()
        
        # 如果没有找到标签，返回整个文本
        return text.strip()
    
    def calculate_format_score(self, generated_text: str) -> float:
        """计算格式分数 (0.0-1.0) - 只检查<answer></answer>格式"""
        # 检查是否包含完整的<answer></answer>格式
        if self.answer_pattern.search(generated_text):
            return 1.0
        else:
            return 0.0
    
    def calculate_em_score(self, generated_answer: str, reference_answer: str) -> float:
        """计算精确匹配分数 (0.0-1.0)"""
        # 标准化答案
        gen_norm = self._normalize_answer(generated_answer)
        ref_norm = self._normalize_answer(reference_answer)
        
        return 1.0 if gen_norm == ref_norm else 0.0
    
    def _normalize_answer(self, answer: str) -> str:
        """标准化答案格式"""
        # 转换为小写
        answer = answer.lower().strip()
        
        # 移除标点符号
        answer = re.sub(r'[^\w\s]', '', answer)
        
        # 移除多余空格
        answer = ' '.join(answer.split())
        
        return answer
    
    def calculate_accuracy(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """计算准确率统计"""
        if len(predictions) != len(references):
            raise ValueError("预测结果和参考答案数量不匹配")
        
        total_samples = len(predictions)
        exact_matches = 0
        format_correct = 0
        
        for pred, ref in zip(predictions, references):
            # 提取答案
            pred_answer = self.extract_answer(pred)
            
            # 计算EM
            if self.calculate_em_score(pred_answer, ref) == 1.0:
                exact_matches += 1
            
            # 计算格式正确性
            if self.calculate_format_score(pred) == 1.0:
                format_correct += 1
        
        return {
            "total_samples": total_samples,
            "exact_match_count": exact_matches,
            "format_correct_count": format_correct,
            "exact_match_accuracy": exact_matches / total_samples if total_samples > 0 else 0.0,
            "format_accuracy": format_correct / total_samples if total_samples > 0 else 0.0,
            "overall_accuracy": exact_matches / total_samples if total_samples > 0 else 0.0
        }


def load_model_and_tokenizer(config: Dict[str, Any]):
    """加载模型和tokenizer"""
    model_path = config.get("model", {}).get("name", "./models/pretrained/Qwen/Qwen2.5-1.5B-Instruct")
    
    print(f"正在加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    return model, tokenizer


def evaluate_predictions(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """评估预测结果 - 使用答案对比器"""
    comparator = AnswerComparator()
    return comparator.calculate_accuracy(predictions, references)


def evaluate_pipeline(pipeline: "TableQAPipeline", test_data: List[Dict]) -> Dict[str, Any]:
    """评估pipeline性能"""
    results = {
        "total_samples": len(test_data),
        "successful_executions": 0,
        "failed_executions": 0,
        "execution_times": [],
        "predictions": [],
        "references": [],
        "strategy_stats": Counter()
    }
    
    for i, sample in enumerate(test_data):
        question = sample['question']
        table = sample['table']
        reference = sample['answer']
        
        print(f"评估样本 {i+1}/{len(test_data)}: {question[:50]}...")
        
        start_time = time.time()
        
        try:
            # 执行pipeline - 使用简化的端到端方法
            result = pipeline.process_question_simple(question, table)
            execution_time = time.time() - start_time
            
            # 提取预测结果 - 获取完整的生成文本
            prediction = result.get('final_answer', '')
            
            # 统计
            results["execution_times"].append(execution_time)
            results["predictions"].append(prediction)
            results["references"].append(reference)
            results["strategy_stats"][result.get('strategy', 'unknown')] += 1
            
            # 检查是否成功 - 端到端方法
            if result.get('strategy') == 'end_to_end':
                # 端到端方法：只要有答案就算成功
                if prediction and len(prediction.strip()) > 0:  # 有答案就算成功
                    results["successful_executions"] += 1
                    print(f"  ✓ 成功 ({execution_time:.2f}s)")
                else:
                    results["failed_executions"] += 1
                    print(f"  ❌ 失败 ({execution_time:.2f}s)")
            else:
                # 原来的子任务检查逻辑
                successful_subtasks = sum(1 for st in result.get('subtask_results', []) if st.get('status') == 'success')
                total_subtasks = len(result.get('subtask_results', []))
                
                if successful_subtasks == total_subtasks and total_subtasks > 0:
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
    
    # 计算评估指标
    eval_metrics = evaluate_predictions(results["predictions"], results["references"])
    results.update(eval_metrics)
    
    # 计算性能指标
    total = results["total_samples"]
    success = results["successful_executions"]
    results["success_rate"] = success / total if total > 0 else 0.0
    results["avg_execution_time"] = sum(results["execution_times"]) / len(results["execution_times"]) if results["execution_times"] else 0.0
    
    return results


def save_results(results: Dict[str, Any], filepath: str):
    """保存评估结果"""
    # 转换Counter为普通字典以便JSON序列化
    if "strategy_stats" in results:
        results["strategy_stats"] = dict(results["strategy_stats"])
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"评估结果已保存到: {filepath}")


def main():
    """主评估函数"""
    # 加载配置
    config = load_config()
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(config)
    
    # 创建MCP工具管理器
    mcp_manager = MCPToolManager()
    
    # 创建pipeline
    pipeline = TableQAPipeline(model, tokenizer, mcp_manager)
    
    # 加载测试数据
    import sys
    import os
    sys.path.append('/home/zyc/TableQA')
    sys.path.append('/home/zyc/TableQA/datasets/tablebench')
    from load_tablebench import TableBenchLoader
    loader = TableBenchLoader()
    test_data = loader.load(version='base', max_samples=50)
    
    # 执行评估
    results = evaluate_pipeline(pipeline, test_data)
    
    # 打印结果
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    print(f"总样本数: {results['total_samples']}")
    print(f"成功率: {results['success_rate']:.3f}")
    print(f"精确匹配准确率: {results['exact_match_accuracy']:.3f}")
    print(f"格式准确率: {results['format_accuracy']:.3f}")
    print(f"整体准确率: {results['overall_accuracy']:.3f}")
    print(f"精确匹配数量: {results['exact_match_count']}/{results['total_samples']}")
    print(f"格式正确数量: {results['format_correct_count']}/{results['total_samples']}")
    print(f"平均执行时间: {results['avg_execution_time']:.2f}s")
    print(f"策略分布: {dict(results['strategy_stats'])}")
    
    # 保存结果
    save_results(results, "evaluation_results.json")
    
    return results


if __name__ == "__main__":
    main()
