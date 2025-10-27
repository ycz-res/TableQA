"""
TableQA主程序
支持训练、评估和推理模式
"""

import yaml
import argparse
from pathlib import Path
from typing import Dict, Any

# 导入模块
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from train import train_model
from eval import evaluate_pipeline, save_results
from pipeline import TableQAPipeline
from mcp_tools import MCPToolManager
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories():
    """设置目录结构"""
    directories = ["./models", "./results", "./logs"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


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


def run_training(config: Dict[str, Any]):
    """运行训练"""
    print("开始训练...")
    
    # 使用train模块
    from src.train import main as train_main
    pipeline = train_main()
    
    print("训练完成")
    return pipeline


def run_evaluation(config: Dict[str, Any]):
    """运行评估"""
    print("开始评估...")
    
    # 使用eval模块
    from src.eval import main as eval_main
    results = eval_main()
    
    print("评估完成")
    return results


def run_inference(config: Dict[str, Any], question: str, table_data: Dict):
    """运行推理"""
    print("开始推理...")
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(config)
    
    # 创建MCP工具管理器
    mcp_manager = MCPToolManager()
    
    # 创建pipeline
    pipeline = TableQAPipeline(model, tokenizer, mcp_manager)
    
    # 处理问题
    result = pipeline.process_question(question, table_data)
    
    print("推理完成")
    return result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="TableQA主程序")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="配置文件路径")
    parser.add_argument("--mode", type=str, default="train",
                       choices=["train", "eval", "inference"],
                       help="运行模式")
    parser.add_argument("--question", type=str,
                       help="推理模式下的问题")
    parser.add_argument("--table", type=str,
                       help="推理模式下的表格数据文件路径")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置目录
    setup_directories()
    
    if args.mode == "train":
        run_training(config)
    elif args.mode == "eval":
        run_evaluation(config)
    elif args.mode == "inference":
        if not args.question:
            print("推理模式需要提供问题 (--question)")
            return
        
        # 加载表格数据
        if args.table:
            import json
            with open(args.table, 'r', encoding='utf-8') as f:
                table_data = json.load(f)
        else:
            # 使用默认测试数据
            table_data = {
                "columns": ["season", "tropical cyclones"],
                "data": [
                    ["1990-91", "10"],
                    ["1991-92", "10"],
                    ["1992-93", "3"]
                ]
            }
        
        result = run_inference(config, args.question, table_data)
        
        print("\n" + "="*60)
        print("推理结果")
        print("="*60)
        print(f"问题: {args.question}")
        print(f"策略: {result['strategy']}")
        print(f"子任务数: {len(result['subtasks'])}")
        print(f"最终答案: {result['final_answer']}")


if __name__ == "__main__":
    main()