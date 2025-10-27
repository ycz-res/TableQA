"""
测试训练流程
"""

import sys
import os
sys.path.append('src')

from src.train import main as train_main
from src.pipeline import TableQAPipeline
from src.mcp_tools import MCPToolManager
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch


def test_training():
    """测试训练流程"""
    print("="*60)
    print("测试TableQA训练流程")
    print("="*60)
    
    try:
        # 运行训练
        print("开始训练...")
        pipeline = train_main()
        print("✓ 训练完成")
        
        # 测试推理
        print("\n测试推理...")
        question = "What is the average number of tropical cyclones per season?"
        table_data = {
            "columns": ["season", "tropical cyclones"],
            "data": [
                ["1990-91", "10"],
                ["1991-92", "10"],
                ["1992-93", "3"]
            ]
        }
        
        result = pipeline.process_question(question, table_data)
        
        print(f"\n问题: {question}")
        print(f"策略: {result['strategy']}")
        print(f"迭代次数: {result['iterations']}")
        print(f"子任务数: {len(result['subtasks'])}")
        print(f"最终答案: {result['final_answer']}")
        
        print("\n✅ 测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_inference_only():
    """只测试推理（不训练）"""
    print("="*60)
    print("测试推理流程（不训练）")
    print("="*60)
    
    try:
        # 加载预训练模型
        model_path = "./models/pretrained/Qwen/Qwen2.5-1.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        # 创建MCP工具管理器
        mcp_manager = MCPToolManager()
        
        # 创建pipeline
        pipeline = TableQAPipeline(model, tokenizer, mcp_manager)
        
        # 测试推理
        question = "What is the average number of tropical cyclones per season?"
        table_data = {
            "columns": ["season", "tropical cyclones"],
            "data": [
                ["1990-91", "10"],
                ["1991-92", "10"],
                ["1992-93", "3"]
            ]
        }
        
        result = pipeline.process_question(question, table_data)
        
        print(f"\n问题: {question}")
        print(f"策略: {result['strategy']}")
        print(f"迭代次数: {result['iterations']}")
        print(f"子任务数: {len(result['subtasks'])}")
        print(f"最终答案: {result['final_answer']}")
        
        print("\n✅ 推理测试完成")
        
    except Exception as e:
        print(f"❌ 推理测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试TableQA训练流程")
    parser.add_argument("--mode", type=str, default="inference", 
                       choices=["train", "inference"],
                       help="测试模式")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        test_training()
    else:
        test_inference_only()

