#!/usr/bin/env python3
"""
快速测试修复后的格式
"""
import sys
sys.path.append('src')
sys.path.append('datasets/tablebench')

import yaml
from load_tablebench import TableBenchLoader
from src.pipeline import TableQAPipeline
from src.mcp_tools import MCPToolManager
from modelscope import AutoModelForCausalLM, AutoTokenizer

def load_config():
    """加载配置"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def test_format_fix():
    """测试格式修复"""
    print("="*60)
    print("测试格式修复")
    print("="*60)
    
    # 加载配置
    config = load_config()
    
    # 加载模型
    print("\n[1/3] 加载模型...")
    model_path = "./models/finetuned"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto"
    )
    print("✓ 模型加载完成")
    
    # 创建pipeline
    print("\n[2/3] 创建Pipeline...")
    mcp_manager = MCPToolManager()
    pipeline = TableQAPipeline(model, tokenizer, mcp_manager)
    print("✓ Pipeline创建完成")
    
    # 测试几个样本
    print("\n[3/3] 测试格式...")
    loader = TableBenchLoader('datasets/tablebench')
    test_data = loader.load(version='base', max_samples=5)
    
    format_correct = 0
    content_correct = 0
    
    for i, sample in enumerate(test_data):
        question = sample['question']
        table = sample['table']
        reference = sample['answer']
        
        print(f"\n测试样本 {i+1}: {question[:40]}...")
        
        try:
            result = pipeline.process_question(question, table)  # 使用完整的问题拆分流程
            prediction = result.get('final_answer', '')
            
            print(f"预测: {repr(prediction)}")
            print(f"参考: {repr(reference)}")
            
            # 显示问题拆分信息
            print(f"策略: {result.get('strategy', 'unknown')}")
            print(f"迭代次数: {result.get('iterations', 0)}")
            print(f"子任务数量: {len(result.get('subtasks', []))}")
            
            # 显示子任务详情
            for j, subtask in enumerate(result.get('subtasks', [])[:3]):  # 只显示前3个
                print(f"  子任务{j+1}: {subtask.get('description', '')[:50]}...")
            
            if len(result.get('subtasks', [])) > 3:
                print(f"  ... 还有 {len(result.get('subtasks', [])) - 3} 个子任务")
            
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
    
    format_accuracy = format_correct / len(test_data)
    content_accuracy = content_correct / len(test_data)
    
    print(f"\n" + "="*60)
    print("测试结果")
    print("="*60)
    print(f"格式准确率: {format_accuracy:.2%} ({format_correct}/{len(test_data)})")
    print(f"内容准确率: {content_accuracy:.2%} ({content_correct}/{len(test_data)})")
    print(f"整体准确率: {content_accuracy:.2%}")
    
    return format_accuracy, content_accuracy

def main():
    """主函数"""
    format_acc, content_acc = test_format_fix()
    
    print("\n" + "="*60)
    print("格式修复测试完成")
    print("="*60)
    
    if format_acc > 0.8:
        print("✓ 格式修复成功！")
        print("建议运行完整评估: python3 run_full_evaluation.py")
    else:
        print("⚠ 格式修复需要进一步改进")
        print("建议检查训练数据和模型输出")

if __name__ == "__main__":
    main()
