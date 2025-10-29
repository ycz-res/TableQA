#!/usr/bin/env python3
"""
测试问题拆分逻辑
"""
import sys
sys.path.append('src')
sys.path.append('datasets/tablebench')

import yaml
import json
from load_tablebench import TableBenchLoader
from src.pipeline import PlanAgent
from modelscope import AutoModelForCausalLM, AutoTokenizer

def load_config():
    """加载配置"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def test_question_splitting():
    """测试问题拆分逻辑"""
    print("="*80)
    print("测试问题拆分逻辑")
    print("="*80)
    
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
    
    # 创建PlanAgent
    print("\n[2/3] 创建PlanAgent...")
    plan_agent = PlanAgent(model, tokenizer)
    print("✓ PlanAgent创建完成")
    
    # 测试不同类型的问题
    print("\n[3/3] 测试问题拆分...")
    
    test_questions = [
        "What is the average number of tropical cyclones per season?",  # 聚合类
        "Which country has higher GDP, China or USA?",  # 比较类
        "What is the trend of sales from 2020 to 2023?",  # 顺序类
        "List all countries with population over 100 million",  # 独立类
        "What is the total GDP of countries with area over 1000000 km2?"  # 桥接类
    ]
    
    # 加载一个表格样本
    loader = TableBenchLoader('datasets/tablebench')
    sample_data = loader.load(version='base', max_samples=1)[0]
    table_info = f"表格列: {sample_data['table']['columns']}"
    
    for i, question in enumerate(test_questions):
        print(f"\n{'='*60}")
        print(f"测试问题 {i+1}: {question}")
        print(f"{'='*60}")
        
        try:
            # 1. 测试问题分类
            strategy = plan_agent.classify_question(question, table_info)
            print(f"分类结果: {strategy}")
            
            # 2. 测试问题拆分
            split_result = plan_agent.split_question(question, table_info)
            print(f"拆分策略: {split_result.get('strategy', 'unknown')}")
            print(f"子任务数量: {len(split_result.get('subtasks', []))}")
            
            # 3. 显示子任务详情
            for j, subtask in enumerate(split_result.get('subtasks', [])):
                print(f"\n子任务 {j+1}:")
                print(f"  ID: {subtask.get('id', 'unknown')}")
                print(f"  描述: {subtask.get('description', 'unknown')}")
                print(f"  类型: {subtask.get('task_type', 'unknown')}")
                print(f"  依赖: {subtask.get('dependencies', [])}")
                print(f"  期望输出: {subtask.get('expected_output', 'unknown')}")
            
            # 4. 验证独立性
            is_valid, issues = plan_agent.validate_independence(split_result.get('subtasks', []))
            print(f"\n独立性验证: {'✓ 通过' if is_valid else '❌ 失败'}")
            if issues:
                for issue in issues:
                    print(f"  问题: {issue}")
            
            # 5. 显示完整的JSON结果
            print(f"\n完整JSON结果:")
            print(json.dumps(split_result, indent=2, ensure_ascii=False))
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    test_question_splitting()
    
    print("\n" + "="*80)
    print("问题拆分逻辑测试完成")
    print("="*80)
    
    print("\n分析结果:")
    print("1. 检查问题分类是否准确")
    print("2. 检查拆分策略是否合理")
    print("3. 检查子任务依赖关系是否正确")
    print("4. 检查独立性验证是否有效")

if __name__ == "__main__":
    main()
