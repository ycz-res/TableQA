#!/usr/bin/env python3
"""
Evaluation script for TableQA
"""
from utils import load_config, AnswerComparator
from models.loader import load_both_agents
from pipeline import TableQAPipeline
from mcp.tools import MCPToolManager
from dataset import TableBenchLoader
import json
from pathlib import Path


def main():
    print("="*60)
    print("TableQA Evaluation")
    print("="*60)
    
    # Load config
    config = load_config()
    print(f"\nModel: {config['model']}")
    print(f"Dataset: {config['dataset']}")
    
    # Load both agents
    print("\nLoading agents...")
    agents = load_both_agents(config, use_plan_lora=True)
    # Use Plan Agent for pipeline (it handles both planning and reasoning)
    model, tokenizer = agents['plan']
    
    # Create pipeline
    mcp_manager = MCPToolManager()
    pipeline = TableQAPipeline(model, tokenizer, mcp_manager)
    
    # Load test data
    print("Loading test data...")
    loader = TableBenchLoader()
    test_data = loader.load(split='test')
    print(f"✓ Loaded {len(test_data)} test samples")
    
    # Evaluate
    print("\nEvaluating...")
    comparator = AnswerComparator()
    results = []
    
    for i, item in enumerate(test_data):
        # Run pipeline
        result = pipeline.process_question(item['question'], item['table'])
        predicted = result['final_answer']
        ground_truth = item['answer']
        
        # Calculate scores
        format_score = comparator.calculate_format_score(predicted)
        em_score = comparator.calculate_em_score(
            comparator.extract_answer(predicted),
            ground_truth
        )
        
        results.append({
            'id': i,
            'question': item['question'],
            'predicted': predicted,
            'ground_truth': ground_truth,
            'format_score': format_score,
            'em_score': em_score
        })
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(test_data)} samples")
    
    # Calculate metrics
    avg_format = sum(r['format_score'] for r in results) / len(results)
    avg_em = sum(r['em_score'] for r in results) / len(results)
    overall = 0.1 * avg_format + 0.9 * avg_em
    
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"\nFormat Accuracy: {avg_format*100:.2f}%")
    print(f"EM Accuracy: {avg_em*100:.2f}%")
    print(f"Overall Score: {overall*100:.2f}%")
    
    # Save results
    output_file = config['evaluation']['results_file']
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': {
                'format_accuracy': avg_format,
                'em_accuracy': avg_em,
                'overall_score': overall
            },
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
