#!/usr/bin/env python3
"""
Inference script for TableQA
"""
import json
import argparse
from pathlib import Path
from utils import load_config
from models.loader import load_both_agents
from pipeline import TableQAPipeline
from mcp.tools import MCPToolManager


def infer_single(question: str, table_data: dict, pipeline: TableQAPipeline):
    """Single question inference"""
    result = pipeline.process_question(question, table_data)
    return result


def infer_batch(input_file: str, output_file: str, pipeline: TableQAPipeline):
    """Batch inference from JSONL file"""
    results = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            question = item['question']
            table_data = item['table']
            
            result = pipeline.process_question(question, table_data)
            results.append({
                'id': item.get('id', ''),
                'question': question,
                'prediction': result['final_answer'],
                'ground_truth': item.get('answer', '')
            })
            
            print(f"Processed: {item.get('id', 'unknown')}")
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Results saved to: {output_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="TableQA Inference")
    parser.add_argument("--question", type=str, help="Single question to answer")
    parser.add_argument("--table", type=str, help="Table data (JSON file or string)")
    parser.add_argument("--input", type=str, help="Input JSONL file for batch inference")
    parser.add_argument("--output", type=str, default="predictions.jsonl", 
                       help="Output file for batch inference")
    
    args = parser.parse_args()
    
    # Load config and agents
    print("Loading agents...")
    config = load_config()
    agents = load_both_agents(config, use_plan_lora=True)
    # Use Plan Agent for pipeline
    model, tokenizer = agents['plan']
    
    # Create pipeline
    mcp_manager = MCPToolManager()
    pipeline = TableQAPipeline(model, tokenizer, mcp_manager)
    
    if args.input:
        # Batch inference
        print(f"\nBatch inference from: {args.input}")
        infer_batch(args.input, args.output, pipeline)
        
    elif args.question:
        # Single inference
        if args.table:
            # Load table from file or parse JSON string
            if Path(args.table).exists():
                with open(args.table, 'r') as f:
                    table_data = json.load(f)
            else:
                table_data = json.loads(args.table)
        else:
            # Default example table
            table_data = {
                "columns": ["name", "value"],
                "data": [["item1", "10"], ["item2", "20"]]
            }
        
        print(f"\nQuestion: {args.question}")
        result = infer_single(args.question, table_data, pipeline)
        
        print("\n" + "="*60)
        print("Inference Result")
        print("="*60)
        print(f"Answer: {result['final_answer']}")
        print(f"Strategy: {result['strategy']}")
        print(f"Subtasks: {len(result.get('subtasks', []))}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
