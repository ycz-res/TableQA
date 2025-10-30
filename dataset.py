#!/usr/bin/env python3
"""
Dataset classes for TableQA
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
from collections import Counter


# ==================== TableBench Dataset ====================

class TableBenchDataset:
    """
    TableBench Dataset
    
    A comprehensive benchmark for table-based question answering with:
    - 4 main categories: NumericalReasoning, DataAnalysis, FactChecking, Visualization
    - 18 subcategories: Aggregation, Comparison, Counting, etc.
    - 886 total samples
    
    Note: To download the dataset, run:
        python3 data/tablebench/download.py
    """
    
    # Available versions
    VERSIONS = {
        'base': 'TableBench.jsonl',
        'DP': 'TableBench_DP.jsonl',
        'PoT': 'TableBench_PoT.jsonl',
        'SCoT': 'TableBench_SCoT.jsonl',
        'TCoT': 'TableBench_TCoT.jsonl',
    }
    
    # Category information
    CATEGORIES = ['NumericalReasoning', 'DataAnalysis', 'FactChecking', 'Visualization']
    SUBCATEGORIES = [
        'Multi-hop NumericalReasoing', 'DescriptiveAnalysis', 'Aggregation',
        'ChartGeneration', 'ImpactAnalysis', 'Multi-hop FactChecking',
        'Counting', 'ArithmeticCalculation', 'CorrelationAnalysis',
        'Ranking', 'TrendForecasting', 'StatisticalAnalysis',
        'Comparison', 'Domain-Specific', 'AnomalyDetection',
        'Time-basedCalculation', 'MatchBased', 'CausalAnalysis'
    ]
    
    def __init__(self, data_dir: str = "./data/tablebench"):
        """
        Initialize TableBench dataset
        
        Args:
            data_dir: Dataset directory path
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {data_dir}\n"
                f"Please download the dataset first:\n"
                f"  python3 data/tablebench/download.py"
            )
    
    def load(
        self,
        version: str = 'base',
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
        qtype: Optional[Union[str, List[str]]] = None,
        qsubtype: Optional[Union[str, List[str]]] = None,
        show_stats: bool = True
    ) -> List[Dict]:
        """
        Load TableBench dataset with filtering
        
        Args:
            version: Dataset version ('base', 'DP', 'PoT', 'SCoT', 'TCoT')
            split: Data split ('train', 'test', 'val'), None for default file
            max_samples: Max samples to load (None for all)
            qtype: Category filter (str or list)
                   Options: 'NumericalReasoning', 'DataAnalysis', 'FactChecking', 'Visualization'
                   Use None or 'all' to load all categories
            qsubtype: Subcategory filter (str or list)
                      See SUBCATEGORIES for options
                      Use None or 'all' to load all subcategories
            show_stats: Whether to show loading statistics
        
        Returns:
            List of data items, each with:
                - id: Unique identifier
                - qtype: Main category
                - qsubtype: Subcategory
                - question: Question text
                - answer: Answer text
                - table: Table data (dict with 'columns' and 'data')
        
        Examples:
            # Load all data
            dataset.load()
            
            # Load specific category
            dataset.load(qtype='NumericalReasoning')
            
            # Load specific subcategory
            dataset.load(qsubtype='Aggregation')
            
            # Load multiple categories
            dataset.load(qtype=['NumericalReasoning', 'DataAnalysis'])
            
            # Load first 10 samples
            dataset.load(max_samples=10)
        """
        # Determine file path
        if version not in self.VERSIONS:
            raise ValueError(f"Unknown version: {version}. Available: {list(self.VERSIONS.keys())}")
        
        if split:
            file_name = f"{split}.jsonl"
        else:
            file_name = self.VERSIONS[version]
        
        file_path = self.data_dir / file_name
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {file_path}\n"
                f"Please download the dataset first:\n"
                f"  python3 data/tablebench/download.py"
            )
        
        # Normalize filters
        if qtype == 'all':
            qtype = None
        if qsubtype == 'all':
            qsubtype = None
        
        qtype_filter = [qtype] if isinstance(qtype, str) else qtype
        qsubtype_filter = [qsubtype] if isinstance(qsubtype, str) else qsubtype
        
        if show_stats:
            print(f"Loading from: {file_path}")
            if qtype_filter:
                print(f"  Category filter: {qtype_filter}")
            if qsubtype_filter:
                print(f"  Subcategory filter: {qsubtype_filter}")
        
        # Load and filter data
        data = []
        total_read = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_read += 1
                item = json.loads(line.strip())
                
                # Apply filters
                if qtype_filter and item['qtype'] not in qtype_filter:
                    continue
                if qsubtype_filter and item['qsubtype'] not in qsubtype_filter:
                    continue
                
                data.append(item)
                
                # Check max_samples
                if max_samples and len(data) >= max_samples:
                    break
        
        if show_stats:
            print(f"✓ Loaded {len(data)} samples (from {total_read} total)")
            
            if len(data) > 0:
                # Show category distribution
                qtypes = Counter(item['qtype'] for item in data)
                qsubtypes = Counter(item['qsubtype'] for item in data)
                
                if len(qtypes) <= 5:
                    print(f"  Categories: {dict(qtypes)}")
                if len(qsubtypes) <= 10:
                    print(f"  Subcategories: {dict(qsubtypes)}")
        
        return data
    
    def get_stats(self, version: str = 'base') -> Dict:
        """
        Get dataset statistics
        
        Args:
            version: Dataset version
        
        Returns:
            Dict with:
                - total_samples: Total number of samples
                - categories: Dict of category counts
                - subcategories: Dict of subcategory counts
        """
        file_path = self.data_dir / self.VERSIONS[version]
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        qtypes = []
        qsubtypes = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                qtypes.append(item['qtype'])
                qsubtypes.append(item['qsubtype'])
        
        return {
            'total_samples': len(qtypes),
            'categories': dict(Counter(qtypes)),
            'subcategories': dict(Counter(qsubtypes))
        }
    
    def format_for_training(self, data: List[Dict], include_table: bool = True) -> List[Dict]:
        """
        Format data for training
        
        Args:
            data: Raw data list
            include_table: Whether to include table in input
        
        Returns:
            Formatted training data with 'input', 'table', 'answer' fields
        """
        formatted_data = []
        
        for item in data:
            formatted_item = {
                'id': item['id'],
                'question': item['question'],
                'answer': item['answer'],
                'qtype': item['qtype'],
                'qsubtype': item['qsubtype'],
            }
            
            if include_table:
                # Store original table data
                formatted_item['table'] = item.get('table', {})
                
                # Create markdown table for input
                table_data = item.get('table', {})
                df = pd.DataFrame(
                    table_data.get('data', []),
                    columns=table_data.get('columns', [])
                )
                table_md = df.to_markdown(index=False)
                formatted_item['input'] = f"Table:\n{table_md}\n\nQuestion: {formatted_item['question']}"
            else:
                formatted_item['input'] = formatted_item['question']
            
            formatted_data.append(formatted_item)
        
        return formatted_data


# ==================== Aliases ====================

# For backward compatibility
TableBenchLoader = TableBenchDataset


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TableBench Dataset Manager")
    parser.add_argument("command", choices=["stats", "sample"],
                       help="Command to run")
    parser.add_argument("--qtype", type=str,
                       help="Category filter (e.g., NumericalReasoning)")
    parser.add_argument("--qsubtype", type=str,
                       help="Subcategory filter (e.g., Aggregation)")
    parser.add_argument("--n", type=int, default=5,
                       help="Number of samples to show (default: 5)")
    
    args = parser.parse_args()
    
    # Create dataset instance
    dataset = TableBenchDataset()
    
    if args.command == "stats":
        stats = dataset.get_stats()
        
        print("="*60)
        print("TableBench Dataset Statistics")
        print("="*60)
        print(f"\nTotal samples: {stats['total_samples']}")
        
        print("\nCategories (qtype):")
        for cat, count in sorted(stats['categories'].items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")
        
        print("\nSubcategories (qsubtype):")
        for subcat, count in sorted(stats['subcategories'].items(), key=lambda x: -x[1]):
            print(f"  {subcat}: {count}")
    
    elif args.command == "sample":
        data = dataset.load(
            qtype=args.qtype,
            qsubtype=args.qsubtype,
            max_samples=args.n
        )
        
        print(f"\n{'='*60}")
        print(f"Sample Data ({len(data)} items)")
        print("="*60)
        
        for i, item in enumerate(data):
            print(f"\n[{i+1}] {item['qtype']} > {item['qsubtype']}")
            print(f"  Q: {item['question'][:60]}...")
            print(f"  A: {item['answer']}")
