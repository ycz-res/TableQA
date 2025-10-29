#!/usr/bin/env python3
"""
Dataset utilities: Download and load TableBench dataset
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


# ==================== Download ====================

def download_tablebench():
    """Download TableBench dataset from HuggingFace"""
    print("Downloading TableBench dataset...")
    print("Source: https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench")
    
    save_dir = Path(__file__).parent / "data" / "tablebench"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from datasets import load_dataset
        
        # Download dataset
        dataset = load_dataset("Multilingual-Multimodal-NLP/TableBench")
        
        # Save to JSONL files
        for split_name, split_data in dataset.items():
            output_file = save_dir / f"{split_name}.jsonl"
            split_data.to_json(output_file)
            print(f"✓ Saved {split_name} to {output_file}")
        
        print(f"\n✓ Dataset downloaded to: {save_dir}")
        
    except ImportError:
        print("\n❌ Please install datasets library:")
        print("   pip install datasets")
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nAlternative: Download manually from:")
        print("https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench")


# ==================== Load ====================

class TableBenchLoader:
    """TableBench Dataset Loader"""
    
    AVAILABLE_VERSIONS = {
        'base': 'TableBench.jsonl',
        'DP': 'TableBench_DP.jsonl',
        'PoT': 'TableBench_PoT.jsonl',
        'SCoT': 'TableBench_SCoT.jsonl',
        'TCoT': 'TableBench_TCoT.jsonl',
    }
    
    def __init__(self, data_dir: str = "./data/tablebench"):
        """
        Initialize loader
        
        Args:
            data_dir: Dataset directory path
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    def load(self, version: str = 'base', max_samples: Optional[int] = None) -> List[Dict]:
        """
        Load dataset
        
        Args:
            version: Dataset version ('base', 'DP', 'PoT', 'SCoT', 'TCoT')
            max_samples: Max samples to load, None for all
        
        Returns:
            List of data items
        """
        if version not in self.AVAILABLE_VERSIONS:
            raise ValueError(f"Unsupported version: {version}. Available: {list(self.AVAILABLE_VERSIONS.keys())}")
        
        file_path = self.data_dir / self.AVAILABLE_VERSIONS[version]
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        print(f"Loading {version} version: {file_path}")
        
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                item = json.loads(line.strip())
                data.append(item)
        
        print(f"✓ Loaded {len(data)} samples")
        return data
    
    def format_for_training(self, data: List[Dict], include_table: bool = True) -> List[Dict]:
        """
        Format data for training
        
        Args:
            data: Raw data list
            include_table: Whether to include table in input
        
        Returns:
            Formatted training data
        """
        formatted_data = []
        
        for item in data:
            formatted_item = {
                'id': item['id'],
                'question': item['question'],
                'answer': item['answer'],
            }
            
            if include_table:
                table_data = item.get('table', {})
                df = pd.DataFrame(table_data.get('data', []), 
                                columns=table_data.get('columns', []))
                formatted_item['table'] = df.to_markdown(index=False)
                formatted_item['input'] = f"Table:\n{formatted_item['table']}\n\nQuestion: {formatted_item['question']}"
            else:
                formatted_item['input'] = formatted_item['question']
            
            formatted_data.append(formatted_item)
        
        return formatted_data


# ==================== Main ====================

if __name__ == "__main__":
    download_tablebench()
