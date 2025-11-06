"""
OTT-QA Dataset Loader
支持加载 dev_linked.json 以及关联的表格和passage数据
"""

import json
from typing import Dict, Optional, Any
from pathlib import Path

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from torch.utils.data import Dataset
except ImportError:
    class Dataset:
        def __len__(self):
            raise NotImplementedError
        def __getitem__(self, idx):
            raise NotImplementedError


def load_config(config_file: str = "config.yaml") -> Dict[str, Any]:
    """加载配置文件"""
    if not YAML_AVAILABLE:
        print("Warning: PyYAML not installed, using default config")
        return {}
    
    config_path = Path(config_file)
    if not config_path.exists():
        print(f"Warning: Config file {config_file} not found, using default config")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


class TableQADataset(Dataset):
    """
    OTT-QA 数据集加载器
    
    Args:
        dataset_name: 数据集名称，对应 config.yaml 中 dataset 下的键（默认: "OTTQA"）
        data_file: 数据文件路径（覆盖配置文件）
        table_dir: 表格文件夹路径（覆盖配置文件）
        passage_dir: passage文件夹路径（覆盖配置文件）
        reference_file: 标准答案文件路径（覆盖配置文件）
        load_tables: 是否加载表格数据（覆盖配置文件）
        load_passages: 是否加载passage数据（覆盖配置文件）
        config_file: 配置文件路径
    """
    
    def __init__(
        self,
        dataset_name: str = "OTTQA",
        data_file: Optional[str] = None,
        table_dir: Optional[str] = None,
        passage_dir: Optional[str] = None,
        reference_file: Optional[str] = None,
        load_tables: Optional[bool] = None,
        load_passages: Optional[bool] = None,
        config_file: str = "config.yaml"
    ):
        # 加载配置文件
        config = load_config(config_file)
        dataset_configs = config.get('dataset', {})
        
        # 获取指定数据集的配置
        if dataset_name not in dataset_configs:
            raise ValueError(
                f"Dataset '{dataset_name}' not found in config. "
                f"Available datasets: {list(dataset_configs.keys())}"
            )
        
        ds_config = dataset_configs[dataset_name]
        
        # 设置基础目录
        base_dir = ds_config.get('base_dir')
        base_path = Path(base_dir) if base_dir else Path.cwd()
        
        # 从配置文件或参数获取路径（参数优先）
        self.data_file = Path(data_file or ds_config.get('data_file', 'data/dev_linked.json'))
        self.table_dir = Path(table_dir or ds_config.get('table_dir', 'data/tables'))
        self.passage_dir = Path(passage_dir or ds_config.get('passage_dir', 'data/passages'))
        ref_file = reference_file or ds_config.get('reference_file')
        self.reference_file = Path(ref_file) if ref_file else None
        
        # 如果设置了base_dir，所有路径相对于base_dir
        if base_dir:
            self.data_file = base_path / self.data_file
            self.table_dir = base_path / self.table_dir
            self.passage_dir = base_path / self.passage_dir
            if self.reference_file:
                self.reference_file = base_path / self.reference_file
        
        # 从配置文件或参数获取选项（参数优先）
        self.load_tables = load_tables if load_tables is not None else ds_config.get('load_tables', True)
        self.load_passages = load_passages if load_passages is not None else ds_config.get('load_passages', True)
        self.verbose = ds_config.get('verbose', True)
        
        # 加载数据
        with open(self.data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        if self.verbose:
            print(f"Loaded {len(self.data)} samples from {self.data_file}")
        
        # 加载参考答案
        self.reference = None
        if self.reference_file and self.reference_file.exists():
            with open(self.reference_file, 'r', encoding='utf-8') as f:
                self.reference = json.load(f).get('reference', {})
            if self.verbose:
                print(f"Loaded {len(self.reference)} reference answers")
    
    def _load_json(self, file_path: Path) -> Optional[Dict]:
        """加载JSON文件"""
        if not file_path.exists():
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def get_reference_answer(self, question_id: str) -> Optional[str]:
        """获取标准答案"""
        return self.reference.get(question_id) if self.reference else None
    
    def get_passage_by_link(self, table_id: str, entity_link: str) -> Optional[str]:
        """根据实体链接获取passage文本"""
        if not self.load_passages:
            return None
        passages = self._load_json(self.passage_dir / f"{table_id}.json")
        return passages.get(entity_link) if passages else None
    
    def get_cell_content(self, table_id: str, row: int, col: int) -> Optional[str]:
        """获取表格单元格内容"""
        if not self.load_tables:
            return None
        table = self._load_json(self.table_dir / f"{table_id}.json")
        if not table:
            return None
        data = table.get('data', [])
        if 0 <= row < len(data) and 0 <= col < len(data[row]):
            cell = data[row][col]
            return cell[0] if isinstance(cell, list) and cell else None
        return None
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        table_id = item['table_id']
        
        sample = {
            'question_id': item['question_id'],
            'question': item['question'],
            'answer_text': item['answer-text'],
            'table_id': table_id,
            'answer_nodes': item.get('answer-node', []),
            'matched_cells': {
                'tf-idf': item.get('tf-idf', []),
                'string-overlap': item.get('string-overlap', []),
                'links': item.get('links', [])
            },
            'difficulty': item.get('type', 'unknown'),
            'answer_source': item.get('where', 'unknown'),
            'question_postag': item.get('question_postag', ''),
            'table': self._load_json(self.table_dir / f"{table_id}.json") if self.load_tables else None,
            'passages': self._load_json(self.passage_dir / f"{table_id}.json") if self.load_passages else None,
            'reference_answer': self.get_reference_answer(item['question_id'])
        }
        return sample
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        from collections import Counter
        return {
            'total_samples': len(self.data),
            'unique_tables': len(set(item['table_id'] for item in self.data)),
            'difficulty_distribution': dict(Counter(item.get('type', 'unknown') for item in self.data)),
            'answer_source_distribution': dict(Counter(item.get('where', 'unknown') for item in self.data)),
            'has_reference': self.reference is not None
        }


if __name__ == "__main__":
    # 示例用法：从配置文件加载（使用默认数据集 OTTQA）
    print("从 config.yaml 加载 OTTQA 数据集...")
    dataset = TableQADataset()  # 默认使用 OTTQA
    
    print("\n统计信息:")
    stats = dataset.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\n第一个样本:")
    sample = dataset[0]
    print(f"  Question: {sample['question'][:60]}...")
    print(f"  Answer: {sample['answer_text']}")
    print(f"  Table: {'✓' if sample['table'] else '✗'}")
    print(f"  Passages: {'✓' if sample['passages'] else '✗'}")
    
    # 示例：指定数据集名称
    print("\n" + "="*50)
    print("指定数据集名称示例:")
    dataset2 = TableQADataset(dataset_name="OTTQA")
    print(f"  Loaded {len(dataset2)} samples")
    
    # 示例：直接指定参数（会覆盖配置文件）
    print("\n" + "="*50)
    print("覆盖配置示例:")
    dataset3 = TableQADataset(
        dataset_name="OTTQA",
        load_tables=False  # 覆盖配置文件中的设置
    )
    print(f"  Loaded {len(dataset3)} samples (tables disabled)")
