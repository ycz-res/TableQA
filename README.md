# OTT-QA Dataset Loader

OTT-QA (Open Table-and-Text Question Answering) 数据集加载器，支持从配置文件加载数据。

## 功能特性

- 📁 支持 YAML 配置文件管理数据集路径
- 🔄 支持多个数据集配置
- 📊 自动加载表格和 passage 数据
- 🔍 支持按需检索 passage 和单元格内容
- 📈 提供数据集统计信息

## 安装

```bash
pip install PyYAML
```

或使用 requirements.txt：

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 配置文件

编辑 `config.yaml`：

```yaml
dataset:
  OTTQA:
    data_file: "data/OTTQA/linked.json"
    table_dir: "data/OTTQA/tables"
    passage_dir: "data/OTTQA/passages"
    reference_file: "data/OTTQA/reference.json"
```

### 2. 使用数据集

```python
from dataset import TableQADataset

# 加载数据集
dataset = TableQADataset()

# 获取样本
sample = dataset[0]
print(sample['question'])
print(sample['answer_text'])

# 获取统计信息
stats = dataset.get_statistics()
print(stats)
```

## API 文档

### TableQADataset

#### 初始化参数

- `dataset_name` (str): 数据集名称，默认 "OTTQA"
- `data_file` (str, optional): 数据文件路径（覆盖配置）
- `table_dir` (str, optional): 表格文件夹路径（覆盖配置）
- `passage_dir` (str, optional): Passage文件夹路径（覆盖配置）
- `reference_file` (str, optional): 标准答案文件路径（覆盖配置）
- `load_tables` (bool, optional): 是否加载表格数据
- `load_passages` (bool, optional): 是否加载passage数据
- `config_file` (str): 配置文件路径，默认 "config.yaml"

#### 主要方法

- `__getitem__(idx)`: 获取指定索引的样本
- `__len__()`: 返回数据集大小
- `get_statistics()`: 获取数据集统计信息
- `get_reference_answer(question_id)`: 获取标准答案
- `get_passage_by_link(table_id, entity_link)`: 根据实体链接获取passage
- `get_cell_content(table_id, row, col)`: 获取表格单元格内容

## 项目结构

```
.
├── config.yaml          # 配置文件
├── dataset.py           # 数据集加载器
├── requirements.txt     # 依赖包
└── data/
    └── OTTQA/
        ├── linked.json
        ├── reference.json
        ├── tables/
        └── passages/
```

## License

MIT

