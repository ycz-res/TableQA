# TableQA

基于多Agent的表格问答系统，实现Plan Agent和Reasoning Agent的迭代式协作。

## 🏗️ 系统架构

```
表格+问题 → Plan Agent(LoRA微调) → 子任务拆分 → Reasoning Agent → MCP工具检索 → 迭代执行 → 最终答案
```

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 下载模型
```bash
python3 download_model.py --model_id "Qwen/Qwen2.5-1.5B-Instruct"
```

### 3. 运行系统
```bash
# 训练Plan Agent
python3 main.py --mode train

# 推理测试
python3 test_training.py --mode inference

# 完整训练+推理
python3 test_training.py --mode train
```

## 🎯 核心特性

- **Plan Agent**: LoRA微调，5种拆分策略
- **Reasoning Agent**: 执行子任务，MCP工具调用
- **迭代协作**: Plan和Reasoning Agent多轮交互
- **上下文管理**: 子任务间信息传递

## 📁 项目结构

```
TableQA/
├── src/
│   ├── pipeline.py        # 核心Pipeline
│   ├── train.py          # LoRA训练
│   ├── eval.py           # 评估
│   └── mcp_tools.py      # MCP工具
├── datasets/tablebench/   # 数据集
├── models/pretrained/     # 预训练模型
└── config.yaml           # 配置
```

## 🔧 配置

```yaml
model:
  name: "./models/pretrained/Qwen/Qwen2.5-1.5B-Instruct"
  max_length: 1024

lora:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

training:
  epochs: 3
  batch_size: 4
  output_dir: "./models/finetuned"
```