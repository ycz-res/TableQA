# TableQA

Table Question Answering system with Plan Agent and Reasoning Agent.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Download resources (optional)
python3 dataset.py
python3 models/download.py

# Train
python3 train.py

# Evaluate
python3 eval.py

# Inference
python3 infer.py --question "What is the average?" --table data.json

# Merge LoRA (optional)
python3 models/utils.py merge
```

## Configuration

Edit `config.yaml`:

```yaml
model: "Qwen/Qwen2.5-1.5B-Instruct"
dataset: "tablebench"
epochs: 3
batch_size: 2
learning_rate: 0.00002
```

## Features

- **Plan Agent**: Question decomposition (5 strategies)
- **Reasoning Agent**: Subtask execution with MCP tools
- **LoRA Training**: SFT cold start + GPRO optimization
- **Answer Format**: `<answer></answer>` output
- **886 Samples**: TableBench dataset

## Project Structure

```
TableQA/
├── train.py              # Training
├── eval.py               # Evaluation
├── infer.py              # Inference
├── dataset.py            # Dataset download & loader
├── utils.py              # Utilities (config, eval)
├── pipeline.py           # Pipeline & Agents
├── config.yaml           # Configuration (12 lines)
├── mcp/                  # MCP tools
│   └── tools.py
├── data/                 # Data files only (gitignored)
│   └── tablebench/
└── models/               # Model-related (gitignored)
    ├── utils.py          # Load, download, merge
    ├── trainer.py        # Training (SFT + GPRO)
    ├── pretrained/
    └── finetuned/
```

## License

MIT
