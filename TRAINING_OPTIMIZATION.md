# TableQA训练优化说明

## 优化概述

根据您的要求，我已经对TableQA训练代码进行了全面优化，实现了以下关键功能：

### 1. 简化的数据格式
- **格式要求**: 只使用 `<answer></answer>` 格式输出答案
- **实现位置**: `TableQADataset.__getitem__()` 方法
- **格式示例**:
```
<answer>
7.67
</answer>
```

### 2. 答案对比机制
- **实现类**: `AnswerComparator`
- **对比指标**:
  - **Format分数**: 检查是否包含完整的 `<answer></answer>` 标签
  - **EM分数**: 精确匹配分数，与标准答案对比
  - **总奖励**: `0.1 * format + 0.9 * EM`
  - **准确率统计**: 提供详细的准确率分析

### 3. LoRA微调优化
- **配置位置**: `config.yaml` 中的 `lora` 部分
- **参数优化**:
  - `r`: 16 (秩)
  - `lora_alpha`: 32
  - `target_modules`: ["q_proj", "v_proj", "k_proj", "o_proj"]
  - `lora_dropout`: 0.1

### 4. SFT冷启动训练
- **实现函数**: `train_sft_cold_start()`
- **训练特点**:
  - 只训练 **1个epoch**
  - 使用标准监督微调
  - 保存为 `_sft` 后缀的模型

### 5. GPRO训练
- **实现类**: `GPROTrainer`
- **奖励机制**: `reward = 0.1 * format + 0.9 * EM`
- **训练特点**:
  - 基于奖励的强化学习
  - 动态调整损失权重
  - 保存为 `_gpro` 后缀的模型

## 训练流程

### 阶段1: SFT冷启动
```python
# 1. 加载预训练模型
# 2. 应用LoRA配置
# 3. 使用标准SFT训练1个epoch
# 4. 保存SFT模型
```

### 阶段2: GPRO训练
```python
# 1. 加载SFT模型
# 2. 创建GPRO训练器
# 3. 基于奖励进行强化学习
# 4. 保存最终GPRO模型
```

## 配置文件更新

### 新增配置项
```yaml
# GPRO训练配置
gpro:
  epochs: 3
  batch_size: 4
  learning_rate: 1e-5
  weight_decay: 0.01
  logging_steps: 10
  # 奖励权重配置
  format_weight: 0.1  # format分数权重
  em_weight: 0.9      # EM分数权重

# 答案对比配置
evaluation:
  answer_comparison:
    normalize_answers: true
    case_sensitive: false
    remove_punctuation: true
```

## 使用方法

### 1. 运行训练
```bash
cd /home/zyc/TableQA
python main.py --mode train
```

### 2. 测试优化功能
```bash
python3 test_optimized_training.py
```

### 3. 演示答案格式和准确率
```bash
python3 demo_answer_format.py
```

### 4. 评估模型
```bash
python3 main.py --mode eval
```

## 关键改进

1. **数据格式标准化**: 确保所有训练数据都使用统一的 `<answer></answer>` 格式
2. **奖励机制**: 实现了基于格式和准确性的复合奖励函数
3. **两阶段训练**: SFT冷启动 + GPRO强化学习
4. **错误处理**: 添加了完善的异常处理和回退机制
5. **配置灵活性**: 支持通过配置文件调整所有训练参数

## 文件结构

```
src/
├── train.py          # 主要训练代码（已优化）
├── eval.py           # 评估代码
├── pipeline.py       # 推理管道
└── mcp_tools.py      # MCP工具

config.yaml           # 配置文件（已更新）
test_optimized_training.py  # 测试脚本（新增）
demo_answer_format.py      # 答案格式演示脚本（新增）
```

## 注意事项

1. **内存使用**: GPRO训练需要更多内存，建议使用GPU
2. **训练时间**: 两阶段训练会增加总训练时间
3. **模型保存**: 会生成多个模型文件（SFT和GPRO版本）
4. **数据格式**: 确保输入数据包含 `question`, `table`, `answer` 字段

## 性能预期

- **SFT阶段**: 快速收敛，建立基础能力
- **GPRO阶段**: 通过奖励机制优化格式和准确性
- **最终效果**: 更好的格式一致性和答案准确性

## 准确率计算

### 评估指标
- **格式准确率**: 使用正确 `<answer></answer>` 格式的比例
- **精确匹配准确率**: 答案内容完全正确的比例
- **整体准确率**: 精确匹配准确率（主要指标）

### 奖励机制
- **格式分数**: 0.0-1.0，检查是否包含 `<answer></answer>` 标签
- **EM分数**: 0.0-1.0，与标准答案的精确匹配
- **总奖励**: `0.1 * 格式分数 + 0.9 * EM分数`

### 示例
```
输入: "What is the average?"
生成: "<answer>\n7.67\n</answer>"
参考: "7.67"
结果: 格式分数=1.0, EM分数=1.0, 总奖励=1.0
```
