# TableQA Pipeline 使用指南

## 🏗️ 系统架构

TableQA Pipeline 是一个基于多Agent的表格问答系统，包含以下核心组件：

```
输入问题 → Plan Agent → 子任务拆分 → Reasoning Agent → MCP工具检索 → 最终答案
```

### 核心组件

1. **Plan Agent** - 问题拆分代理
   - 5种拆分策略自动选择
   - 子任务独立性验证
   - 依赖关系管理

2. **Reasoning Agent** - 推理执行代理
   - 子任务执行
   - MCP工具调用
   - 上下文管理

3. **MCP工具系统** - 检索工具
   - 关键词稀疏检索
   - BM25+BGE-M3稠密检索
   - 结果融合

## 🎯 5种拆分策略

### 1. Aggregation (聚合策略)
**适用场景**: 计算平均值、总和、计数等聚合操作

**示例问题**: "What is the average number of tropical cyclones per season?"

**拆分结果**:
```json
{
  "strategy": "aggregation",
  "subtasks": [
    {
      "id": "task_1",
      "description": "提取tropical cyclones列的所有数值",
      "task_type": "independent",
      "dependencies": [],
      "expected_output": "数值列表"
    },
    {
      "id": "task_2",
      "description": "计算这些数值的平均值",
      "task_type": "aggregate", 
      "dependencies": ["task_1"],
      "expected_output": "平均值结果"
    }
  ]
}
```

### 2. Comparison (比较策略)
**适用场景**: 比较两个或多个实体的指标

**示例问题**: "Which country has higher GDP, China or USA?"

**拆分结果**:
```json
{
  "strategy": "comparison",
  "subtasks": [
    {
      "id": "task_1",
      "description": "计算中国的GDP",
      "task_type": "independent",
      "dependencies": []
    },
    {
      "id": "task_2", 
      "description": "计算美国的GDP",
      "task_type": "independent",
      "dependencies": []
    },
    {
      "id": "task_3",
      "description": "比较两个GDP值",
      "task_type": "compare",
      "dependencies": ["task_1", "task_2"]
    }
  ]
}
```

### 3. Bridge (桥接策略)
**适用场景**: 多步推理，前一步结果影响后一步

**示例问题**: "What is the total revenue of companies with profit > 1000?"

**拆分结果**:
```json
{
  "strategy": "bridge",
  "subtasks": [
    {
      "id": "task_1",
      "description": "筛选出利润大于1000的公司",
      "task_type": "bridge",
      "dependencies": []
    },
    {
      "id": "task_2",
      "description": "计算这些公司的总收入",
      "task_type": "bridge",
      "dependencies": ["task_1"]
    }
  ]
}
```

### 4. Sequential (顺序策略)
**适用场景**: 时间序列或逻辑顺序问题

**示例问题**: "What was the trend of sales from 2020 to 2023?"

**拆分结果**:
```json
{
  "strategy": "sequential",
  "subtasks": [
    {
      "id": "task_1",
      "description": "提取2020年的销售数据",
      "task_type": "sequential",
      "dependencies": []
    },
    {
      "id": "task_2",
      "description": "提取2021年的销售数据", 
      "task_type": "sequential",
      "dependencies": ["task_1"]
    },
    {
      "id": "task_3",
      "description": "分析销售趋势",
      "task_type": "sequential",
      "dependencies": ["task_1", "task_2"]
    }
  ]
}
```

### 5. Independent (独立策略)
**适用场景**: 可以并行执行的独立任务

**示例问题**: "What are the top 3 countries by population and GDP?"

**拆分结果**:
```json
{
  "strategy": "independent",
  "subtasks": [
    {
      "id": "task_1",
      "description": "找出人口最多的3个国家",
      "task_type": "independent",
      "dependencies": []
    },
    {
      "id": "task_2",
      "description": "找出GDP最高的3个国家",
      "task_type": "independent",
      "dependencies": []
    },
    {
      "id": "task_3",
      "description": "汇总两个列表的结果",
      "task_type": "aggregate",
      "dependencies": ["task_1", "task_2"]
    }
  ]
}
```

## 🔧 快速开始

### 1. 基本使用

```python
from src.pipeline import TableQAPipeline
from src.mcp_tools import MCPToolManager
from modelscope import AutoModelForCausalLM, AutoTokenizer

# 加载模型
model = AutoModelForCausalLM.from_pretrained("./models/pretrained/Qwen/Qwen2.5-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("./models/pretrained/Qwen/Qwen2.5-1.5B-Instruct")

# 创建MCP工具管理器
mcp_manager = MCPToolManager()

# 创建pipeline
pipeline = TableQAPipeline(model, tokenizer, mcp_manager)

# 处理问题
question = "What is the average number of tropical cyclones per season?"
table_data = {
    "columns": ["season", "tropical cyclones"],
    "data": [
        ["1990-91", "10"],
        ["1991-92", "10"], 
        ["1992-93", "3"]
    ]
}

result = pipeline.process_question(question, table_data)
print(result['final_answer'])
```

### 2. 测试拆分策略

```bash
# 测试5种拆分策略
python3 test_pipeline.py

# 输出示例:
# ======================================================================
# 测试案例 1: 聚合类问题
# ======================================================================
# 问题: What is the average number of tropical cyclones per season?
# 识别策略: aggregation
# 拆分策略: aggregation
# 
# 子任务 (2 个):
#   1. 提取tropical cyclones列的所有数值
#      类型: independent
#      依赖: []
#      预期输出: 数值列表
#   2. 计算这些数值的平均值
#      类型: aggregate
#      依赖: ['task_1']
#      预期输出: 平均值结果
# 
# 独立性检查: ✓ 通过
```

### 3. 评估Pipeline性能

```bash
# 基于TableBench数据集评估
python3 evaluate_pipeline.py

# 输出示例:
# ================================================================================
# TableQA Pipeline 评估报告
# ================================================================================
# 
# 📊 策略分类效果:
#   准确率: 0.840
#   正确分类: 42/50
# 
# 🔗 子任务独立性:
#   整体通过率: 0.920
#   通过: 46/50
# 
# ⚡ Pipeline性能:
#   成功率: 0.800
#   平均执行时间: 2.34s
```

## 🛠️ 高级配置

### 1. 自定义拆分策略

```python
from src.pipeline import PlanAgent

# 创建Plan Agent
plan_agent = PlanAgent(model, tokenizer)

# 自定义策略提示词
custom_strategy = """
你是一个专业的表格问题分析专家。对于自定义问题，请按以下规则拆分：
...
"""

# 添加自定义策略
plan_agent.strategies["custom"] = custom_strategy
```

### 2. 配置MCP工具

```python
from src.mcp_tools import MCPToolManager, BM25BGERetrieval

# 创建自定义BGE模型路径
bge_retrieval = BM25BGERetrieval(bge_model_path="path/to/bge-model")

# 创建工具管理器
mcp_manager = MCPToolManager()
mcp_manager.tools["custom_bge"] = bge_retrieval

# 使用自定义工具
results = mcp_manager.search("custom_bge", query, table_data)
```

### 3. 子任务独立性验证

```python
from src.pipeline import PlanAgent

plan_agent = PlanAgent(model, tokenizer)

# 拆分问题
split_result = plan_agent.split_question(question, table_data)

# 验证独立性
is_valid, issues = plan_agent.validate_independence(split_result['subtasks'])

if not is_valid:
    print("独立性问题:")
    for issue in issues:
        print(f"  - {issue}")
```

## 📊 性能优化

### 1. 模型选择

| 模型 | 参数量 | 内存需求 | 推理速度 | 推荐场景 |
|------|--------|----------|----------|----------|
| Qwen2.5-0.5B | 0.5B | 1GB | 很快 | 快速测试 |
| Qwen2.5-1.5B | 1.5B | 3GB | 快 | 开发调试 |
| Qwen2.5-7B | 7B | 14GB | 中等 | 生产环境 |
| Qwen2.5-14B | 14B | 28GB | 慢 | 高精度需求 |

### 2. 检索优化

```python
# 调整检索参数
keyword_results = mcp_manager.search("keyword", query, table_data, top_k=5)
bm25_bge_results = mcp_manager.search("bm25_bge", query, table_data, top_k=3)

# 融合权重调整
# 在BM25BGERetrieval._combine_results中修改:
# combined_score = 0.3 * bm25_score + 0.7 * bge_score
```

### 3. 批处理优化

```python
# 批量处理多个问题
questions = ["问题1", "问题2", "问题3"]
results = []

for question in questions:
    result = pipeline.process_question(question, table_data)
    results.append(result)
```

## 🔍 故障排除

### 1. 常见问题

**问题**: 模型加载失败
```bash
# 解决方案: 检查模型路径和依赖
ls -la ./models/pretrained/Qwen/
pip3 install -r requirements.txt
```

**问题**: 子任务独立性检查失败
```python
# 解决方案: 检查拆分策略
is_valid, issues = plan_agent.validate_independence(subtasks)
print("问题:", issues)
```

**问题**: MCP工具检索失败
```python
# 解决方案: 检查工具配置
print("可用工具:", mcp_manager.list_tools())
```

### 2. 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 单步调试
result = pipeline.process_question(question, table_data)
print("拆分结果:", result['subtasks'])
print("子任务结果:", result['subtask_results'])
```

## 📈 评估指标

### 1. 策略分类准确率
- 目标: > 80%
- 计算方法: 正确分类的问题数 / 总问题数

### 2. 子任务独立性通过率
- 目标: > 90%
- 计算方法: 通过独立性检查的样本数 / 总样本数

### 3. Pipeline成功率
- 目标: > 75%
- 计算方法: 成功执行的样本数 / 总样本数

### 4. 平均执行时间
- 目标: < 5秒
- 计算方法: 总执行时间 / 样本数

## 🚀 扩展开发

### 1. 添加新的拆分策略

```python
def _get_custom_strategy(self) -> str:
    return """
    自定义策略提示词...
    """

# 在PlanAgent.__init__中添加
self.strategies["custom"] = self._get_custom_strategy()
```

### 2. 添加新的检索工具

```python
class CustomRetrieval(RetrievalTool):
    def search(self, query: str, table_data: Dict, top_k: int = 5) -> List[RetrievalResult]:
        # 实现自定义检索逻辑
        pass

# 注册到MCP管理器
mcp_manager.tools["custom"] = CustomRetrieval()
```

### 3. 自定义评估指标

```python
def custom_evaluation_metric(result):
    # 实现自定义评估逻辑
    return score

# 在evaluate_pipeline.py中使用
```

## 📚 参考资料

- [TableBench数据集](https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench)
- [Qwen2.5模型](https://github.com/QwenLM/Qwen2.5)
- [MCP协议](https://modelcontextprotocol.io/)
- [BM25算法](https://en.wikipedia.org/wiki/Okapi_BM25)
- [BGE-M3模型](https://huggingface.co/BAAI/bge-m3)

---

**注意**: 本系统仍在开发中，如有问题请查看日志或提交Issue。

