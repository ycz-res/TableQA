"""
TableQA Pipeline: Plan Agent + Reasoning Agent
支持5种问题拆分策略，确保子任务独立性
"""

import json
import re
from typing import List, Dict, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod


class TaskType(Enum):
    """任务类型枚举"""
    INDEPENDENT = "independent"  # 独立任务 (A + B)
    BRIDGE = "bridge"           # 桥接任务 (A -> B -> C)
    COMPARE = "compare"         # 比较任务 (A vs B)
    AGGREGATE = "aggregate"     # 聚合任务 (A + B + C)
    SEQUENTIAL = "sequential"   # 顺序任务 (A -> B)


@dataclass
class SubTask:
    """子任务数据结构"""
    id: str
    description: str
    task_type: TaskType
    dependencies: List[str]  # 依赖的其他子任务ID
    expected_output: str
    reasoning_steps: List[str]


class PlanAgent:
    """计划代理 - 负责问题拆分"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # 5种拆分策略的提示词模板
        self.strategies = {
            "aggregation": self._get_aggregation_strategy(),
            "comparison": self._get_comparison_strategy(), 
            "bridge": self._get_bridge_strategy(),
            "sequential": self._get_sequential_strategy(),
            "independent": self._get_independent_strategy()
        }
    
    def _get_aggregation_strategy(self) -> str:
        """聚合类问题拆分策略"""
        return """
你是一个专业的表格问题分析专家。对于聚合类问题，请按以下规则拆分：

**拆分原则**：
1. 识别需要聚合的数值列
2. 将复杂聚合拆分为简单计算步骤
3. 确保每个子任务只处理一个数值列
4. 最后一步进行聚合计算

**输出格式**：
```json
{
  "strategy": "aggregation",
  "subtasks": [
    {
      "id": "task_1",
      "description": "提取[列名]的所有数值",
      "task_type": "independent",
      "dependencies": [],
      "expected_output": "数值列表",
      "reasoning_steps": ["定位列", "提取数值", "验证数据"]
    },
    {
      "id": "task_2", 
      "description": "计算[聚合类型]",
      "task_type": "aggregate",
      "dependencies": ["task_1"],
      "expected_output": "聚合结果",
      "reasoning_steps": ["接收数值", "执行聚合", "返回结果"]
    }
  ]
}
```

**示例**：
问题：What is the average number of tropical cyclones per season?
拆分：
1. 提取tropical cyclones列的所有数值
2. 计算这些数值的平均值
"""
    
    def _get_comparison_strategy(self) -> str:
        """比较类问题拆分策略"""
        return """
你是一个专业的表格问题分析专家。对于比较类问题，请按以下规则拆分：

**拆分原则**：
1. 识别需要比较的两个或多个实体
2. 为每个实体创建独立的计算任务
3. 最后一步进行比较操作
4. 确保比较任务依赖所有计算任务

**输出格式**：
```json
{
  "strategy": "comparison",
  "subtasks": [
    {
      "id": "task_1",
      "description": "计算[实体A]的[指标]",
      "task_type": "independent", 
      "dependencies": [],
      "expected_output": "实体A的指标值",
      "reasoning_steps": ["定位实体A", "计算指标", "返回结果"]
    },
    {
      "id": "task_2",
      "description": "计算[实体B]的[指标]", 
      "task_type": "independent",
      "dependencies": [],
      "expected_output": "实体B的指标值",
      "reasoning_steps": ["定位实体B", "计算指标", "返回结果"]
    },
    {
      "id": "task_3",
      "description": "比较[实体A]和[实体B]的[指标]",
      "task_type": "compare",
      "dependencies": ["task_1", "task_2"],
      "expected_output": "比较结果",
      "reasoning_steps": ["接收两个指标值", "执行比较", "返回比较结果"]
    }
  ]
}
```

**示例**：
问题：Which country has higher GDP, China or USA?
拆分：
1. 计算中国的GDP
2. 计算美国的GDP  
3. 比较两个GDP值
"""
    
    def _get_bridge_strategy(self) -> str:
        """桥接类问题拆分策略"""
        return """
你是一个专业的表格问题分析专家。对于桥接类问题，请按以下规则拆分：

**拆分原则**：
1. 识别问题中的中间步骤和依赖关系
2. 按依赖顺序创建子任务
3. 每个子任务的输出作为下一个子任务的输入
4. 确保依赖链的完整性

**输出格式**：
```json
{
  "strategy": "bridge",
  "subtasks": [
    {
      "id": "task_1",
      "description": "[第一步描述]",
      "task_type": "bridge",
      "dependencies": [],
      "expected_output": "第一步结果",
      "reasoning_steps": ["执行第一步", "返回结果"]
    },
    {
      "id": "task_2",
      "description": "[第二步描述，依赖task_1结果]",
      "task_type": "bridge", 
      "dependencies": ["task_1"],
      "expected_output": "第二步结果",
      "reasoning_steps": ["接收task_1结果", "执行第二步", "返回结果"]
    }
  ]
}
```

**示例**：
问题：What is the total revenue of companies with profit > 1000?
拆分：
1. 筛选出利润大于1000的公司
2. 计算这些公司的总收入
"""
    
    def _get_sequential_strategy(self) -> str:
        """顺序类问题拆分策略"""
        return """
你是一个专业的表格问题分析专家。对于顺序类问题，请按以下规则拆分：

**拆分原则**：
1. 识别问题中的时间序列或逻辑顺序
2. 按顺序创建子任务
3. 每个子任务可能依赖前一个的结果
4. 保持顺序的完整性

**输出格式**：
```json
{
  "strategy": "sequential",
  "subtasks": [
    {
      "id": "task_1",
      "description": "[第一步描述]",
      "task_type": "sequential",
      "dependencies": [],
      "expected_output": "第一步结果",
      "reasoning_steps": ["执行第一步", "返回结果"]
    },
    {
      "id": "task_2", 
      "description": "[第二步描述]",
      "task_type": "sequential",
      "dependencies": ["task_1"],
      "expected_output": "第二步结果",
      "reasoning_steps": ["接收task_1结果", "执行第二步", "返回结果"]
    }
  ]
}
```

**示例**：
问题：What was the trend of sales from 2020 to 2023?
拆分：
1. 提取2020年的销售数据
2. 提取2021年的销售数据
3. 提取2022年的销售数据
4. 提取2023年的销售数据
5. 分析销售趋势
"""
    
    def _get_independent_strategy(self) -> str:
        """独立类问题拆分策略"""
        return """
你是一个专业的表格问题分析专家。对于独立类问题，请按以下规则拆分：

**拆分原则**：
1. 识别可以并行执行的独立子任务
2. 每个子任务不依赖其他子任务
3. 最后一步汇总所有独立结果
4. 最大化并行性

**输出格式**：
```json
{
  "strategy": "independent",
  "subtasks": [
    {
      "id": "task_1",
      "description": "[独立任务1描述]",
      "task_type": "independent",
      "dependencies": [],
      "expected_output": "任务1结果",
      "reasoning_steps": ["执行任务1", "返回结果"]
    },
    {
      "id": "task_2",
      "description": "[独立任务2描述]",
      "task_type": "independent", 
      "dependencies": [],
      "expected_output": "任务2结果",
      "reasoning_steps": ["执行任务2", "返回结果"]
    },
    {
      "id": "task_3",
      "description": "汇总所有独立任务的结果",
      "task_type": "aggregate",
      "dependencies": ["task_1", "task_2"],
      "expected_output": "汇总结果",
      "reasoning_steps": ["接收所有结果", "执行汇总", "返回最终结果"]
    }
  ]
}
```

**示例**：
问题：What are the top 3 countries by population and GDP?
拆分：
1. 找出人口最多的3个国家
2. 找出GDP最高的3个国家
3. 汇总两个列表的结果
"""
    
    def classify_question(self, question: str, table_info: Dict) -> str:
        """分类问题类型，选择最适合的拆分策略"""
        
        # 关键词匹配规则
        question_lower = question.lower()
        
        # 聚合类关键词
        aggregation_keywords = [
            'average', 'mean', 'total', 'sum', 'count', 'maximum', 'minimum',
            'max', 'min', 'aggregate', 'total number', 'how many'
        ]
        
        # 比较类关键词  
        comparison_keywords = [
            'compare', 'which', 'better', 'higher', 'lower', 'greater', 'less',
            'more than', 'less than', 'versus', 'vs', 'between'
        ]
        
        # 桥接类关键词
        bridge_keywords = [
            'with', 'where', 'that', 'which have', 'whose', 'for', 'of',
            'containing', 'including', 'filtered by'
        ]
        
        # 顺序类关键词
        sequential_keywords = [
            'trend', 'over time', 'from', 'to', 'between', 'during',
            'sequence', 'order', 'chronological'
        ]
        
        # 独立类关键词
        independent_keywords = [
            'list', 'top', 'bottom', 'all', 'each', 'separate',
            'individually', 'respectively'
        ]
        
        # 计算匹配分数
        scores = {
            'aggregation': sum(1 for kw in aggregation_keywords if kw in question_lower),
            'comparison': sum(1 for kw in comparison_keywords if kw in question_lower),
            'bridge': sum(1 for kw in bridge_keywords if kw in question_lower),
            'sequential': sum(1 for kw in sequential_keywords if kw in question_lower),
            'independent': sum(1 for kw in independent_keywords if kw in question_lower)
        }
        
        # 返回得分最高的策略
        best_strategy = max(scores, key=scores.get)
        return best_strategy if scores[best_strategy] > 0 else 'aggregation'
    
    def split_question(self, question: str, table_info: Dict, context: Dict = None) -> Dict[str, Any]:
        """拆分问题为子任务"""
        
        # 1. 分类问题类型
        strategy_name = self.classify_question(question, table_info)
        strategy_prompt = self.strategies[strategy_name]
        
        # 2. 构建上下文信息
        context_str = ""
        if context:
            context_str = "\n当前上下文:\n"
            for key, value in context.items():
                context_str += f"- {key}: {value}\n"
        
        # 3. 构建完整提示词
        full_prompt = f"""
{table_info}

问题: {question}{context_str}

{strategy_prompt}

请根据上述规则拆分这个问题，严格按照以下格式输出JSON结果（不要添加任何其他文字）：

```json
{{
  "strategy": "策略名称",
  "subtasks": [
    {{
      "id": "task_1",
      "description": "任务描述",
      "task_type": "任务类型",
      "dependencies": [],
      "expected_output": "期望输出",
      "reasoning_steps": ["步骤1", "步骤2"]
    }}
  ]
}}
```
"""
        
        # 4. 调用模型进行拆分
        try:
            inputs = self.tokenizer(full_prompt, return_tensors="pt")
            # 确保输入在正确的设备上
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 5. 解析JSON响应
            # 尝试多种方式解析JSON
            json_result = None
            
            # 方法1: 直接查找JSON块
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    json_result = json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # 方法2: 查找```json块
            if not json_result:
                json_block_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
                if json_block_match:
                    try:
                        json_result = json.loads(json_block_match.group(1))
                    except json.JSONDecodeError:
                        pass
            
            # 方法3: 查找```块
            if not json_result:
                block_match = re.search(r'```\s*(\{.*?\})\s*```', response, re.DOTALL)
                if block_match:
                    try:
                        json_result = json.loads(block_match.group(1))
                    except json.JSONDecodeError:
                        pass
            
            if json_result:
                return json_result
            else:
                print(f"JSON解析失败，响应内容: {response[:200]}...")
                return self._get_default_split(question, strategy_name)
                
        except Exception as e:
            print(f"拆分失败: {e}")
            return self._get_default_split(question, strategy_name)
    
    def _get_default_split(self, question: str, strategy: str) -> Dict[str, Any]:
        """获取默认拆分结果"""
        
        # 根据策略类型生成更合理的默认拆分
        if strategy == "aggregation":
            return {
                "strategy": "aggregation",
                "subtasks": [
                    {
                        "id": "task_1",
                        "description": f"提取相关数据列",
                        "task_type": "independent",
                        "dependencies": [],
                        "expected_output": "数据列",
                        "reasoning_steps": ["分析问题", "识别相关列", "提取数据"]
                    },
                    {
                        "id": "task_2",
                        "description": f"执行聚合计算",
                        "task_type": "aggregate",
                        "dependencies": ["task_1"],
                        "expected_output": "聚合结果",
                        "reasoning_steps": ["接收数据", "执行计算", "返回结果"]
                    }
                ]
            }
        elif strategy == "comparison":
            return {
                "strategy": "comparison",
                "subtasks": [
                    {
                        "id": "task_1",
                        "description": f"提取第一个实体的数据",
                        "task_type": "independent",
                        "dependencies": [],
                        "expected_output": "第一个实体的值",
                        "reasoning_steps": ["识别实体1", "提取数据", "返回结果"]
                    },
                    {
                        "id": "task_2",
                        "description": f"提取第二个实体的数据",
                        "task_type": "independent",
                        "dependencies": [],
                        "expected_output": "第二个实体的值",
                        "reasoning_steps": ["识别实体2", "提取数据", "返回结果"]
                    },
                    {
                        "id": "task_3",
                        "description": f"比较两个实体的值",
                        "task_type": "compare",
                        "dependencies": ["task_1", "task_2"],
                        "expected_output": "比较结果",
                        "reasoning_steps": ["接收两个值", "执行比较", "返回结果"]
                    }
                ]
            }
        elif strategy == "sequential":
            return {
                "strategy": "sequential",
                "subtasks": [
                    {
                        "id": "task_1",
                        "description": f"提取时间序列数据",
                        "task_type": "independent",
                        "dependencies": [],
                        "expected_output": "时间序列数据",
                        "reasoning_steps": ["识别时间列", "提取数据", "排序"]
                    },
                    {
                        "id": "task_2",
                        "description": f"分析趋势变化",
                        "task_type": "sequential",
                        "dependencies": ["task_1"],
                        "expected_output": "趋势分析",
                        "reasoning_steps": ["接收数据", "分析趋势", "返回结果"]
                    }
                ]
            }
        elif strategy == "bridge":
            return {
                "strategy": "bridge",
                "subtasks": [
                    {
                        "id": "task_1",
                        "description": f"应用过滤条件",
                        "task_type": "independent",
                        "dependencies": [],
                        "expected_output": "过滤后的数据",
                        "reasoning_steps": ["识别条件", "应用过滤", "返回结果"]
                    },
                    {
                        "id": "task_2",
                        "description": f"对过滤结果进行计算",
                        "task_type": "bridge",
                        "dependencies": ["task_1"],
                        "expected_output": "计算结果",
                        "reasoning_steps": ["接收过滤数据", "执行计算", "返回结果"]
                    }
                ]
            }
        else:  # independent
            return {
                "strategy": "independent",
                "subtasks": [
                    {
                        "id": "task_1",
                        "description": f"提取所有相关数据",
                        "task_type": "independent",
                        "dependencies": [],
                        "expected_output": "数据列表",
                        "reasoning_steps": ["分析需求", "提取数据", "返回结果"]
                    }
                ]
            }
    
    def validate_independence(self, subtasks: List[Dict]) -> Tuple[bool, List[str]]:
        """验证子任务独立性"""
        issues = []
        
        # 检查Compare类型任务的独立性
        compare_tasks = [t for t in subtasks if t.get('task_type') == 'compare']
        for task in compare_tasks:
            deps = task.get('dependencies', [])
            if len(deps) < 2:
                issues.append(f"比较任务 {task['id']} 应该依赖至少2个独立任务")
        
        # 检查Bridge类型任务的链式依赖
        bridge_tasks = [t for t in subtasks if t.get('task_type') == 'bridge']
        for i, task in enumerate(bridge_tasks):
            deps = task.get('dependencies', [])
            if i > 0 and len(deps) == 0:
                issues.append(f"桥接任务 {task['id']} 应该依赖前一个任务")
        
        return len(issues) == 0, issues


class ReasoningAgent:
    """推理代理 - 负责执行子任务"""
    
    def __init__(self, model, tokenizer, mcp_manager=None):
        self.model = model
        self.tokenizer = tokenizer
        self.mcp_manager = mcp_manager
    
    def execute_subtask(self, subtask: Dict, table_data: Dict, context: Dict = None) -> Dict[str, Any]:
        """执行单个子任务"""
        
        task_type = subtask.get('task_type', 'independent')
        description = subtask.get('description', '')
        reasoning_steps = subtask.get('reasoning_steps', [])
        
        # 1. 使用MCP工具进行检索（如果需要）
        retrieval_results = []
        if self.mcp_manager and task_type in ['independent', 'bridge']:
            retrieval_results = self._retrieve_relevant_data(description, table_data)
        
        # 2. 构建执行提示词
        prompt = self._build_execution_prompt(subtask, table_data, context, retrieval_results)
        
        try:
            # 3. 调用模型执行任务
            inputs = self.tokenizer(prompt, return_tensors="pt")
            # 确保输入在正确的设备上
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                "task_id": subtask['id'],
                "status": "success",
                "result": response,
                "reasoning": reasoning_steps,
                "retrieval_results": retrieval_results
            }
            
        except Exception as e:
            return {
                "task_id": subtask['id'],
                "status": "error", 
                "error": str(e),
                "reasoning": reasoning_steps,
                "retrieval_results": retrieval_results
            }
    
    def _retrieve_relevant_data(self, description: str, table_data: Dict) -> List[Dict]:
        """使用MCP工具检索相关数据"""
        if not self.mcp_manager:
            return []
        
        try:
            # 使用关键词检索
            keyword_results = self.mcp_manager.search("keyword", description, table_data, top_k=3)
            
            # 使用BM25+BGE检索
            bm25_bge_results = self.mcp_manager.search("bm25_bge", description, table_data, top_k=3)
            
            # 合并结果
            all_results = keyword_results + bm25_bge_results
            
            # 转换为字典格式
            retrieval_data = []
            for result in all_results:
                retrieval_data.append({
                    "content": result.content,
                    "score": result.score,
                    "source": result.source,
                    "metadata": result.metadata
                })
            
            return retrieval_data
            
        except Exception as e:
            print(f"⚠️ 检索失败: {e}")
            return []
    
    def _build_execution_prompt(self, subtask: Dict, table_data: Dict, context: Dict, retrieval_results: List[Dict] = None) -> str:
        """构建执行提示词"""
        
        table_str = self._format_table(table_data)
        
        # 添加上下文信息
        context_str = ""
        if context:
            context_str = "\n相关上下文:\n"
            for key, value in context.items():
                context_str += f"- {key}: {value}\n"
        
        # 添加检索结果
        retrieval_str = ""
        if retrieval_results:
            retrieval_str = "\n检索到的相关数据:\n"
            for i, result in enumerate(retrieval_results[:3], 1):
                retrieval_str += f"{i}. {result['content']} (分数: {result['score']:.3f})\n"
        
        prompt = f"""
表格数据:
{table_str}{context_str}{retrieval_str}

任务描述: {subtask['description']}
任务类型: {subtask.get('task_type', 'independent')}
预期输出: {subtask.get('expected_output', '')}

推理步骤:
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(subtask.get('reasoning_steps', [])))}

请执行这个任务，并返回结果：
"""
        
        return prompt
    
    def _format_table(self, table_data: Dict) -> str:
        """格式化表格数据"""
        columns = table_data.get('columns', [])
        rows = table_data.get('data', [])
        
        if not columns or not rows:
            return "无表格数据"
        
        # 简单的表格格式化
        table_str = " | ".join(columns) + "\n"
        table_str += " | ".join(["-" * len(col) for col in columns]) + "\n"
        
        for row in rows[:10]:  # 只显示前10行
            table_str += " | ".join(str(cell) for cell in row) + "\n"
        
        if len(rows) > 10:
            table_str += f"... (还有 {len(rows) - 10} 行)\n"
        
        return table_str


class TableQAPipeline:
    """TableQA主管道"""
    
    def __init__(self, model, tokenizer, mcp_manager=None):
        self.plan_agent = PlanAgent(model, tokenizer)
        self.reasoning_agent = ReasoningAgent(model, tokenizer, mcp_manager)
    
    def process_question(self, question: str, table_data: Dict) -> Dict[str, Any]:
        """处理完整的问题回答流程 - 迭代式Plan和Reasoning"""
        
        print(f"🔍 开始处理问题: {question[:50]}...")
        
        # 初始化
        context = {}
        iteration = 0
        max_iterations = 5
        all_subtasks = []
        all_results = []
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- 第 {iteration} 轮迭代 ---")
            
            # 1. Plan Agent: 问题拆分
            print("🔍 Plan Agent: 分析问题并拆分...")
            split_result = self.plan_agent.split_question(question, table_data, context)
            
            # 验证独立性
            is_valid, issues = self.plan_agent.validate_independence(split_result['subtasks'])
            if not is_valid:
                print(f"⚠️ 子任务独立性检查发现问题: {issues}")
            
            # 2. Reasoning Agent: 执行子任务
            print("🧠 Reasoning Agent: 执行子任务...")
            subtask_results = []
            
            for subtask in split_result['subtasks']:
                print(f"  执行任务: {subtask['description']}")
                
                # 检查依赖是否满足
                if self._check_dependencies(subtask, context):
                    result = self.reasoning_agent.execute_subtask(subtask, table_data, context)
                    subtask_results.append(result)
                    
                    # 更新上下文
                    if result['status'] == 'success':
                        context[subtask['id']] = result['result']
                        print(f"    ✓ 成功: {result['result'][:100]}...")
                    else:
                        print(f"    ❌ 失败: {result.get('error', '未知错误')}")
                else:
                    print(f"    ⏳ 等待依赖满足...")
                    subtask_results.append({
                        "task_id": subtask['id'],
                        "status": "pending",
                        "reasoning": ["等待依赖满足"]
                    })
            
            # 记录本轮结果
            all_subtasks.extend(split_result['subtasks'])
            all_results.extend(subtask_results)
            
            # 3. 检查是否完成
            completed_tasks = sum(1 for r in subtask_results if r['status'] == 'success')
            total_tasks = len(subtask_results)
            
            print(f"📊 本轮完成: {completed_tasks}/{total_tasks} 个任务")
            
            # 如果所有任务都完成，或者没有新任务，则结束
            if completed_tasks == total_tasks and total_tasks > 0:
                print("✅ 所有子任务执行完成")
                break
            elif completed_tasks == 0:
                print("❌ 没有任务可以执行，结束迭代")
                break
        
        # 4. 最终汇总
        print("\n📊 汇总最终结果...")
        final_answer = self._aggregate_final_results(all_results, context)
        
        return {
            "question": question,
            "strategy": split_result['strategy'],
            "iterations": iteration,
            "subtasks": all_subtasks,
            "subtask_results": all_results,
            "final_answer": final_answer,
            "context": context,
            "independence_valid": is_valid,
            "independence_issues": issues
        }
    
    def _format_table(self, table_data: Dict) -> str:
        """格式化表格数据"""
        columns = table_data.get('columns', [])
        rows = table_data.get('data', [])
        
        if not columns or not rows:
            return "表格数据为空"
        
        # 构建表格字符串
        table_str = " | ".join(columns) + "\n"
        table_str += " | ".join(["-" * len(str(col)) for col in columns]) + "\n"
        
        for row in rows[:10]:  # 只显示前10行
            row_str = " | ".join([str(cell) for cell in row])
            table_str += row_str + "\n"
        
        if len(rows) > 10:
            table_str += f"... (还有 {len(rows) - 10} 行)\n"
        
        return table_str
    
    def process_question_simple(self, question: str, table_data: Dict) -> Dict[str, Any]:
        """简化的端到端问题处理"""
        print(f"🔍 开始处理问题: {question[:50]}...")
        
        # 格式化表格数据
        table_str = self._format_table(table_data)
        
        # 构建输入
        input_text = f"<|im_start|>user\n表格数据:\n{table_str}\n\n问题: {question}\n<|im_end|>\n<|im_start|>assistant\n"
        
        try:
            # 调用模型生成答案
            inputs = self.plan_agent.tokenizer(input_text, return_tensors="pt")
            inputs = {k: v.to(self.plan_agent.model.device) for k, v in inputs.items()}
            
            outputs = self.plan_agent.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.plan_agent.tokenizer.eos_token_id
            )
            
            response = self.plan_agent.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取assistant的回复
            if "<|im_start|>assistant" in response:
                final_answer = response.split("<|im_start|>assistant")[-1].strip()
            else:
                final_answer = response[len(input_text):].strip()
            
            # 确保输出格式正确 - 如果没有<answer>标签，添加它们
            if not final_answer.startswith('<answer>'):
                # 提取实际答案内容（去掉可能的解释文本）
                import re
                # 尝试提取数字、百分比等答案
                numbers = re.findall(r'\d+\.?\d*%?', final_answer)
                if numbers:
                    # 如果有数字，使用最后一个数字作为答案
                    answer_content = numbers[-1]
                else:
                    # 否则使用整个回答
                    answer_content = final_answer.strip()
                
                final_answer = f"<answer>\n{answer_content}\n</answer>"
            
            print(f"✅ 生成答案: {final_answer[:100]}...")
            
            return {
                "question": question,
                "strategy": "end_to_end",
                "iterations": 1,
                "subtasks": [],
                "subtask_results": [],
                "final_answer": final_answer,
                "context": {},
                "independence_valid": True,
                "independence_issues": []
            }
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            return {
                "question": question,
                "strategy": "end_to_end",
                "iterations": 1,
                "subtasks": [],
                "subtask_results": [],
                "final_answer": f"抱歉，无法处理这个问题: {str(e)}",
                "context": {},
                "independence_valid": True,
                "independence_issues": []
            }
    
    def _check_dependencies(self, subtask: Dict, context: Dict) -> bool:
        """检查子任务依赖是否满足"""
        dependencies = subtask.get('dependencies', [])
        return all(dep in context for dep in dependencies)
    
    def _aggregate_final_results(self, all_results: List[Dict], context: Dict) -> str:
        """汇总最终结果"""
        
        # 获取成功的任务结果
        successful_results = [r for r in all_results if r['status'] == 'success']
        
        if not successful_results:
            return "无法完成问题解答，所有子任务都失败了。"
        
        if len(successful_results) == 1:
            return successful_results[0]['result']
        
        # 多任务汇总 - 让Reasoning Agent进行最终汇总
        summary_prompt = f"""
基于以下子任务的执行结果，请给出最终答案：

"""
        
        for i, result in enumerate(successful_results, 1):
            summary_prompt += f"子任务 {i}: {result['result']}\n"
        
        summary_prompt += "\n请基于以上结果给出最终答案："
        
        try:
            # 使用Reasoning Agent进行最终汇总
            inputs = self.reasoning_agent.tokenizer(summary_prompt, return_tensors="pt")
            # 确保输入在正确的设备上
            inputs = {k: v.to(self.reasoning_agent.model.device) for k, v in inputs.items()}
            outputs = self.reasoning_agent.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True
            )
            
            final_answer = self.reasoning_agent.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 移除输入部分，只保留生成的部分
            final_answer = final_answer[len(summary_prompt):].strip()
            
            return final_answer if final_answer else "基于子任务结果，无法确定最终答案。"
            
        except Exception as e:
            print(f"⚠️ 最终汇总失败: {e}")
            # 回退到简单汇总
            summary = "基于以下子任务的结果:\n\n"
            for i, result in enumerate(successful_results, 1):
                summary += f"{i}. {result['result']}\n"
            return summary
    
    def _aggregate_results(self, subtask_results: List[Dict], split_result: Dict) -> str:
        """汇总子任务结果（保持向后兼容）"""
        return self._aggregate_final_results(subtask_results, {})


