"""
TableQA训练模块
实现Plan Agent的LoRA微调训练
"""

import json
import yaml
import torch
import torch.nn.functional as F
import re
from typing import Dict, Any, List, Tuple
from datasets import Dataset
from transformers import TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
from pipeline import PlanAgent
from mcp_tools import MCPToolManager
from modelscope import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import numpy as np


class TableQADataset(torch.utils.data.Dataset):
    """TableQA训练数据集"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建训练样本
        question = item['question']
        table = item['table']
        answer = item['answer']
        
        # 构建输入文本
        table_str = self._format_table(table)
        input_text = f"<|im_start|>user\n表格数据:\n{table_str}\n\n问题: {question}\n<|im_end|>\n<|im_start|>assistant\n"
        
        # 构建目标文本，只输出<answer></answer>格式
        target_text = f"<answer>\n{answer}\n</answer><|im_end|>"
        
        # 构建完整的训练文本
        full_text = input_text + target_text
        
        # 编码
        inputs = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 创建标签，只对assistant部分计算损失
        labels = inputs["input_ids"].clone()
        input_length = len(self.tokenizer.encode(input_text, add_special_tokens=False))
        labels[:, :input_length] = -100  # 忽略user部分的损失
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }
    
    def _format_table(self, table: Dict) -> str:
        """格式化表格"""
        columns = table.get('columns', [])
        rows = table.get('data', [])
        
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
    
    def _create_target_split(self, question: str, table: Dict, answer: str) -> str:
        """创建目标拆分结果"""
        # 基于问题类型创建简化的拆分
        question_lower = question.lower()
        
        if 'average' in question_lower or 'mean' in question_lower:
            strategy = "aggregation"
            subtasks = [
                {
                    "id": "task_1",
                    "description": "提取相关列的所有数值",
                    "task_type": "independent",
                    "dependencies": [],
                    "expected_output": "数值列表"
                },
                {
                    "id": "task_2",
                    "description": "计算平均值",
                    "task_type": "aggregate",
                    "dependencies": ["task_1"],
                    "expected_output": "平均值结果"
                }
            ]
        elif 'compare' in question_lower or 'which' in question_lower:
            strategy = "comparison"
            subtasks = [
                {
                    "id": "task_1",
                    "description": "计算第一个实体的指标",
                    "task_type": "independent",
                    "dependencies": [],
                    "expected_output": "第一个实体的指标值"
                },
                {
                    "id": "task_2",
                    "description": "计算第二个实体的指标",
                    "task_type": "independent",
                    "dependencies": [],
                    "expected_output": "第二个实体的指标值"
                },
                {
                    "id": "task_3",
                    "description": "比较两个指标",
                    "task_type": "compare",
                    "dependencies": ["task_1", "task_2"],
                    "expected_output": "比较结果"
                }
            ]
        else:
            strategy = "bridge"
            subtasks = [
                {
                    "id": "task_1",
                    "description": "分析问题需求",
                    "task_type": "bridge",
                    "dependencies": [],
                    "expected_output": "分析结果"
                },
                {
                    "id": "task_2",
                    "description": "执行计算",
                    "task_type": "bridge",
                    "dependencies": ["task_1"],
                    "expected_output": "计算结果"
                }
            ]
        
        target = {
            "strategy": strategy,
            "subtasks": subtasks
        }
        
        return json.dumps(target, ensure_ascii=False, indent=2)
    
    def _create_reasoning_process(self, question: str, table: Dict, answer: str) -> str:
        """创建完整的推理过程"""
        question_lower = question.lower()
        
        # 根据问题类型生成不同的推理过程
        if 'average' in question_lower or 'mean' in question_lower:
            return self._create_aggregation_reasoning(question, table, answer)
        elif 'compare' in question_lower or 'which' in question_lower:
            return self._create_comparison_reasoning(question, table, answer)
        elif 'total' in question_lower or 'sum' in question_lower:
            return self._create_sum_reasoning(question, table, answer)
        else:
            return self._create_general_reasoning(question, table, answer)
    
    def _create_aggregation_reasoning(self, question: str, table: Dict, answer: str) -> str:
        """创建聚合类问题的推理过程"""
        columns = table.get('columns', [])
        rows = table.get('data', [])
        
        # 找到数值列
        numeric_columns = []
        for i, col in enumerate(columns):
            if any(char.isdigit() for char in str(col)):
                numeric_columns.append((i, col))
        
        reasoning = f"这是一个计算平均值的问题。\n\n"
        reasoning += f"步骤1: 识别相关列\n"
        reasoning += f"表格列: {', '.join(columns)}\n"
        
        if numeric_columns:
            reasoning += f"数值列: {', '.join([col[1] for col in numeric_columns])}\n\n"
            reasoning += f"步骤2: 提取数值数据\n"
            values = []
            for row in rows:
                for col_idx, col_name in numeric_columns:
                    if col_idx < len(row):
                        try:
                            val = float(str(row[col_idx]).replace(',', ''))
                            values.append(val)
                        except:
                            pass
            
            if values:
                reasoning += f"提取的数值: {values}\n\n"
                reasoning += f"步骤3: 计算平均值\n"
                reasoning += f"总和: {sum(values)}\n"
                reasoning += f"数量: {len(values)}\n"
                reasoning += f"平均值: {sum(values)/len(values):.2f}\n"
            else:
                reasoning += f"未能提取到有效数值\n"
        else:
            reasoning += f"未找到数值列\n"
        
        return reasoning
    
    def _create_comparison_reasoning(self, question: str, table: Dict, answer: str) -> str:
        """创建比较类问题的推理过程"""
        reasoning = f"这是一个比较类问题。\n\n"
        reasoning += f"步骤1: 分析问题要求\n"
        reasoning += f"问题: {question}\n\n"
        reasoning += f"步骤2: 识别比较对象\n"
        reasoning += f"需要比较表格中的不同项目\n\n"
        reasoning += f"步骤3: 提取比较数据\n"
        reasoning += f"从表格中提取相关数据进行对比\n\n"
        reasoning += f"步骤4: 得出结论\n"
        reasoning += f"基于数据对比得出最终结论\n"
        
        return reasoning
    
    def _create_sum_reasoning(self, question: str, table: Dict, answer: str) -> str:
        """创建求和类问题的推理过程"""
        reasoning = f"这是一个求和问题。\n\n"
        reasoning += f"步骤1: 识别需要求和的列\n"
        reasoning += f"问题: {question}\n\n"
        reasoning += f"步骤2: 提取数值数据\n"
        reasoning += f"从表格中提取所有相关数值\n\n"
        reasoning += f"步骤3: 计算总和\n"
        reasoning += f"将所有数值相加得到总和\n"
        
        return reasoning
    
    def _create_general_reasoning(self, question: str, table: Dict, answer: str) -> str:
        """创建一般问题的推理过程"""
        reasoning = f"这是一个表格问答问题。\n\n"
        reasoning += f"步骤1: 理解问题\n"
        reasoning += f"问题: {question}\n\n"
        reasoning += f"步骤2: 分析表格数据\n"
        reasoning += f"查看表格结构和内容\n\n"
        reasoning += f"步骤3: 提取相关信息\n"
        reasoning += f"根据问题要求提取相关数据\n\n"
        reasoning += f"步骤4: 计算或分析\n"
        reasoning += f"对提取的数据进行计算或分析\n"
        
        return reasoning


class AnswerComparator:
    """答案对比器 - 用于计算EM和Format分数"""
    
    def __init__(self):
        self.answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    
    def extract_answer(self, text: str) -> str:
        """从文本中提取答案部分"""
        # 尝试提取<answer>标签中的内容
        answer_match = self.answer_pattern.search(text)
        if answer_match:
            return answer_match.group(1).strip()
        
        # 如果没有找到标签，返回整个文本
        return text.strip()
    
    def calculate_format_score(self, generated_text: str) -> float:
        """计算格式分数 (0.0-1.0) - 只检查<answer></answer>格式"""
        # 检查是否包含完整的<answer></answer>格式
        if self.answer_pattern.search(generated_text):
            return 1.0
        else:
            return 0.0
    
    def calculate_em_score(self, generated_answer: str, reference_answer: str) -> float:
        """计算精确匹配分数 (0.0-1.0)"""
        # 标准化答案
        gen_norm = self._normalize_answer(generated_answer)
        ref_norm = self._normalize_answer(reference_answer)
        
        return 1.0 if gen_norm == ref_norm else 0.0
    
    def _normalize_answer(self, answer: str) -> str:
        """标准化答案格式"""
        # 转换为小写
        answer = answer.lower().strip()
        
        # 移除标点符号
        answer = re.sub(r'[^\w\s]', '', answer)
        
        # 移除多余空格
        answer = ' '.join(answer.split())
        
        return answer
    
    def calculate_reward(self, generated_text: str, reference_answer: str) -> float:
        """计算总奖励分数: 0.1 * format + 0.9 * EM"""
        format_score = self.calculate_format_score(generated_text)
        generated_answer = self.extract_answer(generated_text)
        em_score = self.calculate_em_score(generated_answer, reference_answer)
        
        reward = 0.1 * format_score + 0.9 * em_score
        return reward
    
    def calculate_accuracy(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """计算准确率统计"""
        if len(predictions) != len(references):
            raise ValueError("预测结果和参考答案数量不匹配")
        
        total_samples = len(predictions)
        exact_matches = 0
        format_correct = 0
        
        for pred, ref in zip(predictions, references):
            # 提取答案
            pred_answer = self.extract_answer(pred)
            
            # 计算EM
            if self.calculate_em_score(pred_answer, ref) == 1.0:
                exact_matches += 1
            
            # 计算格式正确性
            if self.calculate_format_score(pred) == 1.0:
                format_correct += 1
        
        return {
            "total_samples": total_samples,
            "exact_match_count": exact_matches,
            "format_correct_count": format_correct,
            "exact_match_accuracy": exact_matches / total_samples if total_samples > 0 else 0.0,
            "format_accuracy": format_correct / total_samples if total_samples > 0 else 0.0,
            "overall_accuracy": exact_matches / total_samples if total_samples > 0 else 0.0
        }


class GPROTrainer:
    """GPRO (Generalized Preference Optimization) 训练器"""
    
    def __init__(self, model, tokenizer, config: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.comparator = AnswerComparator()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get("gpro", {}).get("learning_rate", 1e-5),
            weight_decay=config.get("gpro", {}).get("weight_decay", 0.01)
        )
    
    def generate_response(self, question: str, table_data: Dict) -> str:
        """生成模型响应"""
        table_str = self._format_table(table_data)
        input_text = f"<|im_start|>user\n表格数据:\n{table_str}\n\n问题: {question}\n<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(input_text):].strip()
    
    def compute_gpro_loss(self, batch_data: List[Dict]) -> torch.Tensor:
        """计算GPRO损失"""
        total_loss = 0.0
        batch_size = len(batch_data)
        
        for item in batch_data:
            question = item['question']
            table = item['table']
            reference_answer = item['answer']
            
            # 生成响应
            generated_text = self.generate_response(question, table)
            
            # 计算奖励
            reward = self.comparator.calculate_reward(generated_text, reference_answer)
            
            # 构建训练样本
            table_str = self._format_table(table)
            input_text = f"<|im_start|>user\n表格数据:\n{table_str}\n\n问题: {question}\n<|im_end|>\n<|im_start|>assistant\n"
            
            # 构建目标文本，只使用<answer></answer>格式
            target_text = f"<answer>\n{reference_answer}\n</answer><|im_end|>"
            
            full_text = input_text + target_text
            
            # 编码
            inputs = self.tokenizer(
                full_text,
                max_length=self.config.get("model", {}).get("max_length", 1024),
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # 创建标签
            labels = inputs["input_ids"].clone()
            input_length = len(self.tokenizer.encode(input_text, add_special_tokens=False))
            labels[:, :input_length] = -100
            
            # 前向传播
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            
            # 应用奖励权重
            weighted_loss = loss * reward
            total_loss += weighted_loss
        
        return total_loss / batch_size
    
    def train_step(self, batch_data: List[Dict]) -> Dict[str, float]:
        """执行一步GPRO训练"""
        self.model.train()
        self.optimizer.zero_grad()
        
        loss = self.compute_gpro_loss(batch_data)
        loss.backward()
        self.optimizer.step()
        
        return {"loss": loss.item()}
    
    def _format_table(self, table: Dict) -> str:
        """格式化表格"""
        columns = table.get('columns', [])
        rows = table.get('data', [])
        
        if not columns or not rows:
            return "无表格数据"
        
        table_str = " | ".join(columns) + "\n"
        table_str += " | ".join(["-" * len(col) for col in columns]) + "\n"
        
        for row in rows[:10]:
            table_str += " | ".join(str(cell) for cell in row) + "\n"
        
        if len(rows) > 10:
            table_str += f"... (还有 {len(rows) - 10} 行)\n"
        
        return table_str
    
    def _create_reasoning_process(self, question: str, table: Dict, answer: str) -> str:
        """创建推理过程"""
        question_lower = question.lower()
        
        if 'average' in question_lower or 'mean' in question_lower:
            return self._create_aggregation_reasoning(question, table, answer)
        elif 'compare' in question_lower or 'which' in question_lower:
            return self._create_comparison_reasoning(question, table, answer)
        elif 'total' in question_lower or 'sum' in question_lower:
            return self._create_sum_reasoning(question, table, answer)
        else:
            return self._create_general_reasoning(question, table, answer)
    
    def _create_aggregation_reasoning(self, question: str, table: Dict, answer: str) -> str:
        """创建聚合类问题的推理过程"""
        columns = table.get('columns', [])
        rows = table.get('data', [])
        
        reasoning = f"这是一个计算平均值的问题。\n\n"
        reasoning += f"步骤1: 识别相关列\n"
        reasoning += f"表格列: {', '.join(columns)}\n"
        
        # 找到数值列
        numeric_columns = []
        for i, col in enumerate(columns):
            if any(char.isdigit() for char in str(col)):
                numeric_columns.append((i, col))
        
        if numeric_columns:
            reasoning += f"数值列: {', '.join([col[1] for col in numeric_columns])}\n\n"
            reasoning += f"步骤2: 提取数值数据\n"
            values = []
            for row in rows:
                for col_idx, col_name in numeric_columns:
                    if col_idx < len(row):
                        try:
                            val = float(str(row[col_idx]).replace(',', ''))
                            values.append(val)
                        except:
                            pass
            
            if values:
                reasoning += f"提取的数值: {values}\n\n"
                reasoning += f"步骤3: 计算平均值\n"
                reasoning += f"总和: {sum(values)}\n"
                reasoning += f"数量: {len(values)}\n"
                reasoning += f"平均值: {sum(values)/len(values):.2f}\n"
            else:
                reasoning += f"未能提取到有效数值\n"
        else:
            reasoning += f"未找到数值列\n"
        
        return reasoning
    
    def _create_comparison_reasoning(self, question: str, table: Dict, answer: str) -> str:
        """创建比较类问题的推理过程"""
        reasoning = f"这是一个比较类问题。\n\n"
        reasoning += f"步骤1: 分析问题要求\n"
        reasoning += f"问题: {question}\n\n"
        reasoning += f"步骤2: 识别比较对象\n"
        reasoning += f"需要比较表格中的不同项目\n\n"
        reasoning += f"步骤3: 提取比较数据\n"
        reasoning += f"从表格中提取相关数据进行对比\n\n"
        reasoning += f"步骤4: 得出结论\n"
        reasoning += f"基于数据对比得出最终结论\n"
        
        return reasoning
    
    def _create_sum_reasoning(self, question: str, table: Dict, answer: str) -> str:
        """创建求和类问题的推理过程"""
        reasoning = f"这是一个求和问题。\n\n"
        reasoning += f"步骤1: 识别需要求和的列\n"
        reasoning += f"问题: {question}\n\n"
        reasoning += f"步骤2: 提取数值数据\n"
        reasoning += f"从表格中提取所有相关数值\n\n"
        reasoning += f"步骤3: 计算总和\n"
        reasoning += f"将所有数值相加得到总和\n"
        
        return reasoning
    
    def _create_general_reasoning(self, question: str, table: Dict, answer: str) -> str:
        """创建一般问题的推理过程"""
        reasoning = f"这是一个表格问答问题。\n\n"
        reasoning += f"步骤1: 理解问题\n"
        reasoning += f"问题: {question}\n\n"
        reasoning += f"步骤2: 分析表格数据\n"
        reasoning += f"查看表格结构和内容\n\n"
        reasoning += f"步骤3: 提取相关信息\n"
        reasoning += f"根据问题要求提取相关数据\n\n"
        reasoning += f"步骤4: 计算或分析\n"
        reasoning += f"对提取的数据进行计算或分析\n"
        
        return reasoning


def setup_lora(model, config: Dict[str, Any]):
    """设置LoRA配置"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.get("lora", {}).get("r", 16),
        lora_alpha=config.get("lora", {}).get("lora_alpha", 32),
        target_modules=config.get("lora", {}).get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
        lora_dropout=config.get("lora", {}).get("lora_dropout", 0.1),
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def train_sft_cold_start(train_data: List[Dict], eval_data: List[Dict], config: Dict[str, Any]):
    """SFT冷启动训练 - 只训练一个epoch"""
    print("开始SFT冷启动训练...")
    
    # 加载模型和tokenizer
    model_path = config.get("model", {}).get("name", "./models/pretrained/Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # 设置LoRA
    model = setup_lora(model, config)
    
    # 创建数据集
    train_dataset = TableQADataset(train_data, tokenizer, config.get("model", {}).get("max_length", 1024))
    eval_dataset = TableQADataset(eval_data, tokenizer, config.get("model", {}).get("max_length", 1024))
    
    # SFT训练参数 - 只训练一个epoch
    training_args = TrainingArguments(
        output_dir=config.get("training", {}).get("output_dir", "./models/finetuned"),
        num_train_epochs=1,  # 只训练一个epoch
        per_device_train_batch_size=config.get("training", {}).get("batch_size", 2),
        per_device_eval_batch_size=2,
        warmup_steps=10,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=10,
        save_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        learning_rate=config.get("training", {}).get("learning_rate", 2e-5),
        weight_decay=0.01,
        gradient_accumulation_steps=2,
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )
    
    # 开始SFT训练
    trainer.train()
    
    # 保存SFT模型
    sft_output_dir = config.get("training", {}).get("output_dir", "./models/finetuned") + "_sft"
    trainer.save_model(sft_output_dir)
    tokenizer.save_pretrained(sft_output_dir)
    
    print("SFT冷启动训练完成")
    return model, tokenizer


def train_gpro(model, tokenizer, train_data: List[Dict], eval_data: List[Dict], config: Dict[str, Any]):
    """GPRO训练阶段"""
    print("开始GPRO训练...")
    
    # 创建GPRO训练器
    gpro_trainer = GPROTrainer(model, tokenizer, config)
    
    # 创建数据加载器
    batch_size = config.get("gpro", {}).get("batch_size", 4)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    # GPRO训练参数
    num_epochs = config.get("gpro", {}).get("epochs", 3)
    logging_steps = config.get("gpro", {}).get("logging_steps", 10)
    
    # 训练循环
    model.train()
    total_steps = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # 执行GPRO训练步骤
            metrics = gpro_trainer.train_step(batch)
            epoch_loss += metrics["loss"]
            num_batches += 1
            total_steps += 1
            
            # 日志记录
            if total_steps % logging_steps == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Step {total_steps}, Loss: {metrics['loss']:.4f}")
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch+1}/{num_epochs} 完成, 平均损失: {avg_loss:.4f}")
    
    # 保存GPRO模型
    gpro_output_dir = config.get("training", {}).get("output_dir", "./models/finetuned") + "_gpro"
    model.save_pretrained(gpro_output_dir)
    tokenizer.save_pretrained(gpro_output_dir)
    
    print("GPRO训练完成")
    return model, tokenizer


def train_end_to_end_pipeline(train_data: List[Dict], eval_data: List[Dict], config: Dict[str, Any]):
    """端到端训练整个pipeline - 包含SFT冷启动和GPRO训练"""
    print("开始端到端训练TableQA Pipeline...")
    
    # 1. SFT冷启动训练
    print("\n=== 阶段1: SFT冷启动训练 ===")
    model, tokenizer = train_sft_cold_start(train_data, eval_data, config)
    
    # 2. GPRO训练
    print("\n=== 阶段2: GPRO训练 ===")
    model, tokenizer = train_gpro(model, tokenizer, train_data, eval_data, config)
    
    print("端到端训练完成")
    return model, tokenizer


def train_model(train_data: List[Dict], eval_data: List[Dict], config: Dict[str, Any]):
    """主训练函数"""
    print(f"开始端到端训练，训练数据量: {len(train_data)}, 验证数据量: {len(eval_data)}")
    
    # 端到端训练整个pipeline
    model, tokenizer = train_end_to_end_pipeline(train_data, eval_data, config)
    
    # 创建MCP工具管理器
    mcp_manager = MCPToolManager()
    
    # 创建完整的pipeline
    from pipeline import TableQAPipeline
    pipeline = TableQAPipeline(model, tokenizer, mcp_manager)
    
    print("端到端训练完成")
    return pipeline


def main():
    """主训练函数"""
    # 加载配置
    config = load_config()
    
    # 加载数据集
    import sys
    import os
    sys.path.append('/home/zyc/TableQA')
    sys.path.append('/home/zyc/TableQA/datasets/tablebench')
    
    try:
        from load_tablebench import TableBenchLoader
        loader = TableBenchLoader()
        
        # 加载训练数据
        train_data = loader.load(version='base', max_samples=config.get("data", {}).get("train_samples", 100))
        eval_data = loader.load(version='base', max_samples=config.get("data", {}).get("eval_samples", 20))
        
        # 端到端训练模型
        pipeline = train_model(train_data, eval_data, config)
        
        return pipeline
    except ImportError as e:
        print(f"警告: 无法导入TableBenchLoader: {e}")
        print("使用模拟数据进行训练...")
        
        # 创建模拟数据
        train_data = [
            {
                "question": "What is the average number of tropical cyclones per season?",
                "table": {
                    "columns": ["season", "tropical cyclones"],
                    "data": [
                        ["1990-91", "10"],
                        ["1991-92", "10"],
                        ["1992-93", "3"]
                    ]
                },
                "answer": "7.67"
            }
        ]
        eval_data = train_data[:1]
        
        # 端到端训练模型
        pipeline = train_model(train_data, eval_data, config)
        
        return pipeline


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
