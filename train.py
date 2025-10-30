#!/usr/bin/env python3
"""
Training script for TableQA - Plan Agent Fine-tuning
"""
from utils import load_config
from models.loader import load_plan_agent
from dataset import TableBenchDataset
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer
from pathlib import Path


class PlanAgentDataset(torch.utils.data.Dataset):
    """Dataset for Plan Agent training (question splitting)"""
    
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format input: question + table
        input_text = f"Question: {item['question']}\nTable: {item['table']}\n\nAnswer:"
        target_text = f"<answer>{item['answer']}</answer>"
        
        # Tokenize
        full_text = input_text + target_text
        inputs = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create labels (only compute loss on answer part)
        labels = inputs["input_ids"].clone()
        input_length = len(self.tokenizer.encode(input_text, add_special_tokens=False))
        labels[:, :input_length] = -100  # Ignore input part
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }


def setup_lora(model, config):
    """Setup LoRA for Plan Agent"""
    lora_config = LoraConfig(
        r=config.get("lora", {}).get("r", 16),
        lora_alpha=config.get("lora", {}).get("lora_alpha", 32),
        target_modules=config.get("lora", {}).get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=config.get("lora", {}).get("lora_dropout", 0.1),
        bias=config.get("lora", {}).get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train_plan_agent(model, tokenizer, train_data, config):
    """
    Train Plan Agent with LoRA
    Uses SFT (Supervised Fine-Tuning)
    """
    print("\n" + "="*60)
    print("Training Plan Agent")
    print("="*60)
    
    # Create dataset
    train_dataset = PlanAgentDataset(train_data, tokenizer)
    
    # Training arguments
    output_dir = config.get("output_dir", "./models/finetuned/plan_agent")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get("epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 2),
        learning_rate=float(config.get("learning_rate", 2e-5)),
        weight_decay=config.get("training", {}).get("weight_decay", 0.01),
        logging_steps=config.get("training", {}).get("logging_steps", 5),
        save_steps=config.get("training", {}).get("save_steps", 20),
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    print(f"\nSaving model to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n✓ Training completed!")
    return model


def main():
    print("="*60)
    print("TableQA - Plan Agent Fine-tuning")
    print("="*60)
    
    # Load config
    config = load_config()
    print(f"\nModel: {config['model']}")
    print(f"Dataset: {config['dataset']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    
    # Load data
    print("\n[1/3] Loading dataset...")
    dataset = TableBenchDataset()
    train_data = dataset.load(max_samples=config.get("train_samples"))
    print(f"✓ Loaded {len(train_data)} training samples")
    
    # Load Plan Agent (base model, will add LoRA)
    print("\n[2/3] Loading Plan Agent...")
    model, tokenizer = load_plan_agent(config, use_lora=False)
    model = setup_lora(model, config)
    print("✓ Plan Agent loaded with LoRA")
    
    # Train
    print("\n[3/3] Training...")
    train_plan_agent(model, tokenizer, train_data, config)
    
    print("\n" + "="*60)
    print("✓ All Done!")
    print("="*60)
    print(f"\nPlan Agent LoRA saved to: {config.get('output_dir', './models/finetuned/plan_agent')}")
    print("\nNext steps:")
    print("  1. Evaluate: python3 eval.py")
    print("  2. Inference: python3 infer.py --question 'your question' --table data.json")


if __name__ == "__main__":
    main()
