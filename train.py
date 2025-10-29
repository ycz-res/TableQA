#!/usr/bin/env python3
"""
Training script for TableQA
"""
from utils import load_config
from models.trainer import train_sft_cold_start, train_gpro, setup_lora
from models.utils import load_model_and_tokenizer
from dataset import TableBenchLoader
from torch.utils.data import DataLoader
import torch


class TableQADataset(torch.utils.data.Dataset):
    """Dataset for TableQA training"""
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format input
        input_text = f"Question: {item['question']}\nTable: {item['table']}\n\nAnswer:"
        target_text = f"<answer>{item['answer']}</answer>"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        targets = self.tokenizer(
            target_text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze(),
            "reference_answer": item['answer']
        }


def main():
    print("="*60)
    print("TableQA Training - SFT + GPRO")
    print("="*60)
    
    # Load config
    config = load_config()
    print(f"\nModel: {config['model']}")
    print(f"Dataset: {config['dataset']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    
    # Load data
    print("\n[1/4] Loading dataset...")
    loader = TableBenchLoader()
    train_data = loader.load(split='train')
    print(f"✓ Loaded {len(train_data)} training samples")
    
    # Load model with LoRA
    print("\n[2/4] Loading model with LoRA...")
    model, tokenizer = load_model_and_tokenizer(config, use_lora=False)
    model = setup_lora(model, config)
    print("✓ Model loaded with LoRA")
    
    # Create dataset
    train_dataset = TableQADataset(train_data, tokenizer)
    
    # SFT Cold Start
    print("\n[3/4] SFT Cold Start (1 epoch)...")
    train_sft_cold_start(model, train_dataset, config)
    print("✓ SFT completed")
    
    # GPRO Training
    print("\n[4/4] GPRO Training...")
    train_gpro(model, tokenizer, train_dataset, config)
    print("✓ GPRO completed")
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    print(f"\nModel saved to: {config['output_dir']}")


if __name__ == "__main__":
    main()
