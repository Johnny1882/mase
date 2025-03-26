#!/usr/bin/env python3
import datasets
from datasets import load_dataset as original_load_dataset
import math
import torch
import time
import os
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from chop.tools import get_tokenized_dataset
from chop.passes.module.transforms.attention import fc_transform_pass
from pathlib import Path
import sys
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(Path(__file__).resolve().parents[5].as_posix())

def patched_load_dataset(dataset, *args, **kwargs):
    if dataset == "wikitext" and "config" not in kwargs:
        return original_load_dataset(dataset, "wikitext-2-raw-v1", *args, **kwargs)
    else:
        return original_load_dataset(dataset, *args, **kwargs)

datasets.load_dataset = patched_load_dataset

# --------------------------------------------------
# 1. Dataset preparation functions
# --------------------------------------------------
def prepare_dataset():
    logger.info("Loading and preparing dataset...")
    dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Define tokenization function with block size for autoregressive modeling
    block_size = 128  # Context window size for autoregressive modeling
    
    def tokenize_and_group(examples):
        # Tokenize
        tokenized = tokenizer(examples["text"], truncation=False, add_special_tokens=False)
        
        # Concatenate all texts and split into blocks
        concatenated = {k: sum(tokenized[k], []) for k in tokenized.keys()}
        total_length = len(concatenated["input_ids"])
        
        # Create blocks of block_size
        result = {
            k: [concatenated[k][i: i + block_size] for i in range(0, total_length, block_size) 
                if i + block_size <= total_length]
            for k in concatenated.keys()
        }
        
        # Create labels for autoregressive modeling (shifted input_ids)
        result["labels"] = result["input_ids"].copy()
        
        return result
    
    # Tokenize dataset
    tokenized_dataset = {}
    for split in dataset:
        tokenized_dataset[split] = dataset[split].map(
            tokenize_and_group,
            batched=True,
            remove_columns=["text"],
            desc=f"Tokenizing and grouping dataset - {split}",
        )
    
    # Additional filtering to remove input blocks that are too short
    filtered_dataset = {}
    for split in tokenized_dataset:
        filtered_dataset[split] = tokenized_dataset[split].filter(
            lambda x: len(x["input_ids"]) == block_size, 
            desc=f"Filtering blocks - {split}"
        )
        logger.info(f"{split} dataset size: {len(filtered_dataset[split])}")
    
    return filtered_dataset, tokenizer

# --------------------------------------------------
# 2. Model evaluation function
# --------------------------------------------------
def evaluate_model(model, test_dataset, tokenizer, model_name="Model"):
    """Evaluate a model and return metrics"""
    logger.info(f"Evaluating {model_name}...")
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # No masked language modeling for autoregressive GPT-2
    )
    
    # Set evaluation arguments
    eval_args = TrainingArguments(
        output_dir=f"./results_{model_name.lower().replace(' ', '_')}",
        per_device_eval_batch_size=8,
        do_eval=True,
        eval_strategy="no",
        report_to="none",
        logging_steps=100,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )
    
    # Measure evaluation time
    start_time = time.time()
    eval_results = trainer.evaluate()
    eval_time = time.time() - start_time
    
    # Calculate metrics
    eval_loss = eval_results["eval_loss"]
    perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")
    
    # Print results
    print("\n" + "="*50)
    print(f"{model_name} Evaluation Results:")
    print("="*50)
    print(f"Eval Loss (Cross Entropy): {eval_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Evaluation Time: {eval_time:.2f} seconds")
    print("="*50)
    
    return {
        "model_name": model_name,
        "eval_loss": eval_loss,
        "perplexity": perplexity,
        "eval_time": eval_time,
        "raw_results": eval_results
    }

# --------------------------------------------------
# 3. Model training function
# --------------------------------------------------
def train_model(model, train_dataset, eval_dataset, tokenizer, model_name="Model", num_epochs=1):
    """Train a model and return metrics"""
    logger.info(f"Training {model_name}...")
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # No masked language modeling for autoregressive GPT-2
    )
    
    # Set training arguments
    training_args = TrainingArguments(
        output_dir=f"./trained_{model_name.lower().replace(' ', '_')}",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        gradient_accumulation_steps=4,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        warmup_steps=500,
        lr_scheduler_type="cosine",
        learning_rate=5e-5,
        save_steps=1000,
        save_total_limit=2,
        report_to="none",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Measure training time
    start_time = time.time()
    train_results = trainer.train()
    train_time = time.time() - start_time
    
    # Save model
    trainer.save_model(f"./final_{model_name.lower().replace(' ', '_')}")
    
    # Evaluate after training
    eval_results = trainer.evaluate()
    eval_loss = eval_results["eval_loss"]
    perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")
    
    # Print results
    print("\n" + "="*50)
    print(f"{model_name} Training Results:")
    print("="*50)
    print(f"Training Time: {train_time:.2f} seconds")
    print(f"Final Eval Loss: {eval_loss:.4f}")
    print(f"Final Perplexity: {perplexity:.4f}")
    print("="*50)
    
    return {
        "model_name": model_name,
        "train_time": train_time,
        "final_eval_loss": eval_loss,
        "final_perplexity": perplexity,
        "train_results": train_results,
        "eval_results": eval_results
    }

# --------------------------------------------------
# 4. Main comparison function
# --------------------------------------------------
def compare_models(do_train=True, num_epochs=1, low_rank_ratio=4):
    """Compare original and Low-Rank FC-transformed models"""
    # Prepare dataset
    dataset, tokenizer = prepare_dataset()
    
    # 1. Original Model
    logger.info("Loading original model...")
    original_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    original_model.config.pad_token_id = tokenizer.eos_token_id
    
    # 2. Low-Rank FC-Transformed Model
    logger.info("Loading and transforming model with low-rank FC...")
    transformed_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    transformed_model.config.pad_token_id = tokenizer.eos_token_id
    
    # Get hidden size for rank calculation
    hidden_size = transformed_model.config.hidden_size
    rank = hidden_size // low_rank_ratio  # Controllable rank
    
    # Apply Low-Rank FC transformation
    # We'll transform the last layer of the model
    module_name = "transformer.h.11.attn"
    transformed_model = fc_transform_pass(
        transformed_model, 
        module_name, 
        config={
            "low_rank": True,
            "rank": rank
            # No alpha parameter as it's removed in the updated implementation
        }
    )
    
    # Memory usage analysis
    def get_model_size(model):
        return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)  # Size in MB
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    
    original_size = get_model_size(original_model)
    transformed_size = get_model_size(transformed_model)
    original_params = count_parameters(original_model)
    transformed_params = count_parameters(transformed_model)
    
    print("\n" + "="*50)
    print("Model Analysis:")
    print("="*50)
    print(f"Low-Rank Ratio: 1/{low_rank_ratio} (rank={rank})")
    print(f"Original Model Size: {original_size:.2f} MB")
    print(f"Low-Rank Transformed Model Size: {transformed_size:.2f} MB")
    print(f"Size Reduction: {original_size - transformed_size:.2f} MB ({100 * (original_size - transformed_size) / original_size:.2f}%)")
    print(f"Original Parameters: {original_params:,}")
    print(f"Transformed Parameters: {transformed_params:,}")
    print(f"Parameter Reduction: {original_params - transformed_params:,} ({100 * (original_params - transformed_params) / original_params:.2f}%)")
    print("="*50)
    
    # Evaluation phase
    original_eval = evaluate_model(original_model, dataset["test"], tokenizer, "Original GPT-2")
    transformed_eval = evaluate_model(transformed_model, dataset["test"], tokenizer, "Low-Rank FC GPT-2")
    
    results = {
        "model_config": {
            "low_rank_ratio": low_rank_ratio,
            "rank": rank,
            "hidden_size": hidden_size,
        },
        "model_sizes": {
            "original_mb": original_size,
            "transformed_mb": transformed_size,
            "reduction_percent": 100 * (original_size - transformed_size) / original_size,
            "original_params": original_params,
            "transformed_params": transformed_params,
            "param_reduction_percent": 100 * (original_params - transformed_params) / original_params
        },
        "evaluation": {
            "original": original_eval,
            "transformed": transformed_eval
        }
    }
    
    # Training phase (optional)
    if do_train:
        logger.info("Starting training comparison...")
        
        original_train = train_model(original_model, dataset["train"], dataset["validation"], 
                                    tokenizer, "Original GPT-2", num_epochs)
        
        transformed_train = train_model(transformed_model, dataset["train"], dataset["validation"], 
                                       tokenizer, "Low-Rank FC GPT-2", num_epochs)
        
        results["training"] = {
            "original": original_train,
            "transformed": transformed_train
        }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"lowrank_model_comparison_results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print comparison summary
    print("\n" + "="*50)
    print("Model Comparison Summary:")
    print("="*50)
    print(f"Low-Rank Configuration: 1/{low_rank_ratio} of hidden size (rank={rank})")
    print(f"Original Model Perplexity: {original_eval['perplexity']:.4f}")
    print(f"Low-Rank FC Model Perplexity: {transformed_eval['perplexity']:.4f}")
    print(f"Perplexity Impact: {((transformed_eval['perplexity'] - original_eval['perplexity']) / original_eval['perplexity'] * 100):.2f}%")
    print()
    print(f"Original Model Eval Time: {original_eval['eval_time']:.2f} seconds")
    print(f"Low-Rank FC Model Eval Time: {transformed_eval['eval_time']:.2f} seconds")
    print(f"Speed Improvement: {((original_eval['eval_time'] - transformed_eval['eval_time']) / original_eval['eval_time'] * 100):.2f}%")
    print()
    print(f"Size Reduction: {results['model_sizes']['reduction_percent']:.2f}%")
    print(f"Parameter Reduction: {results['model_sizes']['param_reduction_percent']:.2f}%")
    print("="*50)
    
    if do_train:
        print("\nTraining Results:")
        print(f"Original Model Training Time: {results['training']['original']['train_time']:.2f} seconds")
        print(f"Low-Rank FC Model Training Time: {results['training']['transformed']['train_time']:.2f} seconds")
        print(f"Training Speed Improvement: {((results['training']['original']['train_time'] - results['training']['transformed']['train_time']) / results['training']['original']['train_time'] * 100):.2f}%")
        print()
        print(f"Original Model Final Perplexity: {results['training']['original']['final_perplexity']:.4f}")
        print(f"Low-Rank FC Model Final Perplexity: {results['training']['transformed']['final_perplexity']:.4f}")
        print("="*50)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare original GPT-2 with enhanced Low-Rank FC-transformed version")
    parser.add_argument("--train", action="store_true", help="Include training comparison")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--rank-ratio", type=int, default=4, help="Divisor for hidden size to get rank (higher means more compression)")
    parser.add_argument("--profile", action="store_true", help="Run detailed profiling")
    
    args = parser.parse_args()
    
    # Run comparison
    compare_models(do_train=args.train, num_epochs=args.epochs, low_rank_ratio=args.rank_ratio)
