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
import matplotlib.pyplot as plt
import numpy as np

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
# 2. KV Cache testing function
# --------------------------------------------------
def test_kv_cache(model, tokenizer, prompt, max_new_tokens=50):
    """
    Test the KV cache functionality and measure its size.
    
    Args:
        model: The model to test
        tokenizer: The tokenizer for the model
        prompt: The starting prompt
        max_new_tokens: Number of tokens to generate
        
    Returns:
        dict: Results including KV cache size and generation time
    """
    import torch
    import time
    
    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate with KV cache and measure time
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True
        )
    generation_time = time.time() - start_time
    
    # Extract generated text
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    # Analyze KV cache size
    past_key_values = outputs.past_key_values
    kv_cache_size = 0
    
    if past_key_values:
        num_layers = len(past_key_values)
        
        # Calculate total size
        for layer_idx, (k, v) in enumerate(past_key_values):
            layer_bytes = k.nelement() * k.element_size() + v.nelement() * v.element_size()
            kv_cache_size += layer_bytes
        
        kv_cache_size_mb = kv_cache_size / (1024 * 1024)
    else:
        kv_cache_size_mb = 0
    
    # Perform without cache as comparison
    if past_key_values:
        start_time = time.time()
        with torch.no_grad():
            no_cache_outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                use_cache=False
            )
        no_cache_time = time.time() - start_time
        
        cache_speedup = no_cache_time / generation_time
    else:
        no_cache_time = 0
        cache_speedup = 0
    
    # Return results
    results = {
        "generation_time": generation_time,
        "kv_cache_size_bytes": kv_cache_size,
        "kv_cache_size_mb": kv_cache_size_mb,
        "no_cache_time": no_cache_time,
        "cache_speedup": cache_speedup
    }
    
    return results

# --------------------------------------------------
# 3. Model evaluation function
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
    
    return {
        "model_name": model_name,
        "eval_loss": eval_loss,
        "perplexity": perplexity,
        "eval_time": eval_time
    }

# --------------------------------------------------
# 4. Model training function
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
    output_dir = f"./final_{model_name.lower().replace(' ', '_')}"
    trainer.save_model(output_dir)
    logger.info(f"Saved fine-tuned model to {output_dir}")
    
    # Evaluate after training
    eval_results = trainer.evaluate()
    eval_loss = eval_results["eval_loss"]
    perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")
    
    return {
        "model_name": model_name,
        "train_time": train_time,
        "final_eval_loss": eval_loss,
        "final_perplexity": perplexity
    }

# --------------------------------------------------
# 5. Apply low-rank transformation
# --------------------------------------------------
def apply_lowrank_transformation(model, rank):
    """Apply low-rank factorization to all attention layers"""
    for layer_idx in range(len(model.transformer.h)):
        module_name = f"transformer.h.{layer_idx}.attn"
        model = fc_transform_pass(
            model, 
            module_name, 
            config={
                "low_rank": True,
                "rank": rank
            }
        )
    return model

# --------------------------------------------------
# 6. Main comparison function
# --------------------------------------------------
def compare_models_with_ranks(rank_ratios=[2, 4, 8, 16], do_train=False, num_epochs=1):
    """Compare models with different low-rank configurations"""
    # Prepare dataset
    dataset, tokenizer = prepare_dataset()
    
    # Get test samples from WikiText for KV cache testing
    def get_samples(num_samples=2, seed=42):
        """Get random samples from WikiText-2 dataset with max length 1024."""
        import random
        random.seed(seed)
        
        # Load raw dataset directly for text samples
        raw_dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
        data = raw_dataset["train"]
        indices = random.sample(range(len(data)), min(num_samples, len(data)))
        
        return [data[idx]["text"][:1024] for idx in indices]
    
    test_cases = get_samples(num_samples=2, seed=42)
    
    # Evaluate original model first
    logger.info("Loading and evaluating original model...")
    original_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    original_model.config.pad_token_id = tokenizer.eos_token_id
    
    # Get hidden size for rank calculation
    hidden_size = original_model.config.hidden_size
    
    # Evaluate original model
    original_eval = evaluate_model(original_model, dataset["test"], tokenizer, "Original GPT-2")
    
    # Test KV cache on original model
    original_kv_results = []
    for i, sample in enumerate(test_cases):
        logger.info(f"Testing KV cache on original model - sample {i+1}/{len(test_cases)}")
        original_kv = test_kv_cache(original_model, tokenizer, sample)
        original_kv_results.append(original_kv)
    
    # Calculate averages for original model KV cache
    avg_original_kv_size = sum(r["kv_cache_size_mb"] for r in original_kv_results) / len(original_kv_results)
    avg_original_gen_time = sum(r["generation_time"] for r in original_kv_results) / len(original_kv_results)
    avg_original_speedup = sum(r["cache_speedup"] for r in original_kv_results) / len(original_kv_results)
    
    # Store results for all models
    all_results = {
        "original": {
            "perplexity": original_eval["perplexity"],
            "eval_loss": original_eval["eval_loss"],
            "eval_time": original_eval["eval_time"],
            "kv_cache_size_mb": avg_original_kv_size,
            "generation_time": avg_original_gen_time,
            "cache_speedup": avg_original_speedup
        }
    }
    
    # Test each rank ratio
    for ratio in rank_ratios:
        rank = hidden_size // ratio
        logger.info(f"Testing low-rank ratio 1/{ratio} (rank={rank})...")
        
        # No fine-tuning model
        logger.info(f"Creating low-rank model without fine-tuning...")
        model_no_ft = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        model_no_ft.config.pad_token_id = tokenizer.eos_token_id
        model_no_ft = apply_lowrank_transformation(model_no_ft, rank)
        
        # Evaluate without fine-tuning
        eval_no_ft = evaluate_model(model_no_ft, dataset["test"], tokenizer, f"Low-Rank 1/{ratio} (No FT)")
        
        # Test KV cache on non-fine-tuned model
        kv_results_no_ft = []
        for i, sample in enumerate(test_cases):
            logger.info(f"Testing KV cache on rank 1/{ratio} model - sample {i+1}/{len(test_cases)}")
            kv_no_ft = test_kv_cache(model_no_ft, tokenizer, sample)
            kv_results_no_ft.append(kv_no_ft)
        
        # Calculate averages for non-fine-tuned model
        avg_kv_size_no_ft = sum(r["kv_cache_size_mb"] for r in kv_results_no_ft) / len(kv_results_no_ft)
        avg_gen_time_no_ft = sum(r["generation_time"] for r in kv_results_no_ft) / len(kv_results_no_ft)
        avg_speedup_no_ft = sum(r["cache_speedup"] for r in kv_results_no_ft) / len(kv_results_no_ft)
        
        # Store results for non-fine-tuned model
        all_results[f"rank_1_{ratio}_no_ft"] = {
            "rank_ratio": ratio,
            "rank": rank,
            "perplexity": eval_no_ft["perplexity"],
            "eval_loss": eval_no_ft["eval_loss"],
            "eval_time": eval_no_ft["eval_time"],
            "kv_cache_size_mb": avg_kv_size_no_ft,
            "generation_time": avg_gen_time_no_ft,
            "cache_speedup": avg_speedup_no_ft
        }
        
        # Fine-tuning (if requested)
        if do_train:
            logger.info(f"Creating low-rank model for fine-tuning...")
            model_ft = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
            model_ft.config.pad_token_id = tokenizer.eos_token_id
            model_ft = apply_lowrank_transformation(model_ft, rank)
            
            # Train the model
            train_result = train_model(
                model_ft, 
                dataset["train"], 
                dataset["validation"], 
                tokenizer, 
                f"Low-Rank 1/{ratio} (FT)", 
                num_epochs
            )
            
            # Evaluate after fine-tuning
            eval_ft = evaluate_model(model_ft, dataset["test"], tokenizer, f"Low-Rank 1/{ratio} (FT)")
            
            # Test KV cache on fine-tuned model
            kv_results_ft = []
            for i, sample in enumerate(test_cases):
                logger.info(f"Testing KV cache on fine-tuned rank 1/{ratio} model - sample {i+1}/{len(test_cases)}")
                kv_ft = test_kv_cache(model_ft, tokenizer, sample)
                kv_results_ft.append(kv_ft)
            
            # Calculate averages for fine-tuned model
            avg_kv_size_ft = sum(r["kv_cache_size_mb"] for r in kv_results_ft) / len(kv_results_ft)
            avg_gen_time_ft = sum(r["generation_time"] for r in kv_results_ft) / len(kv_results_ft)
            avg_speedup_ft = sum(r["cache_speedup"] for r in kv_results_ft) / len(kv_results_ft)
            
            # Store results for fine-tuned model
            all_results[f"rank_1_{ratio}_ft"] = {
                "rank_ratio": ratio,
                "rank": rank,
                "perplexity": eval_ft["perplexity"],
                "eval_loss": eval_ft["eval_loss"],
                "eval_time": eval_ft["eval_time"],
                "kv_cache_size_mb": avg_kv_size_ft,
                "generation_time": avg_gen_time_ft,
                "cache_speedup": avg_speedup_ft,
                "training_time": train_result["train_time"]
            }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"lowrank_model_comparison_ranks_{timestamp}.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Print results table
    print("\n" + "="*80)
    print("Model Comparison Across Different Ranks:")
    print("="*80)
    print(f"{'Model':<30} {'Loss':<10} {'Perplexity':<12} {'KV Cache':<10} {'Gen Time':<10} {'Speedup':<10}")
    print("-"*80)
    
    # Original model row
    print(f"{'Original GPT-2':<30} {all_results['original']['eval_loss']:<10.4f} {all_results['original']['perplexity']:<12.4f} {all_results['original']['kv_cache_size_mb']:<10.2f} {all_results['original']['generation_time']:<10.4f} {all_results['original']['cache_speedup']:<10.2f}x")
    
    # Low-rank models without fine-tuning
    for ratio in rank_ratios:
        key = f"rank_1_{ratio}_no_ft"
        if key in all_results:
            print(f"{'Low-Rank 1/'+str(ratio)+' (No FT)':<30} {all_results[key]['eval_loss']:<10.4f} {all_results[key]['perplexity']:<12.4f} {all_results[key]['kv_cache_size_mb']:<10.2f} {all_results[key]['generation_time']:<10.4f} {all_results[key]['cache_speedup']:<10.2f}x")
    
    # Low-rank models with fine-tuning
    if do_train:
        print("-"*80)
        for ratio in rank_ratios:
            key = f"rank_1_{ratio}_ft"
            if key in all_results:
                print(f"{'Low-Rank 1/'+str(ratio)+' (FT)':<30} {all_results[key]['eval_loss']:<10.4f} {all_results[key]['perplexity']:<12.4f} {all_results[key]['kv_cache_size_mb']:<10.2f} {all_results[key]['generation_time']:<10.4f} {all_results[key]['cache_speedup']:<10.2f}x")
    
    print("="*80)
    print("Changes from original (%):")
    print("-"*80)
    
    # Low-rank models without fine-tuning (percentage changes)
    for ratio in rank_ratios:
        key = f"rank_1_{ratio}_no_ft"
        if key in all_results:
            loss_change = (all_results[key]['eval_loss'] - all_results['original']['eval_loss'])
            ppl_change = ((all_results[key]['perplexity'] - all_results['original']['perplexity'])/all_results['original']['perplexity']*100)
            kv_change = ((all_results[key]['kv_cache_size_mb'] - all_results['original']['kv_cache_size_mb'])/all_results['original']['kv_cache_size_mb']*100)
            time_change = ((all_results[key]['generation_time'] - all_results['original']['generation_time'])/all_results['original']['generation_time']*100)
            speedup_change = ((all_results[key]['cache_speedup'] - all_results['original']['cache_speedup'])/all_results['original']['cache_speedup']*100)
            
            print(f"{'Low-Rank 1/'+str(ratio)+' (No FT)':<30} {loss_change:<+10.4f} {ppl_change:<+11.2f}% {kv_change:<+9.2f}% {time_change:<+9.2f}% {speedup_change:<+9.2f}%")
    
    # Low-rank models with fine-tuning (percentage changes)
    if do_train:
        print("-"*80)
        for ratio in rank_ratios:
            key = f"rank_1_{ratio}_ft"
            if key in all_results:
                loss_change = (all_results[key]['eval_loss'] - all_results['original']['eval_loss'])
                ppl_change = ((all_results[key]['perplexity'] - all_results['original']['perplexity'])/all_results['original']['perplexity']*100)
                kv_change = ((all_results[key]['kv_cache_size_mb'] - all_results['original']['kv_cache_size_mb'])/all_results['original']['kv_cache_size_mb']*100)
                time_change = ((all_results[key]['generation_time'] - all_results['original']['generation_time'])/all_results['original']['generation_time']*100)
                speedup_change = ((all_results[key]['cache_speedup'] - all_results['original']['cache_speedup'])/all_results['original']['cache_speedup']*100)
                
                print(f"{'Low-Rank 1/'+str(ratio)+' (FT)':<30} {loss_change:<+10.4f} {ppl_change:<+11.2f}% {kv_change:<+9.2f}% {time_change:<+9.2f}% {speedup_change:<+9.2f}%")
    
    print("="*80)
    
    # Generate visualization
    plot_results(all_results, rank_ratios, do_train, timestamp)
    
    return all_results

# --------------------------------------------------
# 7. Visualization function
# --------------------------------------------------
def plot_results(results, rank_ratios, include_ft, timestamp):
    """Generate plots to visualize the tradeoffs between different ranks"""
    plt.figure(figsize=(15, 10))
    
    # Extract data for plotting
    ranks = [results["original"]["perplexity"]]
    ppl_no_ft = [results["original"]["perplexity"]]
    kv_size_no_ft = [results["original"]["kv_cache_size_mb"]]
    gen_time_no_ft = [results["original"]["generation_time"]]
    
    if include_ft:
        ppl_ft = [results["original"]["perplexity"]]
        kv_size_ft = [results["original"]["kv_cache_size_mb"]]
        gen_time_ft = [results["original"]["generation_time"]]
    
    x_labels = ["Original"]
    
    for ratio in rank_ratios:
        x_labels.append(f"1/{ratio}")
        ranks.append(results[f"rank_1_{ratio}_no_ft"]["rank"])
        
        ppl_no_ft.append(results[f"rank_1_{ratio}_no_ft"]["perplexity"])
        kv_size_no_ft.append(results[f"rank_1_{ratio}_no_ft"]["kv_cache_size_mb"])
        gen_time_no_ft.append(results[f"rank_1_{ratio}_no_ft"]["generation_time"])
        
        if include_ft:
            ppl_ft.append(results[f"rank_1_{ratio}_ft"]["perplexity"])
            kv_size_ft.append(results[f"rank_1_{ratio}_ft"]["kv_cache_size_mb"])
            gen_time_ft.append(results[f"rank_1_{ratio}_ft"]["generation_time"])
    
    # Plot perplexity
    plt.subplot(2, 2, 1)
    plt.plot(x_labels, ppl_no_ft, 'bo-', label='No Fine-tuning')
    if include_ft:
        plt.plot(x_labels, ppl_ft, 'ro-', label='With Fine-tuning')
    plt.title('Perplexity vs Rank Ratio')
    plt.xlabel('Rank Ratio')
    plt.ylabel('Perplexity')
    plt.grid(True)
    plt.legend()
    
    # Plot KV cache size
    plt.subplot(2, 2, 2)
    plt.plot(x_labels, kv_size_no_ft, 'bo-', label='No Fine-tuning')
    if include_ft:
        plt.plot(x_labels, kv_size_ft, 'ro-', label='With Fine-tuning')
    plt.title('KV Cache Size vs Rank Ratio')
    plt.xlabel('Rank Ratio')
    plt.ylabel('KV Cache Size (MB)')
    plt.grid(True)
    plt.legend()
    
    # Plot generation time
    plt.subplot(2, 2, 3)
    plt.plot(x_labels, gen_time_no_ft, 'bo-', label='No Fine-tuning')
    if include_ft:
        plt.plot(x_labels, gen_time_ft, 'ro-', label='With Fine-tuning')
    plt.title('Generation Time vs Rank Ratio')
    plt.xlabel('Rank Ratio')
    plt.ylabel('Generation Time (s)')
    plt.grid(True)
    plt.legend()
    
    # Plot tradeoff between perplexity and KV cache size
    plt.subplot(2, 2, 4)
    plt.scatter(kv_size_no_ft, ppl_no_ft, c='blue', s=100, label='No Fine-tuning')
    if include_ft:
        plt.scatter(kv_size_ft, ppl_ft, c='red', s=100, label='With Fine-tuning')
    
    # Add labels to points
    for i, label in enumerate(x_labels):
        plt.annotate(label, (kv_size_no_ft[i], ppl_no_ft[i]), textcoords="offset points", xytext=(0,10), ha='center')
        if include_ft:
            plt.annotate(label, (kv_size_ft[i], ppl_ft[i]), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.title('Perplexity vs KV Cache Size')
    plt.xlabel('KV Cache Size (MB)')
    plt.ylabel('Perplexity')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"lowrank_comparison_plots_{timestamp}.png")
    logger.info(f"Saved plots to lowrank_comparison_plots_{timestamp}.png")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare original GPT-2 with Low-Rank FC-transformed versions at different ranks")
    parser.add_argument("--train", action="store_true", help="Include fine-tuning of transformed models")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--ranks", type=str, default="2,4,8,16", help="Comma-separated list of rank ratios to test")
    
    args = parser.parse_args()
    
    # Parse rank ratios
    rank_ratios = [int(r) for r in args.ranks.split(',')]
    
    # Run comparison
    compare_models_with_ranks(rank_ratios=rank_ratios, do_train=args.train, num_epochs=args.epochs)