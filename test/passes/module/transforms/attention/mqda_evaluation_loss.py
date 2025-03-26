#!/usr/bin/env python3
import logging
import os
import sys
from pathlib import Path
import math
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset # Added Dataset for potential subsetting
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import gc
import numpy as np
from tqdm.auto import tqdm # Use auto version for better notebook compatibility
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import copy # Needed for deepcopy if preferred over reloading

# Set CUDA device (Consider setting via environment variable externally)
# os.environ["CUDA_VISIBLE_DEVICES"] = "3" # Example

# Chop-specific imports (adjust path as needed)
try:
    # Adjust path based on your project structure
    # sys.path.append(Path(__file__).resolve().parents[2].as_posix())
    from chop.passes.module.transforms import attention_transform_pass
except ImportError:
    print("Warning: 'chop' library not found or path incorrect. MGQA transform will fail.")
    def attention_transform_pass(model, pass_args):
        print("Error: 'chop' library needed for attention_transform_pass.")
        raise ImportError("Cannot perform MGQA transform without 'chop'.")

# --- Helper Functions ---

def apply_mgqa_transform(model, kv_heads=2):
    """Apply MGQA transform with a given 'kv_heads' count."""
    pass_args = {
        "by": "type",
        "gpt2spda": { # Assuming 'gpt2spda' is the correct key
            "config": {
                "name": "mgqa",
                "kv_heads": kv_heads,
            }
        },
    }
    print(f"Applying MGQA transform with kv_heads={kv_heads}...")
    # Apply on CPU to potentially save VRAM during transformation
    transformed_model, _ = attention_transform_pass(model.cpu(), pass_args)
    print("Transformation complete.")
    return transformed_model

def set_reproducible(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Consider commenting out for performance if strict reproducibility isn't paramount
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def tokenize_function(examples, tokenizer, max_length):
    """Tokenization function wrapper."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Main difference from original: max_length is now an argument
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

def fine_tune_model(model, tokenizer, train_dataset, device, args):
    """Fine-tune the model on the provided training dataset."""
    set_reproducible(args.seed) # Reset seed for consistent training start
    model = model.to(device)
    model.train() # Set model to training mode

    # Prepare DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True # Shuffle training data
    )

    # Prepare Optimizer
    # Filter out parameters that don't require gradients (e.g., embeddings if frozen)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate
    )

    # Prepare Scheduler
    num_training_steps = len(train_loader) * args.num_train_epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio) # e.g., 10% warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    print(f"\n--- Starting Fine-tuning ---")
    print(f"  Epochs: {args.num_train_epochs}")
    print(f"  Train Batch Size: {args.train_batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Total Steps: {num_training_steps}")
    print(f"  Warmup Steps: {num_warmup_steps}")

    global_step = 0
    total_loss = 0.0
    log_interval = 50 # Log loss every N steps

    # Clean GPU memory before starting
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    for epoch in range(args.num_train_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")
        epoch_pbar = tqdm(train_loader, desc="Training")
        for step, batch in enumerate(epoch_pbar):
            model.train() # Ensure model is in train mode
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch.get('attention_mask')
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attn_mask, labels=input_ids)
            loss = outputs.loss

            if loss is None:
                 print(f"Warning: Loss is None at step {global_step}. Skipping batch.")
                 continue

            # Backward pass & Optimization
            loss.backward()

            # Optional: Gradient Clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            global_step += 1

            # Update progress bar
            epoch_pbar.set_postfix({'loss': loss.item()})

            # Log loss periodically
            if global_step % log_interval == 0:
                avg_loss_interval = total_loss / log_interval
                print(f"  Step {global_step}/{num_training_steps} - Avg Loss (last {log_interval}): {avg_loss_interval:.4f} - LR: {scheduler.get_last_lr()[0]:.2e}")
                total_loss = 0.0 # Reset interval loss accumulator


    print("Fine-tuning finished.")
    # Clean up memory after training
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return model # Return the fine-tuned model


def evaluate_model(model, dataset, batch_size, device, seed=42):
    """Evaluate CE Loss and Perplexity on the dataset (modified for clarity)."""
    set_reproducible(seed) # Reset seed for consistent evaluation start
    model = model.to(device)
    model.eval() # Set model to evaluation mode

    # Clean up GPU memory before evaluation
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    num_batches = 0

    print(f"\n--- Starting Evaluation ---")
    print(f"  Eval Batch Size: {batch_size}")
    with torch.no_grad(): # Disable gradient calculations
        for batch in tqdm(loader, desc="Evaluating"):
            try:
                input_ids = batch['input_ids'].to(device)
                attn_mask = batch.get('attention_mask')
                if attn_mask is not None:
                    attn_mask = attn_mask.to(device)

                outputs = model(input_ids, attention_mask=attn_mask, labels=input_ids)
                loss = outputs.loss

                if loss is not None:
                    total_loss += loss.item()
                    num_batches += 1
                else:
                    print("Warning: Eval loss is None for a batch.")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nCUDA out of memory during evaluation with batch size {batch_size}. Try reducing --eval_batch_size.")
                    gc.collect(); torch.cuda.empty_cache()
                    raise e
                else: raise e

    if num_batches == 0:
        print("Warning: No batches processed during evaluation.")
        return float('nan'), float('nan')

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)

    print("Evaluation finished.")
    gc.collect(); torch.cuda.empty_cache() # Clean up after evaluation
    return avg_loss, perplexity

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune & Evaluate GPT-2 with MGQA')
    # Model & Data Args
    parser.add_argument('--model_name', type=str, default='gpt2', help='Base model name or path')
    parser.add_argument('--train_dataset_name', type=str, default='wikitext', help='Training dataset name from datasets lib')
    parser.add_argument('--train_dataset_config', type=str, default='wikitext-2-raw-v1', help='Training dataset config')
    parser.add_argument('--train_split', type=str, default='train', help='Split for training data')
    parser.add_argument('--eval_dataset_name', type=str, default='wikitext', help='Evaluation dataset name')
    parser.add_argument('--eval_dataset_config', type=str, default='wikitext-2-raw-v1', help='Evaluation dataset config')
    parser.add_argument('--eval_split', type=str, default='test', help='Split for evaluation data')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length for tokenization')
    parser.add_argument('--kv_heads_list', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], help='List of KV heads to test (Keep short for fine-tuning demo)') # Reduced default list

    # Training Args
    parser.add_argument('--do_finetune', action='store_true', default=True, help='Perform fine-tuning after MGQA transform')
    parser.add_argument('--num_train_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Batch size for training (adjust based on GPU memory)')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for AdamW')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Ratio of training steps for linear warmup')
    parser.add_argument('--max_train_samples', type=int, default=None, help='Limit the number of training samples for quick testing')
    parser.add_argument('--max_eval_samples', type=int, default=None, help='Limit the number of evaluation samples for quick testing')


    # System Args
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    print(f"Using device: {args.device}")
    set_reproducible(args.seed)

    # --- Load Base Model & Tokenizer ---
    print(f"\nLoading base model and tokenizer: {args.model_name}")
    # Load on CPU initially to manage memory, especially if transforming multiple times
    base_model_state_dict = GPT2LMHeadModel.from_pretrained(args.model_name).state_dict()
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        print("Setting pad token to eos token")
        tokenizer.pad_token = tokenizer.eos_token
    print("Base model state dict and tokenizer loaded.")

    # --- Load and Tokenize Datasets ---
    print(f"\nLoading and tokenizing datasets...")

    # Training Data
    if args.do_finetune:
        raw_train_dataset = load_dataset(args.train_dataset_name, args.train_dataset_config, split=args.train_split)
        if args.max_train_samples:
             print(f"Limiting training data to {args.max_train_samples} samples.")
             raw_train_dataset = Dataset.from_dict(raw_train_dataset[:args.max_train_samples]) # Efficiently slice smaller datasets

        print("Tokenizing training data...")
        # Use functools.partial if needing to pass more args to map elegantly
        train_dataset = raw_train_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer, args.max_length),
            batched=True,
            remove_columns=raw_train_dataset.column_names # Remove original text column
        )
        train_dataset.set_format(type="torch")
        print(f"Training dataset size: {len(train_dataset)}")
    else:
        train_dataset = None
        print("Skipping training data loading as --do_finetune is not set.")


    # Evaluation Data
    raw_eval_dataset = load_dataset(args.eval_dataset_name, args.eval_dataset_config, split=args.eval_split)
    if args.max_eval_samples:
        print(f"Limiting evaluation data to {args.max_eval_samples} samples.")
        raw_eval_dataset = Dataset.from_dict(raw_eval_dataset[:args.max_eval_samples])

    print("Tokenizing evaluation data...")
    eval_dataset = raw_eval_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.max_length),
        batched=True,
        remove_columns=raw_eval_dataset.column_names
    )
    eval_dataset.set_format(type="torch")
    print(f"Evaluation dataset size: {len(eval_dataset)}")

    # --- Evaluation Loop ---
    results = {}

    # Helper function to load a fresh base model instance
    def get_fresh_base_model(model_name, state_dict):
         # Instantiate model structure first
         model = GPT2LMHeadModel.from_pretrained(model_name) # Loads config + structure
         # Load the saved state dict to ensure we start from the exact same weights
         model.load_state_dict(copy.deepcopy(state_dict))
         return model

    # --- Evaluate Original Model (Optional but recommended) ---
    print("\n--- Evaluating Original Untransformed Model ---")
    try:
        original_model_instance = get_fresh_base_model(args.model_name, base_model_state_dict)
        original_loss, original_ppl = evaluate_model(
            model=original_model_instance,
            dataset=eval_dataset,
            batch_size=args.eval_batch_size,
            device=args.device,
            seed=args.seed
        )
        print(f"Original Model -> CE Loss: {original_loss:.4f}, Perplexity: {original_ppl:.4f}")
        results["original"] = {"loss": original_loss, "ppl": original_ppl}
        del original_model_instance # Clean up
        gc.collect(); torch.cuda.empty_cache()
    except Exception as e: # Catch broader exceptions during evaluation
        print(f"Evaluation failed for original model: {e}")

    # --- Loop through KV Heads ---
    for kv in args.kv_heads_list:
        print(f"\n===== Processing KV = {kv} =====")

        # 1. Get a fresh copy of the base model
        print("Loading fresh base model instance...")
        current_model = get_fresh_base_model(args.model_name, base_model_state_dict)
        num_attention_heads = current_model.config.n_head

        # 2. Check KV head validity
        if kv > num_attention_heads:
            print(f"Skipping KV={kv}: Exceeds total heads ({num_attention_heads}).")
            del current_model; gc.collect(); torch.cuda.empty_cache()
            continue

        try:
            # 3. Apply MGQA Transform
            transformed_model = apply_mgqa_transform(current_model, kv_heads=kv)
            del current_model # Free memory of the pre-transform CPU model
            gc.collect()

            # 4. Fine-tune (if requested)
            if args.do_finetune:
                if train_dataset is None:
                     print("Error: --do_finetune specified but no training data loaded. Skipping fine-tuning.")
                else:
                     # Move model to device before fine-tuning
                     transformed_model = fine_tune_model(
                         model=transformed_model.to(args.device), # Ensure model is on correct device
                         tokenizer=tokenizer,
                         train_dataset=train_dataset,
                         device=args.device,
                         args=args # Pass all args for training params
                     )
                     # Model is returned on args.device from fine_tune_model
            else:
                 print("Skipping fine-tuning as --do_finetune is not set.")
                 # Move model to device for evaluation if not fine-tuned
                 transformed_model = transformed_model.to(args.device)


            # 5. Evaluate the (potentially fine-tuned) transformed model
            mgqa_loss, mgqa_ppl = evaluate_model(
                model=transformed_model, # Already on args.device
                dataset=eval_dataset,
                batch_size=args.eval_batch_size,
                device=args.device,
                seed=args.seed
            )
            finetuned_tag = " (Fine-tuned)" if args.do_finetune else ""
            print(f"KV = {kv}{finetuned_tag} -> CE Loss: {mgqa_loss:.4f}, Perplexity: {mgqa_ppl:.4f}")
            results[f"kv_{kv}"] = {"loss": mgqa_loss, "ppl": mgqa_ppl, "finetuned": args.do_finetune}

            # Clean up the transformed model explicitly
            del transformed_model
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        except ImportError:
            print("Cannot perform MGQA steps because 'chop' library is missing.")
            break # Stop the loop
        except Exception as e: # Catch broader exceptions during transform/train/eval
            print(f"Processing failed for KV = {kv}: {e}")
            # Attempt cleanup even on error
            if 'transformed_model' in locals(): del transformed_model
            if 'current_model' in locals(): del current_model
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            # Continue to the next KV value? Or break? Let's continue.
            continue


    # --- Summary ---
    print("\n--- Summary ---")
    if "original" in results:
        print(f"Original Model: Loss={results['original']['loss']:.4f}, PPL={results['original']['ppl']:.4f}")

    for kv in args.kv_heads_list:
        key = f"kv_{kv}"
        if key in results:
             finetuned_tag = " (Fine-tuned)" if results[key].get("finetuned", False) else ""
             print(f"MGQA (KV={kv}){finetuned_tag}: Loss={results[key]['loss']:.4f}, PPL={results[key]['ppl']:.4f}")
        else:
             # Check config constraints again to explain potential skips
             try:
                num_attention_heads = GPT2LMHeadModel.from_pretrained(args.model_name, low_cpu_mem_usage=True).config.n_head # Load config only
                if kv > num_attention_heads:
                    print(f"MGQA (KV={kv}): Skipped (Invalid KV head count for model {args.model_name})")
                else:
                    print(f"MGQA (KV={kv}): Processing did not complete successfully.")
             except Exception: # Handle cases where model loading might fail even for config
                 print(f"MGQA (KV={kv}): Processing did not complete successfully.")


    print("\nExperiment finished.")