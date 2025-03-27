# #!/usr/bin/env python3

# import os
# import sys
# import math
# import torch
# import numpy as np
# from datasets import load_dataset
# from transformers import (
#     AutoTokenizer, 
#     AutoModelForCausalLM, 
#     Trainer, 
#     TrainingArguments,
#     DataCollatorForLanguageModeling
# )
# from tqdm.auto import tqdm
# import argparse
# import gc
# import copy
# import logging
# from datetime import datetime

# # Import chop library (adjust path as needed)
# try:
#     # sys.path.append("path/to/chop")  # Uncomment and adjust if needed
#     from chop.passes.module.transforms import attention_transform_pass
# except ImportError:
#     print("Warning: 'chop' library not found. MGQA transform will fail.")
#     def attention_transform_pass(model, pass_args):
#         raise ImportError("Cannot perform MGQA transform without 'chop'.")

# def set_reproducible(seed=42):
#     """Set seeds for reproducibility."""
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# def tokenize_function(examples, tokenizer, max_length):
#     """Tokenize examples with appropriate padding and truncation."""
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
    
#     return tokenizer(
#         examples["text"],
#         padding="max_length",
#         truncation=True,
#         max_length=max_length,
#         return_tensors="pt"
#     )

# def apply_mgqa_transform(model, kv_heads=2):
#     """Apply MGQA transform with a given 'kv_heads' count."""
#     pass_args = {
#         "by": "type",
#         "gpt2spda": {  # Adjust key if needed for your specific model type
#             "config": {
#                 "name": "mgqa",
#                 "kv_heads": kv_heads,
#             }
#         },
#     }
#     print(f"Applying MGQA transform with kv_heads={kv_heads}...")
    
#     # Apply on CPU to potentially save VRAM during transformation
#     transformed_model, _ = attention_transform_pass(model.cpu(), pass_args)
#     print("Transformation complete.")
#     return transformed_model

# def evaluate_model(model, dataset, batch_size, device, seed=42):
#     """Evaluate CE Loss and Perplexity on the dataset."""
#     set_reproducible(seed)
#     model = model.to(device)
#     model.eval()
    
#     # Clean up GPU memory before evaluation
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
    
#     loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
#     total_loss = 0.0
#     num_batches = 0
    
#     print(f"\n--- Starting Evaluation ---")
#     print(f" Eval Batch Size: {batch_size}")
    
#     with torch.no_grad():
#         for batch in tqdm(loader, desc="Evaluating"):
#             try:
#                 input_ids = batch['input_ids'].to(device)
#                 attn_mask = batch.get('attention_mask')
#                 if attn_mask is not None:
#                     attn_mask = attn_mask.to(device)
                
#                 outputs = model(input_ids, attention_mask=attn_mask, labels=input_ids)
#                 loss = outputs.loss
                
#                 if loss is not None:
#                     total_loss += loss.item()
#                     num_batches += 1
#                 else:
#                     print("Warning: Eval loss is None for a batch.")
            
#             except RuntimeError as e:
#                 if "out of memory" in str(e):
#                     print(f"\nCUDA out of memory during evaluation. Try reducing batch size.")
#                     gc.collect()
#                     torch.cuda.empty_cache()
#                     raise e
#                 else:
#                     raise e
    
#     if num_batches == 0:
#         print("Warning: No batches processed during evaluation.")
#         return float('nan'), float('nan')
    
#     avg_loss = total_loss / num_batches
#     perplexity = math.exp(avg_loss)
    
#     print("Evaluation finished.")
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
    
#     return avg_loss, perplexity

# def fine_tune_with_trainer(model, tokenizer, train_dataset, args):
#     """Fine-tune model using Hugging Face Trainer."""
#     print(f"\n--- Starting Fine-tuning with Trainer ---")
    
#     # Setup logging
#     logging.basicConfig(level=logging.INFO)
    
#     # Create a unique output directory
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_dir = f"./fine_tuned_mgqa_{timestamp}"
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Prepare data collator
#     data_collator = DataCollatorForLanguageModeling(
#         tokenizer=tokenizer,
#         mlm=False  # Not using masked language modeling
#     )
    
#     # Setup training arguments
#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         overwrite_output_dir=True,
#         num_train_epochs=args.num_train_epochs,
#         per_device_train_batch_size=args.train_batch_size,
#         save_steps=args.save_steps,
#         save_total_limit=2,
#         prediction_loss_only=True,
#         learning_rate=args.learning_rate,
#         weight_decay=args.weight_decay,
#         warmup_ratio=args.warmup_ratio,
#         logging_dir=f"{output_dir}/logs",
#         logging_steps=args.logging_steps,
#         report_to=None if args.disable_wandb else "wandb",
#         seed=args.seed
#     )
    
#     # Initialize Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         data_collator=data_collator,
#         train_dataset=train_dataset,
#     )
    
#     # Start training
#     print(f" Training with batch size {args.train_batch_size} for {args.num_train_epochs} epochs")
#     trainer.train()
    
#     # Save model if requested
#     if args.save_model:
#         print(f"Saving fine-tuned model to {output_dir}")
#         trainer.save_model(output_dir)
    
#     print("Fine-tuning complete")
#     return model

# def main():
#     parser = argparse.ArgumentParser(description='Fine-tune and evaluate models with MGQA transformation')
    
#     # Model & Data Args
#     parser.add_argument('--model_name', type=str, default='gpt2', help='Base model name or path')
#     parser.add_argument('--train_dataset_name', type=str, default='wikitext', help='Training dataset name')
#     parser.add_argument('--train_dataset_config', type=str, default='wikitext-2-raw-v1', help='Training dataset config')
#     parser.add_argument('--train_split', type=str, default='train', help='Split for training data')
#     parser.add_argument('--eval_dataset_name', type=str, default='wikitext', help='Evaluation dataset name')
#     parser.add_argument('--eval_dataset_config', type=str, default='wikitext-2-raw-v1', help='Evaluation dataset config')
#     parser.add_argument('--eval_split', type=str, default='test', help='Split for evaluation data')
#     parser.add_argument('--max_length', type=int, default=512, help='Max sequence length for tokenization')
#     parser.add_argument('--kv_heads_list', nargs='+', type=int, default=[6, 10], 
#                         help='List of KV heads to test')
    
#     # Fine-tuning Args
#     parser.add_argument('--do_finetune', action='store_true',default=True, help='Perform fine-tuning after MGQA transform')
#     parser.add_argument('--num_train_epochs', type=int, default=1, help='Number of training epochs')
#     parser.add_argument('--train_batch_size', type=int, default=8, help='Batch size for training')
#     parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for fine-tuning')
#     parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for regularization')
#     parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Ratio of training steps for warmup')
#     parser.add_argument('--max_train_samples', type=int, default=None, help='Limit training samples')
#     parser.add_argument('--logging_steps', type=int, default=500, help='Logging frequency during training')
#     parser.add_argument('--save_steps', type=int, default=1000, help='Checkpoint save frequency')
#     parser.add_argument('--save_model', action='store_true', help='Save the fine-tuned model')
#     parser.add_argument('--disable_wandb', action='store_true', help='Disable Weights & Biases logging')
    
#     # Evaluation Args
#     parser.add_argument('--eval_batch_size', type=int, default=16, help='Batch size for evaluation')
#     parser.add_argument('--max_eval_samples', type=int, default=None, 
#                        help='Limit the number of evaluation samples')
    
#     # System Args
#     parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
#     parser.add_argument('--seed', type=int, default=42)
    
#     args = parser.parse_args()
    
#     print(f"Using device: {args.device}")
#     set_reproducible(args.seed)
    
#     # --- Load Base Model & Tokenizer ---
#     print(f"\nLoading base model and tokenizer: {args.model_name}")
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
#     if tokenizer.pad_token is None:
#         print("Setting pad token to eos token")
#         tokenizer.pad_token = tokenizer.eos_token
    
#     # Load training dataset if fine-tuning is enabled
#     train_dataset = None
#     if args.do_finetune:
#         print(f"\nLoading and tokenizing training dataset...")
#         raw_train_dataset = load_dataset(
#             args.train_dataset_name,
#             args.train_dataset_config,
#             split=args.train_split
#         )
        
#         if args.max_train_samples:
#             print(f"Limiting training data to {args.max_train_samples} samples.")
#             raw_train_dataset = raw_train_dataset.select(range(min(len(raw_train_dataset), args.max_train_samples)))
        
#         print("Tokenizing training data...")
#         train_dataset = raw_train_dataset.map(
#             lambda examples: tokenize_function(examples, tokenizer, args.max_length),
#             batched=True,
#             remove_columns=raw_train_dataset.column_names
#         )
#         train_dataset.set_format(type="torch")
#         print(f"Training dataset size: {len(train_dataset)}")
    
#     # Load evaluation dataset
#     print(f"\nLoading and tokenizing evaluation dataset...")
#     raw_eval_dataset = load_dataset(
#         args.eval_dataset_name, 
#         args.eval_dataset_config, 
#         split=args.eval_split
#     )
    
#     if args.max_eval_samples:
#         print(f"Limiting evaluation data to {args.max_eval_samples} samples.")
#         raw_eval_dataset = raw_eval_dataset.select(range(min(len(raw_eval_dataset), args.max_eval_samples)))
    
#     print("Tokenizing evaluation data...")
#     eval_dataset = raw_eval_dataset.map(
#         lambda examples: tokenize_function(examples, tokenizer, args.max_length),
#         batched=True,
#         remove_columns=raw_eval_dataset.column_names
#     )
#     eval_dataset.set_format(type="torch")
#     print(f"Evaluation dataset size: {len(eval_dataset)}")
    
#     # --- Setup evaluation results tracking ---
#     results = {}
    
#     # --- Evaluate Original Model ---
#     print("\n--- Evaluating Original Model ---")
#     try:
#         # Load base model (keeping separate from transformation to ensure clean state)
#         base_model = AutoModelForCausalLM.from_pretrained(args.model_name)
        
#         # Evaluate original model
#         original_loss, original_ppl = evaluate_model(
#             model=base_model,
#             dataset=eval_dataset,
#             batch_size=args.eval_batch_size,
#             device=args.device,
#             seed=args.seed
#         )
        
#         print(f"Original Model -> CE Loss: {original_loss:.4f}, Perplexity: {original_ppl:.4f}")
#         results["original"] = {"loss": original_loss, "ppl": original_ppl}
        
#         # Clean up
#         del base_model
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
            
#     except Exception as e:
#         print(f"Evaluation failed for original model: {e}")
    
#     # --- Loop through KV Heads ---
#     for kv in args.kv_heads_list:
#         print(f"\n===== Processing KV = {kv} =====")
        
#         try:
#             # Load a fresh copy of the base model for each transformation
#             current_model = AutoModelForCausalLM.from_pretrained(args.model_name)
#             num_attention_heads = getattr(current_model.config, "n_head", 
#                                           getattr(current_model.config, "num_attention_heads", 0))
            
#             # Check KV head validity
#             if kv > num_attention_heads:
#                 print(f"Skipping KV={kv}: Exceeds total heads ({num_attention_heads}).")
#                 del current_model
#                 gc.collect()
#                 if torch.cuda.is_available():
#                     torch.cuda.empty_cache()
#                 continue
            
#             # Apply MGQA Transform
#             transformed_model = apply_mgqa_transform(current_model, kv_heads=kv)
#             del current_model  # Free memory
            
#             # Fine-tune if requested
#             if args.do_finetune and train_dataset is not None:
#                 print(f"\n--- Fine-tuning model with KV = {kv} ---")
#                 try:
#                     transformed_model = fine_tune_with_trainer(
#                         model=transformed_model,
#                         tokenizer=tokenizer,
#                         train_dataset=train_dataset,
#                         args=args
#                     )
#                 except Exception as e:
#                     print(f"Fine-tuning failed for KV = {kv}: {e}")
#                     print("Continuing with evaluation of the transformed (but not fine-tuned) model.")
            
#             # Evaluate the transformed model
#             mgqa_loss, mgqa_ppl = evaluate_model(
#                 model=transformed_model,
#                 dataset=eval_dataset,
#                 batch_size=args.eval_batch_size,
#                 device=args.device,
#                 seed=args.seed
#             )
            
#             # Add fine-tuning info to results
#             finetuned_tag = " (Fine-tuned)" if args.do_finetune else ""
#             print(f"KV = {kv}{finetuned_tag} -> CE Loss: {mgqa_loss:.4f}, Perplexity: {mgqa_ppl:.4f}")
#             results[f"kv_{kv}"] = {
#                 "loss": mgqa_loss, 
#                 "ppl": mgqa_ppl,
#                 "finetuned": args.do_finetune
#             }
            
#             # Clean up
#             del transformed_model
#             gc.collect()
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
                
#         except ImportError:
#             print("Cannot perform MGQA steps because 'chop' library is missing.")
#             break
            
#         except Exception as e:
#             print(f"Processing failed for KV = {kv}: {e}")
#             # Clean up on error
#             if 'transformed_model' in locals():
#                 del transformed_model
#             if 'current_model' in locals():
#                 del current_model
#             gc.collect()
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
    
#     # --- Summary ---
#     print("\n------ Results Summary ------")
#     if "original" in results:
#         print(f"Original Model: Loss={results['original']['loss']:.4f}, PPL={results['original']['ppl']:.4f}")
    
#     for kv in args.kv_heads_list:
#         key = f"kv_{kv}"
#         if key in results:
#             rel_ppl = (results[key]['ppl'] / results.get("original", {}).get("ppl", 1.0)) * 100 - 100
#             print(f"MGQA (KV={kv}): Loss={results[key]['loss']:.4f}, PPL={results[key]['ppl']:.4f} ({rel_ppl:+.2f}% rel. to original)")
    
#     print("\nEvaluation complete.")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3

import os
import math
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from tqdm.auto import tqdm
import argparse
import gc
import logging
from datetime import datetime

def set_reproducible(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def tokenize_function(examples, tokenizer, max_length):
    """Tokenize examples with appropriate padding and truncation."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

def evaluate_model(model, dataset, batch_size, device, seed=42):
    """Evaluate CE Loss and Perplexity on the dataset."""
    set_reproducible(seed)
    model = model.to(device)
    model.eval()
    
    # Clean up GPU memory before evaluation
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    num_batches = 0
    
    print(f"\n--- Starting Evaluation ---")
    print(f" Eval Batch Size: {batch_size}")
    
    with torch.no_grad():
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
                    print(f"\nCUDA out of memory during evaluation. Try reducing batch size.")
                    gc.collect()
                    torch.cuda.empty_cache()
                    raise e
                else:
                    raise e
    
    if num_batches == 0:
        print("Warning: No batches processed during evaluation.")
        return float('nan'), float('nan')
    
    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    
    print("Evaluation finished.")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return avg_loss, perplexity

def fine_tune_with_trainer(model, tokenizer, train_dataset, args):
    """Fine-tune model using Hugging Face Trainer."""
    print(f"\n--- Starting Fine-tuning with Trainer ---")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create a unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./fine_tuned_original_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Not using masked language modeling
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        save_steps=args.save_steps,
        save_total_limit=2,
        prediction_loss_only=True,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_dir=f"{output_dir}/logs",
        logging_steps=args.logging_steps,
        report_to=None if args.disable_wandb else "wandb",
        seed=args.seed
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # Start training
    print(f" Training with batch size {args.train_batch_size} for {args.num_train_epochs} epochs")
    trainer.train()
    
    # Save model if requested
    if args.save_model:
        print(f"Saving fine-tuned model to {output_dir}")
        trainer.save_model(output_dir)
    
    print("Fine-tuning complete")
    return model

def main():
    parser = argparse.ArgumentParser(description='Fine-tune and evaluate original model')
    
    # Model & Data Args
    parser.add_argument('--model_name', type=str, default='gpt2', help='Base model name or path')
    parser.add_argument('--train_dataset_name', type=str, default='wikitext', help='Training dataset name')
    parser.add_argument('--train_dataset_config', type=str, default='wikitext-2-raw-v1', help='Training dataset config')
    parser.add_argument('--train_split', type=str, default='train', help='Split for training data')
    parser.add_argument('--eval_dataset_name', type=str, default='wikitext', help='Evaluation dataset name')
    parser.add_argument('--eval_dataset_config', type=str, default='wikitext-2-raw-v1', help='Evaluation dataset config')
    parser.add_argument('--eval_split', type=str, default='test', help='Split for evaluation data')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length for tokenization')
    
    # Fine-tuning Args
    parser.add_argument('--num_train_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for fine-tuning')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for regularization')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Ratio of training steps for warmup')
    parser.add_argument('--max_train_samples', type=int, default=None, help='Limit training samples')
    parser.add_argument('--logging_steps', type=int, default=500, help='Logging frequency during training')
    parser.add_argument('--save_steps', type=int, default=1000, help='Checkpoint save frequency')
    parser.add_argument('--save_model', action='store_true', help='Save the fine-tuned model')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable Weights & Biases logging')
    
    # Evaluation Args
    parser.add_argument('--eval_batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--max_eval_samples', type=int, default=None, 
                       help='Limit the number of evaluation samples')
    
    # System Args
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    set_reproducible(args.seed)
    
    # --- Load Base Model & Tokenizer ---
    print(f"\nLoading base model and tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if tokenizer.pad_token is None:
        print("Setting pad token to eos token")
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load training dataset
    print(f"\nLoading and tokenizing training dataset...")
    raw_train_dataset = load_dataset(
        args.train_dataset_name,
        args.train_dataset_config,
        split=args.train_split
    )
    
    if args.max_train_samples:
        print(f"Limiting training data to {args.max_train_samples} samples.")
        raw_train_dataset = raw_train_dataset.select(range(min(len(raw_train_dataset), args.max_train_samples)))
    
    print("Tokenizing training data...")
    train_dataset = raw_train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.max_length),
        batched=True,
        remove_columns=raw_train_dataset.column_names
    )
    train_dataset.set_format(type="torch")
    print(f"Training dataset size: {len(train_dataset)}")
    
    # Load evaluation dataset
    print(f"\nLoading and tokenizing evaluation dataset...")
    raw_eval_dataset = load_dataset(
        args.eval_dataset_name, 
        args.eval_dataset_config, 
        split=args.eval_split
    )
    
    if args.max_eval_samples:
        print(f"Limiting evaluation data to {args.max_eval_samples} samples.")
        raw_eval_dataset = raw_eval_dataset.select(range(min(len(raw_eval_dataset), args.max_eval_samples)))
    
    print("Tokenizing evaluation data...")
    eval_dataset = raw_eval_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.max_length),
        batched=True,
        remove_columns=raw_eval_dataset.column_names
    )
    eval_dataset.set_format(type="torch")
    print(f"Evaluation dataset size: {len(eval_dataset)}")
    
    # --- Evaluate Original Model (Before Fine-tuning) ---
    print("\n--- Evaluating Original Model (Before Fine-tuning) ---")
    try:
        # Load base model
        original_model = AutoModelForCausalLM.from_pretrained(args.model_name)
        
        # Evaluate original model (before fine-tuning)
        original_loss, original_ppl = evaluate_model(
            model=original_model,
            dataset=eval_dataset,
            batch_size=args.eval_batch_size,
            device=args.device,
            seed=args.seed
        )
        
        print(f"Original Model (Before Fine-tuning) -> CE Loss: {original_loss:.4f}, Perplexity: {original_ppl:.4f}")
            
    except Exception as e:
        print(f"Evaluation failed for original model: {e}")
        # If evaluation fails, try to continue with fine-tuning
        original_loss, original_ppl = float('nan'), float('nan')
    
    # --- Fine-tune Original Model ---
    print("\n--- Fine-tuning Original Model ---")
    try:
        # Original model is already loaded from previous step
        # If previous evaluation failed, reload the model
        if math.isnan(original_loss):
            original_model = AutoModelForCausalLM.from_pretrained(args.model_name)
        
        # Fine-tune original model
        fine_tuned_model = fine_tune_with_trainer(
            model=original_model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            args=args
        )
        
        # Clean up original model reference to save memory
        del original_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Evaluate fine-tuned model
        fine_tuned_loss, fine_tuned_ppl = evaluate_model(
            model=fine_tuned_model,
            dataset=eval_dataset,
            batch_size=args.eval_batch_size,
            device=args.device,
            seed=args.seed
        )
        
        print(f"Fine-tuned Original Model -> CE Loss: {fine_tuned_loss:.4f}, Perplexity: {fine_tuned_ppl:.4f}")
            
    except Exception as e:
        print(f"Fine-tuning or evaluation failed: {e}")
        fine_tuned_loss, fine_tuned_ppl = float('nan'), float('nan')
    
    # --- Summary ---
    print("\n------ Results Summary ------")
    print(f"Original Model (Before Fine-tuning): Loss={original_loss:.4f}, PPL={original_ppl:.4f}")
    print(f"Fine-tuned Original Model: Loss={fine_tuned_loss:.4f}, PPL={fine_tuned_ppl:.4f}")
    
    # Calculate improvement if both evaluations succeeded
    if not (math.isnan(original_ppl) or math.isnan(fine_tuned_ppl)):
        ppl_change = ((fine_tuned_ppl / original_ppl) * 100) - 100
        print(f"Change after fine-tuning: {ppl_change:.2f}% in perplexity")
    
    # Save results to CSV
    try:
        import pandas as pd
        results_data = [
            {
                "model": "Original (Before Fine-tuning)",
                "loss": original_loss,
                "perplexity": original_ppl,
                "ppl_change_pct": 0.0
            },
            {
                "model": "Original (After Fine-tuning)",
                "loss": fine_tuned_loss,
                "perplexity": fine_tuned_ppl,
                "ppl_change_pct": ((fine_tuned_ppl / original_ppl) * 100) - 100 if not (math.isnan(original_ppl) or math.isnan(fine_tuned_ppl)) else float('nan')
            }
        ]
        
        # Create DataFrame and save to CSV
        results_df = pd.DataFrame(results_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"original_finetune_results_{timestamp}.csv"
        results_df.to_csv(csv_filename, index=False)
        print(f"\nResults saved to {csv_filename}")
    except Exception as e:
        print(f"Failed to save results to CSV: {e}")
    
    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()