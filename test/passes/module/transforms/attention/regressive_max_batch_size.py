import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

#!/usr/bin/env python3
import logging
import os
import sys
import dill
import time
from pathlib import Path
import math
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer
import bisect
import gc
import numpy as np

# Chop-specific imports (adjust path as needed)
sys.path.append(Path(__file__).resolve().parents[5].as_posix())
from chop.tools import get_tokenized_dataset, get_trainer
from chop.passes.module.transforms import attention_transform_pass

import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset


def apply_mgqa_transform(model, kv_heads=2):
    """
    Apply MGQA transform with a given 'kv_heads' count.
    """
    pass_args = {
        "by": "type",
        "gpt2spda": {
            "config": {
                "name": "mgqa",
                "kv_heads": kv_heads,
            }
        },
    }
    transformed_model, _ = attention_transform_pass(model, pass_args)
    return transformed_model

def set_reproducible(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    torch.backends.cudnn.deterministic = True
    
def test_batch_size(model, dataset, batch_size, device):
    """Test if a batch size is feasible (no OOM)."""
    try:
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        batch = next(iter(loader))
        
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch.get('attention_mask')
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)
            
        with torch.no_grad():
            _ = model(input_ids, attention_mask=attn_mask, labels=input_ids)
        return True
    except RuntimeError as e:
        return False if "out of memory" in str(e) else None
        
def autoregressive_test(model, dataset, reference_index=0, device='cuda', seed=42):
    """Evaluate perplexity on a single reference and find max batch size."""
    set_reproducible(seed)
    model = model.to(device)
        
    # Find max batch size using bisect
    def batch_feasible(size):
        result = test_batch_size(model, dataset, size, device)
        return True if result else False
    
    # Use bisect to find the boundary between feasible and infeasible
    max_size = 1
    while batch_feasible(max_size):
        max_size *= 2
    
    max_batch = bisect.bisect_left(
        range(1, max_size),
        True,
        key=lambda x: not batch_feasible(x)
    )
    
    return max_batch


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Autoregressive testing for GPT-2')
    parser.add_argument('--model_name', type=str, default='gpt2', help='Model name or path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ref_index', type=int, default=0, help='Index of reference to test')
    args = parser.parse_args()
    
    # Load model & dataset
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # Run test
    # max_batch_size = autoregressive_test(
    #     model=model,
    #     dataset=tokenized_dataset,
    #     reference_index=args.ref_index,
    #     device=args.device,
    #     seed=args.seed
    # )
    # print(f"Maximum batch size: {max_batch_size}")
    
    for kv in [1, 2, 3, 4, 6, 12]:
        mgqa_model = apply_mgqa_transform(model, kv_heads=kv)
        max_batch_size = autoregressive_test(
            model=mgqa_model,
            dataset=tokenized_dataset,
            reference_index=args.ref_index,
            device=args.device,
            seed=args.seed
        )
        
        print(f"KV = {kv}, Maximum batch size: {max_batch_size}")