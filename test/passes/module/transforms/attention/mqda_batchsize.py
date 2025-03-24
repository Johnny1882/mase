#!/usr/bin/env python3
import logging
import os
import sys
import dill
import time
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer

# Chop-specific imports (adjust path as needed)
sys.path.append(Path(__file__).resolve().parents[5].as_posix())
from chop.tools import get_tokenized_dataset, get_trainer
from chop.passes.module.transforms import attention_transform_pass

# 1. Bisection to Find Maximum Batch Size Quickly
def find_max_batch_size_bisect(
    model,
    tokenizer,
    text,
    start_bs=1,
    end_bs=8192,
    device="cuda:0"
):
    """
    Performs a binary search between [start_bs .. end_bs] to find
    the largest batch size that fits in GPU memory without OOM.
    Returns the maximum feasible batch size.
    """
    model.to(device)
    model.eval()

    low = start_bs
    high = end_bs
    best = low

    while low <= high:
        mid = (low + high) // 2
        try:
            inputs = tokenizer(
                [text] * mid,
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(device)

            with torch.no_grad():
                _ = model(**inputs)

            # Success => can try bigger batch size
            best = mid
            low = mid + 1

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Too big => reduce upper bound
                high = mid - 1
            else:
                # Some other error => raise
                raise e

    return best


# 2. Example: Load Dataset & Tokenizer & model
checkpoint = "openai-community/gpt2"
tokenizer_checkpoint = "openai-community/gpt2"
dataset_name = "imdb"

# Using CHOP's convenience function to get a tokenized dataset + tokenizer
dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)
tokenizer.pad_token = tokenizer.eos_token


model_path = f"{Path.home()}/Projects/mase/mase_output/bert-uncased-2epoch.pkl"
with open(model_path, "rb") as f:
    original_model = dill.load(f)

# We'll test with a single text prompt to see how large a batch can fit
sample_text = "This is a sample text for measuring batch-size capacity."


# 4. Baseline Measurement (No Transform)
print("\n=== BASELINE (No MGQA Transform) ===")
max_bs_baseline = find_max_batch_size_bisect(
    model=original_model,
    tokenizer=tokenizer,
    text=sample_text,
    start_bs=1000,
    end_bs=100000,  # Adjust the upper bound for your GPU
    device="cuda:0"
)
print(f"Max batch size (no transform) = {max_bs_baseline}\n")


# 5. MGQA Transform Pass Helper
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

if __name__ == "__main__":
    results = {}
    for kv in range(1, 2):
        print(f"Testing MGQA transform with kv_heads={kv}...")
        # Reload the original model each time (fresh weights)
        with open(model_path, "rb") as f:
            current_model = dill.load(f)

        # Apply transform
        current_model = apply_mgqa_transform(current_model, kv_heads=kv)

        # Find max batch size with bisection
        max_bs_kv = find_max_batch_size_bisect(
            model=current_model,
            tokenizer=tokenizer,
            text=sample_text,
            start_bs=100,
            end_bs=100000, 
            device="cuda:0"
        )

        # Estimate memory improvement in percentage
        if max_bs_baseline > 0:
            mem_improv_pct = ((max_bs_kv - max_bs_baseline) / max_bs_baseline) * 100
        else:
            # If baseline was 0 for some reason, define improvement as 0
            mem_improv_pct = 0.0

        results[kv] = {
            "max_bs": max_bs_kv,
            "mem_improvement_pct": mem_improv_pct
        }
        print(f"kv_heads={kv} => max batch size = {max_bs_kv}, ~{mem_improv_pct:.1f}% memory improvement\n")

    # 7. Print Final Summary
    print("\n================= Final Results (Max Batch Size) =================")
    print(f"Baseline (no transform): {max_bs_baseline}")
    print("kv_heads | Max Batch Size | Memory Improvement (%)")
    print("---------+---------------+------------------------")
    for kv in sorted(results.keys()):
        print(
            f"{kv:8d} | "
            f"{results[kv]['max_bs']:13d} | "
            f"{results[kv]['mem_improvement_pct']:22.1f}"
        )
    print("==================================================================\n")
