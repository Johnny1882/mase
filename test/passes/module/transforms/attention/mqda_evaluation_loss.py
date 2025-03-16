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

# Chop-specific imports
sys.path.append(Path(__file__).resolve().parents[5].as_posix())
from chop.tools import get_tokenized_dataset, get_trainer
from chop.passes.module.transforms import attention_transform_pass
from chop import MaseGraph
import chop.passes as passes


#########################################
# 1. Inference Speed Test
#########################################
def measure_inference_speed(
    model,
    tokenizer,
    sample_text,
    device="cuda",
    num_warmup=5,
    num_runs=20
):
    """
    Measures *average inference time* per run (in seconds).
    """
    model.to(device)
    model.eval()

    # Prepare inputs
    inputs = tokenizer(
        sample_text,
        return_tensors='pt',
        padding=True,
        truncation=True
    ).to(device)

    # Warm-up (not timed)
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(**inputs)

    # Timed runs
    torch.cuda.synchronize(device)
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(**inputs)
    torch.cuda.synchronize(device)
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    return avg_time

#########################################
# 2. Evaluate Helper (Cross-Entropy & Perplexity)
#########################################
def evaluate_ce_and_ppl(trainer, eval_dataset):
    """
    Given a Hugging Face trainer and an eval dataset, compute cross-entropy & perplexity.
    """
    predictions, labels, _ = trainer.predict(eval_dataset)
    logits_tensor = torch.tensor(predictions, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    loss_fn = nn.CrossEntropyLoss()
    cross_entropy_val = loss_fn(logits_tensor, labels_tensor)
    perplexity_val = torch.exp(cross_entropy_val)

    return cross_entropy_val.item(), perplexity_val.item()

#########################################
# 3. Load Dataset & Tokenizer
#########################################
checkpoint = "openai-community/gpt2"
tokenizer_checkpoint = "openai-community/gpt2"
dataset_name = "imdb"

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 usually has no pad token

#########################################
# 4. Load Original Model (Dill)
#########################################
model_path = f"{Path.home()}/adls/mase/mase_output/bert-uncased-2epoch.pkl"
with open(model_path, "rb") as f:
    original_model = dill.load(f)

sample_text = "This is a test input to check inference speed."

#########################################
# 5. Measure Speed BEFORE Transform (One Time)
#########################################
speed_before = measure_inference_speed(
    original_model, tokenizer, sample_text, device='cuda:0'
)
print(f"Inference speed (original, BEFORE any MLA transform): {speed_before:.6f} seconds/run")

#########################################
# 6. Loop Over kv_heads = 1..12
#########################################
results = []  # to store (kv_heads, speed_after_transform, speed_after_finetune, ce, ppl)

def apply_mla_transform(model, kv_heads=2):
    """
    Transform the GPT2 MHA to MGQA with specified kv_heads grouping.
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

for kv in range(1, 13):
    print("="*60)
    print(f"Testing MLA transform with kv_heads = {kv}")

    # 1) Reload the original model (so each kv config starts from baseline)
    with open(model_path, "rb") as f:
        tmp_model = dill.load(f)

    # 2) Apply MLA transform
    tmp_model = apply_mla_transform(tmp_model, kv_heads=kv)

    # 3) Measure speed after transform (before finetuning)
    speed_after_transform = measure_inference_speed(
        tmp_model, tokenizer, sample_text, device='cuda:0'
    )
    print(f" Speed AFTER MLA transform (kv_heads={kv}, no finetuning): {speed_after_transform:.6f} s/run")

    # 4) Finetune
    trainer = get_trainer(
        model=tmp_model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,  # Use a small # for quick example
    )
    print(" Starting finetuning...")
    trainer.train()
    print(" Finetuning complete.")

    # 5) Evaluate cross-entropy & perplexity AFTER finetuning
    ce_after, ppl_after = evaluate_ce_and_ppl(trainer, dataset["test"])
    print(f" Cross-Entropy (kv_heads={kv}, after finetuning): {ce_after:.6f}")
    print(f" Perplexity    (kv_heads={kv}, after finetuning): {ppl_after:.6f}")

    # 6) Measure inference speed after finetuning
    speed_after_finetune = measure_inference_speed(
        tmp_model, tokenizer, sample_text, device='cuda:0'
    )
    print(f" Inference speed AFTER finetuning (kv_heads={kv}): {speed_after_finetune:.6f} s/run")

    # Store results
    results.append((kv, speed_after_transform, speed_after_finetune, ce_after, ppl_after))

#########################################
# 7. Print Summary of All kv_heads Tests
#########################################
print("\n\n===== Summary of MLA Transform Tests (kv_heads = 1..12) =====")
print("kv_heads | Speed After Transform (s/run) | Speed After Finetune (s/run) | Cross-Entropy | Perplexity")
for (kv, sp_trans, sp_ft, ce_val, ppl_val) in results:
    print(f"{kv:>7d} | {sp_trans:>28.6f} | {sp_ft:>28.6f} | {ce_val:>13.6f} | {ppl_val:>10.6f}")