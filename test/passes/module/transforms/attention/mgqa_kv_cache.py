checkpoint = "openai-community/gpt2"
tokenizer_checkpoint = "openai-community/gpt2"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from chop.passes.module.transforms import quantize_module_transform_pass, attention_transform_pass
from chop.passes.module.transforms.attention.attention_transform_helper import MGQAWrapper
from datasets import load_dataset
import random

# Load GPT-2 model and tokenizer
model = AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
tokenizer.pad_token = tokenizer.eos_token

from transformers.models.gpt2.modeling_gpt2 import (
    GPT2SdpaAttention,
    GPT2Block,
)

def get_samples(dataset, num_samples=10, seed=42):
    """Get random samples from WikiText-2 dataset with max length 1024."""
    import random
    random.seed(seed)
    
    data = dataset["train"]
    indices = random.sample(range(len(data)), min(num_samples, len(data)))
    
    return [data[idx]["text"][:1024] for idx in indices]


def spda_transform_pass(model, pass_args):
    model, _ = attention_transform_pass(model, pass_args)
    return model

# Function to perform autoregressive generation
def generate_text(model, tokenizer, text, max_length=50):
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        num_beams=5,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Evaluate kv cache used
def evaluate_kv_cache(model, tokenizer, text, max_length=200):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"], 
            max_length=max_length, 
            use_cache=True, 
            # output_attentions=True, 
            # output_hidden_states=True,
            return_dict_in_generate=True
        )
    
    past_key_values = outputs.past_key_values

    # Print basic KV cache info
    print("\n----- KV Cache Info -----")
    if past_key_values:
        num_layers = len(past_key_values)
        layer_0_keys = past_key_values[0][0]  # First layer's keys
        
        # Calculate total size
        total_bytes = 0
        for layer_idx, (k, v) in enumerate(past_key_values):
            layer_bytes = k.nelement() * k.element_size() + v.nelement() * v.element_size()
            total_bytes += layer_bytes
        
        print(f"Number of layers: {num_layers}")
        print(f"Cache shape (Layer 0 Keys): {layer_0_keys.shape}")
        print(f"Total KV cache size: {total_bytes / (1024 * 1024):.2f} MB")

    return outputs

def evaluate_kv_cache_custom(model, tokenizer, text, max_new_tokens=100):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"], 
            use_cache=True, 
            # output_attentions=True, 
            # output_hidden_states=True,
            return_dict_in_generate=True
        )

    kv_cache_size = 0.0
    for n, m in model.named_modules():
        module = MGQAWrapper
        if isinstance(m, module):
            kv_cache_size += m.kv_cache

    print(f"current KV cache size: {kv_cache_size:.2f} MB")

    return kv_cache_size


if __name__ == "__main__":
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    num_samples = 2
    test_cases = get_samples(dataset, num_samples = num_samples, seed=42)

    for kv_heads in [1, 2, 3, 4, 6, 12]:
        pass_args = {
            "by": "type",
            "gpt2spda": {
                "config": {
                    "name": "mgqa",
                    "kv_heads": kv_heads,
                }
            },
        }
        mgqa_model = spda_transform_pass(model, pass_args)
        
        for i, review in enumerate(test_cases):
            total_kv_cache = evaluate_kv_cache_custom(mgqa_model, tokenizer, review)
        print(f"kv heads = {kv_heads} for {num_samples} samples: {total_kv_cache}, average: {total_kv_cache/num_samples}")
