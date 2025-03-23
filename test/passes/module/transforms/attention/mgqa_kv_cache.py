checkpoint = "openai-community/gpt2"
tokenizer_checkpoint = "openai-community/gpt2"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from chop.passes.module.transforms import quantize_module_transform_pass, attention_transform_pass
from chop.passes.module.transforms.attention.attention_transform_helper import MGQAWrapper

# Load GPT-2 model and tokenizer
model = AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
tokenizer.pad_token = tokenizer.eos_token

from transformers.models.gpt2.modeling_gpt2 import (
    GPT2SdpaAttention,
    GPT2Block,
)

# Create a simple test case
test_cases = [
    "Hallo, wie geht es dir?",
    "Das Wetter ist heute schön."
]

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
def evaluate_kv_cache(model, tokenizer, text, max_length=100):
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

def evaluate_kv_cache_custom(model, tokenizer, text, max_length=100):
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

    kv_cache_size = 0.0
    for n, m in model.named_modules():
        module = MGQAWrapper
        if isinstance(m, module):
            kv_cache_size += m.kv_cache

    print(f"Total KV cache size: {kv_cache_size:.2f} MB")

    return outputs


if __name__ == "__main__":
    sample_text = "Dies ist ein Beispieltext für die Übersetzung Dies ist ein Beispieltext für die Übersetzung Dies ist ein Beispieltext für die Übersetzung."
    for kv_heads in [1, 2, 4, 8]:
        pass_args = {
            "by": "type",
            "gpt2spda": {
                "config": {
                    "name": "mgqa",
                    "kv_heads": 2,
                }
            },
        }
        # kv_cache_outputs = evaluate_kv_cache(model, tokenizer, sample_text)
        mgqa_model = spda_transform_pass(model, pass_args)
        kv_cache_outputs = evaluate_kv_cache_custom(mgqa_model, tokenizer, sample_text)

    # for text in test_cases:
    #     generated_text = generate_text(model, tokenizer, text)
    # print(f"\nInput: {text}")
    # print(f"Generated: {generated_text}")