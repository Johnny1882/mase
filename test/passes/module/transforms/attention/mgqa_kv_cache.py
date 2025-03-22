checkpoint = "openai-community/gpt2"
tokenizer_checkpoint = "openai-community/gpt2"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from chop.passes.module.transforms import quantize_module_transform_pass, attention_transform_pass

# Load GPT-2 model and tokenizer
model = AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
tokenizer.pad_token = tokenizer.eos_token

# Create a simple test case
test_cases = [
    "Hallo, wie geht es dir?",
    "Das Wetter ist heute schön."
]

def spda_transform_pass(model):
    pass_args = {
        "by": "type",
        "gpt2spda": {
            "config": {
                "name": "mgqa",
                # "kv_heads": 2,
            }
        },
    }
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
def evaluate_kv_cache(model, tokenizer, text, max_length=50):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"], 
            max_length=max_length, 
            use_cache=True, 
            output_attentions=True, 
            output_hidden_states=True,
            return_dict_in_generate=True
        )
    return outputs

if __name__ == "__main__":
    sample_text = "Dies ist ein Beispieltext für die Übersetzung."
    # kv_cache_outputs = evaluate_kv_cache(model, tokenizer, sample_text)
    model = spda_transform_pass(model)
    # kv_cache_outputs = evaluate_kv_cache(model, tokenizer, sample_text)
    for text in test_cases:
        generated_text = generate_text(model, tokenizer, text)
    print(f"\nInput: {text}")
    print(f"Generated: {generated_text}")