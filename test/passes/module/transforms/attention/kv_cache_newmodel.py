import torch
import transformers

# Set model name
# allenai/longformer-base-4096
model_name = 'mosaicml/mpt-7b-storywriter'
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
from torch.nn import MultiheadAttention

# Configure the model with optimizations
config = transformers.AutoConfig.from_pretrained(
    model_name, 
    trust_remote_code=True
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    # torch_dtype=torch.bfloat16,  # Uncomment for BF16 precision to save memory
    trust_remote_code=True
)
print(model)