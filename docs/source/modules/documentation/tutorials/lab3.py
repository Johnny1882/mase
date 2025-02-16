import os
import logging
import traceback
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = "cuda:0"

save_path = "/adls/mase/docs/source/modules/documentation/tutorials/"


from transformers import AutoModel
from pathlib import Path
import dill
from chop.tools import get_tokenized_dataset

checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"

save_path = "/adls/mase/docs/source/modules/documentation/tutorials/cw_submission/models"    
model = AutoModel.from_pretrained(checkpoint)
with open(f"{Path.home()}{save_path}/tutorial_5_best_model.pkl", "rb") as f:
    base_model = dill.load(f)

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)




import torch
from chop.nn.quantized.modules.linear import (
    LinearInteger,
    LinearMinifloatDenorm,
    LinearMinifloatIEEE,
    LinearLog,
    LinearBlockFP,
    LinearBlockMinifloat,
    LinearBlockLog,
    LinearBinary,
    LinearBinaryScaling,
    LinearBinaryResidualSign,
)
from chop.tools.utils import deepsetattr
from copy import deepcopy
from chop.tools import get_trainer
import random


search_space = {
    "linear_layer_choices": [
        torch.nn.Linear,
        LinearInteger,
        LinearMinifloatDenorm,
        LinearMinifloatIEEE,
        LinearLog,
        LinearBlockFP,
        LinearBlockMinifloat,
        LinearBlockLog,
        LinearBinary,
        LinearBinaryScaling,
        LinearBinaryResidualSign,
    ],
    "width_choices": [8, 16, 32],
    "frac_width_choices": [2, 4, 8],
    # "block_size_choices": [[1], [2], [4]],
}

################################################################################
## construct model function
################################################################################
def construct_model(trial):

    # Fetch the model
    trial_model = deepcopy(base_model)

    # Quantize layers according to optuna suggestions
    for name, layer in trial_model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            new_layer_cls = trial.suggest_categorical(
                f"{name}_type",
                search_space["linear_layer_choices"],
            )

            if new_layer_cls == torch.nn.Linear:
                continue

            kwargs = {
                "in_features": layer.in_features,
                "out_features": layer.out_features,
            }

            chosen_width = trial.suggest_categorical(
                    f"{name}_width",
                    search_space["width_choices"]
                )
            chosen_frac_width = trial.suggest_categorical(
                f"{name}_frac_width",
                search_space["frac_width_choices"]
            )

            # If the chosen layer is integer, define the low precision config
            if new_layer_cls == LinearInteger:

                kwargs["config"] = {
                    "data_in_width": chosen_width,
                    "data_in_frac_width": chosen_frac_width,
                    "weight_width": chosen_width,
                    "weight_frac_width": chosen_frac_width,
                    "bias_width": chosen_width,
                    "bias_frac_width": chosen_frac_width,
                }

            elif new_layer_cls == LinearMinifloatDenorm:
                # Example: Minifloat (denormal) config
                exponent_bits = chosen_width - chosen_frac_width - 1
                bypass = exponent_bits < 1

                kwargs["config"] = {
                    # Optional bypass flag — if True, no quantization is applied
                    "bypass": bypass,
                    "weight_width": chosen_width,
                    "weight_exponent_width": chosen_frac_width,
                    "weight_exponent_bias": exponent_bits,
                    "data_in_width": chosen_width,
                    "data_in_exponent_width": chosen_frac_width,
                    "data_in_exponent_bias": exponent_bits,
                    "bias_width": chosen_width,
                    "bias_exponent_width": chosen_frac_width,
                    "bias_exponent_bias": exponent_bits,
                }

            elif new_layer_cls == LinearMinifloatIEEE:
                exponent_bits = chosen_width - chosen_frac_width - 1
                bypass = exponent_bits < 1

                kwargs["config"] = {
                    # If True, skip all quantization entirely
                    "bypass": bypass,
                    "weight_width": chosen_width,
                    "weight_exponent_width": exponent_bits,
                    "weight_exponent_bias": "None",
                    "data_in_width": chosen_width,
                    "data_in_exponent_width": exponent_bits,
                    "data_in_exponent_bias": "None",
                    "bias_width": chosen_width,
                    "bias_exponent_width": exponent_bits,
                    "bias_exponent_bias": "None",
                }

            elif new_layer_cls == LinearLog:
                # Example: Log domain quant
                kwargs["config"] = {
                    "bypass": False,
                    "weight_width": chosen_width,
                    "weight_exponent_bias": "None",
                    "data_in_width": chosen_width,
                    "data_in_exponent_bias": "None",
                    "bias_width": chosen_width,
                    "bias_exponent_bias": "None",
                }

            elif new_layer_cls == LinearBlockFP:
                exponent_bits = chosen_width - chosen_frac_width - 1
                bypass = exponent_bits < 1
                chosen_block_size = 2
                # chosen_block_size = trial.suggest_categorical(
                #     f"{name}_block_size",
                #     search_space["block_size_choices"]
                # )
                
                kwargs["config"] = {
                    # Optional bypass flag to skip quantization entirely
                    "bypass": False,

                    "weight_width": chosen_width,           # total bits (sign + exponent + fraction)
                    "weight_exponent_width": exponent_bits,  # bits devoted to exponent
                    "weight_exponent_bias": None,   # numeric exponent bias (typical for 4-bit exponent)
                    "weight_block_size": chosen_block_size,     # group size (block) along which to share exponent

                    "data_in_width": chosen_width,
                    "data_in_exponent_width": exponent_bits,
                    "data_in_exponent_bias": None,
                    "data_in_block_size": chosen_block_size,     
                    # Whether to ignore the first dimension (batch dimension) 
                    # when grouping values in blocks
                    "data_in_skip_first_dim": True,  

                    "bias_width": chosen_width,
                    "bias_exponent_width": exponent_bits,
                    "bias_exponent_bias": None,
                    "bias_block_size": chosen_block_size,        # often 1 for biases (they’re smaller in size)
                }

            elif new_layer_cls == LinearBlockMinifloat:
                exponent_bits = chosen_width - chosen_frac_width - 1
                bypass = exponent_bits < 1
                chosen_block_size = 2
                # chosen_block_size = trial.suggest_categorical(
                #     f"{name}_block_size",
                #     search_space["block_size_choices"]
                # )
                
                kwargs["config"] = {
                    # Optional: if True, skip all quantization and pass values as-is
                    "bypass": False,

                    "weight_width": chosen_width,
                    "weight_exponent_width": exponent_bits,
                    "weight_exponent_bias_width": 1,
                    "weight_block_size": chosen_block_size,

                    "data_in_width": chosen_width,
                    "data_in_exponent_width": exponent_bits,
                    "data_in_exponent_bias_width": 1,
                    "data_in_block_size": chosen_block_size,
                    "data_in_skip_first_dim": True,

                    "bias_width": chosen_width,
                    "bias_exponent_width": exponent_bits,
                    "bias_exponent_bias_width": 1,
                    "bias_block_size": chosen_block_size,
                }

            elif new_layer_cls == LinearBlockLog:
                exponent_bits = chosen_width - chosen_frac_width - 1
                bypass = exponent_bits < 1
                chosen_block_size = 2
                # chosen_block_size = trial.suggest_categorical(
                #     f"{name}_block_size",
                #     search_space["block_size_choices"]
                # )
                kwargs["config"] = {
                    "bypass": False,

                    "weight_width": chosen_width,
                    "weight_exponent_bias_width": exponent_bits,  # bits to store the exponent bias for each block
                    "weight_block_size": chosen_block_size,          # each block has 2 elements sharing the same exponent bias

                    "data_in_width": chosen_width,
                    "data_in_exponent_bias_width": exponent_bits,
                    "data_in_block_size": chosen_block_size,
                    # Often skip first dimension (batch) to block only within each sample/channel
                    "data_in_skip_first_dim": True,  

                    "bias_width": chosen_width,
                    "bias_exponent_bias_width": exponent_bits,
                    "bias_block_size": chosen_block_size,
                }

            elif new_layer_cls == LinearBinary:
                # Example: Basic binary quant
                kwargs["config"] = {
                    "weight_stochastic": True,
                    "weight_bipolar": True,
                }

            elif new_layer_cls == LinearBinaryScaling:
                # Example: Binary scaling with a trainable scaling parameter
                kwargs["config"] = {
                    "data_in_stochastic": True,  
                    "bias_stochastic": False,   
                    "weight_stochastic": True,   

                    "data_in_bipolar": True,      
                    "bias_bipolar": False,        
                    "weight_bipolar": True,  

                    "binary_training": False,     
                }

            elif new_layer_cls == LinearBinaryResidualSign:
                # Example: Residual sign approach
                kwargs["config"] = {
                    "bypass": False,
                    "data_in_stochastic": True,
                    "bias_stochastic": False,
                    "weight_stochastic": True,
                    "data_in_bipolar": True,
                    "bias_bipolar": False,
                    "weight_bipolar": True,
                    "binary_training": True,
                    "data_in_levels": 3,
                    "data_in_residual_sign": True,
                }

            # Create the new layer (copy the weights)
            new_layer = new_layer_cls(**kwargs)
            new_layer.weight.data = layer.weight.data

            # Replace the layer in the model
            deepsetattr(trial_model, name, new_layer)

    return trial_model


################################################################################
## objective function
################################################################################

def objective(trial):

    # Define the model
    model = construct_model(trial)

    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    trial.set_user_attr("model", model)

    return eval_results["eval_accuracy"]


# if __name__ == "main":
if True:
    from optuna.samplers import GridSampler, RandomSampler, TPESampler
    import optuna

    sampler = RandomSampler()

    study = optuna.create_study(
        direction="maximize",
        study_name="bert-tiny-nas-study",
        sampler=sampler,
    )

    study.optimize(
        objective,
        n_trials=30,
        timeout=60 * 60 * 24,
    )
    
    import random
    import json

    trial_results = []
    for trial in study.trials:
        trial_data = {
            "trial_number": trial.number,
            "params": trial.params,
            "accuracy": trial.value,   # or trial.values if multi-objective
        }
        trial_results.append(trial_data)

    # Save the results to a JSON file
    with open("trial_results.json", "w") as f:
        json.dump(trial_results, f, indent=2)

    print("Results have been written to trial_results.json.")


######################################################################
## Plot the data
######################################################################

import matplotlib.pyplot as plt
import numpy as np

# Gather all trials
trials = study.trials

# 1. Build a dictionary: precision_name -> list of (trial_number, best_accuracy_so_far)
precision_curves = {}

for t in trials:
    if t.value is None:
        continue  # skip failed or incomplete trials

    chosen_keys = [
        k for k in t.params.keys() 
        if k.endswith("_type")
    ]
    if not chosen_keys:
        continue
    
    # For demonstration, let's just use the first linear layer's type
    chosen_type = str(t.params[chosen_keys[0]])
    
    # Initialize if not present
    if chosen_type not in precision_curves:
        precision_curves[chosen_type] = []
    
    # The "best accuracy so far" for that precision is the max among all prior trials of the same type
    old_best = precision_curves[chosen_type][-1][1] if precision_curves[chosen_type] else 0.0
    new_best = max(old_best, t.value)
    
    precision_curves[chosen_type].append((t.number, new_best))

    # 2. Plot each precision’s curve
    plt.figure(figsize=(8,6))

    for prec_type, data in precision_curves.items():
        data_sorted = sorted(data, key=lambda x: x[0])  # sort by trial number
        x_vals = [d[0] for d in data_sorted]
        y_vals = [d[1] for d in data_sorted]
        plt.plot(x_vals, y_vals, label=prec_type)

    plt.xlabel("Trial Number")
    plt.ylabel("Max Accuracy So Far")
    plt.title("Optuna Search: Accuracy by Precision")
    plt.legend()
    plt.show()
