import torch

def printer(variable, name):
    # print("==== Intermediate Sign (16-bit) ====")
    sign_u16 = variable.view(torch.uint16)
    for i, val in enumerate(sign_u16.flatten()):
        print(f"{name}[{i}]: {val.item():016b}")

def dequantize1d_simulated_with_binary_print(
    input: torch.Tensor, 
    scale: torch.Tensor, 
    group_size: int
) -> torch.Tensor:
    assert input.ndim == 1, "Input tensor must be 1D"
    assert scale.ndim == 1, "Scale tensor must be 1D"
    input = input.view(torch.int8)
    scale = scale.view(torch.uint8)

    numel = input.numel()
    num_groups = numel // group_size

    mantissa = input.reshape(num_groups, group_size)
    scales = scale.reshape(num_groups, 1)

    sign = (mantissa & 0x80).to(torch.int16) << 8
    exp = scales.to(torch.int16) << 7
    mantissa_abs = mantissa.abs()
    printer((mantissa_abs & 0x3F).view(torch.uint8).to(torch.int16), "mantissa_abs")
    frac = ((mantissa_abs & 0x3F) << 1).view(torch.uint8).to(torch.int16)
    printer(sign, "sign")
    printer(exp, "exp")
    printer(frac, "frac")


    output = (sign | exp | frac).view(torch.bfloat16)
    printer(output, "output")

    dont_need_bias = (mantissa_abs & 0x40).bool()
    print("dont_need_bias: ", dont_need_bias)

    bias = (sign | exp | 0x00).view(torch.bfloat16)
    printer(bias, "bias")

    output = torch.where(dont_need_bias, output, output - bias).flatten()
    printer(output, "output")
    print(output)
    return output

single_int8_value = torch.tensor([63], dtype=torch.int8)   # e.g. +63
single_scale_value = torch.tensor([127], dtype=torch.uint8) # e.g. scale=10
group_size = 1

output_single = dequantize1d_simulated_with_binary_print(
    single_int8_value, 
    single_scale_value, 
    group_size
)