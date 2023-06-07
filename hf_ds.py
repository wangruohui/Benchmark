import torch

from utils import timehere as t


def init_model():
    from transformers import AutoModelForCausalLM
    import deepspeed

    torch.set_default_tensor_type(torch.cuda.HalfTensor)

    t()
    model_id = "decapoda-research/llama-7b-hf"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    t("load model")

    ds_model = deepspeed.init_inference(
        model=model,  # Transformers models
        mp_size=1,  # Number of GPU
        dtype=torch.float16,  # dtype of the weights (fp16)
        replace_method="auto",  # Lets DS autmatically identify the layer to replace
        replace_with_kernel_inject=True,  # replace the model with the kernel injector
        max_out_tokens=2048,
    )
    print(f"model is loaded on device {ds_model.module.device}")

    return ds_model