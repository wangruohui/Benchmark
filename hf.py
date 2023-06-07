import torch

from utils import timehere


def init_model():
    from transformers import AutoModelForCausalLM

    torch.set_default_tensor_type(torch.cuda.HalfTensor)

    timehere()
    model_id = "decapoda-research/llama-7b-hf"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    timehere("load model")

    return model
