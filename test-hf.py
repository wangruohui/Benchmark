import torch

from utils import timehere


def init_model():
    from transformers import AutoModelForCausalLM

    torch.set_default_tensor_type(torch.cuda.HalfTensor)

    timehere()
    model_id = "/nvme/wangruohui/llama-65b-hf"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    timehere("load model")

    return model


if __name__ == "__main__":
    model = init_model()
