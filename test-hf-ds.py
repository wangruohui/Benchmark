import deepspeed
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import timehere as t
import os

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))


def init_model():
    t()
    # model_id = "/nvme/wangruohui/llama-65b-hf"
    # model_id = "/nvme/wangruohui/llama-7b-hf"
    # model_id = "/share_140/InternLM/7B/0703/hf"
    model_id = "llama-2-7b-chat"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    t("load model")

    # return model, tokenizer
    print(model)
    ds_model = deepspeed.init_inference(
        model=model,  # Transformers models
        mp_size=1,  # Number of GPU
        dtype=torch.float16,  # dtype of the weights (fp16)
        replace_with_kernel_inject=True,  # replace the model with the kernel injector
        max_out_tokens=4096,
    )
    print(f"model is loaded on device {ds_model.module.device}")

    return ds_model, tokenizer


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.cuda.HalfTensor)

    model, tokenizer = init_model()

    prompt = "I believe the meaning of life is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    print(input_ids)

    torch.set_grad_enabled(False)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
        record_shapes=True,
    ) as prof:
        out = model.generate(input_ids.repeat(1, 1), max_length=10)

    if LOCAL_RANK == 0:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        prof.export_chrome_trace("test-hf-ds.json")

    out = model.generate(input_ids.repeat(1, 1), max_length=48)

    print(out)
    print(tokenizer.decode(out[0].tolist()))
