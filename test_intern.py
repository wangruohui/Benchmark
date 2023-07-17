import deepspeed
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from mystat import stat
from timing import TimingStreamer, timehere
from wrappers import init_deepspeed

from contextlib import nullcontext


def default_chat(model_path="internlm", dtype=torch.float16):
    timehere()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True
    )
    model = model.eval()
    # model = model.cuda()
    timehere("Load model")
    print(model.__class__)

    response, history = model.chat(tokenizer, "hello", history=[])
    print(response)
    timehere("First response")

    response, history = model.chat(
        tokenizer,
        "please provide three suggestions about time management",
        history=history,
    )
    print(response)
    timehere("Second response")


def default_generate(model_path="internlm", dtype=torch.float16):
    timehere()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True
    )
    model = model.eval()
    timehere("Load model")

    print(model.__class__)

    prompt = "<|User|>:自我介绍一下吧<eoh>"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    timehere()
    output = model.generate(input_ids)
    timehere("Generate")

    print(output)
    print(tokenizer.decode(output[0].tolist()))

    # <s><|User|>:自我介绍一下吧<eoh>
    # <|Bot|>:你好，我是一名
    return model.model, tokenizer, input_ids

def deepspeed_generate(model_path="internlm", dtype=torch.float16):
    timehere()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True
    )
    model = model.eval()
    orig_model = model

    timehere("Load model")

    print(deepspeed.__file__)
    print(model.__class__)
    print(model.lm_head.weight.dtype)

    ds_model = deepspeed.init_inference(
        model=model.model,  # Transformers models
        tensor_parallel=dict(tp_size=1),  # Number of GPU
        dtype=dtype,  # dtype of the weights (fp16)
        max_out_tokens=1024,
        replace_with_kernel_inject=True,  # replace the model with the kernel injector
    )

    model.model = ds_model
    print(model.lm_head.weight.dtype)

    prompt = "<|User|>:自我介绍一下吧<eoh>"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()

    timehere()
    output = model.generate(input_ids, max_new_tokens=20)
    timehere("Generate")

    print(output)
    print(tokenizer.decode(output[0].tolist()))


    # model = model
    # <s><|User|>:自我介绍一下吧<eoh>
    # <|Bot|>:你好，我是一名

    # model = model.model, model.model = ds_model
    # <s><|User|>:自我介绍一下吧<eoh>
    # <|Bot|>:你好，我是一名


    return ds_model, tokenizer, input_ids


def deepspeed_chat(model_path="internlm", dtype=torch.float16, with_prof=False):
    timehere()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True
    )
    model = model.eval()
    # model = model.cuda()
    timehere("Load model")
    print(deepspeed.__file__)
    print(model.__class__)

    ds_model = deepspeed.init_inference(
        model=model.model,  # Transformers models
        tensor_parallel=dict(tp_size=1),  # Number of GPU
        dtype=torch.float16,  # dtype of the weights (fp16)
        max_out_tokens=1024,
        replace_with_kernel_inject=True,  # replace the model with the kernel injector
    )

    model.model = ds_model

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        with_stack=True,
    ) if with_prof else nullcontext() as prof:
        response, history = model.chat(tokenizer, "hello hello", history=[])
        print(response)
        timehere("First response")

    if with_prof:
        return prof.export_chrome_trace("intern_deepspeed.json")

    response, history = model.chat(
        tokenizer,
        "please provide three suggestions about time management management management",
        history=history,
    )
    print(response)
    timehere("Second response")


if __name__ == "__main__":
    device = 0

    torch.set_default_device(device)
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)

    default_generate()
    # deepspeed_generate()
    deepspeed_chat(with_prof=False)

# model.model = ds_model

# response, history = model.chat(tokenizer, "hello", history=[], do_sample=False)
# print(response)

# response, history = model.chat(
#     tokenizer,
#     "please provide three suggestions about time management",
#     history=history,
#     do_sample=False
# )
# print(response)

# exit()

# ts = TimingStreamer()
# torch.cuda.synchronize()
# fake_inputs = torch.arange(2, device=device).repeat(1, 1) + 66
# fake_inputs = fake_inputs.to(device)
# fake_outputs = model.generate(
#     fake_inputs,
#     GenerationConfig(max_length=6, do_sample=False, eos_token_id=[-1]),
#     streamer=ts,
# )

# raw = ts.raw_times()
# print(stat(raw))

# with profile(
#     activities=[
#         ProfilerActivity.CPU,
#         ProfilerActivity.CUDA,
#     ],
# ) as prof:
#     # response, history = model.stream_chat(tokenizer, "hello", history=[])
#     # print(response)
#     fake_inputs = torch.arange(2, device=device).repeat(1, 1) + 66
#     fake_inputs = fake_inputs.to(device)
#     fake_outputs = model.generate(
#         fake_inputs,
#         GenerationConfig(max_length=6, do_sample=False, eos_token_id=[-1]),
#     )

# prof.export_chrome_trace("tmp2.json")

# bs = 16
# prompt_len = 1
# total_len = 2048

# name_or_path = "decapoda-research/llama-7b-hf"

# # torch.set_default_tensor_type(torch.cuda.HalfTensor)
# # model = AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=torch.float16)

# # ds_model = deepspeed.init_inference(
# #     model=model,  # Transformers models
# #     mp_size=mp_size,  # Number of GPU
# #     dtype=torch.float16,  # dtype of the weights (fp16)
# #     replace_method="auto",  # Lets DS autmatically identify the layer to replace
# #     replace_with_kernel_inject=True,  # replace the model with the kernel injector
# #     max_out_tokens=2049,
# # )
# # print(f"model is loaded on device {ds_model.module.device}")

# benckmarker = init_deepspeed("decapoda-research/llama-7b-hf", mp_size=1)
# ds_model = benckmarker.model

# for _ in range(3):
#     fake_inputs = torch.arange(prompt_len).repeat(bs, 1)
#     ts = TimingStreamer()
#     torch.cuda.synchronize()
#     fake_outputs = ds_model.generate(fake_inputs, max_length=total_len, streamer=ts)
#     raw = ts.raw_times()
#     print(stat(raw))


# # benckmarker = init_deepspeed("decapoda-research/llama-7b-hf")
# # fake_inputs = torch.arange(prompt_len).repeat(bs, 1)
# # ts = TimingStreamer()
# # fake_outputs = benckmarker.model.generate(
# #     fake_inputs, max_length=total_len, streamer=ts
# # )

# # print(stat(ts.raw_times()))
# # fake_inputs = torch.arange(prompt_len).repeat(bs, 1)
# # ts = TimingStreamer()
# # fake_outputs = benckmarker.model.generate(
# #     fake_inputs, max_length=total_len, streamer=ts
# # )

# # print(stat(ts.raw_times()))

# # fake_inputs = torch.arange(prompt_len).repeat(bs, 1)
# # ts = TimingStreamer()
# # fake_outputs = benckmarker.model.generate(
# #     fake_inputs, max_length=total_len, streamer=ts
# # )

# # print(stat(ts.raw_times()))

# # fake_inputs = torch.arange(prompt_len).repeat(bs, 1)
# # ts = TimingStreamer()
# # fake_outputs = benckmarker.model.generate(
# #     fake_inputs, max_length=total_len, streamer=ts
# # )

# # print(stat(ts.raw_times()))


# # times = benckmarker.benchmark_and_time(bs, prompt_len, total_len)
# # times = benckmarker.benchmark_and_time(bs, prompt_len, total_len)
