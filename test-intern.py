# import deepspeed
import torch
# from transformers import AutoModelForCausalLM

# from mystat import stat
# from timing import TimingStreamer
# from timing import timehere as t
# from wrappers import init_deepspeed

from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "InternLM/7B/0705/5299"

torch.set_default_device("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
model = model.cuda()
model = model.eval()
print(model.__class__)

response, history = model.stream_chat(tokenizer, "hello", history=[])

exit()

with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ],
    with_stack=True,
) as prof:
    response, history = model.stream_chat(tokenizer, "hello", history=[])
    # print(response)

    # response, history = model.stream_chat(
    #     tokenizer, "please provide three suggestions about time management", history=history
    # )
    # print(response)

prof.export_chrome_trace("intern.json")

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
