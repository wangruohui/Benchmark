import os

from vllm import LLM
from vllm.sampling_params import SamplingParams

import torch
import time

s = SamplingParams(ignore_eos=False, max_tokens=66, temperature=0)
prompts = ["Hello, my name is", "The capital of France is"]  # Sample prompts.
llm = LLM(
    model="/nvme/wangruohui/hfcache/hub/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348"
)  # Create an LLM.
outputs0 = llm.generate(prompts, s)  # Generate texts from the prompts.
outputs1 = llm.generate(prompts, s)  # Generate texts from the prompts.

# make sure no sample happens
for o1, o2 in zip(outputs0, outputs1):
    assert len(o1.outputs[0].token_ids) == 6
    assert len(o2.outputs[0].token_ids) == 6

    assert o1.outputs[0].token_ids == o2.outputs[0].token_ids

s = SamplingParams(ignore_eos=True, max_tokens=2048, temperature=0)

for bs in [1, 4, 8, 16, 24, 32, 40, 48]:
    print("bs = ", bs)
    torch.cuda.synchronize()
    start = time.monotonic()
    prompt_token_ids = [[x+99] for x in range(bs)]
    outputs = llm.generate(
        prompt_token_ids=prompt_token_ids, sampling_params=s
    )  # Generate texts from the prompts.
    torch.cuda.synchronize()
    end = time.monotonic()

    print(end - start)
