import csv
import os
import time

import torch
from vllm import LLM
from vllm.sampling_params import SamplingParams

s = SamplingParams(ignore_eos=False, max_tokens=66, temperature=0)
prompts = ["Hello, my name is", "The capital of France is"]  # Sample prompts.
llm = LLM(
    model="/nvme/wangruohui/hfcache/hub/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348"
    # model = "/nvme/wangruohui/llama-65b-hf",
    # tensor_parallel_size=8
)  # Create an LLM.
outputs0 = llm.generate(prompts, s)  # Generate texts from the prompts.
outputs1 = llm.generate(prompts, s)  # Generate texts from the prompts.

# make sure no sample happens
for o1, o2 in zip(outputs0, outputs1):
    assert len(o1.outputs[0].token_ids) == 66
    assert len(o2.outputs[0].token_ids) == 66

    assert o1.outputs[0].token_ids == o2.outputs[0].token_ids


for bs in [1, 16, 32, 64]:
    print("bs = ", bs)
    for pl, gl in [(128, 512), (512, 512), (1024, 1024), (1, 2048)]:
        tl = pl + gl
        s = SamplingParams(ignore_eos=True, max_tokens=tl, temperature=0)

        prompt_token_ids = torch.randint(10, 10000, size=(bs, pl)).tolist()

        # exclude preparation time
        for token_id in prompt_token_ids:
            llm._add_request(
                prompt=None,
                prompt_token_ids=token_id,
                sampling_params=s,
            )

        torch.cuda.synchronize()
        start = time.monotonic()
        # outputs = llm.generate(
        #     prompt_token_ids=prompt_token_ids, sampling_params=s
        # )  # Generate texts from the prompts.

        llm._run_engine(use_tqdm=True)
        torch.cuda.synchronize()
        end = time.monotonic()
        total_time = end - start    # time in s -> time in ms
        line = ["vLLM", bs, pl, gl, tl, 0, 0, 0, 0, total_time*1000]
        file = "b2tmp.csv"
        csv.writer(open(file, "a")).writerow(line)
