
import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from turbollm.models.llama import ModelArgs, Transformer, Tokenizer

from utils import timehere


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
        initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
):
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    # torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    # generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return model


def init_generator():
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    model = load(
        "/nvme/wangruohui/Downloads/LLaMA/7B",
        "/nvme/wangruohui/Downloads/LLaMA/tokenizer.model",
        local_rank,
        world_size,
        2048,
        64,
    )

    return model

def temp():
    bsz = len(prompts)
    params = self.model.params
    assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

    prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

    min_prompt_size = min([len(t) for t in prompt_tokens])
    max_prompt_size = max([len(t) for t in prompt_tokens])

    total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

    tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t).long()
    input_text_mask = tokens != self.tokenizer.pad_id
    start_pos = min_prompt_size
    prev_pos = 0
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []
    warmups = 100
    count = 0

    static_input = (tokens[:, prev_pos:start_pos], torch.tensor(prev_pos))
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            self.model(*static_input)
    torch.cuda.current_stream().wait_stream(s)



    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_output = self.model.forward(*static_input)


    for cur_pos in range(start_pos, total_len):
        count += 1
        starter.record()
        # static_input[0].copy_(tokens[:, prev_pos:cur_pos])
        # static_input[1].copy_(torch.tensor(prev_pos))
        # g.replay()
        # logits = static_output
        logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        # only replace token if prompt has already been generated
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        tokens[:, cur_pos] = next_token
        prev_pos = cur_pos

        ender.record()
        torch.cuda.synchronize() # 等待GPU任务完成
        curr_time = starter.elapsed_time(ender)
        print(f'iter-{count}: {curr_time}')
        if count > warmups:
            timings.append(curr_time)

    stats = np.asarray(timings, dtype=np.float32)
    print(f'avg: {stats.sum()/len(stats)}; tokens/s: {1000.0 * len(stats) / stats.sum()}')

    decoded = []
    for i, t in enumerate(tokens.tolist()):
        # cut to max gen len
        t = t[: len(prompt_tokens[i]) + max_gen_len]
        # cut to eos tok if any
        try:
            t = t[: t.index(self.tokenizer.eos_id)]
        except ValueError:
            pass
        decoded.append(self.tokenizer.decode(t))
    return decoded
