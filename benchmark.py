import torch
from torch.cuda import Event
from utils import timehere
import csv
import os


def benchmark_generator(model, bs, prompt_len, total_len):
    """
    Benchmark the model on a sequence of length seq_len

    Args:
    """

    tokens = torch.arange(total_len).repeat(bs, 1)

    torch.cuda.synchronize()

    events = []
    e = Event(enable_timing=True)
    e.record()
    events.append(e)

    prev_pos = 0
    for cur_pos in range(prompt_len, total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        tokens[:, cur_pos] = next_token
        prev_pos = cur_pos
        e = Event(enable_timing=True)
        e.record()
        events.append(e)

    torch.cuda.synchronize()

    return tokens, events


def benchmark_model(model, bs, prompt_len, total_len):
    tokens = torch.arange(prompt_len).repeat(bs, 1)

    torch.cuda.synchronize()
    start = Event(enable_timing=True)
    start.record()
    res = model.generate(tokens, do_sample=False, max_length=total_len)
    end = Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize()
    el = start.elapsed_time(end)
    return res, el


if __name__ == "__main__":
    import hf, llama_single, hf_ds, llama_official, llama_turbo

    model_inits = {
        # "HF": hf,
        # "HF_DS": hf_ds,
    }

    generator_inits = {
        # "LLAMA_turbo": llama_turbo,
        "LLAMA_official": llama_official,
        # "LLAMA_single": llama_single,
    }

    header = [
        "model",
        "batch_size",
        "prompt_len",
        "total_len",
        "total_time",
        "total_time_aux",
        "time_prompt",
        "time_inc",
    ]
    csv.writer(open("tmp.csv", "a")).writerow(header)
    records = []

    for name, module in generator_inits.items():
        generator = module.init_generator()
        #warmup
        tokens, events = benchmark_generator(generator, 2, 2, 20)
        print(tokens[::2, ::64].cpu().numpy().tolist())
        for bs in [4**i for i in range(1,3)]:
            for prompt_len in [8**i for i in range(4)]:
                total_len = 512 if prompt_len<512 else 2048
                print(f"Benchmarking {name} using bs={bs}, prompt_len={prompt_len}, total_len={total_len}")
                timehere()
                tokens, events = benchmark_generator(generator, bs, prompt_len, total_len)
                print(tokens[::2, ::64].cpu().numpy().tolist())
                elapsed = timehere("total time: ")
                ets = []
                for i in range(len(events) - 1):
                    ets.append(events[i].elapsed_time(events[i + 1]))
                print(ets[0], sum(ets[1:]) / len(ets[1:]))
                print(f"Time by Events: {sum(ets)} ms")

                record = [name, bs, prompt_len, total_len, elapsed, sum(ets), ets[0], sum(ets[1:]) / len(ets[1:])]
                records.append(record)
                csv.writer(open("tmp.csv", "a")).writerow(record)
        del generator

    for name, module in model_inits.items():
        model = module.init_model()
        # warmup
        tokens, events = benchmark_model(model, 2, 2, 20)
        print(tokens[::2, ::64].cpu().numpy().tolist())
        for bs in [4**i for i in range(0, 4)]:
            for prompt_len in [8**i for i in range(4)]:
                total_len = 512 if prompt_len < 512 else 2048
                print(
                    f"Benchmarking {name} using bs={bs}, prompt_len={prompt_len}, total_len={total_len}"
                )
                timehere()
                tokens, el = benchmark_model(model, bs, prompt_len, total_len)
                print(tokens[::2, ::64].cpu().numpy().tolist())
                elapsed = timehere("total time: ")
                print(f"Time by Events: {el} ms")

                record = [name, bs, prompt_len, total_len, elapsed, el, "", ""]
                records.append(record)
                csv.writer(open("tmp.csv", "a")).writerow(record)
        del model

    csv.writer(open("benchmark.csv", "w")).writerows([header] + records)
