import argparse
import csv
import os
import sys
from collections import namedtuple

import torch
import yaml

from mystat import stat
from wrappers import SUPPPORTED_MODELS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default="config.yaml", type=str)
    parser.add_argument("--local_rank", type=str, required=False, default=None)
    return parser.parse_args()


Spec = namedtuple("Spec", ["bs", "prompt_len", "gen_len", "total_len"])


def _expand_config(config):
    res = []
    for bs in config["bss"]:
        for tokens in config["tokens"]:
            prompt_len, gen_len, total_len = tokens
            if total_len is None:
                assert (
                    gen_len is not None
                ), "Either total_len or gen_len must be specified"
                total_len = prompt_len + gen_len
            res.append(Spec(bs, prompt_len, gen_len, total_len))

    return res


def cprint(s):
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    print(f"{OKGREEN}{s}{ENDC}")


def _max_of_outer(lst):
    key = lambda x: x if x is not None else -999
    return [max(x, key=key) for x in zip(*lst)]


class Logger:
    def __init__(
        self,
        file="b2tmp.csv",
        header=[
            "model",
            "batch_size",
            "prompt_len",
            "gen_len",
            "total_len",
            "first_time",
            "inc_time",
            "inc_time2",
            "error",
            "total time",
            "throughput(total)",
            "throughput(gen)",
        ],
    ):
        if self.on_master:
            self.file = file
            csv.writer(open(file, "a")).writerow(header)

    def log(self, line):
        if self.on_master:
            csv.writer(open(self.file, "a")).writerow(line)

    @property
    def on_master(self):
        # return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return local_rank == 0


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config, "r"))
    assert "models" in config, "No model defined"
    assert "specs" in config, "No specs defined"

    specs = _expand_config(config["specs"])

    x = _max_of_outer(specs)
    max_batch_size, _, _, max_seq_len = x

    cprint(f"Max batch size: {max_batch_size}")
    cprint(f"Max seq len: {max_seq_len}")

    logger = Logger()
    print(logger.on_master)

    for model, kwargs in config["models"].items():
        kwargs["max_seq_len"] = max_seq_len
        kwargs["max_batch_size"] = max_batch_size

        cprint(f"Initializing {model} with {kwargs}")
        init_fun = SUPPPORTED_MODELS[model]
        benckmarker = init_fun(**kwargs)

        # cprint(
        #     f"Warm up {model}, with specs {Spec(max_batch_size, 1, None, max_seq_len)}"
        # )
        # benckmarker.benchmark_and_time(max_batch_size, 1, max_seq_len)

        for c in specs:
            cprint(f"Benchmarking {model}, with specs {c}")
            bs, prompt_len, gen_len, total_len = c

            torch.cuda.synchronize()
            times = benckmarker.benchmark_and_time(bs, prompt_len, total_len)
            torch.cuda.synchronize()
            print("\t".join(f"{t:4.1f}" for t in times))

            first, inc, inc2, error = stat(times)
            tt = bs * total_len * 1000 / sum(times)
            tg = bs * gen_len * 1000 / sum(times)
            cprint(
                f"First/ms: {first:5.1f}, Inc/ms: {inc:5.1f}, "
                f"Inc2/ms: {inc2:5.3f}, IncStd/ms: {error:5.3f}, "
                f"Total Time/ms: {sum(times):5.3f}, "
                f"Throughput Total: {tt:5.3f}, "
                f"Throughput Gen: {tg:5.3f}"
            )

            # fmt: off
            logger.log([model, bs, prompt_len, gen_len, total_len,
                        first, inc, inc2, error, sum(times), tt, tg])
            # fmt: on


if __name__ == "__main__":
    main()
