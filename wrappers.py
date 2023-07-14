import json
import os
import time
from functools import wraps
from pathlib import Path

import torch

from timing import TimingStreamer, _get_times
from timing import timehere as t
from utils import LoadWoInit

from transformers import GenerationConfig

SUPPPORTED_MODELS = {}


def _register_as(model_name):
    def decorator(init_func):
        assert (
            model_name not in SUPPPORTED_MODELS
        ), f"Model {model_name} already registered"
        SUPPPORTED_MODELS[model_name] = init_func
        return init_func

    return decorator


def _register(init_func):
    model_name = init_func.__name__.split("_", maxsplit=1)[1]
    assert model_name not in SUPPPORTED_MODELS, f"Model {model_name} already registered"
    SUPPPORTED_MODELS[model_name] = init_func
    return init_func


class HuggingFaceModelBenchmarker:
    def __init__(self, model):
        self.model = model
        self.device = (
            self.model.module.device
            if "InferenceEngine" in self.model.__class__.__name__
            else self.model.device
        )

    def benchmark_and_time(self, bs, prompt_len, total_len):
        # input_id = 0 cause some cuda error
        fake_inputs = torch.arange(prompt_len, device=self.device).repeat(bs, 1) + 66
        fake_inputs = fake_inputs.to(self.device)

        ts = TimingStreamer()
        fake_outputs = self.model.generate(
            fake_inputs,
            GenerationConfig(max_length=total_len, do_sample=False, eos_token_id=[-1]),
            streamer=ts,
        )
        print(fake_outputs.size())

        return ts.raw_times()


class LLaMAModelBenchmarker:
    def __init__(self, model):
        self.model = model

    def benchmark_and_time(self, bs, prompt_len, total_len):
        fake_tokens = torch.arange(total_len).repeat(bs, 1)

        torch.cuda.synchronize()

        events = []
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        events.append(e)

        prev_pos = 0
        for cur_pos in range(prompt_len, total_len):
            logits = self.model.forward(fake_tokens[:, prev_pos:cur_pos], prev_pos)
            next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            fake_tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            events.append(e)

        torch.cuda.synchronize()

        return _get_times(events)


@_register
def init_hf_baseline(name_or_path, **kwawrgs):
    from transformers import AutoModelForCausalLM

    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    t()
    # model_id = "decapoda-research/llama-7b-hf"
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    t("load model")

    return HuggingFaceModelBenchmarker(model)


@_register
def init_deepspeed(name_or_path, mp_size, max_seq_len, init_on_gpu=True, **kwawrgs):
    tp_size = mp_size
    import deepspeed
    from transformers import AutoModelForCausalLM

    if init_on_gpu:
        # torch.set_default_tensor_type(torch.cuda.HalfTensor)
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        torch.set_default_device(local_rank)

    # from accelerate import init_empty_weights
    #     with init_empty_weights():
    # with deepspeed.OnDevice(dtype=torch.float16, device="meta"):
    print("Loading")
    with LoadWoInit():
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path, torch_dtype=torch.float16
        )
    # model = AutoModelForCausalLM.from_pretrained(
    #     name_or_path, torch_dtype=torch.float16
    # )
    t("load model")
    print(f"max_seq_len = {max_seq_len}")

    # tp_config= DeepSpeedTPConfig
    ds_model = deepspeed.init_inference(
        model=model,  # Transformers models
        tensor_parallel={"tp_size": tp_size},
        dtype=torch.float16,  # dtype of the weights (fp16)
        replace_method="auto",  # Lets DS autmatically identify the layer to replace
        replace_with_kernel_inject=True,  # replace the model with the kernel injector
        max_out_tokens=max_seq_len,
    )
    print(f"model is loaded on device {ds_model.module.device}")

    return HuggingFaceModelBenchmarker(ds_model)


def _setup_model_parallel():
    from fairscale.nn.model_parallel.initialize import initialize_model_parallel

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
        initialize_model_parallel(world_size)
    print(f"local_rank = {local_rank}")
    torch.cuda.set_device(int(local_rank))

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def _load_llama(
    ModelArgs,
    Transformer,
    ckpt_dir: str,
    # local_rank: int,
    # world_size: int,
    max_seq_len: int = 2048,
    max_batch_size: int = 32,
    vocab_size: int = 32000,
):
    local_rank, world_size = _setup_model_parallel()
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
    model_args.vocab_size = vocab_size
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    # torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    # generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return model


@_register
def init_llama_baseline(
    name_or_path: str, max_seq_len: int = 2048, max_batch_size: int = 32, **kwawrgs
):
    from llama import ModelArgs, Transformer

    model = _load_llama(
        ModelArgs=ModelArgs,
        Transformer=Transformer,
        ckpt_dir=name_or_path,
        # local_rank=0,
        # world_size=1,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    return LLaMAModelBenchmarker(model)
