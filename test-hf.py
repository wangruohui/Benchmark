import torch

from utils import timehere
from timing import TimingStreamer, timehere
from torch.profiler import ProfilerActivity, profile, record_function

from transformers import AutoModelForCausalLM, GenerationConfig


def init_model(model_id="/nvme/wangruohui/llama-7b-hf"):
    torch.set_default_tensor_type(torch.cuda.HalfTensor)

    timehere()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, trust_remote_code=True
    )
    timehere("load model")

    return model


def generate(model, prompt_len=2, total_len=6, device="cuda"):
    ts = TimingStreamer()
    torch.cuda.synchronize()
    fake_inputs = torch.arange(prompt_len, device=device).repeat(1, 1) + 66
    fake_inputs = fake_inputs.to(device)
    fake_outputs = model.generate(
        fake_inputs,
        GenerationConfig(max_length=total_len, do_sample=False, eos_token_id=[-1]),
        streamer=ts,
    )

    raw = ts.raw_times()

    return raw


if __name__ == "__main__":
    device = 7
    model_id = "internlm"
    torch.set_default_device(device)

    model = init_model(model_id=model_id)

    print("=" * 20 + "warm-up" + "=" * 20)
    raw_ts = generate(model, device=device)
    print(raw_ts)

    print("=" * 20 + "gen" + "=" * 20)
    raw_ts = generate(model, device=device)
    print(raw_ts)

    print("=" * 20 + "profile" + "=" * 20)
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        with_stack=True,
    ) as prof:
        raw_ts = generate(model, device=device)
    print(raw_ts)

    prof.export_chrome_trace("tmp_30ms_streamer.json")
