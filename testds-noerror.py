import deepspeed

from transformers import AutoModelForCausalLM
from transformers.generation.streamers import BaseStreamer

import torch

torch.set_default_tensor_type(torch.cuda.HalfTensor)

model_id = "decapoda-research/llama-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

ds_model = deepspeed.init_inference(
    model=model,  # Transformers models
    mp_size=1,  # Number of GPU
    dtype=torch.float16,  # dtype of the weights (fp16)
    replace_method="auto",  # Lets DS autmatically identify the layer to replace
    replace_with_kernel_inject=True,  # replace the model with the kernel injector
    max_out_tokens=2048,
)
print(f"model is loaded on device {ds_model.module.device}")


class TimingStreamer(BaseStreamer):
    """Timing helper for HuggingFace models."""

    def __init__(self) -> None:
        self.token_cache = []
        self.tokens = None

        # torch.cuda.synchronize()

        self.evts = []
        self.value = 0

    def put(self, value):
        """
        Notes:
            When `put` is called for the first time, no prompt is feed to the model yet.
            When `put` is called later, event is recorded for the previous generation,
                which means the second event records the time for the first prompt.
        """
        self.value += 1
        self.token_cache.append(value)
        evt = torch.cuda.Event(enable_timing=True)
        evt.record()
        self.evts.append(evt)

    def end(self):
        torch.cuda.synchronize()
        # self.tokens = torch.hstack([_atleast_2d(v) for v in self.token_cache])

    # def raw_times(self):
    #     return _get_times(self.evts)


bs = 16
prompt_len = 1
totel_len = 2048

print(1)
fake_inputs = torch.arange(prompt_len).repeat(bs, 1)
ds_model.generate(fake_inputs, max_length=totel_len, streamer=TimingStreamer())
print(2)
fake_inputs = torch.arange(prompt_len).repeat(bs, 1)
ds_model.generate(fake_inputs, max_length=totel_len, streamer=TimingStreamer())
