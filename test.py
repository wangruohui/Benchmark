import torch
from torch.cuda import Event
from utils import timehere


from transformers import AutoModelForCausalLM
from transformers.generation.streamers import BaseStreamer

import numpy as np
def _atleast_2d(tensor):
    if tensor.dim() == 1:
        return tensor.unsqueeze(-1)
    return tensor

def _std(seq):
    return np.std(seq, ddof=1)

def _get_times(evts):
    return [evts[i].elapsed_time(evts[i + 1]) for i in range(len(evts) - 1)]

class TimingStreamer(BaseStreamer):
    def __init__(self) -> None:
        self.token_cache = []
        self.tokens = None

        torch.cuda.synchronize()

        # evt = Event(enable_timing=True)
        # evt.record()  # record before first prompt input
        # self.evts = [evt]
        self.evts = []

    def put(self, value):
        """
        Notes:
            When `put` is called for the first time, no prompt is feed to the model yet.
            When `put` is called later, event is recorded for the previous generation,
                which means the second event records the time for the first prompt.
        """
        self.token_cache.append(value)
        evt = Event(enable_timing=True)
        evt.record()
        self.evts.append(evt)

    def end(self):
        torch.cuda.synchronize()
        self.tokens = torch.hstack([_atleast_2d(v) for v in self.token_cache])

    def get_times(self):
        """
        Returns:
          a tuple of times in ms for (first prompt, avg next token, total time)
        """
        first = self.evts[0].elapsed_time(self.evts[1])
        rest = [
            self.evts[i].elapsed_time(self.evts[i + 1])
            for i in range(1, len(self.evts) - 1)
        ]
        avg = sum(rest) / len(rest)
        std = _std(rest)
        return first + sum(rest), first, avg, std

    def raw_times(self):
        return _get_times(self.evts)



torch.set_default_device("cuda:6")
torch.set_default_tensor_type(torch.cuda.HalfTensor)

timehere()
model_id = "decapoda-research/llama-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
timehere("load model")

ts = TimingStreamer()
model.generate(torch.arange(1, 10, 1).repeat(2, 1), streamer=ts)
print(ts.tokens)
print(ts.get_times())
