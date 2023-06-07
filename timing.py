from time import time

import numpy as np
import torch
from transformers.generation.streamers import BaseStreamer


def _atleast_2d(tensor):
    if tensor.dim() == 1:
        return tensor.unsqueeze(-1)
    return tensor


def _std(seq):
    return np.std(seq, ddof=1)


def _get_times(evts):
    return [evts[i].elapsed_time(evts[i + 1]) for i in range(len(evts) - 1)]


# def _get_times(evts):
#     return [evts[i + 1] - evts[i] for i in range(len(evts) - 1)]


class TimingStreamer(BaseStreamer):
    """Timing helper for HuggingFace models."""

    def __init__(self) -> None:
        self.token_cache = []
        self.tokens = None

        torch.cuda.synchronize()

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


class TimeHere:
    """A simple timer"""

    def __init__(self) -> None:
        self.start = None

    def __call__(self, checkpoint=""):
        if self.start is None:
            self.start = time()
        else:
            elapsed = time() - self.start
            print(f"{checkpoint}: {elapsed}")
            self.start = time()
            return elapsed


timehere = TimeHere()
