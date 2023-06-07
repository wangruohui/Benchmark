from time import time


class TimeHere:
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
