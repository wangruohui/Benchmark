class BatchGenerator:

    def __call__(self, prompts):
        pass


class SeqGenerator:

    def __call__(self, prompts):

        pass

    def __iter__(self):
        return self

    def __next__(self):
        print('next')
        yield 1

if __name__ == "__main__":
    s = SeqGenerator()
    for ss in s:
        print(s)