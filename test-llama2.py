import torch
from llama import Llama

# torch.set_default_tensor_type(torch.cuda.HalfTensor)
# local_rank = int(os.environ.get("LOCAL_RANK", 0))
# torch.cuda.set_device(local_rank)

TARGET_FOLDER = "llama2/facebook"
ckpt_dir = TARGET_FOLDER + "/llama-2-7b"
tokenizer_path = TARGET_FOLDER + "/tokenizer.model"
inptu_len = 1024
max_seq_len = 4096
max_batch_size = 1

generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
)

prompt_tokens = torch.randint(10, 10000, size=(max_batch_size, inptu_len))

generation_tokens, generation_logprobs = generator.generate(
    prompt_tokens=prompt_tokens,
    max_gen_len=max_seq_len - inptu_len,
    temperature=0,
)


print(generation_tokens)
