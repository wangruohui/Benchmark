
import torch
from single import ModelArgs, Transformer
from utils import timehere

def init_generator():
    args = ModelArgs()
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(args)
    timehere(
        f"create model \t {next(iter(model.parameters())).device} {next(iter(model.parameters())).dtype}"
    )

    state_dict = torch.load("full_fused.pth", map_location="cuda")
    model.load_state_dict(state_dict, strict=False)
    timehere(
        f"load model \t {next(iter(model.parameters())).device} {next(iter(model.parameters())).dtype}"
    )

    model = model.eval()

    return model
