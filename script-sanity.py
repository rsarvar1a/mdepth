import torch

from mdepth.model import MonocularDepth
from mdepth.utils import inspect


def sanity_check():
    """
    Ensures the model can output an image.
    """
    net = MonocularDepth()
    net.eval()
    print("learnable parameters:", inspect.get_num_params(net))

    t = torch.rand([1, 3, 64, 64])
    out = net.forward(t)
    pred = out[0]

    print("disparity map shapes:", *[list(o.shape[-3:]) for o in out])
    print(
        "size of prediction matches input shape:",
        t.shape[-2:] == pred.shape[-2:] and pred.shape[1] == 1,
    )


if __name__ == "__main__":
    sanity_check()
