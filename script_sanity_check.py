
import torch

from mdepth.model import MonocularDepth
from mdepth.utils.inspect import get_num_params


def main():
    
    net = MonocularDepth()
    print("learnable parameters:", get_num_params(net))
    
    net.eval()
    t = torch.rand([1, 3, 64, 64])
    out = net(t)
    pred = out[0]
    
    print('disparity maps:', *[list(o.shape[-3:]) for o in out])
    print(f'size of prediction {"matches" if t.shape[-2:] == pred.shape[-2:] else "does not match"} input shape')


if __name__ == "__main__":
    main()
