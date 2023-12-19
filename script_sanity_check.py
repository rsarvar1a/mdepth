
import torch

from mdepth.model import Monodepth, MonodepthVGG
from mdepth.utils.inspect import get_num_params


def sanity_check_model(net):

    print("learnable parameters:", get_num_params(net))
    
    net.eval()
    t = torch.rand([1, 3, 256, 512])
    out = net(t)
    pred = out[0]
    
    print('disparity maps:', *[list(o.shape[-3:]) for o in out])
    print(f'size of prediction {"matches" if t.shape[-2:] == pred.shape[-2:] else "does not match"} input shape')


def main ():
    
    print('monodepth --------------------------------')
    net = Monodepth()
    sanity_check_model(net)
    print('------------------------------------------')
    print()
    
    print('monodepthvgg -----------------------------')
    net = MonodepthVGG()
    sanity_check_model(net)
    print('------------------------------------------')

if __name__ == "__main__":
    main()
