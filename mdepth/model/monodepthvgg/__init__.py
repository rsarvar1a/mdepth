from torch import nn

from .components import VGGEncoder, VGGDecoder


class MonodepthVGG(nn.Module):
    """
    A monocular depth model with a VGG encoder.
    """

    def __init__(self):
        super().__init__()
        self.encoder = VGGEncoder()
        self.decoder = VGGDecoder()

    def forward(self, left):
        skips = self.encoder(left)
        disps = self.decoder(skips)
        return disps
