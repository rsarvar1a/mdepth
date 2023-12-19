from torch import nn, load, save

from .components import VGGEncoder, VGGDecoder


class MonodepthVGG(nn.Module):
    """
    A monocular depth model with a VGG encoder.
    """

    def __init__(self, batch_norm=False):
        super().__init__()
        self.encoder = VGGEncoder(batch_norm=batch_norm)
        self.decoder = VGGDecoder()

    def forward(self, left):
        skips = self.encoder(left)
        disps = self.decoder(skips)
        return disps

    def load(self, path):
        """
        Loads the model from disk.
        """
        self.load_state_dict(load(path))

    def save(self, path):
        """
        Saves a model to a PyTorch model file.
        """
        save(self.state_dict(), path)