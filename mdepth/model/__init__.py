from torch import nn, load, save, Tensor

from .encoder import ImageEncoder
from .decoder import DepthDecoder

class MonocularDepth(nn.Module):
    """
    A deep inference model that generates a depth image for a given RGB image.
    """

    def __init__(self) -> None:
        """
        Constructs an untrained MonocularDepth model.
        """
        super().__init__()

        self.encoder = ImageEncoder()
        self.decoder = DepthDecoder()

    def forward(self, x: Tensor) -> list[Tensor]:
        """
        Returns a set of disparity maps for each input image.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

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
