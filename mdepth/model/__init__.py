from torch import nn, load, save, Tensor
from torch.nn import functional as F

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
        self.decoder = DepthDecoder(channels=self.encoder.channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Returns a depth image for each input image.
        """
        size = x.shape[-2:]
        encoded = [self._interpolate(e, size=size) for e in self.encoder(x)]
        decoded = self._interpolate(self.decoder(encoded), size=size)
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

    def _interpolate(self, x: Tensor, *, size) -> Tensor:
        """
        Performs bilinear interpolation.
        """
        return F.interpolate(x, size=size, mode="bilinear")
