from torch import nn, Tensor
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
        encoded = list(map(lambda e: self._interpolate(e, size=size), self.encoder(x)))
        decoded = self.decoder(encoded)
        return decoded

    def _interpolate(self, x: Tensor, *, size) -> Tensor:
        """
        Performs bilinear interpolation.
        """
        return F.interpolate(x, size=size, mode="bilinear")
