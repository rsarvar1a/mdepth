from torch import cat, nn, Tensor
from torch.nn import functional as F


class Block(nn.Sequential):
    """
    A convolutional block in the decoder model.
    """

    def __init__(self, ich, och) -> None:
        """
        Constructs a new block.
        """
        super().__init__(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ich, och, kernel_size=3),
            nn.ReLU(inplace=True),
        )


class DepthDecoder(nn.Module):
    """
    A model that produces monocular depth estimations from a series of feature images.
    """

    def __init__(self, *, channels) -> None:
        """
        Constructs a new DepthDecoder.
        """
        super().__init__()

        self.length = len(channels)
        self.ichannels = channels
        self.ochannels = channels

        conv1s = []
        conv2s = []

        for i in range(self.length):
            conv1s.append(
                Block(
                    self.ichannels[-1]
                    if i == self.length - 1
                    else self.ochannels[i + 1],
                    self.ochannels[i],
                )
            )

            conv2s.append(
                Block(
                    self.ochannels[i] + (self.ichannels[i - 1] if i > 0 else 0),
                    self.ochannels[i],
                )
            )

        self.conv1s = nn.ModuleList(conv1s)
        self.conv2s = nn.ModuleList(conv2s)
        self.reduce = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.ochannels[0], 2, kernel_size=3),
            nn.Sigmoid(),
        )

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        """
        Given a set of image features, computes a set of disparity images.
        """
        x = features[-1]
        size = x.shape[-2:]

        for i in range(self.length - 1, -1, -1):
            x = self.conv1s[i](x)
            x = [self._interpolate(x, size=size)]
            x += [features[i - 1]] if i > 0 else []
            x = cat(x, 1)
            x = self.conv2s[i](x)

        # Drop the last one.
        return self.reduce(x)

    def _interpolate(self, x: Tensor, *, size) -> Tensor:
        """
        Performs bilinear interpolation with a scale factor of 2.
        """
        return F.interpolate(x, size=size, mode="bilinear")
