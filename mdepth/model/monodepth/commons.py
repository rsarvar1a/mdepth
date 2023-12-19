import numpy as np

from torch import nn, Tensor
from torch.nn import functional as F


class Padder(nn.Module):
    """
    An equal padder.
    """

    def __init__(self, *, size):
        """
        Initializes a new padder for the given padding size.
        """
        super().__init__()

        size = int(np.floor((size - 1) / 2))
        self.padding = (size, size, size, size)

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, self.padding)
        return x


class Pool(nn.Module):
    """
    A maxpool layer that pads.
    """

    def __init__(self, *, kernel):
        """
        Initializes a new pooling layer.
        """
        super().__init__()

        self.pad2d = Padder(size=kernel)
        self.pool1 = nn.MaxPool2d(kernel_size=kernel, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pad2d(x)
        x = self.pool1(x)
        return x


class Conv(nn.Module):
    """
    A single convolution with a norm and activation.
    """

    def __init__(self, *, ich, och, kernel, stride):
        """
        Initializes a new convolutional layer.
        """
        super().__init__()

        self.pad2d = Padder(size=kernel)
        self.conv1 = nn.Conv2d(ich, och, kernel_size=kernel, stride=stride)
        self.norm1 = nn.BatchNorm2d(och)
        self.activ = nn.ELU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pad2d(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activ(x)
        return x
