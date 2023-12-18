from torch import nn, Tensor

from .commons import Conv, Pool


class ConvBlock(nn.Module):
    """
    A convolutional block in the network.
    """

    def __init__(self, *, ich, och, kernel, stride):
        """
        Initializes a new convolutional block.
        """
        super().__init__()

        self.conv1 = Conv(ich=ich, och=och, kernel=kernel, stride=1)
        self.conv2 = Conv(ich=och, och=och, kernel=kernel, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResidualConv(nn.Module):
    """
    A convolutional layer for a residual block.
    """

    def __init__(self, *, ich, och, stride):
        """
        Initializes a new residual layer.
        """
        super().__init__()

        self.conv1 = Conv(ich=ich, och=och, kernel=1, stride=1)
        self.conv2 = Conv(ich=och, och=och, kernel=3, stride=stride)
        self.conv3 = nn.Conv2d(och, 4 * och, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(ich, 4 * och, kernel_size=1, stride=stride)
        self.norm1 = nn.BatchNorm2d(4 * och)
        self.activ = nn.ELU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        t = self.conv1(x)
        t = self.conv2(t)
        t = self.conv3(t)
        x = self.conv4(x)
        x = x + t
        x = self.norm1(x)
        x = self.activ(x)
        return x


class ResidualBlock(nn.Module):
    """
    A residual block in the network.
    """

    def __init__(self, *, ich, och, num_blocks, stride):
        """
        Initializes a new residual block.
        """
        super().__init__()

        layers = [ResidualConv(ich=ich, och=och, stride=stride)]
        for _ in range(1, num_blocks - 1):
            layers.append(ResidualConv(ich=4 * och, och=och, stride=1))
        layers.append(ResidualConv(ich=4 * och, och=och, stride=1))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ImageEncoder(nn.Module):
    """
    Takes in an tensor of RGB images, and outputs a list of feature tensors.
    """

    def __init__(self) -> None:
        """
        Constructs a new ImageEncoder by scrapping a ResNet101.
        """
        super().__init__()

        self.layers = nn.ModuleList(
            [
                Conv(ich=3, och=64, kernel=7, stride=2),
                Pool(kernel=3),
                ResidualBlock(ich=64, och=64, num_blocks=3, stride=2),
                ResidualBlock(ich=256, och=128, num_blocks=4, stride=2),
                ResidualBlock(ich=512, och=256, num_blocks=6, stride=2),
                ResidualBlock(ich=1024, och=512, num_blocks=3, stride=2),
            ]
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        """
        Returns a list of feature images.
        """
        feature_maps: list[Tensor] = []
        for layer in self.layers:
            x = layer(x)
            feature_maps.append(x)
        return feature_maps
