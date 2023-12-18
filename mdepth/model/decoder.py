from torch import cat, nn, Tensor
from torch.nn import functional as F

from .commons import Conv, Padder


class ConvUpscale(nn.Module):
    """
    A convolution and upscale paired together.
    """

    def __init__(self, *, ich, och, kernel, scale):
        """
        Initializes a new upscaling convolution.
        """
        super().__init__()

        self.conv1 = Conv(ich=ich, och=och, kernel=kernel, stride=1)
        self.scale = lambda x: F.interpolate(
            x, scale_factor=scale, mode="bilinear", align_corners=True
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.scale(x)
        x = self.conv1(x)
        return x


class Disparities(nn.Module):
    """
    A layer that produces a disparity map output.
    """

    def __init__(self, *, ich):
        """
        Initializes a new disparity layer.
        """
        super().__init__()

        self.pad2d = Padder(size=3)
        self.conv1 = nn.Conv2d(ich, 2, kernel_size=3, stride=1)
        self.norm1 = nn.BatchNorm2d(2)
        self.activ = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x = self.pad2d(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = 0.3 * x
        return x


class DepthDecoder(nn.Module):
    """
    A model that produces monocular depth estimations from a series of feature images.
    """

    def __init__(self) -> None:
        """
        Constructs a new DepthDecoder.
        """
        super().__init__()

        self.uconvs = nn.ModuleList(
            [
                ConvUpscale(ich=32, och=16, kernel=3, scale=2),
                ConvUpscale(ich=64, och=32, kernel=3, scale=2),
                ConvUpscale(ich=128, och=64, kernel=3, scale=2),
                ConvUpscale(ich=256, och=128, kernel=3, scale=2),
                ConvUpscale(ich=512, och=256, kernel=3, scale=2),
                ConvUpscale(ich=2048, och=512, kernel=3, scale=2),
            ]
        )

        self.iconvs = nn.ModuleList(
            [
                Conv(ich=16 + 2 + 0, och=16, kernel=3, stride=1),
                Conv(ich=64 + 32 + 2, och=32, kernel=3, stride=1),
                Conv(ich=64 + 64 + 2, och=64, kernel=3, stride=1),
                Conv(ich=256 + 128 + 0, och=128, kernel=3, stride=1),
                Conv(ich=512 + 256 + 0, och=256, kernel=3, stride=1),
                Conv(ich=1024 + 512 + 0, och=512, kernel=3, stride=1),
            ]
        )

        self.odisps = nn.ModuleList(
            [
                Disparities(ich=16),
                Disparities(ich=32),
                Disparities(ich=64),
                Disparities(ich=128),
                None,
                None,
            ]
        )

        self.scaler = lambda x: F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=True
        )

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        """
        Given a set of image features, computes a set of disparity images.
        """
        output = []

        prev_disparity = None
        x = features[-1]
        l = len(self.uconvs)

        for i in range(l - 1, -1, -1):
                        
            # Perform the upscaling convolution.
            uconvx = self.uconvs[i](x)

            print(f"shapes at {i}: x {list(x.shape[-3:])}, u {list(uconvx.shape[-3:])}")

            # Take the result of the upconv, and add in whatever we have available:
            # If the last set of layers produced an auxiliary disparity map, include it.
            # If we have a skip connection from the previous encoding layer, use it.
            cat_list = [uconvx]
            if i > 0:
                cat_list.append(features[i - 1])
            if prev_disparity is not None:
                cat_list.append(prev_disparity)
            if len(cat_list) == 1:
                cat_list = cat_list[0]
            concat = cat(cat_list, dim=1)

            x = self.iconvs[i](concat)

            # If there is a disparity mapper attached to this layer, produce the
            # disparity map, and forward the upscaled version to the next layer.
            if self.odisps[i] is not None:
                disp = self.odisps[i](x)
                prev_disparity = self.scaler(disp)
                output.append(disp)

        # We appended disparities in reverse order, so reverse the list.
        return output[::-1]
