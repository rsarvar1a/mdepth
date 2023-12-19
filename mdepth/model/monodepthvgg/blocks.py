import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(
        self, ich, och, kernel_size, stride, batch_norm=False, activation=F.elu
    ):
        super().__init__()

        padding = np.floor((kernel_size - 1) / 2).astype(np.int32)
        self.conv_layer = nn.Conv2d(
            ich, och, kernel_size, stride=stride, padding=padding
        )

        self.do_bn = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm2d(och)

        self.activation = activation

    def forward(self, x):
        x = self.conv_layer(x)

        if self.do_bn:
            x = self.bn(x)

        return self.activation(x)


class ConvBlock(nn.Module):
    def __init__(self, ich, och, kernel_size):
        super().__init__()

        self.conv1 = Conv(ich, och, kernel_size, 1)
        self.conv2 = Conv(och, och, kernel_size, 2)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        return conv2


class UpConv(nn.Module):
    def __init__(self, ich, och, kernel_size=3, scale=2):
        super().__init__()

        self.scale = lambda x: F.interpolate(x, scale_factor=scale, mode="bilinear")
        self.conv = Conv(ich, och, kernel_size, 1)

    def forward(self, x):
        upsample = self.scale(x)
        return self.conv(upsample)


class IConv(nn.Module):
    def __init__(self, ich, och):
        super().__init__()
        self.conv = Conv(ich, och, 3, 1)

    def forward(self, x):
        return self.conv(x)


class GetDisp(nn.Module):
    def __init__(self, ich):
        super().__init__()

        self.conv = Conv(ich, 2, 3, 1, activation=F.sigmoid)

    def forward(self, x):
        return 0.3 * self.conv(x)
