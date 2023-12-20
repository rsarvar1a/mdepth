import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import *


class VGGEncoder(nn.Module):
    def __init__(self, batch_norm=False):
        super().__init__()

        self.encoder = nn.ModuleList(
            [
                ConvBlock(3, 32, 7, batch_norm=batch_norm),  # conv1b
                ConvBlock(32, 64, 5, batch_norm=batch_norm),  # conv2b
                ConvBlock(64, 128, 3, batch_norm=batch_norm),  # conv3b
                ConvBlock(128, 256, 3, batch_norm=batch_norm),  # conv4b
                ConvBlock(256, 512, 3, batch_norm=batch_norm),  # conv5b
                ConvBlock(512, 512, 3, batch_norm=batch_norm),  # conv6b
                ConvBlock(512, 512, 3, batch_norm=batch_norm),  # conv7b
            ]
        )

    def forward(self, x):
        skips = []

        for block in self.encoder:
            x = block(x)
            skips.append(x)

        return skips


class VGGDecoder(nn.Module):
    def __init__(self, batch_norm=False):
        super().__init__()

        self.upconv7 = UpConv(512, 512)
        self.iconv7 = IConv(512 + 512, 512)

        self.upconv6 = UpConv(512, 512)
        self.iconv6 = IConv(512 + 512, 512)

        self.upconv5 = UpConv(512, 256)
        self.iconv5 = IConv(256 + 256, 256)

        self.upconv4 = UpConv(256, 128)
        self.iconv4 = IConv(128 + 128, 128)
        self.disp4 = GetDisp(128)

        self.upconv3 = UpConv(128, 64)
        self.iconv3 = IConv(64 + 64 + 2, 64)
        self.disp3 = GetDisp(64)

        self.upconv2 = UpConv(64, 32)
        self.iconv2 = IConv(32 + 32 + 2, 32)
        self.disp2 = GetDisp(32)

        self.upconv1 = UpConv(32, 16)
        self.iconv1 = IConv(16 + 2, 16)
        self.disp1 = GetDisp(16)

    def forward(self, x):
        up_2x = lambda x: F.interpolate(x, scale_factor=2, mode="bilinear")
        cat = lambda x: torch.cat(x, dim=1)
        skip1, skip2, skip3, skip4, skip5, skip6, skip7 = x

        upconv7 = self.upconv7(skip7)
        iconv7 = self.iconv7(cat((upconv7, skip6)))

        upconv6 = self.upconv6(iconv7)
        iconv6 = self.iconv6(cat((upconv6, skip5)))

        upconv5 = self.upconv5(iconv6)
        iconv5 = self.iconv5(cat((upconv5, skip4)))

        upconv4 = self.upconv4(iconv5)
        iconv4 = self.iconv4(cat((upconv4, skip3)))
        disp4 = self.disp4(iconv4)

        upconv3 = self.upconv3(iconv4)
        iconv3 = self.iconv3(cat((upconv3, skip2, up_2x(disp4))))
        disp3 = self.disp3(iconv3)

        upconv2 = self.upconv2(iconv3)
        iconv2 = self.iconv2(cat((upconv2, skip1, up_2x(disp3))))
        disp2 = self.disp2(iconv2)

        upconv1 = self.upconv1(iconv2)
        iconv1 = self.iconv1(cat((upconv1, up_2x(disp2))))
        disp1 = self.disp1(iconv1)

        return [disp1, disp2, disp3, disp4]
