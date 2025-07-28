from typing import Sequence

import numpy as np
from torch import nn

CONVOPS = nn.Conv2d
NORMOPS = nn.BatchNorm2d
ACTOPS = nn.ReLU


class DepthwiseCA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1):
        super(DepthwiseCA, self).__init__()
        self.depthwise = CONVOPS(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = CONVOPS(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.act = ACTOPS()

    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        return self.act(x)


class DepthwiseCNA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1):
        super(DepthwiseCNA, self).__init__()
        self.depthwise = CONVOPS(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = CONVOPS(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.norm = NORMOPS(out_channels)
        self.act = ACTOPS()

    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)


class CNA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, padding="same", bias=True):
        super().__init__()
        self.conv = CONVOPS(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.norm = NORMOPS(out_channels)
        self.act = ACTOPS()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class CN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, padding="same", bias=True):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = CONVOPS(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.norm = NORMOPS(out_channels)

    def forward(self, x):
        return self.norm(self.conv(x))


class SpaceToDepth3D(nn.Module):
    def __init__(self, divide_factor: Sequence[int]):
        super().__init__()
        self.divide_factor = divide_factor
        assert len(divide_factor) == 3

    def forward(self, x):
        batch_size, in_channels, in_depth, in_height, in_width = x.size()
        out_channels, out_depth, out_height, out_width = (
            np.prod(self.divide_factor) * in_channels,
            in_depth // self.divide_factor[0],
            in_width // self.divide_factor[1],
            in_height // self.divide_factor[2],
        )
        x = x.view(
            batch_size,
            in_channels,
            self.divide_factor[0],
            out_depth,
            self.divide_factor[1],
            out_height,
            self.divide_factor[2],
            out_width,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)
        x = x.reshape(batch_size, out_channels, out_depth, out_height, out_width)
        return x
