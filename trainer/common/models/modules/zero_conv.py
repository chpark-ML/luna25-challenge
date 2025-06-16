import torch
import torch.nn as nn


class ZeroConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        torch.nn.init.constant_(self.conv.weight, 0)
        if self.conv.bias is not None:
            torch.nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class ZeroConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        torch.nn.init.constant_(self.conv.weight, 0)
        if self.conv.bias is not None:
            torch.nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)
    