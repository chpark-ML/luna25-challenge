import torch.nn as nn
from trainer.common.models.modules.bifpn.activated_batch_norm import ABN


def conv3x3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )


def conv1x1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class PointWiseConvBlock(nn.Sequential):
    """point-wise conv with BN."""

    def __init__(
        self,
        in_channels,
        out_channels,
        norm_layer=ABN,
        norm_act="relu",
        use_norm=True,
    ):
        modules = [
            # Do we need normalization here? If yes why? If no why?
            # bias is needed for EffDet because in head conv is separated from normalization
            conv1x1x1(in_channels, out_channels, bias=not use_norm),
            norm_layer(out_channels, activation=norm_act) if use_norm else nn.Identity(),
        ]
        super().__init__(*modules)


class DepthwiseSeparableConv(nn.Sequential):
    """Depthwise separable conv with BN after depthwise & pointwise."""

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        dilation=1,
        norm_layer=ABN,
        norm_act="relu",
        use_norm=True,
    ):
        modules = [
            conv3x3x3(in_channels, in_channels, stride=stride, groups=in_channels, dilation=dilation),
            # Do we need normalization here? If yes why? If no why?
            # bias is needed for EffDet because in head conv is separated from normalization
            conv1x1x1(in_channels, out_channels, bias=not use_norm),
            norm_layer(out_channels, activation=norm_act) if use_norm else nn.Identity(),
        ]
        super().__init__(*modules)
