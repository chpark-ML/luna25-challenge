""" ResNeSt Models
Paper: `ResNeSt: Split-Attention Networks` - https://arxiv.org/abs/2004.08955
Adapted from original PyTorch impl w/ weights at https://github.com/zhanghang1989/ResNeSt by Hang Zhang
Modified for torchscript compat, and consistency with timm by Ross Wightman

Split Attention Conv3d (for ResNeSt Models)

Paper: `ResNeSt: Split-Attention Networks` - /https://arxiv.org/abs/2004.08955
Adapted from original PyTorch impl at https://github.com/zhanghang1989/ResNeSt
Modified for torchscript compat, performance, and consistency with timm by Ross Wightman
"""
import torch
import torch.nn.functional as F
from torch import nn


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class RadixSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttn(nn.Module):
    """Split-Attention (aka Splat)"""

    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        bias=False,
        radix=2,
        rd_ratio=0.25,
        rd_channels=None,
        rd_divisor=8,
        act_layer=nn.ReLU,
        norm_layer=None,
        drop_layer=None,
        **kwargs,
    ):
        super(SplitAttn, self).__init__()
        out_channels = out_channels or in_channels
        self.radix = radix
        mid_chs = out_channels * radix
        if rd_channels is None:
            attn_chs = make_divisible(
                in_channels * radix * rd_ratio, min_value=32, divisor=rd_divisor
            )
        else:
            attn_chs = rd_channels * radix

        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv3d(
            in_channels,
            mid_chs,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=groups * radix,
            bias=bias,
            **kwargs,
        )
        self.bn0 = norm_layer(mid_chs) if norm_layer else nn.Identity()
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act0 = act_layer(inplace=True)
        self.fc1 = nn.Conv3d(out_channels, attn_chs, 1, groups=groups)
        self.bn1 = norm_layer(attn_chs) if norm_layer else nn.Identity()
        self.act1 = act_layer(inplace=True)
        self.fc2 = nn.Conv3d(attn_chs, mid_chs, 1, groups=groups)
        self.rsoftmax = RadixSoftmax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn0(x)
        x = self.drop(x)
        x = self.act0(x)

        B, RC, H, W, D = x.shape
        if self.radix > 1:
            x = x.reshape((B, self.radix, RC // self.radix, H, W, D))
            x_gap = x.sum(dim=1)
        else:
            x_gap = x
        x_gap = x_gap.mean((2, 3, 4), keepdim=True)
        x_gap = self.fc1(x_gap)
        x_gap = self.bn1(x_gap)
        x_gap = self.act1(x_gap)
        x_attn = self.fc2(x_gap)

        x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1, 1)
        if self.radix > 1:
            out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1, 1))).sum(dim=1)
        else:
            out = x * x_attn
        return out.contiguous()


def downsample_avg(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    first_dilation=None,
    norm_layer=None,
):
    norm_layer = norm_layer or nn.BatchNorm3d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = nn.AvgPool3d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(
        *[
            pool,
            nn.Conv3d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            norm_layer(out_channels),
        ]
    )


class ResNestBasicBlock(nn.Module):
    """ResNet Bottleneck"""

    # pylint: disable=unused-argument
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        radix=1,
        cardinality=1,
        base_width=64,
        avd=False,
        avd_first=False,
        is_first=False,
        reduce_first=1,
        dilation=1,
        first_dilation=None,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm3d,
        attn_layer=None,
        aa_layer=None,
        drop_block=None,
        drop_path=None,
    ):
        super(ResNestBasicBlock, self).__init__()
        assert reduce_first == 1  # not supported
        assert attn_layer is None  # not supported
        assert aa_layer is None  # TODO not yet supported
        assert drop_path is None  # TODO not yet supported

        group_width = int(planes * (base_width / 64.0)) * cardinality
        first_dilation = first_dilation or dilation
        if avd and (stride > 1 or is_first):
            avd_stride = stride
            stride = 1
        else:
            avd_stride = 0
        self.radix = radix

        self.conv1 = nn.Conv3d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.act1 = act_layer(inplace=True)
        self.avd_first = (
            nn.AvgPool3d(3, avd_stride, padding=1) if avd_stride > 0 and avd_first else None
        )

        if self.radix >= 1:
            self.conv2 = SplitAttn(
                group_width,
                group_width,
                kernel_size=3,
                stride=stride,
                padding=first_dilation,
                dilation=first_dilation,
                groups=cardinality,
                radix=radix,
                norm_layer=norm_layer,
                drop_layer=drop_block,
            )
            self.bn2 = nn.Identity()
            self.drop_block = nn.Identity()
        else:
            self.conv2 = nn.Conv3d(
                group_width,
                group_width,
                kernel_size=3,
                stride=stride,
                padding=first_dilation,
                dilation=first_dilation,
                groups=cardinality,
                bias=False,
            )
            self.bn2 = norm_layer(group_width)
            self.drop_block = drop_block() if drop_block is not None else nn.Identity()

        self.avd_last = (
            nn.AvgPool3d(3, avd_stride, padding=1) if avd_stride > 0 and not avd_first else None
        )
        self.act2 = act_layer(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        if self.avd_first is not None:
            out = self.avd_first(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop_block(out)

        if self.avd_last is not None:
            out = self.avd_last(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out += shortcut
        out = self.act2(out)
        return out


class ResNestBottleneck(nn.Module):
    """ResNet Bottleneck"""

    # pylint: disable=unused-argument
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        radix=1,
        cardinality=1,
        base_width=64,
        avd=False,
        avd_first=False,
        is_first=False,
        reduce_first=1,
        dilation=1,
        first_dilation=None,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm3d,
        attn_layer=None,
        aa_layer=None,
        drop_block=None,
        drop_path=None,
    ):
        super(ResNestBottleneck, self).__init__()
        assert reduce_first == 1  # not supported
        assert attn_layer is None  # not supported
        assert aa_layer is None  # TODO not yet supported
        assert drop_path is None  # TODO not yet supported

        group_width = int(planes * (base_width / 64.0)) * cardinality
        first_dilation = first_dilation or dilation
        if avd and (stride > 1 or is_first):
            avd_stride = stride
            stride = 1
        else:
            avd_stride = 0
        self.radix = radix

        self.conv1 = nn.Conv3d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.act1 = act_layer(inplace=True)
        self.avd_first = (
            nn.AvgPool3d(3, avd_stride, padding=1) if avd_stride > 0 and avd_first else None
        )

        if self.radix >= 1:
            self.conv2 = SplitAttn(
                group_width,
                group_width,
                kernel_size=3,
                stride=stride,
                padding=first_dilation,
                dilation=first_dilation,
                groups=cardinality,
                radix=radix,
                norm_layer=norm_layer,
                drop_layer=drop_block,
            )
            self.bn2 = nn.Identity()
            self.drop_block = nn.Identity()
            self.act2 = nn.Identity()
        else:
            self.conv2 = nn.Conv3d(
                group_width,
                group_width,
                kernel_size=3,
                stride=stride,
                padding=first_dilation,
                dilation=first_dilation,
                groups=cardinality,
                bias=False,
            )
            self.bn2 = norm_layer(group_width)
            self.drop_block = drop_block() if drop_block is not None else nn.Identity()
            self.act2 = act_layer(inplace=True)
        self.avd_last = (
            nn.AvgPool3d(3, avd_stride, padding=1) if avd_stride > 0 and not avd_first else None
        )

        self.conv3 = nn.Conv3d(group_width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.act3 = act_layer(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        if self.avd_first is not None:
            out = self.avd_first(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop_block(out)
        out = self.act2(out)

        if self.avd_last is not None:
            out = self.avd_last(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out += shortcut
        out = self.act3(out)
        return out


class ResNest(nn.Module):
    """ResNest Implementation.
    Reference:
        BottleNeck: https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/resnest.py#L53
        SplitAttn : https://github.com/rwightman/pytorch-image-models/blob/main/timm/layers/split_attn.py#L33

    Basic ResNest starts with depth=50,
    which means BottleNeck code was only available.
    BasicBlocks for depth=10, 18, 34 were reimplemented base on BottleNeck
    """

    def __init__(
        self,
        block,
        layers,
        block_inplanes,
        n_input_channels=1,
        conv1_t_size=7,
        conv1_t_stride=1,
        no_max_pool=False,
        shortcut_type="B",
        widen_factor=1.0,
        block_cfg: dict = None,
    ):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        # First convolution layer with kernel_size=7 is substited as the following
        # due to resnest paper
        self.conv1 = nn.Sequential(
            *[
                nn.Conv3d(n_input_channels, self.in_planes, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(self.in_planes),
                nn.ReLU(inplace=True),
                nn.Conv3d(self.in_planes, self.in_planes, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(self.in_planes),
                nn.ReLU(inplace=True),
                nn.Conv3d(self.in_planes, self.in_planes, 3, stride=1, padding=1, bias=False),
            ]
        )
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.AvgPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block, block_inplanes[0], layers[0], shortcut_type, block_cfg=block_cfg
        )
        self.layer2 = self._make_layer(
            block, block_inplanes[1], layers[1], shortcut_type, stride=2, block_cfg=block_cfg
        )
        self.layer3 = self._make_layer(
            block, block_inplanes[2], layers[2], shortcut_type, stride=2, block_cfg=block_cfg
        )

        self.more_depth = len(block_inplanes) == 4
        if self.more_depth:
            self.layer4 = self._make_layer(
                block, block_inplanes[3], layers[3], shortcut_type, stride=2, block_cfg=block_cfg
            )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, block_cfg: dict = None):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = downsample_avg(
                in_channels=self.in_planes,
                out_channels=planes * block.expansion,
                kernel_size=1,
                stride=stride,
            )
        layers = []
        layers.append(
            block(
                inplanes=self.in_planes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                **block_cfg,
            )
        )
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, **block_cfg))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        if self.more_depth:
            x4 = self.layer4(x3)
            return x1, x2, x3, x4
        else:
            return x1, x2, x3


def generate_resnest(model_depth, in_planes, block_cfg: dict = None, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]
    if block_cfg is None:
        block_cfg = dict(cardinality=1, avd=True, avd_first=False, radix=2)

    if model_depth == 10:
        model = ResNest(ResNestBasicBlock, [1, 1, 1, 1], in_planes, block_cfg=block_cfg, **kwargs)
    elif model_depth == 18:
        model = ResNest(ResNestBasicBlock, [2, 2, 2, 2], in_planes, block_cfg=block_cfg, **kwargs)
    elif model_depth == 34:
        model = ResNest(ResNestBasicBlock, [3, 4, 6, 3], in_planes, block_cfg=block_cfg, **kwargs)
    elif model_depth == 50:
        model = ResNest(ResNestBottleneck, [3, 4, 6, 3], in_planes, block_cfg=block_cfg, **kwargs)
    elif model_depth == 101:
        model = ResNest(ResNestBottleneck, [3, 4, 23, 3], in_planes, block_cfg=block_cfg, **kwargs)
    elif model_depth == 152:
        model = ResNest(ResNestBottleneck, [3, 8, 36, 3], in_planes, block_cfg=block_cfg, **kwargs)
    elif model_depth == 200:
        model = ResNest(ResNestBottleneck, [3, 24, 36, 3], in_planes, block_cfg=block_cfg, **kwargs)
    return model
