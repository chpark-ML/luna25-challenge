import torch.nn as nn

from trainer.common.models.modules.bifpn.activated_batch_norm import ABN
from trainer.common.models.modules.bifpn.bifpn_3d import BiFPN
from trainer.common.models.modules.coord_conv import CoordConv3d


class GateBlock(nn.Module):
    def __init__(
        self,
        in_planes: list,
        pyramid_channels: int,
        num_fpn_layers: int,
        drop_prob: float,
        use_coord: bool,
        use_fusion: bool,
        target_attr_total: list,
    ):
        super(GateBlock, self).__init__()
        self.num_features = len(in_planes)
        self.use_coord = use_coord
        self.use_fusion = use_fusion
        self.target_attr_total = target_attr_total
        self.pyramid_channels = pyramid_channels
        self.num_fpn_layers = num_fpn_layers
        self.embedding_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    (
                        CoordConv3d(
                            in_channels=in_plane,
                            out_channels=in_plane * 2,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            dilation=1,
                            groups=1,
                            bias=True,
                            with_r=True,
                            device=None,
                        )
                        if use_coord
                        else nn.Conv3d(in_plane, in_plane * 2, kernel_size=(1, 1, 1), bias=False)
                    ),
                    nn.BatchNorm3d(in_plane * 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout3d(p=drop_prob),
                    nn.Conv3d(
                        in_plane * 2,
                        in_plane,
                        kernel_size=(1, 1, 1),
                        bias=False,
                    ),
                    nn.BatchNorm3d(in_plane),
                    nn.ReLU(inplace=True),
                )
                for in_plane in in_planes
            ]
        )

        if self.use_fusion:
            norm_layer = ABN
            bn_args = dict(norm_layer=norm_layer, norm_act="swish")
            self.bifpn = BiFPN(
                encoder_channels=in_planes[::-1],  # Number of channels for each feature map from low res to high res
                pyramid_channels=self.pyramid_channels,
                num_layers=num_fpn_layers,
                **bn_args,
            )
        else:
            self.bifpn = None

        self.gate_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout3d(p=drop_prob),
                    nn.Conv3d(
                        self.pyramid_channels if self.use_fusion else in_plane,
                        self.pyramid_channels // 2 if self.use_fusion else in_plane // 2,
                        kernel_size=(1, 1, 1),
                        bias=False,
                    ),
                    nn.BatchNorm3d(self.pyramid_channels // 2 if self.use_fusion else in_plane // 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout3d(p=drop_prob),
                    nn.Conv3d(
                        self.pyramid_channels // 2 if self.use_fusion else in_plane // 2,
                        1,
                        kernel_size=(1, 1, 1),
                        bias=True,
                    ),
                    nn.Sigmoid(),
                )
                for in_plane in in_planes
            ]
        )

    def forward(self, fmaps: list):
        assert len(fmaps) == self.num_features

        embedded = list()
        for i in range(self.num_features):
            embedded.append(self.embedding_blocks[i](fmaps[i]))

        if self.use_fusion:
            embedded = self.bifpn(embedded[::-1])[::-1]

        gate = list()
        for idx_fmap, (x, gate_layer) in enumerate(zip(embedded, self.gate_layers)):
            gate.append(gate_layer(x))

        return gate
