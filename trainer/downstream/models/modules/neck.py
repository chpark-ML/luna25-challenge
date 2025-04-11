from typing import List

import torch
import torch.nn as nn

from trainer.common.models.modules.activated_batch_norm import ABN
from trainer.common.models.bifpn.bifpn_3d import BiFPN


class Neck(nn.Module):
    def __init__(
        self,
        encoder_channels: list,
        widen_factor: float = 1.0,
        pyramid_channels: int = 2,
        num_fpn_layers: int = 1,
        output_feature_index: int = -1,
    ):
        super().__init__()
        encoder_channels = [int(x * widen_factor) for x in encoder_channels][::-1]

        norm_layer = ABN
        bn_args = dict(norm_layer=norm_layer, norm_act="swish")
        self.bifpn = BiFPN(
            encoder_channels=encoder_channels,
            pyramid_channels=pyramid_channels,
            num_layers=num_fpn_layers,
            **bn_args,
        )
        self.output_feature_index = output_feature_index

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        # x:
        # [0]: (B, 48, 24, 40, 40)
        # [1]: (B, 64, 12, 20, 20)
        # [2]: (B, 96,  6, 10, 10)
        x = self.bifpn(x[::-1])

        # x:
        # [0]: (B, 96,  6, 10, 10)
        # [1]: (B, 64, 12, 20, 20)
        # [2]: (B, 48, 24, 40, 40)
        return x[self.output_feature_index]
