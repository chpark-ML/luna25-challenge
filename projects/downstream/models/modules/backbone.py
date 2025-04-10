import torch.nn as nn

from projects.common.models.third_party.resnest import generate_resnest


class BackBone(nn.Module):
    def __init__(self, model_depth, in_planes, widen_factor, block_cfg):
        super().__init__()
        self.backbone = generate_resnest(
            model_depth=model_depth, in_planes=in_planes, widen_factor=widen_factor, block_cfg=block_cfg
        )

    def forward(self, x):
        # x1: (B, 48, 24, 40, 40)
        # x2: (B, 64, 12, 20, 20)
        # x3: (B, 96, 6, 10, 10)
        x1, x2, x3 = self.backbone(x)

        return x1, x2, x3
