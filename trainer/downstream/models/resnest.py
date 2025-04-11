import torch
from torch import nn

from trainer.common.models.third_party.resnest import generate_resnest


class ResNest(nn.Module):
    def __init__(self, model_depth, in_planes, widen_factor, block_cfg):
        super().__init__()
        self.backbone = generate_resnest(
            model_depth=model_depth, in_planes=in_planes, widen_factor=widen_factor, block_cfg=block_cfg
        )
        in_planes = [int(x * widen_factor) for x in in_planes]
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.01),
            nn.Linear(in_planes[-1], 1, bias=True),
        )

    def forward(self, x):
        # backbone
        # x1: (B, 48, 24, 40, 40)
        # x2: (B, 64, 12, 20, 20)
        # x3: (B, 96, 6, 10, 10)
        x1, x2, x3 = self.backbone(x)

        x = x3
        x = nn.AvgPool3d(x.size()[-3:])(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)

        return logits
