import torch
from torch import nn


class ResBiNet(nn.Module):
    def __init__(self, backbone: nn.Module, neck: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x):
        # backbone
        # x1: (B, 48, 24, 40, 40)
        # x2: (B, 64, 12, 20, 20)
        # x3: (B, 96, 6, 10, 10)
        x1, x2, x3 = self.backbone(x)

        # neck
        x = self.neck((x1, x2, x3))  # (B, 80, 24, 40, 40)

        x = nn.AvgPool3d(x.size()[-3:])(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.head(x)

        return logits
