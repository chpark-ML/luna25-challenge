import torch.nn as nn

from trainer.common.models.modules.blocks import PointWiseConvBlock, conv1x1x1


class Head(nn.Module):
    def __init__(self, in_planes_head, drop_out_prob=0.05):
        super(Head, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_out_prob),
            nn.Linear(in_planes_head, 1, bias=True),
        )

    def forward(self, x):
        heatmap_output = self.classifier(x)

        return heatmap_output
