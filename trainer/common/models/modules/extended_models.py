import torch.nn as nn

from trainer.common.models.third_party.monai import swin_unetr as model_swin


class SwinUETRWithActivation(model_swin.SwinUNETR):
    def __init__(
        self,
        xy_size,
        z_size,
        in_channels,
        out_channels,
        feature_size,
        norm_name,
        use_checkpoint,
        final_sigmoid,
    ):
        super().__init__(
            [xy_size, xy_size, z_size],
            in_channels,
            out_channels,
            feature_size=feature_size,
            norm_name=norm_name,
            use_checkpoint=use_checkpoint,
        )
        self.final_activation = nn.Sigmoid() if final_sigmoid else nn.Softmax(dim=1)

    def forward(self, x):
        x = super().forward(x)

        # Apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network
        # outputs logits
        if not self.training and self.final_activation:
            x = self.final_activation(x)

        return x
