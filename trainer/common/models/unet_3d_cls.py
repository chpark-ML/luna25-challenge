import torch
import torch.nn as nn

from trainer.common.models.modules.unet_modules import DoubleConv, create_encoders


class Model(nn.Module):
    def __init__(
        self,
        in_channels,
        basic_module=DoubleConv,
        f_maps=64,
        layer_order="gcr",
        num_groups=8,
        num_levels=4,
        conv_kernel_size=3,
        pool_kernel_size=2,
        conv_padding=1,
    ):
        super(Model, self).__init__()

        if isinstance(f_maps, int):
            f_maps = [f_maps * 2**k for k in range(num_levels)]  # e.g., [24, 48, 96, 192]

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        # create encoder path
        self.encoders = create_encoders(
            in_channels,
            f_maps,
            basic_module,
            conv_kernel_size,
            conv_padding,
            layer_order,
            num_groups,
            pool_kernel_size,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.01),
            nn.Linear(f_maps[-1], 1, bias=True),
        )

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.append(x)
            # Feature map size : (B, 24, 48, 72, 72), (B, 48, 24, 36, 36), (B, 96, 12, 18, 18), (B, 192, 6, 9, 9)
            # RF and padding : (14, 2), (32, 4), (68, 8)

        x = encoders_features[-1]
        x = nn.AvgPool3d(x.size()[-3:])(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)

        return logits
