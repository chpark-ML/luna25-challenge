import torch.nn as nn

from shared_lib.model_output import ModelOutputCls
from trainer.common.constants import LOGIT_KEY
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
        classifier=None,
        return_downstream_logit=False,
        return_named_tuple=False,
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
        self.classifier = classifier

        self.return_downstream_logit = return_downstream_logit
        self.return_named_tuple = return_named_tuple

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.append(x)
            # Feature map size : (B, 24, 48, 72, 72), (B, 48, 24, 36, 36), (B, 96, 12, 18, 18), (B, 192, 6, 9, 9)
            # RF and padding : (14, 2), (32, 4), (68, 8)

        result = self.classifier(encoders_features)

        if self.return_downstream_logit:
            return result[LOGIT_KEY][self.classifier.target_attr_downstream]

        if self.return_named_tuple:
            merged_dict = {**result[LOGIT_KEY]}
            return ModelOutputCls(**merged_dict)
        else:
            return result
