import torch.nn as nn

from shared_lib.model_output import ModelOutputCls
from trainer.common.constants import LOGIT_KEY
from trainer.common.models.modules.unet_modules import DoubleConv, create_encoders


class DualScaleModel(nn.Module):
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
        super(DualScaleModel, self).__init__()

        if isinstance(f_maps, int):
            f_maps = [f_maps * 2**k for k in range(num_levels)]

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        # Create encoder path for large scale image only
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
        
        # Classifier
        self.classifier = classifier
        self.return_downstream_logit = return_downstream_logit
        self.return_named_tuple = return_named_tuple

    def forward(self, x_large, patch_features):
        # Get encoder features for large scale only
        encoders_features_large = []
        x_l = x_large
        for encoder in self.encoders:
            x_l = encoder(x_l)
            encoders_features_large.append(x_l)
        
        # Get image features
        image_features = encoders_features_large[-1].mean(dim=[2,3,4])  # (B, C)
        
        # Get prediction using dual scale classifier
        result = self.classifier(patch_features, image_features)
        
        if self.return_downstream_logit:
            return result[LOGIT_KEY][self.classifier.target_attr_downstream]
        
        if self.return_named_tuple:
            return ModelOutputCls(**result[LOGIT_KEY])
        else:
            return result 
