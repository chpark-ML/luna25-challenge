import torch.nn as nn

from shared_lib.model_output import ModelOutputClsSeg
from trainer.common.constants import LOGIT_KEY, SEG_LOGIT_KEY
from trainer.common.models.modules.unet_modules import DoubleConv, create_decoders, create_encoders


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
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
        **kwargs,
    ):
        super(UNet3D, self).__init__()

        if isinstance(f_maps, int):
            f_maps = [f_maps * 2**k for k in range(num_levels)]  # number_of_features_per_level

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

        # create decoder path
        self.decoders = create_decoders(
            f_maps,
            basic_module,
            conv_kernel_size,
            conv_padding,
            layer_order,
            num_groups,
            upsample=True,
        )

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        # multi-label classifier
        self.classifier = classifier

        self.return_downstream_logit = return_downstream_logit
        self.return_named_tuple = return_named_tuple

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

            # Feature map size :
            # (B, 192, 6, 9, 9), (B, 96, 12, 18, 18), (B, 48, 24, 36, 36), (B, 24, 48, 72, 72),

            # RF and padding :
            # (14, 2), (32, 4), (68, 8)

        reversed_features = list(reversed(encoders_features))
        result = self.classifier(reversed_features)

        if self.return_downstream_logit:
            return result[LOGIT_KEY][self.classifier.target_attr_downstream]

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)
        result[SEG_LOGIT_KEY] = x

        if self.return_named_tuple:
            merged_dict = {**result[LOGIT_KEY], "c_segmentation_logistic": x}
            return ModelOutputClsSeg(**merged_dict)
        else:
            return result
