import torch.nn as nn

from projects.common.models.modules.unet_modules import DoubleConv, create_decoders, create_encoders


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        final_sigmoid,
        basic_module=DoubleConv,
        f_maps=64,
        layer_order="gcr",
        num_groups=8,
        num_levels=4,
        is_segmentation=True,
        conv_kernel_size=3,
        pool_kernel_size=2,
        conv_padding=1,
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

        # semantic segmentation problem
        if is_segmentation:
            self.final_activation = nn.Sigmoid() if final_sigmoid else nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # Apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network
        # outputs probs
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)

        return x
