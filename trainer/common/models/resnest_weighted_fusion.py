import logging

import torch
import torch.nn as nn

from shared_lib.model_output import ModelOutputCls
from trainer.common.constants import LOGIT_KEY
from trainer.common.models.modules.zero_conv import ZeroConv3d

logger = logging.getLogger(__name__)


class WeightedFusionModel(nn.Module):
    def __init__(
        self,
        model_patch,
        model_image,
        in_channels=1,
        out_channels=1,
        fusion_channels=192,  # Default to 192 to match UNet3DMS
        path_patch_model=None,
        widen_factor=1.0,  # Get from config
        in_planes=[96, 128, 192],  # Get from config
    ):
        super().__init__()
        self.model_patch = model_patch
        self.model_image = model_image

        # Load pretrained weights for patch model if provided
        if path_patch_model is not None:
            logger.info(f"Loading pretrained weights for patch-level model from: {path_patch_model}")
            checkpoint = torch.load(path_patch_model)
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            self.model_patch.load_state_dict(state_dict, strict=False)
            logger.info("Successfully loaded patch level model")

        # Freeze patch-level model
        for param in self.model_patch.parameters():
            param.requires_grad = False

        # Freeze image model's classifier, only train encoder
        for param in self.model_image.classifier.parameters():
            param.requires_grad = False

        # Get the last channel size from ResNest config
        last_channel = int(in_planes[-1] * widen_factor)
        logger.info(f"ResNest last channel size: {last_channel}")

        # Channel conversion layer to match UNet3DMS channels
        self.channel_conv = nn.Conv3d(last_channel, fusion_channels, kernel_size=1)

        # Fusion layer to combine features
        self.zero_conv = ZeroConv3d(fusion_channels, fusion_channels)

        # Use patch model's classifier for final prediction
        self.classifier = self.model_patch.classifier

    def forward(self, patch_image, image_large):
        # Extract features from patch model (UNet3DMS)
        patch_features = []
        x = patch_image
        for encoder in self.model_patch.encoders:
            x = encoder(x)
            patch_features.append(x)
        patch_features = patch_features[-1]  # Get the last encoder output

        # Extract features from image model (ResNest)
        image_x1, image_x2, image_x3 = self.model_image.backbone(image_large)

        # Use the last layer features for fusion
        # Reshape image features to match patch features
        b, c, d, h, w = patch_features.shape
        image_x3 = image_x3.mean(dim=[2, 3, 4], keepdim=True)  # Global average pooling
        image_x3 = self.channel_conv(image_x3)  # Convert to 192 channels
        image_x3 = image_x3.repeat(1, 1, d, h, w)  # Repeat to match patch feature dimensions

        # Fuse features
        fused_features = patch_features + self.zero_conv(image_x3)

        # Use patch model's classifier for final prediction
        logits = self.model_patch.classifier([fused_features])

        if self.model_patch.return_downstream_logit:
            return logits[LOGIT_KEY][self.model_patch.classifier.target_attr_downstream]

        if self.model_patch.return_named_tuple:
            return ModelOutputCls(**logits[LOGIT_KEY])
        else:
            return logits
