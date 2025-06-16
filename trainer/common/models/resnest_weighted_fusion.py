import torch
import torch.nn as nn
from trainer.common.models.resnest import ResNest
import logging

logger = logging.getLogger(__name__)


class ZeroConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        torch.nn.init.constant_(self.conv.weight, 0)
        if self.conv.bias is not None:
            torch.nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class WeightedFusionModel(nn.Module):
    def __init__(
        self,
        model_patch,
        model_image,
        in_channels=1,
        out_channels=1,
        fusion_channels=96,  # ResNest의 마지막 채널 수와 동일
        path_patch_model=None,
        path_image_model=None,
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

        # Load pretrained weights for image model if provided
        if path_image_model is not None:
            logger.info(f"Loading pretrained weights for image-level model from: {path_image_model}")
            checkpoint = torch.load(path_image_model)
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            self.model_image.load_state_dict(state_dict, strict=False)
            logger.info("Successfully loaded image level model")

        # Freeze patch-level model
        for param in self.model_patch.parameters():
            param.requires_grad = False

        # Freeze image model's classifier, only train encoder
        for param in self.model_image.classifier.parameters():
            param.requires_grad = False

        # Fusion layer to combine features
        self.zero_conv = ZeroConv3d(fusion_channels, fusion_channels)

        # Use patch model's classifier for final prediction
        self.classifier = self.model_patch.classifier

    def forward(self, patch_image, image_large):
        # Extract features from both models' backbones
        patch_x1, patch_x2, patch_x3 = self.model_patch.backbone(patch_image)
        image_x1, image_x2, image_x3 = self.model_image.backbone(image_large)

        # Use the last layer features (x3) for fusion
        # Global average pooling for image features
        b, c, d, h, w = patch_x3.shape
        image_x3 = image_x3.mean(dim=[2, 3, 4], keepdim=True)  # Global average pooling
        image_x3 = image_x3.expand(b, c, d, h, w)  # Tile to match patch feature dimensions

        # Fuse features
        fused_features = patch_x3 + self.zero_conv(image_x3)

        # Global average pooling and classification
        x = nn.AvgPool3d(fused_features.size()[-3:])(fused_features)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)

        return logits
