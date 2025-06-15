import torch
import torch.nn as nn
from trainer.common.models.unet_3d_MS import Model as Unet3DMS
from trainer.common.constants import LOGIT_KEY
from shared_lib.model_output import ModelOutputCls

import logging

logger = logging.getLogger(__name__)

class WeightedFusionModel(nn.Module):
    def __init__(
        self,
        model_patch,
        model_image,
        in_channels=1,
        out_channels=1,
        fusion_channels=64,
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
        self.fusion_conv = nn.Conv3d(fusion_channels, fusion_channels, kernel_size=1)

    def forward(self, patch_image, image_large):
        # Extract features from both models' encoders
        patch_features = []
        x = patch_image
        for encoder in self.model_patch.encoders:
            x = encoder(x)
            patch_features.append(x)
        patch_features = patch_features[-1]  # Get the last encoder output
        
        image_features = []
        x = image_large
        for encoder in self.model_image.encoders:
            x = encoder(x)
            image_features.append(x)
        image_features = image_features[-1]  # Get the last encoder output
        
        # Tile image features to match patch feature dimensions
        b, c, d, h, w = patch_features.shape
        image_features = image_features.mean(dim=[2, 3, 4], keepdim=True)  # Global average pooling
        image_features = image_features.expand(b, c, d, h, w)  # Tile to match patch feature dimensions
        
        # Fuse features
        fused_features = patch_features + self.fusion_conv(image_features)
        
        # Use patch model's classifier for final prediction
        logits = self.model_patch.classifier([fused_features])
        
        if self.model_patch.return_downstream_logit:
            return logits[LOGIT_KEY][self.model_patch.classifier.target_attr_downstream]
        
        if self.model_patch.return_named_tuple:
            return ModelOutputCls(**logits[LOGIT_KEY])
        else:
            return logits
    