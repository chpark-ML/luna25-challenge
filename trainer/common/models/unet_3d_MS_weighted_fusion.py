import torch
import torch.nn as nn
from trainer.common.models.unet_3d_MS import Model as Unet3DMS
from trainer.common.constants import LOGIT_KEY
from trainer.common.models.unet_3d_MS import ModelOutputCls

class WeightedFusionModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        f_maps: int = 24,
        num_levels: int = 4,
        num_groups: int = 4,
        layer_order: str = "gcr",
        return_downstream_logit: bool = True,
        return_named_tuple: bool = False,
        fusion_weight: float = 0.1,  # Initial weight for image-level features
        weight_increment: float = 0.01,  # How much to increment the weight per epoch
        path_patch_model: str = None,  # Path to pretrained patch-level model
        classifier: dict = None,  # Classifier configuration
    ):
        super().__init__()
        
        # Initialize main Unet_3d_MS model
        self.model = Unet3DMS(
            in_channels=in_channels,
            f_maps=f_maps,
            num_levels=num_levels,
            num_groups=num_groups,
            layer_order=layer_order,
            return_downstream_logit=return_downstream_logit,
            return_named_tuple=return_named_tuple,
            classifier=classifier,  # Pass classifier configuration
        )
        
        # Load pretrained weights if provided
        if path_patch_model:
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
            self.model.load_state_dict(state_dict, strict=False)
            
        # Initialize fusion weight and increment
        self.fusion_weight = fusion_weight
        self.weight_increment = weight_increment
        
        # Zero convolution for feature fusion
        self.fusion_conv = nn.Conv3d(f_maps * 2**(num_levels-1), f_maps * 2**(num_levels-1), 1)
        nn.init.zeros_(self.fusion_conv.weight)
        nn.init.zeros_(self.fusion_conv.bias)
        
    def forward(self, patch_image, image_level_image):
        # Get encoder features from patch-level model
        patch_features = []
        x = patch_image
        for encoder in self.model.encoders:
            x = encoder(x)
            patch_features.append(x)
        patch_features = patch_features[-1]  # Get the last encoder output
        
        # Get encoder features from image-level model
        image_features = []
        x = image_level_image
        for encoder in self.model.encoders:
            x = encoder(x)
            image_features.append(x)
        image_features = image_features[-1]  # Get the last encoder output
        
        # Resize image features to match patch features' spatial dimensions
        image_features = torch.nn.functional.interpolate(
            image_features,
            size=patch_features.shape[2:],  # Match spatial dimensions (6, 9, 9)
            mode='trilinear',
            align_corners=False
        )
        
        # Fuse features with weighted sum
        fused_features = patch_features + self.fusion_weight * self.fusion_conv(image_features)
        
        # Get final prediction using the classifier
        result = self.model.classifier([fused_features])
        
        if self.model.return_downstream_logit:
            return result[LOGIT_KEY][self.model.classifier.target_attr_downstream]
        
        if self.model.return_named_tuple:
            return ModelOutputCls(**result[LOGIT_KEY])
        else:
            return result
        
    def increment_fusion_weight(self):
        """Increment the fusion weight by the specified amount"""
        self.fusion_weight = min(1.0, self.fusion_weight + self.weight_increment)
        return self.fusion_weight 