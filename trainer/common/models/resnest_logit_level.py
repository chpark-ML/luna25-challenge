import logging
from typing import Optional

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig

from trainer.common.models.modules.classifier import Classifier
from trainer.common.models.modules.zero_conv import ZeroConv3d
from trainer.common.utils.enums import LOGIT_KEY
from trainer.common.utils.model_output import ModelOutputCls

logger = logging.getLogger(__name__)


class LogitLevelFusionModel(nn.Module):
    def __init__(
        self,
        model_patch: DictConfig,
        model_image: DictConfig,
        fusion_channels: int,
        path_patch_model: Optional[str] = None,
    ) -> None:
        super().__init__()
        
        # Initialize patch model (UNet3DMS)
        self.model_patch = hydra.utils.instantiate(model_patch)
        if path_patch_model is not None:
            logger.info(f"Loading patch model from {path_patch_model}")
            self.model_patch.load_state_dict(torch.load(path_patch_model)["model"])
        
        # Initialize image model (ResNest)
        self.model_image = hydra.utils.instantiate(model_image)
        
        # Initialize image classifier
        self.image_classifier = Classifier(
            in_planes=fusion_channels,
            out_planes=1,
            drop_prob=0.05,
            target_attr_total=self.model_patch.classifier.target_attr_total,
            target_attr_to_train=self.model_patch.classifier.target_attr_to_train,
            target_attr_downstream=self.model_patch.classifier.target_attr_downstream,
            return_logit=False
        )
        
        # Zero conv for logit-level fusion
        self.zero_conv = ZeroConv3d(1, 1)  # 1 channel for binary classification
        
    def forward(self, patch_image, image_large):
        # Patch model forward
        patch_features = []
        x = patch_image
        for encoder in self.model_patch.encoders:
            x = encoder(x)
            patch_features.append(x)
        patch_features = patch_features[-1]
        patch_logits = self.model_patch.classifier([patch_features])
        
        # Image model forward
        image_x1, image_x2, image_x3 = self.model_image.backbone(image_large)
        image_x3 = image_x3.mean(dim=[2, 3, 4], keepdim=True)  # Global average pooling
        image_logits = self.image_classifier(image_x3)
        
        # Logit-level fusion with zero conv
        adjusted_image_logits = self.zero_conv(image_logits)
        final_logits = patch_logits + adjusted_image_logits
        
        if self.model_patch.return_downstream_logit:
            return final_logits[LOGIT_KEY][self.model_patch.classifier.target_attr_downstream]
        
        if self.model_patch.return_named_tuple:
            return ModelOutputCls(**final_logits[LOGIT_KEY])
        else:
            return final_logits
