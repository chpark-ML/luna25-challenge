import logging

import torch
import torch.nn as nn

from shared_lib.model_output import ModelOutputCls
from trainer.common.constants import LOGIT_KEY
from trainer.common.models.modules.classifier import Classifier
from trainer.common.models.modules.zero_conv import ZeroConv3d

logger = logging.getLogger(__name__)


class LogitLevelFusionModel(nn.Module):
    def __init__(
        self,
        model_patch,
        model_image,
        fusion_channels=192,
        path_patch_model=None,
        use_zero_conv_classifier=False,
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

        # Initialize image classifier
        if use_zero_conv_classifier:
            self.image_classifier = ZeroConv3d(fusion_channels, 1)  # (f, 1) zero conv
            self.use_zero_conv = True
        else:
            self.image_classifier = Classifier(
                in_planes=fusion_channels,
                out_planes=1,
                drop_prob=0.05,
                target_attr_total=self.model_patch.classifier.target_attr_total,
                target_attr_to_train=self.model_patch.classifier.target_attr_to_train,
                target_attr_downstream=self.model_patch.classifier.target_attr_downstream,
                return_logit=False,
            )
            self.use_zero_conv = False

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

        if self.use_zero_conv:
            if image_x3.dim() == 2:
                image_x3 = image_x3.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            elif image_x3.dim() == 3:
                image_x3 = image_x3.unsqueeze(-1).unsqueeze(-1)
            elif image_x3.dim() == 4:
                image_x3 = image_x3.unsqueeze(-1)
            # 이제 [B, C, 1, 1, 1] 보장
            image_logit = self.image_classifier(image_x3)  # [B, 1, 1, 1, 1]
            image_logit = image_logit.squeeze(-1).squeeze(-1).squeeze(-1)  # [B, 1]
            adjusted_image_logits = image_logit  # 추가 zero conv 없음
        else:
            # Use Classifier
            image_logits = self.image_classifier([image_x3])
            image_logit = image_logits[LOGIT_KEY][self.model_patch.classifier.target_attr_downstream]
            image_logit = image_logit.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [B, 1, 1, 1, 1]
            adjusted_image_logits = self.zero_conv(image_logit)
            adjusted_image_logits = adjusted_image_logits.squeeze(-1).squeeze(-1).squeeze(-1)  # [B, 1]

        # Only fuse the target attribute logit
        target_attr = self.model_patch.classifier.target_attr_downstream
        final_logits = {LOGIT_KEY: {target_attr: patch_logits[LOGIT_KEY][target_attr] + adjusted_image_logits}}

        if self.model_patch.return_downstream_logit:
            return final_logits[LOGIT_KEY][target_attr]

        if self.model_patch.return_named_tuple:
            return ModelOutputCls(**final_logits[LOGIT_KEY])
        else:
            return final_logits
