from pathlib import Path
from typing import Union

import torch

from shared_lib.model_runner.base_runner import ModelBaseTorchscript


class ImageLevelRunner(ModelBaseTorchscript):
    def __init__(self, root_path, exp_name, file_name, device: Union[str, torch.device] = None):
        checkpoint_path = Path(root_path) / exp_name / file_name
        super().__init__(checkpoint_path=checkpoint_path, device=device)

    @torch.no_grad()
    def get_prediction(self, patch_image, image_large) -> torch.Tensor:
        """
        Perform inference with no gradient tracking.
        Args:
            patch_image (torch.Tensor): Patch input tensor [B, C, D, H, W]
            image_large (torch.Tensor): Large image input tensor [B, C, D, H, W]
        Returns:
            torch.Tensor: Model output.
        """
        patch_image = patch_image.to(self.device)
        image_large = image_large.to(self.device)

        # Forward pass with both inputs
        output = self.model(patch_image, image_large)

        return output
