import logging
from pathlib import Path

import numpy as np
import torch

from shared_lib.tools.image_parser import clip_and_scale, extract_patch


class MalignancyProcessor:
    """
    Loads a chest CT scan, and predicts the malignancy around a nodule
    """

    def __init__(self, models=None, mode="3D", device=torch.device("cuda:0"), suppress_logs=False):
        self.device = device
        self.size_px_xy = 72
        self.size_px_z = 48
        self.size_mm = 50
        self.order = 1

        self.suppress_logs = suppress_logs
        if not self.suppress_logs:
            logging.info("Initializing the deep learning system")

        self.mode = mode
        self.models = models

    def _extract_patch(self, image, header, coord, output_shape, mode) -> np.array:
        patch = extract_patch(
            CTData=image,
            coord=coord,
            srcVoxelOrigin=header["origin"],
            srcWorldMatrix=header["transform"],
            srcVoxelSpacing=header["spacing"],
            output_shape=output_shape,
            voxel_spacing=(
                self.size_mm / self.size_px_z,
                self.size_mm / self.size_px_xy,
                self.size_mm / self.size_px_xy,
            ),
            coord_space_world=True,
            mode=mode,
            order=self.order,
        )  # (1, w, h, d)

        # ensure same datatype...
        patch = patch.astype(np.float32)

        # clip and scale...
        patch = clip_and_scale(patch)
        return patch  # (1, w, h, d)

    def _prepare_patch(self, image, header, coord, mode="3D") -> torch.Tensor:
        if not self.suppress_logs:
            logging.info("Processing in " + mode)

        assert mode == "3D"
        output_shape = [self.size_px_z, self.size_px_xy, self.size_px_xy]
        patch = self._extract_patch(image, header, coord, output_shape, mode=mode)  # (1, w, h, d)
        patch = torch.from_numpy(patch).to(self.device)

        return patch.unsqueeze(0)  # 1, 1, w, h, d

    def predict(self, numpy_image, header, coord):
        patch = self._prepare_patch(numpy_image, header, coord, self.mode)

        probs = list()
        for model_name, model in self.models.items():
            logits = model.get_prediction(patch)
            logits = logits.data.cpu().numpy()
            probs.append(torch.sigmoid(torch.from_numpy(logits)).numpy())

        probs = np.stack(probs, axis=0)  # shape: (num_models, ...)
        mean_probs = np.mean(probs, axis=0)

        return mean_probs
