import logging
from abc import ABC, abstractmethod

import numpy as np
import torch

from shared_lib.tools.image_parser import clip_and_scale, extract_patch


class BaseProcessor(ABC):
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

    def prepare_patch(self, image, header, coord, mode="3D") -> torch.Tensor:
        if not self.suppress_logs:
            logging.info("Processing in " + mode)

        assert mode == "3D"
        output_shape = [self.size_px_z, self.size_px_xy, self.size_px_xy]
        patch = self._extract_patch(image, header, coord, output_shape, mode=mode)  # (1, w, h, d)
        patch = torch.from_numpy(patch).to(self.device)

        return patch.unsqueeze(0)  # 1, 1, w, h, d

    @abstractmethod
    def predict(self, numpy_image, header, coord):
        """
        Perform model inference on the given input image and coordinate.
        """
        pass

    @abstractmethod
    def inference(self, loader, mode: str, sanity_check: bool = False):
        """
        Perform inference using the provided data loader in either 2D or 3D mode.

        This method is designed for use in analysis pipelines, where model predictions
        are required for a given dataset. It supports both 2D and 3D inference modes,
        and can optionally perform a quick sanity check run.

        Args:
            loader: A PyTorch DataLoader that provides the input data.
            mode (str): Inference mode. Must be one of:
                - '2D': for slice-wise or image-wise inference. (currently, not implemented)
                - '3D': for volumetric or sequential inference.
            sanity_check (bool, optional): If True, performs a minimal run to verify
                that the pipeline works correctly (e.g., a few samples). Defaults to False.

        Returns:
            The inference results, in a format defined by the subclass implementation.
        """
        pass
