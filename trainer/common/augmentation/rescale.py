import random

import numpy as np
import torch
import torch.nn.functional as F


class RescaleImage:
    def __init__(
        self,
        p: float = 0.5,
        is_same_across_axes: bool = False,
        min_scale_factor: float = 1.0,
        max_scale_factor: float = 1.0,
    ):
        assert 0.0 <= p <= 1.0
        self.p = p
        self.is_same_across_axes = is_same_across_axes
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor

    def __call__(self, img: np.ndarray, mask: np.ndarray = None):
        assert img.ndim == 3
        if mask is not None:
            assert mask.ndim == 3
            assert img.shape == mask.shape

        if random.random() <= self.p:
            # random rescale
            if self.is_same_across_axes:
                random_scale_factor = (
                    tuple(
                        np.random.uniform(
                            low=self.min_scale_factor,
                            high=self.max_scale_factor,
                            size=1,
                        )
                    )
                    * 3
                )
            else:
                random_scale_factor = tuple(
                    np.random.uniform(low=self.min_scale_factor, high=self.max_scale_factor, size=3)
                )  # (z_scale_factor, y_, x_)

            # interpolation
            _img = F.interpolate(
                torch.tensor(img).unsqueeze(0).unsqueeze(0),
                scale_factor=random_scale_factor,
                mode="trilinear",
                align_corners=True,
            )  # (1, 1, d, h, w)
            if mask is not None:
                _mask = F.interpolate(
                    torch.tensor(mask).unsqueeze(0).unsqueeze(0),
                    scale_factor=random_scale_factor,
                    mode="trilinear",
                    align_corners=True,
                )  # (1, 1, d, h, w)

            # get padding size, p3d
            image_shape = img.shape
            rescaled_image_shape = np.array(_img.shape[-3:])
            diff = image_shape - rescaled_image_shape
            p3d = tuple(np.array([[int(i // 2), int(i - i // 2)] for i in diff]).ravel()[::-1])

            # zero padding
            img = F.pad(_img, p3d, "constant", 0).squeeze().numpy()
            if mask is not None:
                mask = F.pad(_mask, p3d, "constant", 0).squeeze().numpy()

        if mask is not None:
            return img, mask
        else:
            return img
