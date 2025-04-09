from typing import List

import numpy as np
import torch


def world_to_voxel(world_coords: List[float], origin: List[float], spacing: List[float]) -> List[float]:
    return (np.abs(np.array(world_coords) - np.array(origin)) / np.array(spacing)).tolist()


def map_coord_to_resampled(d_coord, orig_shape, spacing, resampled_spacing=(1.0, 1.0, 1.0)):
    scale_factors = [spacing[i] / resampled_spacing[i] for i in range(3)]
    new_shape = [int(orig_shape[i] * scale_factors[i]) for i in range(3)]
    new_coord = tuple(d_coord[i] * (new_shape[i] - 1) / (orig_shape[i] - 1) for i in range(3))

    return new_coord


def resample_image(orig_image, spacing, resampled_spacing=(1.0, 1.0, 1.0)):
    vol_tensor = torch.tensor(orig_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, z, y, x)
    scale_factors = torch.tensor(
        [spacing[0] / resampled_spacing[0], spacing[1] / resampled_spacing[1], spacing[2] / resampled_spacing[2]]
    )
    size = [int(dim * scale) for dim, scale in zip(orig_image.shape, scale_factors)]

    # Resample the volume to the target size using trilinear interpolation.
    # With align_corners=True, the interpolation maps the corner voxels of the input
    # and output volumes to exactly the same spatial locations. This ensures that the
    # spatial relationship (e.g., physical alignment) is preserved between the original
    # and resampled volumes — important for medical imaging and coordinate mapping.
    resampled = torch.nn.functional.interpolate(vol_tensor, size=size, mode="trilinear", align_corners=True)

    return resampled.squeeze().numpy()
