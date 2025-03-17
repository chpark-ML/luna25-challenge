from typing import Optional, Sequence, Union

import numpy as np


def patch_extract(
    resampled_ct: np.ndarray,
    center_coord: Optional[Sequence[int]] = None,
    voxel_width: Union[int, Sequence[int]] = 32,
    floor: bool = True,
    pad_value: Union[int, float] = 0,
) -> np.ndarray:
    """Extract a 3D patch from the given CT volume.

    Not creating a separate copy of the patch by design.
    The `patch` may be a view of the `volume` array.

    Args:
        resampled_ct: Input volume. May also be a HDF5 memory map.
        center_coord: Center coordinate for patch extraction.
        voxel_width: Patch size.
        floor: Wheter to apply floor function when
            cropping patch. Round otherwise.
        pad_value: Value to pad external region.
            Zero-padding by default.

    Returns:
        A 3D patch extracted from the given volume.
    """
    if not isinstance(voxel_width, int):
        assert len(voxel_width) == 3
    else:
        voxel_width = (voxel_width, voxel_width, voxel_width)

    depth_ct, height_ct, width_ct = resampled_ct.shape  # Forces 3D input.
    if center_coord is None:
        center_coord = [depth_ct // 2, height_ct // 2, width_ct // 2]

    depth_center, height_center, width_center = center_coord

    depth, height, width = voxel_width

    depth_left = depth_center - depth // 2  # Left of depth axis
    height_left = height_center - height // 2  # Left of height axis
    width_left = width_center - width // 2  # Left of width axis
    depth_right = depth_left + depth
    height_right = height_left + height
    width_right = width_left + width

    # `max(x, 0)` prevents negative indexing from creating a length 0 slice.
    # Indexing with values larger than the maximum index is legal in numpy.
    padding = 0.5 if floor else 0
    patch = resampled_ct[
        int(max(0, depth_left) + padding) : int(depth_right + padding),
        int(max(0, height_left) + padding) : int(height_right + padding),
        int(max(0, width_left) + padding) : int(width_right + padding),
    ]

    if patch.shape != voxel_width:
        dp_left = int(abs(min(depth_left, 0)) + 0.5)
        hp_left = int(abs(min(height_left, 0)) + 0.5)
        wp_left = int(abs(min(width_left, 0)) + 0.5)

        dp_right = int(max(depth_right - depth_ct, 0) + 0.5)
        hp_right = int(max(height_right - height_ct, 0) + 0.5)
        wp_right = int(max(width_right - width_ct, 0) + 0.5)
        ddd, hhh, www = patch.shape

        assert (
            dp_left + ddd + dp_right == depth
        ), f"Incorrect depth - dpl:{dp_left} ddd:{ddd} dpr:{dp_right} sum:{dp_left + ddd + dp_right} dp:{depth}"
        assert (
            hp_left + hhh + hp_right == height
        ), f"Incorrect height - hpl:{hp_left} hhh:{hhh} hpr:{hp_right} sum:{hp_left + hhh + hp_right} hp:{height}"
        assert (
            wp_left + www + wp_right == width
        ), f"Incorrect width - wpl:{wp_left} www:{www} wpr:{wp_right} sum:{wp_left + www + wp_right} wp:{width}"

        pad_width = ((dp_left, dp_right), (hp_left, hp_right), (wp_left, wp_right))
        patch = np.pad(patch, pad_width=pad_width, mode="constant", constant_values=pad_value)

    assert patch.shape == voxel_width, f"{patch.shape} != {voxel_width}. Sanity check failed."
    return patch.copy()


def normalize_planes(npzarray: np.ndarray, min_hu: float = -1000.0, max_hu: float = 600.0) -> np.ndarray:
    if min_hu > max_hu:
        raise ValueError(f"minHU should be smaller than maxHU, but got {min_hu} > {max_hu}")

    npzarray = (npzarray - min_hu) / (max_hu - min_hu)
    return np.clip(npzarray, 0, 1)
