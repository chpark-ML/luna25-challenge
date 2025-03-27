from enum import Enum
from typing import Optional, Sequence

import numpy as np

_EPSILON = 1.0e-8


class BitDepth(Enum):
    BIT_DEPTH_8 = "uint8"
    BIT_DEPTH_16 = "uint16"


_MAX_PIXEL_VALUES = {BitDepth.BIT_DEPTH_8: 255, BitDepth.BIT_DEPTH_16: 65535}


def get_pixels_hu(slices):
    try:
        spacing_between_slices = np.array([slices[0].SpacingBetweenSlices], dtype=np.float32)
    except:
        spacing_between_slices = np.array([slices[0].SliceThickness], dtype=np.float32)

    pixel_spacing = np.array(slices[0].PixelSpacing, dtype=np.float32)
    original_spacing = np.concatenate([spacing_between_slices, pixel_spacing])
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            # note: Applying the slope to the values here.
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    image = np.array(image, dtype=np.int16)
    image[image < -1024] = -1024

    return image, original_spacing


def patch_extract(
    resampled_ct: np.ndarray,
    center_coord: Optional[Sequence[int]] = None,
    voxel_width: int | Sequence[int] = 32,
    floor: bool = True,
    pad_value: int | float = 0,
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


def normalize_planes(
    npzarray: np.ndarray, min_hu: float = -1000.0, max_hu: float = 600.0
) -> np.ndarray:
    if min_hu > max_hu:
        raise ValueError(f"minHU should be smaller than maxHU, but got {min_hu} > {max_hu}")

    npzarray = (npzarray - min_hu) / (max_hu - min_hu)
    return np.clip(npzarray, 0, 1)


def normalize_dicom_values(img, lower, upper, bit_depth_type: BitDepth = None):
    x = np.clip(img.copy(), lower, upper)
    x = x - np.min(x)
    x = x / (np.max(x) + _EPSILON)

    if bit_depth_type:
        x = (x * _MAX_PIXEL_VALUES[bit_depth_type]).astype(bit_depth_type.value)
    return x


def normalize_dicom_values_by_fixed_range(
    img: np.array, lower: float, upper: float, bit_depth_type: BitDepth = None
):
    """
    1. clip HU values
    2. 고정 범위를 (HU windowing values) 통해 minmax 정규화 수행
    """
    x = np.clip(img.copy(), lower, upper)

    x = x - lower
    x = x / ((upper - lower) or _EPSILON)  # guarantee non-zero denominatez

    if bit_depth_type:
        x = (x * _MAX_PIXEL_VALUES[bit_depth_type]).astype(bit_depth_type.value)
    return x


class DicomWindowing:
    def __init__(self, hu_range, fixed=False):
        assert len(hu_range) == 2
        self.lower = hu_range[0]
        self.upper = hu_range[1]
        # 정규화를 위한 min, max value 설정 방법에 따라 두 가지 방법으로 나눠진 기능을 선택, 수행
        self.normalize = normalize_dicom_values_by_fixed_range if fixed else normalize_dicom_values

    def __call__(self, img):
        return self.normalize(img, self.lower, self.upper)
