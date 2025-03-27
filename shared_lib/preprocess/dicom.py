from enum import Enum

import numpy as np


class DicomMode(Enum):
    BONE = 1
    LUNG = 2
    MEDIASTINAL = 3
    VUNO = 4


# Values are a tuple of dicom value's window level and window width.
DICOM_VALUE_RANGES = {
    DicomMode.BONE: (400, 1800),  # -500 ~ 1300
    DicomMode.LUNG: (-600, 1500),  # -1350 ~ 150
    DicomMode.MEDIASTINAL: (50, 350),  # -125 ~ 225
    DicomMode.VUNO: (-200, 1600),  # -1000 ~ 600
}
_EPSILON = 1.0e-8


class BitDepth(Enum):
    BIT_DEPTH_8 = "uint8"
    BIT_DEPTH_16 = "uint16"


_MAX_PIXEL_VALUES = {BitDepth.BIT_DEPTH_8: 255, BitDepth.BIT_DEPTH_16: 65535}


def _windowing(img, mode: DicomMode, bit_depth_type: BitDepth = None):
    """
    Shift and normalize dicom values based on the given mode to 0 to 1 or scaled to the given BitDepth
    Args:
        img: image to convert
        mode: supported dicom images.
        bit_depth_type: multiply by max value of the bit depth type if defined, otherwise values are 0 to 1
    Returns:
        converted image.
    """
    window_level, window_width = DICOM_VALUE_RANGES[mode]
    lower = window_level - window_width // 2
    upper = window_level + window_width // 2
    return normalize_dicom_values(img, lower, upper, bit_depth_type)


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


def reshape_window(img, mode: DicomMode):
    img = _windowing(img, mode=mode)
    img = np.expand_dims(img, axis=0)
    return img


def apply_all_modes_windowing(img):
    """
    Apply voxel-wise windowing. Slice-wise windowing may cause inconsistent # windowing, leading to nan values.
    Args:
        img: image to apply windowing.
    Returns:
        windowed voxel
    """
    # Create output of shape (C, D, H, W)
    d0 = _windowing(img, DicomMode.BONE)
    d1 = _windowing(img, DicomMode.LUNG)
    d2 = _windowing(img, DicomMode.MEDIASTINAL)
    img = np.concatenate((d0[np.newaxis, ...], d1[np.newaxis, ...], d2[np.newaxis, ...]), axis=0)
    return img
