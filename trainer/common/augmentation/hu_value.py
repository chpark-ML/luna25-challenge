import random

import numpy as np

from trainer.common.enums import BitDepth

_EPSILON = 1.0e-8
_MAX_PIXEL_VALUES = {BitDepth.BIT_DEPTH_8: 255, BitDepth.BIT_DEPTH_16: 65535}


def normalize_dicom_values(img: np.array, lower: float, upper: float, bit_depth_type: BitDepth = None):
    x = np.clip(img.copy(), lower, upper)

    x = x - lower
    x = x / ((upper - lower) or _EPSILON)  # guarantee non-zero denominatez

    if bit_depth_type:
        x = (x * _MAX_PIXEL_VALUES[bit_depth_type]).astype(bit_depth_type.value)
    return x


class DicomWindowing:
    def __init__(self, hu_range):
        assert len(hu_range) == 2
        self.lower = hu_range[0]
        self.upper = hu_range[1]
        self.normalize = normalize_dicom_values

    def __call__(self, img):
        return self.normalize(img, self.lower, self.upper)


class RandomDicomWindowing:
    def __init__(
        self,
        hu_range: tuple,
        min_width_scale: float = 0.9,
        max_width_scale: float = 1.1,
        min_level_shift: int = -100,
        max_level_shift: int = 100,
        p=0.5,
    ):
        assert len(hu_range) == 2
        assert 0.0 <= p <= 1.0
        assert min_width_scale <= max_width_scale
        assert min_level_shift <= max_level_shift
        self.p = p
        self.hu_range = hu_range
        self.min_width_scale = min_width_scale
        self.max_width_scale = max_width_scale
        self.min_level_shift = min_level_shift
        self.max_level_shift = max_level_shift
        self.normalize = normalize_dicom_values

    def __call__(self, img):
        if random.random() <= self.p:
            width = self.hu_range[1] - self.hu_range[0]
            width_rand = np.random.uniform(width * self.min_width_scale, width * self.max_width_scale)

            level = (self.hu_range[0] + self.hu_range[1]) / 2.0
            level_rand = np.random.uniform(level + self.min_level_shift, level + self.max_level_shift)

            lower = level_rand - width_rand / 2
            upper = level_rand + width_rand / 2
        else:
            lower = self.hu_range[0]
            upper = self.hu_range[1]
        return self.normalize(img, lower, upper)
