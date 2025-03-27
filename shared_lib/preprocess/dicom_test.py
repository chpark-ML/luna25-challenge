import numpy as np
import pytest

import shared_lib.preprocess.dicom as preprocess_dicom


@pytest.mark.parametrize(
    "bit_depth_type, expected",
    [(None, [0, 0.382, 0.872, 1]), (preprocess_dicom.BitDepth.BIT_DEPTH_8, [0, 97, 222, 254])],
)
def test_windowing(bit_depth_type, expected):
    # image of rows of different values
    in_values = [-1400, -777, -42, 167]
    image = np.stack([np.full(10, val) for val in in_values])

    converted_image = preprocess_dicom._windowing(
        image, preprocess_dicom.DicomMode.LUNG, bit_depth_type=bit_depth_type
    )

    for i, in_values in enumerate(in_values):
        np.testing.assert_array_almost_equal(converted_image[i], expected[i])
