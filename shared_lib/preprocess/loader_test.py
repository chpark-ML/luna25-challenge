from unittest import mock

import pytest

import shared_lib.preprocess.loader as loader
from shared_lib.preprocess.loader import DicomLoader

_DICOM_DATA_DIR = "/data/unknown/shared/test_dicoms/loader_test_cases"


@pytest.fixture
def dicom_loader():
    return DicomLoader()


@mock.patch(
    "os.path.getsize",
    side_effect=[
        loader.DicomHandler.MAX_FILE_SIZE - 10,
        loader.DicomHandler.MAX_FILE_SIZE + 10,
        loader.DicomHandler.MAX_FILE_SIZE - 10,
        loader.DicomHandler.MAX_FILE_SIZE + 10,
    ],
)
@mock.patch("os.path.isfile", return_value=True)
@mock.patch(
    "os.walk",
    return_value=[
        ("data_root", ["x", "y", "z"], ["file1"]),
        ("data_root/x", [], ["file2"]),
        ("data_root/y", ["y1", "y2"], ["file3"]),
        ("data_root/z", ["z1", "z2"], ["file4"]),
    ],
)
def test_get_dicom_list_read_dcm_only_not(mock_listdir, mock_isfile, mock_size):
    handler = loader.DicomHandler()
    assert not handler.params.read_dcm_only
    path_to_dir = "data_root"
    with pytest.warns(
        UserWarning, match="There were 2 files that were filtered out. File size threshold: 4194304"
    ):
        dicom_list = handler._get_dicom_list(path_to_dir)
    assert dicom_list == ["data_root/file1", "data_root/y/file3"]


def test_weird_case_float_image_orientation(dicom_loader):
    """DicomLoader compares ImageOrientation metadata with integer array [1, 1, 0, 0, 1].
    However, there may be some cases that contain float array, which is close to our target.
    Below dicom file contains float ImageOrientation and DicomLoader should read this dicom without error
    """
    dcm_file = (
        f"{_DICOM_DATA_DIR}/segmed_float_image_orientation/"
        f"1.3.6.1.4.1.55648.21660590669081573413953782385479867559.2.63.green.dcm"
    )

    dicom_loader._get_grouped_list([dcm_file], "0x0020000E")
    assert len(dicom_loader.dicom_groups) == 1
