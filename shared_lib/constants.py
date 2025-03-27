""" This module contains commonly used constants."""

from shared_lib.preprocess.dicom import DICOM_VALUE_RANGES, DicomMode

# Standard voxel spacing
DEFAULT_PIXEL_SPACING = 0.67
DEFAULT_SPACING_BETWEEN_SLICES = 1.0
RESAMPLED_SPACING_ZYX = [
    DEFAULT_SPACING_BETWEEN_SLICES,
    DEFAULT_PIXEL_SPACING,
    DEFAULT_PIXEL_SPACING,
]

# model encryption key
ENCRYPTION_KEY = b"LJI0Glqx63Xtfi_zAWunSOEGNE_tKf_T7ZLKOZaLYpY="

# HU windows
HU_WINDOW = {
    "VUNO": DICOM_VALUE_RANGES[DicomMode.VUNO],
    "LUNG": DICOM_VALUE_RANGES[DicomMode.LUNG],
    "MEDIASTINAL": DICOM_VALUE_RANGES[DicomMode.MEDIASTINAL],
    "BONE": DICOM_VALUE_RANGES[DicomMode.BONE],
}


class DataLakeKeyDict:
    h5_file_path = "h5_path"  # e.g., /nvme1/...
    hfile_image_key = "hfile_image_key"  # e.g., dicom_pixels, dicom_pixels_resampled, etc.
    spacing = "spacing"
    origin = "origin"
    constant_mapper = "constant_mapper"
    field_name_mapper = "field_name_mapper"  # feature mapper for dataset-specific variations with identical context.


class DatasetInfoKey:
    TOTAL_FOLD = "total_fold"
    VALIDATE_FOLD = "val_fold"
    TEST_FOLD = "test_fold"
    COLLECTION_NAME = "collection_name"
    QUERY = "query"
    FOLD = "fold"
    DATASET = "dataset"
