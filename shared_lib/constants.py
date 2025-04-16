""" This module contains commonly used constants."""


class DataLakeKeyDict:
    HFILE_PATH = "h5_path"  # e.g., /nvme1/...
    HFILE_IMAGE = "hfile_image_key"  # e.g., dicom_pixels, dicom_pixels_resampled, etc.
    SPACING = "spacing"
    ORIGIN = "origin"
    CONSTANT_MAPPER = "constant_mapper"
    FIELD_NAME_MAPPER = "field_name_mapper"  # feature mapper for dataset-specific variations with identical context.


class DatasetInfoKey:
    TOTAL_FOLD = "total_fold"
    VALIDATE_FOLD = "val_fold"
    TEST_FOLD = "test_fold"
    COLLECTION_NAME = "collection_name"
    QUERY = "query"
    FOLD = "fold"
    DATASET = "dataset"
