""" This module contains commonly used constants."""


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
