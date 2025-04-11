"""
개별 spacing 정보를 받아서 cupy interpolation을 통해서 dicom_pixels와 mask의 spacing 정보를
일관되게 (1, 0.67, 0.67)로 프로세싱하고 저장합니다.
"""

import argparse
import logging
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from data_lake.constants import DEFAULT_RESAMPLED_SPACING, TARGET_DB, DataLakeKey
from data_lake.lidc.constants import CollectionName, HFileKey, ImageLevelInfo
from data_lake.utils.client import get_client
from data_lake.utils.resample_image import resample_image
from shared_lib.utils.utils_logger import setup_logger

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="resampling to the consistent spacing")
    args = parser.parse_args()

    logger.info("start image resampling")

    # MongoDB client
    client = get_client()

    # get pylidc-images
    projection = {DataLakeKey.DOC_ID: 0}  # do not show '_id' of documents
    pylidc_images = [x for x in client[TARGET_DB][CollectionName.IMAGE].find({}, projection)]

    for pylidc_image in tqdm(pylidc_images):
        spacing_between_slices = pylidc_image[ImageLevelInfo.SPACING_BETWEEN_SLICES]
        pixel_spacing = pylidc_image[ImageLevelInfo.PIXEL_SPACING]
        original_spacing_zyx = [
            spacing_between_slices,
            pixel_spacing,
            pixel_spacing,
        ]
        h5_file_path = pylidc_image[ImageLevelInfo.H5_FILE_PATH]

        # save result into h5py file, which is indicated by lct/pylidc-image in mongoDB
        with h5py.File(h5_file_path, mode="a", libver="latest") as hf:
            dicom_pixels = hf[HFileKey.HFileAttrName.DICOM_PIXELS][:]
            mask = hf[HFileKey.HFileAttrName.MASK_ANNOTATION][:]

            # resampling
            dicom_pixels_resampled = resample_image(dicom_pixels, original_spacing_zyx, DEFAULT_RESAMPLED_SPACING)
            mask_annotation_resampled = resample_image(mask, original_spacing_zyx, DEFAULT_RESAMPLED_SPACING)

            # save results
            if HFileKey.HFileAttrName.DICOM_PIXELS_RESAMPLED in hf.keys():
                del hf[HFileKey.HFileAttrName.DICOM_PIXELS_RESAMPLED]
            hf.create_dataset(
                name=HFileKey.HFileAttrName.DICOM_PIXELS_RESAMPLED,
                data=dicom_pixels_resampled,
                dtype=np.float16,
                shuffle=True,
                compression="gzip",
                compression_opts=1,
            )
            if HFileKey.HFileAttrName.MASK_ANNOTATION_RESAMPLED in hf.keys():
                del hf[HFileKey.HFileAttrName.MASK_ANNOTATION_RESAMPLED]
            hf.create_dataset(
                name=HFileKey.HFileAttrName.MASK_ANNOTATION_RESAMPLED,
                data=mask_annotation_resampled,
                dtype=np.float16,
                shuffle=True,
                compression="gzip",
                compression_opts=1,
            )


if __name__ == "__main__":
    _THIS_DIR = os.path.dirname(os.path.realpath(__file__))
    setup_logger(save_path=os.path.join(_THIS_DIR, f"{Path(__file__).stem}.log"))
    main()
