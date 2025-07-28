"""
This code converts 3D volumes and segmentation annotations obtained from pylidc into training data and stores them.
input: mongoDB (pylidc-image, pylidc-nodule-cluster)
output: h5 files (mask annotation), updated pylidc-image
"""

import argparse
import logging
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from data_lake.constants import TARGET_DB, DataLakeKey
from data_lake.lidc.constants import ClusterLevelInfo, CollectionName, HFileKey, ImageLevelInfo
from data_lake.utils.client import get_client
from shared_lib.utils.utils_logger import setup_logger

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="processing patch-level mask annotation to image-level")
    parser.add_argument("--do_save_h5", type=bool, default=True)
    args = parser.parse_args()
    logger.info("start mask processing")

    # MongoDB client
    client = get_client()

    # set projection
    projection = {DataLakeKey.DOC_ID: 0}  # do not show '_id' of documents

    # get pylidc-images
    pylidc_images = [x for x in client[TARGET_DB][CollectionName.IMAGE].find({}, projection)]

    for pylidc_image in tqdm(pylidc_images):
        # get pylidc-clusters given series instance UID
        series_uid = pylidc_image[ImageLevelInfo.SEREIS_INSTANCE_UID]
        query = {f"{ClusterLevelInfo.DICOM_SERIES_INFO}.{ImageLevelInfo.SEREIS_INSTANCE_UID}": {"$in": [series_uid]}}
        pylidc_clusters = [x for x in client[TARGET_DB][CollectionName.CLUSTER].find(query, projection)]

        # get empty mask image
        h5_file_path = pylidc_image[ImageLevelInfo.H5_FILE_PATH]
        with h5py.File(h5_file_path, mode="r") as hf:
            img = hf[HFileKey.HFileAttrName.DICOM_PIXELS]
            mask = np.zeros_like(img, dtype=np.int16)

        # fill out mask image
        for pylidc_cluster in pylidc_clusters:
            mask_zyx = np.asarray(pylidc_cluster[ClusterLevelInfo.MASK_ZYX], dtype=np.int16)
            bbox_zyx = pylidc_cluster[ClusterLevelInfo.BBOX_ZYX]

            z_slice = slice(bbox_zyx[0]["start"], bbox_zyx[0]["stop"], bbox_zyx[0]["step"])
            y_slice = slice(bbox_zyx[1]["start"], bbox_zyx[1]["stop"], bbox_zyx[1]["step"])
            x_slice = slice(bbox_zyx[2]["start"], bbox_zyx[2]["stop"], bbox_zyx[2]["step"])
            mask[z_slice, y_slice, x_slice] = mask_zyx

        # save result into h5py file, which is indicated by lct/pylidc-image in mongoDB
        if args.do_save_h5:
            with h5py.File(h5_file_path, mode="a", libver="latest") as hf:
                hf.create_dataset(
                    name=HFileKey.HFileAttrName.MASK_ANNOTATION,
                    data=mask,
                    dtype=np.int16,
                    shuffle=True,
                    compression="gzip",
                    compression_opts=1,
                )


if __name__ == "__main__":
    _THIS_DIR = os.path.dirname(os.path.realpath(__file__))
    setup_logger(save_path=os.path.join(_THIS_DIR, f"{Path(__file__).stem}.log"))
    main()
