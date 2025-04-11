"""
pylidc에서 제공하는 consensus bbox 및 mask를 database에 격납합니다.

1. database: lct / collection: pylidc-image / pylidc-nodule / pylidc-nodule-cluster
2. whole CT image는 nvme1에 저장 / array는 list로 저장 / slice object는 dictionary로 저장

# pylidc의 numpy issue 해결: pip install numpy==1.23.0
"""

import argparse
import functools
import logging
import multiprocessing
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pylidc as pl
from tqdm import tqdm

from data_lake.constants import DEFAULT_RESAMPLED_SPACING, TARGET_DB, DataLakeKey
from data_lake.lidc.constants import (
    ClusterLevelInfo,
    CollectionName,
    HFileKey,
    ImageLevelInfo,
    LidcKeyDict,
    NoduleLevelInfo,
)
from data_lake.lidc.enums import NoduleAttribute
from data_lake.utils.client import get_client
from shared_lib.utils.utils_logger import setup_logger

logger = logging.getLogger(__name__)

# the latest pylidc library still use the deprecated numpy operators, e.g., np.int, np.float, etc.
np.int = np.int32
np.float = np.float32
np.bool = np.bool_

_EXPECTED_NUM_OF_SUBJECTS = 1308
_EXPECTED_NUM_OF_UNIQUE_SUBJECTS = 1010


def _permute_yxz_to_zyx(np_array):
    return np.transpose(np_array, (2, 0, 1))


def _get_image_info(study_uid, series_uid, scan, df_meta_data):
    assert len(df_meta_data[df_meta_data[LidcKeyDict.SERIES_ID] == series_uid][LidcKeyDict.MANUFACTURER]) == 1
    manufacturer = df_meta_data[df_meta_data[LidcKeyDict.SERIES_ID] == series_uid][LidcKeyDict.MANUFACTURER].iloc[0]
    h5_file_path = os.path.join(
        HFileKey.HFilePath.IMAGE_COLLECTION_PATH, f"{study_uid}/{series_uid}/{HFileKey.HFileName}.h5"
    )
    return {
        ImageLevelInfo.STUDY_INSTANCE_UID: getattr(scan, ImageLevelInfo.STUDY_INSTANCE_UID),
        ImageLevelInfo.SEREIS_INSTANCE_UID: series_uid,
        ImageLevelInfo.PATIENT_ID: getattr(scan, ImageLevelInfo.PATIENT_ID),
        ImageLevelInfo.SLICE_THICKNESS: getattr(scan, ImageLevelInfo.SLICE_THICKNESS),
        ImageLevelInfo.SPACING_BETWEEN_SLICES: float(scan.slice_spacing),
        ImageLevelInfo.PIXEL_SPACING: getattr(scan, ImageLevelInfo.PIXEL_SPACING),
        ImageLevelInfo.CONTRAST_USED: getattr(scan, ImageLevelInfo.CONTRAST_USED),
        ImageLevelInfo.NUM_ANNOTATION: len(scan.annotations),
        ImageLevelInfo.NUM_CLUSTER: len(scan.cluster_annotations()),
        ImageLevelInfo.MANUFACTURER: manufacturer,
        ImageLevelInfo.H5_FILE_PATH: h5_file_path,
    }


def _process(input_index, inputs, df_meta_data, is_sanity=False, do_save_h5=False):
    if is_sanity:
        logger.info(f"--start sanity check--")
    else:
        logger.info(f"--start process")

    # get input
    inputs = [inputs[idx] for idx in input_index]

    # MongoDB client
    client = get_client()

    # Loop for series uid
    for series_uid in tqdm(inputs):
        # 1. get image-level info
        scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == series_uid).first()
        if scan is None:
            logger.info(f"scan is None object. series uid : {series_uid}")
        else:
            study_uid = getattr(scan, ImageLevelInfo.STUDY_INSTANCE_UID)
            image_info = _get_image_info(study_uid, series_uid, scan, df_meta_data)
            h5_path_image = image_info[ImageLevelInfo.H5_FILE_PATH]

            # create dataset
            # root_path/study_UID/series_UID/dicom_pixels.h5
            if do_save_h5:
                # dicom path should be defined in advance on ~/.pylidcrc as followed:
                # [dicom]
                # path = /team/team_blu3/lung/data/2_public/LIDC-IDRI-new/volumes/manifest-1600709154662/LIDC-IDRI
                # warn = True
                vol = scan.to_volume()  # numpy int16, e.g., (512, 512, 98) let the each axis y, x, z, respectively.
                vol_zyx = _permute_yxz_to_zyx(vol)  # (98, 512, 512) (z, y, x)

                os.makedirs(Path(h5_path_image).parents[0], exist_ok=True)
                with h5py.File(h5_path_image, mode="w", libver="latest") as hf:
                    hf.create_dataset(
                        name=HFileKey.HFileAttrName.DICOM_PIXELS,
                        data=vol_zyx,
                        dtype=np.int16,
                        shuffle=True,
                        compression="gzip",
                        compression_opts=1,
                    )
                    for i_key, i_value in image_info.items():
                        hf.attrs[i_key] = i_value

            # check if series uid is a unique feature
            collection = client[TARGET_DB][CollectionName.IMAGE]
            docs = [x for x in collection.find({ImageLevelInfo.SEREIS_INSTANCE_UID: {"$in": [series_uid]}})]
            assert len(docs) <= 1

            # update to db
            if len(docs) == 1:
                _filter = {DataLakeKey.DOC_ID: docs[0][DataLakeKey.DOC_ID]}
                newvalues = {"$set": image_info}
                collection.update_one(_filter, newvalues)
            else:
                # upload to db
                collection.insert_one(image_info)

            # 2. get nodule-level info
            feature_names = pl.annotation_feature_names
            for idx_ann, ann in enumerate(scan.annotations):
                # individual annotation
                features = dict()
                features[NoduleLevelInfo.DICOM_SERIES_INFO] = image_info
                features[NoduleLevelInfo.ANNOTATION_ID] = idx_ann

                # features
                for feature_name in feature_names:
                    assert (
                        feature_name in NoduleAttribute._value2member_map_
                    ), f"feature name, {feature_name} is not in dataclass."
                    features[f"{feature_name}"] = getattr(ann, feature_name)

                # diameters, volume
                features[NoduleLevelInfo.DIAMETER] = getattr(ann, NoduleLevelInfo.DIAMETER)
                features[NoduleLevelInfo.VOLUME] = getattr(ann, NoduleLevelInfo.VOLUME)

                # axial의 resampled space에서 diameter / 3D resampled space에서 volume 업데이트
                # diameter, volume의 단위는 각각 mm, mm^3.
                _orig_spacing_mm = [1.0, 1.0, 1.0]
                spacing_ratio = np.divide(_orig_spacing_mm, DEFAULT_RESAMPLED_SPACING)
                features[NoduleLevelInfo.DIAMETER_RESAMPLED] = features[NoduleLevelInfo.DIAMETER] * np.prod(
                    spacing_ratio[1:3]
                ) ** (1 / 2)
                features[NoduleLevelInfo.VOLUME_RESAMPLED] = features[NoduleLevelInfo.VOLUME] * np.prod(spacing_ratio)

                # mask, bbox
                mask = ann.boolean_mask(pad=[(2, 2), (2, 2), (2, 2)])
                bbox = ann.bbox(pad=[(2, 2), (2, 2), (2, 2)])
                mask_zyx = np.array(_permute_yxz_to_zyx(mask), dtype=np.int16)
                bbox_zyx = (bbox[2], bbox[0], bbox[1])
                features[NoduleLevelInfo.MASK_ZYX] = mask_zyx.tolist()
                features[NoduleLevelInfo.BBOX_ZYX] = [
                    {"start": int(s.start), "stop": int(s.stop), "step": s.step} for s in bbox_zyx
                ]

                # check if series uid is a unique feature
                collection = client[TARGET_DB][CollectionName.NODULE]
                docs = [
                    x
                    for x in collection.find(
                        {
                            f"{NoduleLevelInfo.DICOM_SERIES_INFO}.{ImageLevelInfo.SEREIS_INSTANCE_UID}": {
                                "$in": [series_uid]
                            },
                            f"{NoduleLevelInfo.ANNOTATION_ID}": {"$in": [idx_ann]},
                        }
                    )
                ]
                assert len(docs) <= 1

                # update to db
                if len(docs) == 1:
                    _filter = {DataLakeKey.DOC_ID: docs[0][DataLakeKey.DOC_ID]}
                    newvalues = {"$set": features}
                    collection.update_one(_filter, newvalues)
                else:
                    # upload to db
                    collection.insert_one(features)

            # 3. consensus annotation
            nods = scan.cluster_annotations()
            for idx_c_ann, c_ann in enumerate(nods):
                features = dict()
                features[ClusterLevelInfo.DICOM_SERIES_INFO] = image_info
                features[ClusterLevelInfo.CLUSTER_ID] = idx_c_ann

                # cmask : (34, 30, 15)
                # cbbox : (slice(253, 287, None), slice(387, 417, None), slice(55, 70, None))
                # masks : list of (34, 30, 15)
                cmask, cbbox, masks = pl.utils.consensus(c_ann, clevel=0.5, pad=[(2, 2), (2, 2), (2, 2)])
                cmask_zyx = np.array(_permute_yxz_to_zyx(cmask), dtype=np.int16)
                cbbox_zyx = (cbbox[2], cbbox[0], cbbox[1])
                cmask_idx_zyx = np.where(cmask_zyx == 1)
                center_coordinate = [int(cbbox_zyx[i].start + round(cmask_idx_zyx[i].mean())) for i in range(3)]
                num_mask = len(masks)

                feature_names = pl.annotation_feature_names
                feature_names = feature_names + (ClusterLevelInfo.DIAMETER, ClusterLevelInfo.VOLUME)
                for feature_name in feature_names:
                    list_attr = []
                    for i_ann in c_ann:
                        list_attr.append(getattr(i_ann, feature_name))
                    features[f"{feature_name}"] = list_attr

                features[ClusterLevelInfo.D_COORD_ZYX] = center_coordinate
                features[ClusterLevelInfo.NUM_MASK] = num_mask
                features[ClusterLevelInfo.MASK_ZYX] = cmask_zyx.tolist()
                features[ClusterLevelInfo.BBOX_ZYX] = [
                    {"start": int(s.start), "stop": int(s.stop), "step": s.step} for s in cbbox_zyx
                ]

                collection = client[TARGET_DB][CollectionName.CLUSTER]
                docs = [
                    x
                    for x in collection.find(
                        {
                            f"{ClusterLevelInfo.DICOM_SERIES_INFO}.{ImageLevelInfo.SEREIS_INSTANCE_UID}": {
                                "$in": [series_uid]
                            },
                            f"{ClusterLevelInfo.CLUSTER_ID}": {"$in": [idx_c_ann]},
                        }
                    )
                ]
                assert len(docs) <= 1
                # update to db
                if len(docs) == 1:
                    _filter = {DataLakeKey.DOC_ID: docs[0][DataLakeKey.DOC_ID]}
                    newvalues = {"$set": features}
                    collection.update_one(_filter, newvalues)
                else:
                    # upload to db
                    collection.insert_one(features)


def main():
    parser = argparse.ArgumentParser(description="Inference tool for batch inputs")
    parser.add_argument("--limit", type=int, help="number of lines to read from the input file")
    parser.add_argument("--sanity_check", type=bool, default=False)
    parser.add_argument("--clean_documents", type=bool, default=True)
    parser.add_argument("--do_save_h5", type=bool, default=True)
    parser.add_argument("--num_shards", default=16, type=int)
    args = parser.parse_args()
    logger.info(f"Start preparing inputs of multiprocessing")

    # Load metadata
    df = pd.read_csv(LidcKeyDict.LIDC_METADATA_PATH)

    # Check the number of subject
    assert len(df[LidcKeyDict.SUBJECT_ID].values) == _EXPECTED_NUM_OF_SUBJECTS
    assert len(np.unique(df[LidcKeyDict.SUBJECT_ID].values)) == _EXPECTED_NUM_OF_UNIQUE_SUBJECTS
    series_id = np.unique(df[LidcKeyDict.SERIES_ID].values)

    # split input
    inputs = series_id[: args.limit] if args.limit else series_id
    logger.info(f"number of unique patients : {len(inputs)}.")
    input_index_chunks = np.array_split(range(len(inputs)), args.num_shards)

    # sanity check
    if args.sanity_check:
        _process(input_index_chunks[-1], inputs=inputs, df_meta_data=df, is_sanity=True, do_save_h5=False)

    # clean documents in target db, target col with target data source
    if args.clean_documents:
        client = get_client()
        for collection in [CollectionName.IMAGE, CollectionName.NODULE, CollectionName.CLUSTER]:
            col = client[TARGET_DB][collection]
            col.delete_many({})

    # multiprocessing for loading dicom loader
    logger.info(f"Start adding documentation")
    with multiprocessing.Pool(args.num_shards) as p:
        p.map(
            functools.partial(_process, inputs=inputs, df_meta_data=df, is_sanity=False, do_save_h5=args.do_save_h5),
            input_index_chunks,
        )

    logger.info(f"Done.")


if __name__ == "__main__":
    _THIS_DIR = os.path.dirname(os.path.realpath(__file__))
    setup_logger(save_path=os.path.join(_THIS_DIR, f"{Path(__file__).stem}.log"))
    main()
