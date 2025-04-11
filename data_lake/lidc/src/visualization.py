import argparse
import functools
import logging
import os
from pathlib import Path

import h5py
import pymongo
from joblib import Parallel, delayed

from data_lake.constants import DB_ADDRESS, TARGET_DB
from data_lake.lidc.constants import ClusterLevelInfo, CollectionName, HFileKey, ImageLevelInfo
from data_lake.lidc.enums import NoduleAttribute
from shared_lib.utils.utils_logger import setup_logger
from shared_lib.utils.utils_vis import save_plot

logger = logging.getLogger(__name__)

_THRESHOLD = 0.5
_LUNG_DB = DB_ADDRESS
_OUTPUT_DIR = f"./fig_volume/c_lobulation_1"


def fn_save_fig(target_data):
    # make directory to save visualization results
    r_coord = target_data[ClusterLevelInfo.R_COORD_ZYX]
    series_uid = target_data[ClusterLevelInfo.DICOM_SERIES_INFO][ImageLevelInfo.SEREIS_INSTANCE_UID]
    save_dir = Path(
        os.path.join(
            _OUTPUT_DIR,
            f"{series_uid}/{r_coord[0]}_{r_coord[1]}_{r_coord[2]}.png",
        )
    )
    os.makedirs(save_dir.parents[0], exist_ok=True)

    # save visualization result
    figure_title = ""
    attr = {ClusterLevelInfo.NUM_MASK: target_data[ClusterLevelInfo.NUM_MASK]}
    for i_attr in NoduleAttribute:
        attr[i_attr] = target_data[f"{i_attr}"]
    with h5py.File(target_data[ClusterLevelInfo.DICOM_SERIES_INFO][ImageLevelInfo.H5_FILE_PATH]) as hfile:
        dicom_voxels = hfile[HFileKey.HFileAttrName.DICOM_PIXELS_RESAMPLED][:]
        if _THRESHOLD:
            mask = hfile[HFileKey.HFileAttrName.MASK_ANNOTATION_RESAMPLED][:] > _THRESHOLD
        else:
            mask = hfile[HFileKey.HFileAttrName.MASK_ANNOTATION_RESAMPLED][:]
        save_plot(
            dicom_voxels,
            mask_image=mask,
            nodule_zyx=r_coord,
            figure_title=figure_title,
            meta=attr,
            use_norm=True,
            save_dir=save_dir,
            dpi=60,
        )

    logger.info(f"{target_data['dicom_series_info']['series_instance_uid']}, done!")


def main():
    parser = argparse.ArgumentParser(description="volume visualization tool")
    args = parser.parse_args()

    # mongoDB, get nodule samples
    client = pymongo.MongoClient(_LUNG_DB)
    projection = {t: 1 for t in []}
    projection["_id"] = 0  # do not show '_id' of documents

    # set query
    query = {
        "c_lobulation": {"$in": [1]},
        "num_mask": {"$gte": 2},
    }
    data = [x for x in client[TARGET_DB][CollectionName.CLUSTER].find(query, projection)]
    logger.info(f"num data : {len(data)}")

    # run visualization func
    func = delayed(functools.partial(fn_save_fig))
    with Parallel(n_jobs=1, prefer="threads", verbose=True) as parallel:
        parallel(func(target_data) for target_data in data)


if __name__ == "__main__":
    _THIS_DIR = os.path.dirname(os.path.realpath(__file__))
    setup_logger(save_path=os.path.join(_THIS_DIR, f"{Path(__file__).stem}.log"))
    main()
