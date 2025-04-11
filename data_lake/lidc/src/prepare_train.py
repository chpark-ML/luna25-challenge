import argparse
import logging
import os
from functools import partial
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from data_lake.constants import TARGET_DB, DataLakeKey, DBKey
from data_lake.dataset_handler import DatasetHandler
from data_lake.lidc.constants import ClusterLevelInfo, CollectionName, HFileKey, ImageLevelInfo, NoduleLevelInfo
from data_lake.lidc.enums import NoduleAttribute
from data_lake.lidc.src.utils import fn_save_corr_heatmap, fn_save_histograms, save_hist_for_target_field
from data_lake.utils.client import get_client
from trainer.common.utils.utils_logger import setup_logger

logger = logging.getLogger(__name__)


class PlotMode:
    HIST = "histogram"
    CORR = "correlation"


def get_patient_level_infos(target_field, target_field_nodule) -> pd.DataFrame:
    dataset_handler = DatasetHandler()

    # Patient-wise feature representation
    dict_patient_info = {ImageLevelInfo.PATIENT_ID: None}
    for i_field in target_field:
        dict_patient_info[i_field] = None
    for i_field in target_field_nodule:
        dict_patient_info[i_field] = None

    PIDs = dataset_handler.fetch_documents(
        collection=CollectionName.IMAGE, query={}, projection={}, field_name=ImageLevelInfo.PATIENT_ID
    )

    df = None
    for idx, i_PID in enumerate(np.unique(PIDs)):
        image_info = dataset_handler.fetch_documents(
            collection=CollectionName.IMAGE, query={ImageLevelInfo.PATIENT_ID: i_PID}, projection={}
        )

        dict_patient_info = {ImageLevelInfo.PATIENT_ID: i_PID}
        for i_field in target_field:
            tmp_list = list()
            for i_image_info in image_info:
                tmp_list.append(i_image_info[i_field])

            if isinstance(i_image_info[i_field], str):
                dict_patient_info[i_field] = tmp_list[0]
            elif isinstance(i_image_info[i_field], bool):
                dict_patient_info[i_field] = any(tmp_list)
            else:
                dict_patient_info[i_field] = sum(tmp_list) / len(tmp_list)

        cluster_info = dataset_handler.fetch_documents(
            collection=CollectionName.CLUSTER,
            query={f"{ClusterLevelInfo.DICOM_SERIES_INFO}.{ImageLevelInfo.PATIENT_ID}": i_PID},
            projection={},
        )
        for i_field in target_field_nodule:
            tmp_list = list()
            for i_cluster_info in cluster_info:
                tmp_list.append(i_cluster_info[i_field])

            if i_field == NoduleAttribute.INTERNAL_STRUCTURE.value:
                dict_patient_info[i_field] = any([1 if item != 1 else 0 for sublist in tmp_list for item in sublist])
            elif i_field == NoduleAttribute.CALCIFICATION.value:
                dict_patient_info[i_field] = any([1 if item != 6 else 0 for sublist in tmp_list for item in sublist])
            elif len(cluster_info) == 0:
                dict_patient_info[i_field] = None
            else:
                flatten_tmp_list = [item for sublist in tmp_list for item in sublist]
                dict_patient_info[i_field] = sum(flatten_tmp_list) / len(flatten_tmp_list)

        if df is not None:
            df_new = pd.DataFrame.from_dict(dict_patient_info, orient="index").T
            df = pd.concat([df, df_new])
        else:
            df = pd.DataFrame.from_dict(dict_patient_info, orient="index").T

    return df


def merge_features(series: pd.Series, strat_features: list):
    """Merge features with underscore into single column."""
    values = []
    for feat in strat_features:
        if feat + "_bin" in series.index:
            feat = feat + "_bin"

        values.append(str(series[feat]))
    return "_".join(values)


def split_features(
    df: pd.DataFrame,
    strat_features: list,
    n_splits: int = 7,
    random_state: int = 1,
    shuffle: bool = True,
) -> Tuple[dict, pd.DataFrame]:
    """Split input df, stratifying `strat_features`"""
    df[DBKey.FOLD] = -1
    df["stratify"] = df.apply(partial(merge_features, strat_features=strat_features), axis=1)

    X = df
    y = df["stratify"]

    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
    test_folds = {}
    # fold dictionaries containing 7 folds, each element being tuple of (train, test) dataframes
    for fold_num, (train_index, test_index) in enumerate(skf.split(X, y)):
        test_folds[fold_num] = (X.iloc[test_index], y.iloc[test_index])
        df.loc[test_index, DBKey.FOLD] = fold_num  # starts from 1

    return test_folds, df


def split_fold(df, n_splits=7) -> pd.DataFrame:
    ## Dataframe을 적어도 하나의 노듈이 있는 patient와 아닌 patient로 나눠서 작업을 수행
    df_with_nodule = df.dropna().copy()
    df_with_no_nodule = df.drop(index=df_with_nodule.index).copy()
    df_with_nodule = df_with_nodule.copy().reset_index(drop=True)
    df_with_no_nodule = df_with_no_nodule.copy().reset_index(drop=True)

    assert len(df_with_nodule) == 875
    assert len(df_with_no_nodule) == 135

    # patient no nodule
    df_with_no_nodule["strat_slice_thickness"] = df_with_no_nodule[ImageLevelInfo.SLICE_THICKNESS].apply(
        lambda x: int(x > 1.5)
    )
    df_with_no_nodule["strat_spacing_between_slices"] = df_with_no_nodule[ImageLevelInfo.SPACING_BETWEEN_SLICES].apply(
        lambda x: int(x > 1.5)
    )
    df_with_no_nodule["strat_pixel_spacing"] = df_with_no_nodule[ImageLevelInfo.PIXEL_SPACING].apply(
        lambda x: int(x > 0.7)
    )
    df_with_no_nodule["strat_contrast_used"] = df_with_no_nodule[ImageLevelInfo.CONTRAST_USED].apply(lambda x: int(x))
    df_with_no_nodule["strat_num_cluster"] = df_with_no_nodule[ImageLevelInfo.NUM_CLUSTER].apply(lambda x: int(x > 3.0))

    df_with_no_nodule["strat_is_GE"] = df_with_no_nodule[ImageLevelInfo.MANUFACTURER].apply(
        lambda x: int(x > "GE MEDICAL SYSTEMS")
    )
    df_with_no_nodule["strat_is_PH"] = df_with_no_nodule[ImageLevelInfo.MANUFACTURER].apply(
        lambda x: int(x > "Philips")
    )
    df_with_no_nodule["strat_is_SI"] = df_with_no_nodule[ImageLevelInfo.MANUFACTURER].apply(
        lambda x: int(x > "SIEMENS")
    )
    df_with_no_nodule["strat_is_TO"] = df_with_no_nodule[ImageLevelInfo.MANUFACTURER].apply(
        lambda x: int(x > "TOSHIBA")
    )

    strat_features = list()
    for i_key in df_with_no_nodule.keys():
        if "strat_" in i_key:
            strat_features.append(i_key)
    _, fold_df_with_no_nodule = split_features(
        df_with_no_nodule.copy(), strat_features, n_splits=n_splits, random_state=1
    )

    # patient with nodules
    df_with_nodule["strat_slice_thickness"] = df_with_nodule[ImageLevelInfo.SLICE_THICKNESS].apply(
        lambda x: int(x > 1.5)
    )
    df_with_nodule["strat_spacing_between_slices"] = df_with_nodule[ImageLevelInfo.SPACING_BETWEEN_SLICES].apply(
        lambda x: int(x > 1.5)
    )
    df_with_nodule["strat_pixel_spacing"] = df_with_nodule[ImageLevelInfo.PIXEL_SPACING].apply(lambda x: int(x > 0.7))
    df_with_nodule["strat_contrast_used"] = df_with_nodule[ImageLevelInfo.CONTRAST_USED].apply(lambda x: int(x))
    df_with_nodule["strat_num_cluster"] = df_with_nodule[ImageLevelInfo.NUM_CLUSTER].apply(lambda x: int(x > 3.0))

    df_with_nodule["strat_is_GE"] = df_with_nodule[ImageLevelInfo.MANUFACTURER].apply(
        lambda x: int(x > "GE MEDICAL SYSTEMS")
    )
    df_with_nodule["strat_is_PH"] = df_with_nodule[ImageLevelInfo.MANUFACTURER].apply(lambda x: int(x > "Philips"))
    df_with_nodule["strat_is_SI"] = df_with_nodule[ImageLevelInfo.MANUFACTURER].apply(lambda x: int(x > "SIEMENS"))
    df_with_nodule["strat_is_TO"] = df_with_nodule[ImageLevelInfo.MANUFACTURER].apply(lambda x: int(x > "TOSHIBA"))

    df_with_nodule["strat_subtlety_0"] = df_with_nodule[NoduleAttribute.SUBTLETY.value].apply(lambda x: int(x > 0.5))
    df_with_nodule["strat_subtlety_2"] = df_with_nodule[NoduleAttribute.SUBTLETY.value].apply(lambda x: int(x > 2.5))
    df_with_nodule["strat_subtlety_4"] = df_with_nodule[NoduleAttribute.SUBTLETY.value].apply(lambda x: int(x > 4.5))

    df_with_nodule["strat_internalStructure"] = df_with_nodule[NoduleAttribute.INTERNAL_STRUCTURE.value].apply(
        lambda x: int(x == 1.0)
    )
    df_with_nodule["strat_calcification"] = df_with_nodule[NoduleAttribute.CALCIFICATION.value].apply(
        lambda x: int(x == 1.0)
    )

    df_with_nodule["strat_sphericity_0"] = df_with_nodule[NoduleAttribute.SPHERICITY.value].apply(
        lambda x: int(x > 0.5)
    )
    df_with_nodule["strat_sphericity_2"] = df_with_nodule[NoduleAttribute.SPHERICITY.value].apply(
        lambda x: int(x > 2.5)
    )
    df_with_nodule["strat_sphericity_4"] = df_with_nodule[NoduleAttribute.SPHERICITY.value].apply(
        lambda x: int(x > 4.5)
    )

    df_with_nodule["strat_margin_0"] = df_with_nodule[NoduleAttribute.MARGIN.value].apply(lambda x: int(x > 0.5))
    df_with_nodule["strat_margin_2"] = df_with_nodule[NoduleAttribute.MARGIN.value].apply(lambda x: int(x > 2.5))
    df_with_nodule["strat_margin_4"] = df_with_nodule[NoduleAttribute.MARGIN.value].apply(lambda x: int(x > 4.5))

    df_with_nodule["strat_lobulation"] = df_with_nodule[NoduleAttribute.LOBULATION.value].apply(lambda x: int(x > 3.0))
    df_with_nodule["strat_spiculation"] = df_with_nodule[NoduleAttribute.SPICULATION.value].apply(
        lambda x: int(x > 3.0)
    )

    df_with_nodule["strat_texture_0"] = df_with_nodule[NoduleAttribute.TEXTURE.value].apply(lambda x: int(x > 0.5))
    df_with_nodule["strat_texture_2"] = df_with_nodule[NoduleAttribute.TEXTURE.value].apply(lambda x: int(x > 2.5))
    df_with_nodule["strat_texture_4"] = df_with_nodule[NoduleAttribute.TEXTURE.value].apply(lambda x: int(x > 4.5))

    df_with_nodule["strat_malignancy_0"] = df_with_nodule[NoduleAttribute.MALIGNANCY.value].apply(
        lambda x: int(x > 0.5)
    )
    df_with_nodule["strat_malignancy_2"] = df_with_nodule[NoduleAttribute.MALIGNANCY.value].apply(
        lambda x: int(x > 2.5)
    )
    df_with_nodule["strat_malignancy_4"] = df_with_nodule[NoduleAttribute.MALIGNANCY.value].apply(
        lambda x: int(x > 4.5)
    )

    df_with_nodule["strat_diameter_1"] = df_with_nodule[NoduleLevelInfo.DIAMETER].apply(lambda x: int(x < 6))

    df_with_nodule["strat_volume"] = df_with_nodule[NoduleLevelInfo.VOLUME].apply(lambda x: int(x < 1000))

    strat_features = list()
    for i_key in df_with_nodule.keys():
        if "strat_" in i_key:
            strat_features.append(i_key)
    _, fold_df_with_nodule = split_features(df_with_nodule.copy(), strat_features, n_splits=n_splits, random_state=1)

    fold_df = pd.concat([fold_df_with_nodule, fold_df_with_no_nodule])

    return fold_df


def main():
    parser = argparse.ArgumentParser(description="prepare for train")
    args = parser.parse_args()
    logger.info("start to prepare for train.")

    # Get image-level infos.
    target_fields = [
        "slice_thickness",
        "spacing_between_slices",
        "pixel_spacing",
        "contrast_used",
        "num_cluster",
        "manufacturer",
    ]
    save_hist_for_target_field(
        CollectionName.IMAGE, target_fields, fig_size=(50, 4), save_dir=Path("./outputs/image_level_info.jpg")
    )
    logger.info("hist for image-level info has been saved.")

    # Get nodule-level infos.
    target_fields_nodule = [
        "subtlety",
        "internalStructure",
        "calcification",
        "sphericity",
        "margin",
        "lobulation",
        "spiculation",
        "texture",
        "malignancy",
        "diameter",
        "volume",
    ]
    save_hist_for_target_field(
        CollectionName.NODULE, target_fields_nodule, fig_size=(70, 4), save_dir=Path("./outputs/nodule_level_info.jpg")
    )
    logger.info("hist for nodule-level info has been saved.")

    # Get patient-level infos.
    df = get_patient_level_infos(target_fields, target_fields_nodule)
    df = df.reset_index(drop=True).copy()
    assert len(df) == 1010

    # split fold index
    fold_df = split_fold(df, n_splits=NUM_FOLD)
    logger.info("spliting fold index has been done.")

    # visualization image-level infos by fold index
    fn_save_histograms(
        fold_df,
        target_fields=target_fields,
        fig_size=(70, 50),
        num_fold=NUM_FOLD,
        dpi=100,
        save_dir=Path("./outputs/image_level_info_by_fold.jpg"),
    )
    logger.info("hist for image-level info by fold index has been saved.")

    # visualization nodule-level infos by fold index
    fn_save_histograms(
        fold_df,
        target_fields=target_fields_nodule,
        fig_size=(90, 50),
        num_fold=NUM_FOLD,
        dpi=100,
        save_dir=Path("./outputs/nodule_level_info_by_fold.jpg"),
    )
    logger.info("hist for nodule-level info by fold index has been saved.")

    # correlation via image-level infos.
    corr_df = fold_df.copy()
    corr_df[ImageLevelInfo.MANUFACTURER] = corr_df[ImageLevelInfo.MANUFACTURER].map(
        lambda x: {"GE MEDICAL SYSTEMS": 0, "SIEMENS": 1, "TOSHIBA": 2, "Philips": 3}[x]
    )
    fn_save_corr_heatmap(
        corr_df, target_fields=target_fields, save_dir=Path("./outputs/image_level_correlation_heatmap.jpg")
    )

    # correlation via patient-level nodule infos.
    corr_df = fold_df.copy()
    fn_save_corr_heatmap(
        corr_df,
        target_fields=target_fields_nodule,
        save_dir=Path("./outputs/patient_level_nodule_info_correlation_heatmap.jpg"),
    )

    # correlation via nodule-level nodule infos.
    dataset_handler = DatasetHandler()
    projection = {t: 1 for t in target_fields_nodule}
    nodule_infos = dataset_handler.fetch_documents(collection=CollectionName.NODULE, query={}, projection=projection)
    df_nodule = pd.DataFrame(nodule_infos)
    fn_save_corr_heatmap(
        df_nodule,
        target_fields=target_fields_nodule,
        save_dir=Path("./outputs/nodule_level_nodule_info_correlation_heatmap.jpg"),
    )
    logger.info("correlation heatmap has been saved.")

    # update mongoDB (fold index, r_coord_zyx)
    client = get_client()
    for idx, row in fold_df.iterrows():
        PID = row[ImageLevelInfo.PATIENT_ID]
        fold = row[DBKey.FOLD]
        collection = client[TARGET_DB][CollectionName.IMAGE]
        docs = [x for x in collection.find({ImageLevelInfo.PATIENT_ID: PID})]
        for doc in docs:
            _filter = {DataLakeKey.DOC_ID: doc[DataLakeKey.DOC_ID]}
            newvalues = {
                "$set": {
                    DBKey.FOLD: fold,
                }
            }
            collection.update_one(_filter, newvalues)

        collection = client[TARGET_DB][CollectionName.CLUSTER]
        docs = [x for x in collection.find({f"{ClusterLevelInfo.DICOM_SERIES_INFO}.{ImageLevelInfo.PATIENT_ID}": PID})]
        for doc in docs:
            spacing_between_slices = doc[ClusterLevelInfo.DICOM_SERIES_INFO][ImageLevelInfo.SPACING_BETWEEN_SLICES]
            pixel_spacing = doc[ClusterLevelInfo.DICOM_SERIES_INFO][ImageLevelInfo.PIXEL_SPACING]
            h5_path = doc[ClusterLevelInfo.DICOM_SERIES_INFO][ImageLevelInfo.H5_FILE_PATH]
            d_coord = doc[ClusterLevelInfo.D_COORD_ZYX]
            original_spacing = [spacing_between_slices, pixel_spacing, pixel_spacing]
            new_spacing = [1.0, 0.67, 0.67]
            resize_factor = [a / b for a, b in zip(original_spacing, new_spacing)]

            with h5py.File(h5_path, mode="r", libver="latest") as hf:
                image_shape = hf[HFileKey.HFileAttrName.DICOM_PIXELS].shape

            # image_shape
            new_real_shape = [((a * b) + 1) for a, b in zip([s - 1 for s in image_shape], resize_factor)]
            new_shape = tuple([int(round(a)) for a in new_real_shape])
            real_resize_factor = [(a - 1) / (b - 1) for a, b in zip(list(new_shape), list(image_shape))]

            _filter = {DataLakeKey.DOC_ID: doc[DataLakeKey.DOC_ID]}
            newvalues = {
                "$set": {
                    DBKey.FOLD: fold,
                    ClusterLevelInfo.R_COORD_ZYX: [round(c * r) for c, r in zip(d_coord, real_resize_factor)],
                }
            }
            collection.update_one(_filter, newvalues)
    logger.info("update MongoDB for the fields e.g., fold and r_coord_zyx info.")


if __name__ == "__main__":
    _THIS_DIR = os.path.dirname(os.path.realpath(__file__))
    setup_logger(save_path=os.path.join(_THIS_DIR, f"{Path(__file__).stem}.log"))
    main()
