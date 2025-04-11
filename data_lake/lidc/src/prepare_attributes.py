import argparse
import collections
import logging
import os
from pathlib import Path

import numpy as np

from data_lake.constants import DEFAULT_RESAMPLED_SPACING, NUM_FOLD, TARGET_DB, DataLakeKey
from data_lake.lidc.constants import (
    CLASSIFICATION_TASK_POSTFIX,
    LOGISTIC_TASK_POSTFIX,
    PREFIX_CONSENSUS,
    RESAMPLED_FEATURE_POSTFIX,
    ClusterLevelInfo,
    CollectionName,
)
from data_lake.lidc.enums import NoduleAttribute
from data_lake.lidc.src.utils import save_hist_for_target_field
from data_lake.utils.client import get_client
from trainer.common.utils.utils_logger import setup_logger

logger = logging.getLogger(__name__)

_MAX_VALUE_FOR_NORMALIZED_ATTR = 5
_TARGET_FIELD_LOGISTIC = ["subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture", "malignancy"]
_TARGET_FIELD_MULTI_CLASS = ["internalStructure", "calcification"]  # range = {1,2,3,4}  # range = {1,2,3,4,5,6}
_TARGET_FIELD_REGRESSION = ["diameter", "volume"]


def get_most_frequent(data):
    counter = collections.Counter(data)
    # check if the plural maximum values exist
    sorted_values = sorted(list(counter.values()))
    if len(sorted_values) > 1:
        if sorted_values[-1] == sorted_values[-2]:
            return None
    return max(data, key=data.count)


def one_hot_encode(lst, num_class=6):
    encoded = [0] * num_class  # 최댓값 + 1 크기의 리스트를 생성하고 초기화
    for i in range(num_class):
        encoded[i] = lst.count(i + 1) / len(lst)
    return encoded


def update_docs(collection, nodule_cluster_docs):
    for doc in nodule_cluster_docs:
        new_dict = {}

        # get the binary-class feature
        for i_attr in _TARGET_FIELD_LOGISTIC:
            averaged = np.mean(doc[i_attr])
            new_dict[f"{PREFIX_CONSENSUS}{i_attr}{LOGISTIC_TASK_POSTFIX}"] = (averaged - 1) / (
                _MAX_VALUE_FOR_NORMALIZED_ATTR - 1
            )

        # get the multi-class feature
        for i_attr in _TARGET_FIELD_MULTI_CLASS:
            max_value = get_most_frequent(doc[i_attr])
            if i_attr == NoduleAttribute.CALCIFICATION.value:
                new_dict[f"{PREFIX_CONSENSUS}{i_attr}{CLASSIFICATION_TASK_POSTFIX}"] = one_hot_encode(
                    doc[i_attr], num_class=6
                )  # at least one number other than absent
                new_dict[f"{PREFIX_CONSENSUS}{i_attr}{LOGISTIC_TASK_POSTFIX}"] = (
                    any([i_value != 6 for i_value in doc[i_attr]]) * 1.0
                )

            elif i_attr == NoduleAttribute.INTERNAL_STRUCTURE.value:
                new_dict[f"{PREFIX_CONSENSUS}{i_attr}{CLASSIFICATION_TASK_POSTFIX}"] = one_hot_encode(
                    doc[i_attr], num_class=4
                )  # at least one number other than soft tissue
                new_dict[f"{PREFIX_CONSENSUS}{i_attr}{LOGISTIC_TASK_POSTFIX}"] = (
                    any([i_value != 1 for i_value in doc[i_attr]]) * 1.0
                )

        # get the regression feature
        for i_attr in _TARGET_FIELD_REGRESSION:
            averaged = np.mean(doc[i_attr])

            # axial의 resampled space에서 diameter / 3D resampled space에서 volume 업데이트
            _orig_spacing_mm = [1.0, 1.0, 1.0]
            spacing_ratio = np.divide(_orig_spacing_mm, DEFAULT_RESAMPLED_SPACING)
            if i_attr == ClusterLevelInfo.DIAMETER:
                averaged = averaged * np.prod(spacing_ratio[1:3]) ** (1 / 2)
            elif i_attr == ClusterLevelInfo.VOLUME:
                averaged = averaged * np.prod(spacing_ratio)

            new_dict[f"{PREFIX_CONSENSUS}{i_attr}{RESAMPLED_FEATURE_POSTFIX}"] = averaged

        _filter = {DataLakeKey.DOC_ID: doc[DataLakeKey.DOC_ID]}
        newvalues = {"$set": new_dict}
        collection.update_one(_filter, newvalues)


def main():
    parser = argparse.ArgumentParser(description="prepare attributes")
    args = parser.parse_args()
    logger.info("start to prepare for nodule attributes.")

    # clean if the field has prefix for consensus
    client = get_client()
    collection = client[TARGET_DB][CollectionName.CLUSTER]
    nodule_cluster_docs = [x for x in collection.find({}, {})]
    keys_to_del = [i_key for i_key in nodule_cluster_docs[0].keys() if PREFIX_CONSENSUS in i_key]
    for i_key in keys_to_del:
        collection.update_many({}, {"$unset": {i_key: ""}})

    # update mongoDB
    update_docs(collection, nodule_cluster_docs)

    # check num nodules
    expected_results = [2651, 1880, 1392, 911]
    for i, expected_result in zip([1, 2, 3, 4], expected_results):
        query = {ClusterLevelInfo.NUM_MASK: {"$gte": i}}
        nodule_cluster_docs = [x for x in collection.find(query, {})]
        assert len(nodule_cluster_docs) == expected_result
        logger.info(f"num cluster gte {i} mask: {len(nodule_cluster_docs)}")

    # visualization
    # logistic regression
    save_hist_for_target_field(
        CollectionName.CLUSTER,
        _TARGET_FIELD_LOGISTIC,
        prefix=PREFIX_CONSENSUS,
        suffix=LOGISTIC_TASK_POSTFIX,
        fig_size=(50, 40),
        num_fold=NUM_FOLD,
        save_dir="./outputs/hist_consensus_logistic.jpg",
    )

    # class
    save_hist_for_target_field(
        CollectionName.CLUSTER,
        _TARGET_FIELD_MULTI_CLASS,
        prefix=PREFIX_CONSENSUS,
        suffix=CLASSIFICATION_TASK_POSTFIX,
        fig_size=(50, 20),
        num_fold=NUM_FOLD,
        save_dir="./outputs/hist_consensus_class.jpg",
    )


if __name__ == "__main__":
    _THIS_DIR = os.path.dirname(os.path.realpath(__file__))
    setup_logger(save_path=os.path.join(_THIS_DIR, f"{Path(__file__).stem}.log"))
    main()
