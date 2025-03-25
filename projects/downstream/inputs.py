import itertools
import logging
import math
import os
import random
from pathlib import Path
from typing import List, Union

import h5py
import hydra
import numpy as np
import pandas as pd
import pymongo
import sklearn.utils as sk_u
from omegaconf import DictConfig, OmegaConf
import omegaconf
from torch.utils.data import Dataset
from projects.common.constant import DB_ADDRESS, TARGET_DB 

from projects.common.enums import RunMode


logger = logging.getLogger(__name__)
_VUNO_LUNG_DB = DB_ADDRESS

def _get_data_df(
    mode: RunMode,
    target_dataset: List,
    dataset_infos: dict,
    dataset_size_scale_factor: float = 1.0,
    val_shift_scale: int = 4,
):
    
    dfs = list()
    for dataset in target_dataset:
        
        dataset_info = dataset_infos[dataset]
        
        if hasattr(dataset_info, "collection_name"):
            client = pymongo.MongoClient(_VUNO_LUNG_DB)
            projection = {t: 1 for t in []}
            projection["_id"] = 0  # do not show '_id' of documents
            if hasattr(dataset_info, "query"):
                query = (
                    OmegaConf.to_container(dataset_info["query"], resolve=True)
                    if dataset_info["query"]
                    else {}
                )
                data = [
                    x
                    for x in client["lct"][dataset_info["collection_name"]].find(query, projection)
                ]
            else:
                # query 없는 경우는 모두 가져오기
                data = [
                    x for x in client["lct"][dataset_info["collection_name"]].find({}, projection)
                ]
                
            # dataset-specific processing
            df = pd.DataFrame(data)
    
        # get fold indices depending on the "mode"
        if mode == RunMode.TRAIN:
            total_fold = dataset_info["total_fold"]
            val_fold = dataset_info["val_fold"]
            test_fold = dataset_info["test_fold"]
            fold_indices_total = OmegaConf.to_container(total_fold, resolve=True)
            fold_indices_val = OmegaConf.to_container(val_fold, resolve=True)
            fold_indices_test = OmegaConf.to_container(test_fold, resolve=True)
            fold_indices = [
                item
                for item in fold_indices_total
                if item not in fold_indices_val and item not in fold_indices_test
            ]
        elif mode == RunMode.VALIDATE:
            val_fold = dataset_info["val_fold"]
            fold_indices = OmegaConf.to_container(val_fold, resolve=True)
        elif mode == RunMode.TEST:
            test_fold = dataset_info["test_fold"]
            fold_indices = OmegaConf.to_container(test_fold, resolve=True)
        else:
            assert False, "fold selection did not work as intended."

        # get specific size of samples from each fold index
        for i_fold in fold_indices:
            _df = df[df["fold"] == i_fold].copy()
            _df["center_shift_zyx"] = str([0, 0, 0])  # ast.literal_eval()을 통해 decoding
            _df = _df[: int(len(_df) * dataset_size_scale_factor)]

            # Checkpoints must be chosen regardless of whether the center of the object is given exactly.
            if mode == RunMode.VALIDATE or mode == RunMode.TEST:
                dfs_center_shift = list()
                center_shift_scale_factor = val_shift_scale
                shift_direction = (
                    [[0, 0, 0]] if val_shift_scale == 0 else [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]
                )
                for i_shift in shift_direction:
                    _df_tmp = _df.copy()
                    i_shift_scaled = [
                        i_coord_shift * center_shift_scale_factor for i_coord_shift in i_shift
                    ]
                    _df_tmp["center_shift_zyx"] = str(i_shift_scaled)
                    dfs_center_shift.append(_df_tmp)
                _df = pd.concat(dfs_center_shift, ignore_index=True)

            # append dataframes
            _df["dataset"] = dataset
            logger.info(f"the number of samples for {dataset}, fold: {i_fold}: {len(_df)}")
            dfs.append(_df)

        df = pd.concat(dfs, ignore_index=True)
        
        return df
