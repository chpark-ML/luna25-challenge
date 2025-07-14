import os
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import pymongo
from h5py import File
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from data_lake.constants import DB_ADDRESS
from data_lake.dataset_handler import DatasetHandler
from shared_lib.enums import RunMode
from trainer.common.augmentation.coarse_drop import CoarseDropout3D
from trainer.common.augmentation.compose import ComposeAugmentation
from trainer.common.augmentation.crop import RandomCrop3DDeprecated
from trainer.common.augmentation.flip_rotate import Flip3D, FlipXY, RandomRotate903D
from trainer.common.augmentation.hu_value import DicomWindowing, RandomDicomWindowing
from trainer.common.augmentation.rescale import RescaleImage
from trainer.common.constants import HU_WINDOW

_LUNG_DB = DB_ADDRESS


def _get_normalized_tensor(
    mode: RunMode,
    hu_range,
    min_width_scale,
    max_width_scale,
    min_level_shift,
    max_level_shift,
    p,
):
    if mode == RunMode.TRAIN:
        return RandomDicomWindowing(
            hu_range=hu_range,
            min_width_scale=min_width_scale,
            max_width_scale=max_width_scale,
            min_level_shift=min_level_shift,
            max_level_shift=max_level_shift,
            p=p,
        )
    else:
        return DicomWindowing(hu_range=hu_range)


def _get_meta_df(mode: RunMode, target_dataset: List, dataset_info: dict):
    target_dataset_infos = {dataset: dataset_info[dataset] for dataset in target_dataset}
    df = DatasetHandler().fetch_multiple_datasets(dataset_infos=target_dataset_infos, mode=mode)

    return df


class LctDataset(Dataset):
    def __init__(
        self,
        mode: Union[str, RunMode],
        patch_size,
        size_mm,
        dicom_window,
        buffer,
        augmentation,
        dataset_size_scale_factor=None,
        do_random_balanced_sampling=None,
        target_dataset=None,
        dataset_info=None,
        use_weighted_sampler=None,
    ):
        self.mode: RunMode = RunMode(mode) if isinstance(mode, str) else mode
        if isinstance(patch_size, int):
            patch_size_list = [patch_size] * 3
        elif isinstance(patch_size, (list, tuple)):
            patch_size_list = list(patch_size)
        else:
            raise ValueError(f"Unsupported type for patch_size: {type(patch_size)}")
        assert (
            len(patch_size_list) == 3 and patch_size_list[1] == patch_size_list[2]
        ), f"patch_size is {patch_size_list}."

        if isinstance(size_mm, int):
            size_mm_list = [size_mm] * 3
        elif isinstance(size_mm, (list, tuple)):
            size_mm_list = list(size_mm)
        else:
            raise ValueError(f"Unsupported type for patch_size: {type(patch_size)}")

        self.use_weighted_sampler = use_weighted_sampler

        # set configuration
        self.xy_size = patch_size_list[1]
        self.z_size = patch_size_list[0]
        self.patch_size = patch_size_list
        self.size_mm = size_mm_list
        self.dicom_window = OmegaConf.to_container(dicom_window, resolve=True)
        self.buffer = buffer
        self.dataset_size_scale_factor = dataset_size_scale_factor
        self.do_random_balanced_sampling = do_random_balanced_sampling

        # dataset
        self.target_dataset = OmegaConf.to_container(target_dataset, resolve=True)
        self.dataset_info = dataset_info
        if self.mode != RunMode.TRAIN:
            self.buffer = 0  # Do not shift data in inference phase

        # load dataset
        self.meta_df = _get_meta_df(mode, target_dataset, dataset_info)

        # data augmentation
        if self.mode == RunMode.TRAIN:
            self.transform = ComposeAugmentation(
                transform=[
                    FlipXY(p=augmentation["flip_xy"]["aug_prob"]),
                    Flip3D(p=augmentation["flip_3d"]["aug_prob"]),
                    RandomRotate903D(p=augmentation["random_rotate_3d"]["aug_prob"]),
                    CoarseDropout3D(
                        p=augmentation["coarse_dropout_3d"]["aug_prob"],
                        max_holes=augmentation["coarse_dropout_3d"]["max_holes"],
                        size_limit=(0.08, 0.16),
                        patch_size=(self.z_size, self.xy_size, self.xy_size),
                    ),
                    RescaleImage(
                        p=augmentation["rescale_image_3d"]["aug_prob"],
                        is_same_across_axes=augmentation["rescale_image_3d"]["is_same_across_axes"],
                        min_scale_factor=augmentation["rescale_image_3d"]["scale_factor"]["min"],
                        max_scale_factor=augmentation["rescale_image_3d"]["scale_factor"]["max"],
                    ),
                ]
            )

        # data normalization
        selected_hu_range = [
            (
                HU_WINDOW[i_window][0] - HU_WINDOW[i_window][1] // 2,
                HU_WINDOW[i_window][0] + HU_WINDOW[i_window][1] // 2,
            )
            for i_window in self.dicom_window
        ]
        self.dicom_windowing = [
            _get_normalized_tensor(
                self.mode,
                hu_range,
                p=augmentation["dicom_windowing"]["aug_prob"],
                min_width_scale=augmentation["dicom_windowing"]["width_scale"]["min"],
                max_width_scale=augmentation["dicom_windowing"]["width_scale"]["max"],
                min_level_shift=augmentation["dicom_windowing"]["level_shift"]["min"],
                max_level_shift=augmentation["dicom_windowing"]["level_shift"]["max"],
            )
            for hu_range in selected_hu_range
        ]

    def __len__(self):
        return len(self.meta_df)
