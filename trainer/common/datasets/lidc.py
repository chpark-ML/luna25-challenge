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


def patch_extract_3d_pylidc(
    h5_path,
    r_coord,
    xy_size: int = 72,
    z_size: int = 72,
    center_shift_zyx: list = [0, 0, 0],
    fill: float = 0,
) -> np.ndarray:
    """
    Extract a 3D patch from the given CT volume.
    Args:
        h5_path
        r_coord
        xy_size
        z_size
        fill (float): constant value for np.pad
        mode
        index : TODO: hotfix for validation image translation
    Returns:
        np.ndarray: A 3D cube-shaped patch extracted from the given volume.
    """
    # Load cached data
    hf_file = File(h5_path, "r")
    patchsize = (z_size, xy_size, xy_size)
    repr_center = [x + y for x, y in zip(r_coord, center_shift_zyx)]

    file_shape = hf_file["dicom_pixels_resampled"].shape
    rlower, rupper, dlower, dupper = get_patch_extract_3d_meta(file_shape, repr_center, patchsize=patchsize)

    # Load ROI only
    file = hf_file["dicom_pixels_resampled"][rlower[0] : rupper[0], rlower[1] : rupper[1], rlower[2] : rupper[2]]
    mask = hf_file["mask_annotation_resampled"][rlower[0] : rupper[0], rlower[1] : rupper[1], rlower[2] : rupper[2]]

    if file.shape != patchsize:
        pad_width = [pair for pair in zip(dlower, dupper)]
        file = np.pad(file, pad_width=pad_width, mode="constant", constant_values=fill)
        mask = np.pad(mask, pad_width=pad_width, mode="constant", constant_values=0.0)

    file = file.astype("float32")
    mask = mask.astype("float32")

    # FIXME: nested scope?
    # with open() as hf_file:
    #     with open() as hf_mask:
    hf_file.close()

    assert file.shape == (
        z_size,
        xy_size,
        xy_size,
    ), f"Sanity check failed: {file.shape} == ({z_size}, {xy_size}, {xy_size})"
    assert mask.shape == (
        z_size,
        xy_size,
        xy_size,
    ), f"Sanity check failed: {mask.shape} == ({z_size}, {xy_size}, {xy_size})"
    return file, mask


def get_patch_extract_3d_meta(
    image_shape=None,
    center: Optional[Sequence[float]] = None,
    patchsize: Union[int, Sequence[int]] = 72,
):
    # FIXME: This supports square patch size only
    if isinstance(patchsize, int):
        patchsize = (patchsize, patchsize, patchsize)

    center = np.array(center)
    patchsize = np.array(patchsize)

    half = patchsize // 2
    lower = np.rint(center - half).astype(int)
    # lower = (center - half).astype(int)
    upper = lower + patchsize

    # real
    rlower = np.maximum(lower, 0).astype(int).tolist()
    rupper = np.minimum(upper, image_shape).astype(int).tolist()

    # diff
    dlower = np.maximum(-lower, 0)
    dupper = np.maximum(upper - image_shape, 0)

    return rlower, rupper, dlower, dupper


def _get_meta_df(mode: RunMode, target_dataset: List, dataset_info: dict):
    target_dataset_infos = {dataset: dataset_info[dataset] for dataset in target_dataset}
    df = DatasetHandler().fetch_multiple_datasets(dataset_infos=target_dataset_infos, mode=mode)

    return df


class LctDataset(Dataset):
    def __init__(
        self,
        mode: Union[str, RunMode],
        patch_size,
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
        assert len(patch_size) == 3 and patch_size[1] == patch_size[2], f"patch_size is {patch_size}."
        self.use_weighted_sampler = use_weighted_sampler

        # set configuration
        self.xy_size = patch_size[1]
        self.z_size = patch_size[0]
        self.patch_size = OmegaConf.to_container(patch_size, resolve=True)
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
                    RandomCrop3DDeprecated(
                        p=augmentation["random_crop_3d"]["aug_prob"],
                        xy_size=self.xy_size,
                        z_size=self.z_size,
                        buffer=self.buffer,
                    ),
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
