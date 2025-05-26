from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from h5py import File
from sklearn.utils import resample

from data_lake.constants import DB_ADDRESS
from data_lake.lidc.constants import LOGISTIC_TASK_POSTFIX, RESAMPLED_FEATURE_POSTFIX, ClusterLevelInfo
from shared_lib.constants import DataLakeKeyDict
from shared_lib.enums import RunMode
from trainer.common.constants import ATTR_ANNOTATION_KEY, INPUT_PATCH_KEY, SEG_ANNOTATION_KEY
from trainer.common.datasets.lidc import LctDataset

_LUNG_DB = DB_ADDRESS


def _get_balanced_df(df, target_attr_to_train: list):
    assert len(target_attr_to_train) == 1
    _class = target_attr_to_train[0]

    assert (LOGISTIC_TASK_POSTFIX in _class) or (RESAMPLED_FEATURE_POSTFIX in _class)
    if LOGISTIC_TASK_POSTFIX in _class:
        pos_df = df[df[_class] > 0.5]
        neg_df = df[df[_class] < 0.5]
        n_pos = len(pos_df)
        n_neg = len(neg_df)
        max_count = max([n_pos, n_neg])

        if n_pos < max_count:
            oversampled_data = resample(pos_df, replace=True, n_samples=max_count - n_pos, random_state=1111)
            df = pd.concat([df, oversampled_data], ignore_index=True)

        if n_neg < max_count:
            oversampled_data = resample(neg_df, replace=True, n_samples=max_count - n_neg, random_state=1111)
            df = pd.concat([df, oversampled_data], ignore_index=True)
    return df


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


def patch_extract_3d_pylidc(
    h5_path,
    r_coord,
    xy_size: int = 72,
    z_size: int = 72,
    center_shift_zyx: list = [0, 0, 0],
    fill: float = 0,
    do_segmentation: bool = False,
) -> np.ndarray:
    patchsize = (z_size, xy_size, xy_size)
    repr_center = [x + y for x, y in zip(r_coord, center_shift_zyx)]

    with File(h5_path, "r") as hf_file:
        file_shape = hf_file["dicom_pixels_resampled"].shape
        rlower, rupper, dlower, dupper = get_patch_extract_3d_meta(file_shape, repr_center, patchsize=patchsize)

        patch_image = hf_file["dicom_pixels_resampled"][rlower[0]:rupper[0], rlower[1]:rupper[1], rlower[2]:rupper[2]]
        if do_segmentation:
            mask = hf_file["mask_annotation_resampled"][rlower[0]:rupper[0], rlower[1]:rupper[1], rlower[2]:rupper[2]]

    if patch_image.shape != patchsize:
        pad_width = [pair for pair in zip(dlower, dupper)]
        patch_image = np.pad(patch_image, pad_width=pad_width, mode="constant", constant_values=fill)

        if do_segmentation:
            mask = np.pad(mask, pad_width=pad_width, mode="constant", constant_values=0.0)

    assert patch_image.shape == patchsize, f"Sanity check failed: {patch_image.shape} != {patchsize}"
    if do_segmentation:
        assert mask.shape == patchsize, f"Sanity check failed: {mask.shape} != {patchsize}"
        return patch_image, mask
    else:
        return patch_image, None


class Dataset(LctDataset):
    def __init__(
        self,
        mode: Union[str, RunMode],
        patch_size,
        dicom_window,
        buffer,
        augmentation,
        dataset_size_scale_factor=None,
        do_random_balanced_sampling=None,
        do_segmentation=False,
        target_dataset=None,
        dataset_info=None,
        use_weighted_sampler=None,
    ):
        super().__init__(
            mode,
            patch_size,
            dicom_window,
            buffer,
            augmentation,
            dataset_size_scale_factor,
            do_random_balanced_sampling,
            target_dataset,
            dataset_info,
            use_weighted_sampler,
        )
        self.do_segmentation = do_segmentation
        self.target_attr_total = dataset_info["pylidc"]["target_attr_total"]
        self.target_attr_to_train = dataset_info["pylidc"]["target_attr_to_train"]

    def __getitem__(self, index):
        """
        (Resize) -> Augmentation -> Windowing
        """
        elem = self.meta_df.iloc[index]
        dataset = elem["dataset"]

        # Extract voxel of shape (D, H, W)
        if dataset == "pylidc":
            h5_path = elem[DataLakeKeyDict.HFILE_PATH]
            r_coord = elem[ClusterLevelInfo.R_COORD_ZYX]
            center_shift_zyx = [0, 0, 0]
            img_path = h5_path
            mask_path = h5_path
            img, mask = patch_extract_3d_pylidc(
                h5_path,
                r_coord,
                xy_size=self.xy_size + self.buffer,
                z_size=self.z_size + self.buffer,
                center_shift_zyx=center_shift_zyx,
                fill=-3024.0,
                do_segmentation=self.do_segmentation,
            )

            attributes = dict()
            for i_attr in self.target_attr_total:
                attributes[i_attr] = elem[i_attr]

        else:
            assert False, "_getitem_ did not work as intended."

        # Data augmentation
        if self.mode == RunMode.TRAIN:
            if self.do_segmentation:
                img, mask = self.transform(img, mask)
                mask = mask[None, ...]  # (1, 48, 72, 72)
            else:
                img = self.transform(img)

        # Data preprocessing
        img = [fn(img)[np.newaxis, ...] for fn in self.dicom_windowing]  # [(1, 48, 72, 72), ...]
        img = np.concatenate(img, axis=0)  # (n, 48, 72, 72)

        return {
            INPUT_PATCH_KEY: img,
            SEG_ANNOTATION_KEY: mask if self.do_segmentation else None,
            ATTR_ANNOTATION_KEY: attributes,
            "file_path": img_path,
            "mask_path": mask_path,
            "index": index,
        }

    def random_balanced_sampling(self):
        if len(self.target_attr_to_train) == 1:
            self.meta_df = _get_balanced_df(self.meta_df, self.target_attr_to_train)
