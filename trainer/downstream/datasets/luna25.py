import ast
import logging
from pathlib import Path
from typing import Union

import hydra
import numpy as np
import torch
import torch.utils.data as data
from h5py import File
from omegaconf import OmegaConf

from data_lake.constants import DataLakeKey, DBKey, H5DataKey
from data_lake.dataset_handler import DatasetHandler
from shared_lib.enums import RunMode
from shared_lib.tools.image_parser import extract_patch
from trainer.common.augmentation.coarse_drop import CoarseDropout3D
from trainer.common.augmentation.compose import ComposeAugmentation
from trainer.common.augmentation.flip_rotate import Flip3D, FlipXY
from trainer.common.augmentation.hu_value import DicomWindowing, RandomDicomWindowing
from trainer.common.augmentation.rescale import RescaleImage
from trainer.common.constants import DB_ADDRESS, HU_WINDOW
from trainer.downstream.datasets.constants import DataLoaderKeys

logger = logging.getLogger(__name__)
_VUNO_LUNG_DB = DB_ADDRESS


def _get_3d_patch(image_shape=None, center=None, patchsize=None):
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


def extract_patch_dicom_space(
    h5_path,
    coord,
    xy_size: int = 128,
    z_size: int = 64,
    center_shift_zyx: list = [0, 0, 0],
    fill: float = -3024.0,
) -> np.ndarray:
    # Load cached data
    hf_file = File(h5_path, "r")
    patchsize = (z_size, xy_size, xy_size)
    repr_center = [x + y for x, y in zip(coord, center_shift_zyx)]

    file_shape = hf_file[H5DataKey.image].shape
    rlower, rupper, dlower, dupper = _get_3d_patch(file_shape, repr_center, patchsize=patchsize)

    # Load ROI only
    file = hf_file[H5DataKey.image][rlower[0] : rupper[0], rlower[1] : rupper[1], rlower[2] : rupper[2]]

    if file.shape != patchsize:
        pad_width = [pair for pair in zip(dlower, dupper)]
        file = np.pad(file, pad_width=pad_width, mode="constant", constant_values=fill)

    file = file.astype("float32")
    hf_file.close()

    assert file.shape == (
        z_size,
        xy_size,
        xy_size,
    ), f"Sanity check failed: {file.shape} == ({z_size}, {xy_size}, {xy_size})"

    return file


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


class CTCaseDataset(data.Dataset):
    def __init__(
        self,
        mode: Union[str, RunMode],
        mode_model: str = "2D",
        data_dir: str = None,
        fetch_from_patch: bool = True,
        dicom_window: list = None,
        translations: bool = None,
        rotations: tuple = None,
        size_xy: int = 128,
        size_z: int = 64,
        size_px_xy: int = 72,
        size_px_z: int = 48,
        size_mm: int = 50,
        interpolate_order: int = 1,
        dataset_infos=None,
        target_dataset_train=None,
        target_dataset_val_test=None,
        augmentation=None,
        use_weighted_sampler=None,
    ):
        self.mode: RunMode = RunMode(mode) if isinstance(mode, str) else mode
        self.mode_model = mode_model
        self.use_weighted_sampler = use_weighted_sampler

        # load dataset
        if self.mode == RunMode.TRAIN:
            self.target_dataset = OmegaConf.to_container(target_dataset_train, resolve=True)
        else:
            self.target_dataset = OmegaConf.to_container(target_dataset_val_test, resolve=True)
        self.dataset = self.get_meta_df(dataset_infos=dataset_infos)

        self.data_dir = Path(data_dir) if data_dir else None
        self.fetch_from_patch = fetch_from_patch

        self.dicom_window = dicom_window
        self.patch_size = [size_z, size_xy, size_xy]
        self.rotations = ast.literal_eval(rotations) if isinstance(rotations, str) else rotations
        self.translations = translations
        self.size_xy = size_xy
        self.size_z = size_z
        self.size_px_xy = size_px_xy
        self.size_px_z = size_px_z
        self.size_mm = size_mm
        self.order = interpolate_order
        # data augmentation
        if self.mode == RunMode.TRAIN:
            self.transform = ComposeAugmentation(
                transform=[
                    FlipXY(p=augmentation["flip_xy"]["aug_prob"]),
                    Flip3D(p=augmentation["flip_3d"]["aug_prob"]),
                    CoarseDropout3D(
                        p=augmentation["coarse_dropout_3d"]["aug_prob"],
                        max_holes=augmentation["coarse_dropout_3d"]["max_holes"],
                        size_limit=(0.08, 0.16),
                        patch_size=(self.size_px_z, self.size_px_xy, self.size_px_xy),
                    ),
                    RescaleImage(
                        p=augmentation["rescale_image_3d"]["aug_prob"],
                        is_same_across_axes=augmentation["rescale_image_3d"]["is_same_across_axes"],
                        min_scale_factor=augmentation["rescale_image_3d"]["scale_factor"]["min"],
                        max_scale_factor=augmentation["rescale_image_3d"]["scale_factor"]["max"],
                    ),
                ]
            )

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

    def get_meta_df(self, dataset_infos: dict):
        target_dataset_infos = {dataset: dataset_infos[dataset] for dataset in self.target_dataset}
        df = DatasetHandler().fetch_multiple_datasets(dataset_infos=target_dataset_infos, mode=self.mode)

        return df

    def __getitem__(self, idx):  # caseid, z, y, x, label, radius
        elem = self.dataset.iloc[idx]

        doc_id = elem[DataLakeKey.DOC_ID]
        collection_id = elem[DataLakeKey.COLLECTION]
        label = elem[DBKey.LABEL]
        annotation_id = elem[DBKey.ANNOTATION_ID]
        origin = np.array(elem[DBKey.ORIGIN])
        spacing = np.array(elem[DBKey.SPACING])
        transform = np.array(elem[DBKey.TRANSFORM])
        d_coord_zyx = np.array(elem[DBKey.D_COORD_ZYX])

        # fetch large patch
        if self.fetch_from_patch:
            image_path = self.data_dir / "image" / f"{annotation_id}.npy"
            img = np.load(image_path, mmap_mode="r")
        else:
            h5_path = elem[DBKey.H5_PATH_LOCAL]
            img = extract_patch_dicom_space(
                h5_path,
                d_coord_zyx,
                xy_size=self.size_xy,
                z_size=self.size_z,
                center_shift_zyx=[0, 0, 0],
                fill=-3024.0,
            )

        translations = None
        if self.translations == True:
            radius = 2.5
            translations = radius if radius > 0 else None

        if self.mode_model == "2D":
            output_shape = (1, self.size_px_xy, self.size_px_xy)
        else:
            output_shape = (self.size_px_z, self.size_px_xy, self.size_px_xy)

        patch = extract_patch(
            CTData=img,
            coord=tuple(np.array(self.patch_size) // 2),
            srcVoxelOrigin=origin,
            srcWorldMatrix=transform,
            srcVoxelSpacing=spacing,
            output_shape=output_shape,
            voxel_spacing=(
                self.size_mm / self.size_px_z,
                self.size_mm / self.size_px_xy,
                self.size_mm / self.size_px_xy,
            ),
            rotations=self.rotations,
            translations=translations,
            coord_space_world=False,
            mode=self.mode_model,
            order=self.order,
        )

        # ensure same datatype...
        patch = patch.astype(np.float32)

        # Data augmentation
        if self.mode == RunMode.TRAIN:
            if self.mode_model == "3D":
                patch = np.squeeze(patch, axis=0)
                patch = self.transform(patch)
                patch = np.expand_dims(patch, axis=0)

        # Data preprocessing
        patch = [fn(patch) for fn in self.dicom_windowing]
        patch = np.concatenate(patch, axis=0)

        target = torch.ones((1,)) * label

        sample = {
            DataLoaderKeys.COLLECTION_ID: collection_id,
            DataLoaderKeys.DOC_ID: str(doc_id),
            DataLoaderKeys.IMAGE: torch.from_numpy(patch).float(),  # float32
            DataLoaderKeys.LABEL: target.long(),
            DataLoaderKeys.ID: annotation_id,
        }

        return sample

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        return fmt_str


if __name__ == "__main__":
    with hydra.initialize_config_module(config_module="trainer.downstream.configs", version_base=None):
        config = hydra.compose(config_name="config")

    run_modes = [RunMode(m) for m in config.run_modes] if "run_modes" in config else [x for x in RunMode]
    loaders = {
        mode: hydra.utils.instantiate(config.loader, dataset={"mode": mode}, drop_last=False, shuffle=False)
        for mode in run_modes
    }
