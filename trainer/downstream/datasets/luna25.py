import ast
import logging
from pathlib import Path
from typing import List, Union

import hydra
import numpy as np
import numpy.linalg as npl
import pandas as pd
import pymongo
import scipy.ndimage as ndi
import torch
import torch.utils.data as data
from h5py import File
from omegaconf import OmegaConf

from data_lake.constants import DBKey, H5DataKey
from data_lake.dataset_handler import DatasetHandler
from shared_lib.enums import RunMode
from trainer.common.augmentation.coarse_drop import CoarseDropout3D
from trainer.common.augmentation.compose import ComposeAugmentation
from trainer.common.augmentation.flip_rotate import Flip3D, FlipXY
from trainer.common.augmentation.hu_value import DicomWindowing, RandomDicomWindowing
from trainer.common.augmentation.rescale import RescaleImage
from trainer.common.constants import DB_ADDRESS, HU_WINDOW

logger = logging.getLogger(__name__)
_VUNO_LUNG_DB = DB_ADDRESS


class DataLoaderKeys:
    IMAGE = "image"
    LABEL = "label"
    ID = "ID"


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


def _extract_patch(
    h5_path,
    coord,
    xy_size: int = 72,
    z_size: int = 72,
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


def _calculateAllPermutations(itemList):
    if len(itemList) == 1:
        return [[i] for i in itemList[0]]
    else:
        sub_permutations = _calculateAllPermutations(itemList[1:])
        return [[i] + p for i in itemList[0] for p in sub_permutations]


def volumeTransform(
    image,
    voxel_spacing,
    transform_matrix,
    center=None,
    output_shape=None,
    output_voxel_spacing=None,
    **argv,
):
    """
    Parameters
    ----------
      image : a numpy.ndarray
          The image that should be transformed

      voxel_spacing : a vector
          This vector describes the voxel spacing between individual pixels. Can
          be filled with (1,) * image.ndim if unknown.

      transform_matrix : a Nd x Nd matrix where Nd is the number of image dimensions
          This matrix governs how the output image will be oriented. The x-axis will be
          oriented along the last row vector of the transform_matrix, the y-Axis along
          the second-to-last row vector etc. (Note that numpy uses a matrix ordering
          of axes to index image axes). The matrix must be square and of the same
          order as the dimensions of the input image.

          Typically, this matrix is the transposed mapping matrix that maps coordinates
          from the projected image to the original coordinate space.

      center : vector (default: None)
          The center point around which the transform_matrix pivots to extract the
          projected image. If None, this defaults to the center point of the
          input image.

      output_shape : a list of integers (default None)
          The shape of the image projection. This can be used to limit the number
          of pixels that are extracted from the orignal image. Note that the number
          of dimensions must be equal to the number of dimensions of the
          input image. If None, this defaults to dimenions needed to enclose the
          whole inpput image given the transform_matrix, center, voxelSPacings,
          and the output_shape.

      output_voxel_spacing : a vector (default: None)
          The interleave at which points should be extracted from the original image.
          None, lets the function default to a (1,) * output_shape.ndim value.

      **argv : extra arguments
          These extra arguments are passed directly to scipy.ndimage.affine_transform
          to allow to modify its behavior. See that function for an overview of optional
          paramters (other than offset and output_shape which are used by this function
          already).
    """
    if "offset" in argv:
        raise ValueError("Cannot supply 'offset' to scipy.ndimage.affine_transform - already used by this function")
    if "output_shape" in argv:
        raise ValueError(
            "Cannot supply 'output_shape' to scipy.ndimage.affine_transform - already used by this function"
        )

    if image.ndim != len(voxel_spacing):
        raise ValueError("Voxel spacing must have the same dimensions")

    if center is None:
        voxelCenter = (np.array(image.shape) - 1) / 2.0
    else:
        if len(center) != image.ndim:
            raise ValueError("center point has not the same dimensionality as the image")

        # Transform center to voxel coordinates
        voxelCenter = np.asarray(center) / voxel_spacing

    transform_matrix = np.asarray(transform_matrix)
    if output_voxel_spacing is None:
        if output_shape is None:
            output_voxel_spacing = np.ones(transform_matrix.shape[0])
        else:
            output_voxel_spacing = np.ones(len(output_shape))
    else:
        output_voxel_spacing = np.array(output_voxel_spacing)

    if transform_matrix.shape[1] != image.ndim:
        raise ValueError(
            "transform_matrix does not have the correct number of columns (does not match image dimensionality)"
        )
    if transform_matrix.shape[0] != image.ndim:
        raise ValueError(
            "Only allowing square transform matrices here, even though this is unneccessary. However, one will need an algorithm here to create full rank-square matrices. 'QR decomposition with Column Pivoting' would probably be a solution, but the author currently does not know what exactly this is, nor how to do this..."
        )

    # Normalize the transform matrix
    transform_matrix = np.array(transform_matrix)
    transform_matrix = (transform_matrix.T / np.sqrt(np.sum(transform_matrix * transform_matrix, axis=1))).T
    transform_matrix = np.linalg.inv(transform_matrix.T)  # Important normalization for shearing matrices!!

    # The forwardMatrix transforms coordinates from input image space into result image space
    forward_matrix = np.dot(
        np.dot(np.diag(1.0 / output_voxel_spacing), transform_matrix),
        np.diag(voxel_spacing),
    )

    if output_shape is None:
        # No output dimensions are specified
        # Therefore we calculate the region that will span the whole image
        # considering the transform matrix and voxel spacing.
        image_axes = [[0 - o, x - 1 - o] for o, x in zip(voxelCenter, image.shape)]
        image_corners = _calculateAllPermutations(image_axes)

        transformed_image_corners = map(lambda x: np.dot(forward_matrix, x), image_corners)
        output_shape = [
            1 + int(np.ceil(2 * max(abs(x_max), abs(x_min))))
            for x_min, x_max in zip(
                np.amin(transformed_image_corners, axis=0),
                np.amax(transformed_image_corners, axis=0),
            )
        ]
    else:
        # Check output_shape
        if len(output_shape) != transform_matrix.shape[1]:
            raise ValueError("output dimensions must match dimensionality of the transform matrix")
    output_shape = np.array(output_shape)

    # Calculate the backwards matrix which will be used for the slice extraction
    backwards_matrix = npl.inv(forward_matrix)
    target_image_offset = voxelCenter - backwards_matrix.dot((output_shape - 1) / 2.0)

    return ndi.affine_transform(
        image,
        backwards_matrix,
        offset=target_image_offset,
        output_shape=output_shape,
        **argv,
    )


def rotateMatrixX(cosAngle, sinAngle):
    return np.asarray([[1, 0, 0], [0, cosAngle, -sinAngle], [0, sinAngle, cosAngle]])


def rotateMatrixY(cosAngle, sinAngle):
    return np.asarray([[cosAngle, 0, sinAngle], [0, 1, 0], [-sinAngle, 0, cosAngle]])


def rotateMatrixZ(cosAngle, sinAngle):
    return np.asarray([[cosAngle, -sinAngle, 0], [sinAngle, cosAngle, 0], [0, 0, 1]])


def sample_random_coordinate_on_sphere(radius):
    # Generate three random numbers x,y,z using Gaussian distribution
    random_nums = np.random.normal(size=(3,))

    # You should handle what happens if x=y=z=0.
    if np.all(random_nums == 0):
        return np.zeros((3,))

    # Normalise numbers and multiply number by radius of sphere
    return random_nums / np.sqrt(np.sum(random_nums * random_nums)) * radius


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


def extract_patch(
    CTData,
    coord,
    srcVoxelOrigin,
    srcWorldMatrix,
    srcVoxelSpacing,
    output_shape=(64, 64, 64),
    voxel_spacing=(50.0 / 64, 50.0 / 64, 50.0 / 64),
    rotations=None,
    translations=None,
    coord_space_world=False,
    mode="2D",
):
    transform_matrix = np.eye(3)

    if rotations is not None:
        (zmin, zmax), (ymin, ymax), (xmin, xmax) = rotations

        # add random rotation
        angleX = np.multiply(np.pi / 180.0, np.random.randint(xmin, xmax, 1))[0]
        angleY = np.multiply(np.pi / 180.0, np.random.randint(ymin, ymax, 1))[0]
        angleZ = np.multiply(np.pi / 180.0, np.random.randint(zmin, zmax, 1))[0]

        transformMatrixAug = np.eye(3)
        transformMatrixAug = np.dot(transformMatrixAug, rotateMatrixX(np.cos(angleX), np.sin(angleX)))
        transformMatrixAug = np.dot(transformMatrixAug, rotateMatrixY(np.cos(angleY), np.sin(angleY)))
        transformMatrixAug = np.dot(transformMatrixAug, rotateMatrixZ(np.cos(angleZ), np.sin(angleZ)))

        transform_matrix = np.dot(transform_matrix, transformMatrixAug)

    if translations is not None:
        # add random translation
        radius = np.random.random_sample() * translations
        offset = sample_random_coordinate_on_sphere(radius=radius)
        offset = offset * (1.0 / srcVoxelSpacing)

        coord = np.array(coord) + offset

    # Normalize transform matrix
    thisTransformMatrix = transform_matrix

    thisTransformMatrix = (thisTransformMatrix.T / np.sqrt(np.sum(thisTransformMatrix * thisTransformMatrix, axis=1))).T

    invSrcMatrix = np.linalg.inv(srcWorldMatrix)

    # world coord sampling
    if coord_space_world:
        overrideCoord = invSrcMatrix.dot(coord - srcVoxelOrigin)
    else:
        # image coord sampling
        overrideCoord = coord * srcVoxelSpacing
    overrideMatrix = (invSrcMatrix.dot(thisTransformMatrix.T) * srcVoxelSpacing).T

    patch = volumeTransform(
        CTData,
        srcVoxelSpacing,
        overrideMatrix,
        center=overrideCoord,
        output_shape=np.array(output_shape),
        output_voxel_spacing=np.array(voxel_spacing),
        order=1,
        prefilter=False,
    )

    if mode == "2D":
        # replicate the channel dimension
        patch = np.repeat(patch, 3, axis=0)

    else:
        patch = np.expand_dims(patch, axis=0)

    return patch


class CTCaseDataset(data.Dataset):
    def __init__(
        self,
        mode: Union[str, RunMode],
        mode_model: str = "2D",
        data_dir: str = None,
        fetch_from_patch: bool = True,
        dicom_window: list = None,
        patch_size: list = None,
        translations: bool = None,
        rotations: tuple = None,
        size_xy: int = 128,
        size_z: int = 64,
        size_px: int = 64,
        size_mm: int = 50,
        dataset_infos=None,
        target_dataset_train=None,
        target_dataset_val_test=None,
        augmentation=None,
    ):
        self.mode: RunMode = RunMode(mode) if isinstance(mode, str) else mode
        self.mode_model = mode_model

        # load dataset
        if self.mode == RunMode.TRAIN:
            self.target_dataset = OmegaConf.to_container(target_dataset_train, resolve=True)
        else:
            self.target_dataset = OmegaConf.to_container(target_dataset_val_test, resolve=True)
        self.dataset = self.get_meta_df(dataset_infos=dataset_infos)

        self.data_dir = Path(data_dir) if data_dir else None
        self.fetch_from_patch = fetch_from_patch

        self.dicom_window = dicom_window
        self.patch_size = patch_size
        self.rotations = ast.literal_eval(rotations) if isinstance(rotations, str) else rotations
        self.translations = translations
        self.size_xy = size_xy
        self.size_z = size_z
        self.size_px = size_px
        self.size_mm = size_mm

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
                        patch_size=(self.size_px, self.size_px, self.size_px),
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
            img = _extract_patch(
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
            output_shape = (1, self.size_px, self.size_px)
        else:
            output_shape = (self.size_px, self.size_px, self.size_px)

        patch = extract_patch(
            CTData=img,
            coord=tuple(np.array(self.patch_size) // 2),
            srcVoxelOrigin=origin,
            srcWorldMatrix=transform,
            srcVoxelSpacing=spacing,
            output_shape=output_shape,
            voxel_spacing=(
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
            ),
            rotations=self.rotations,
            translations=translations,
            coord_space_world=False,
            mode=self.mode_model,
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
            DataLoaderKeys.IMAGE: torch.from_numpy(patch),
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
        mode: hydra.utils.instantiate(config.inputs, dataset={"mode": mode}, drop_last=False, shuffle=False)
        for mode in run_modes
    }
