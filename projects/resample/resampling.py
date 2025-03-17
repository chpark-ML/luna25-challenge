import os

import h5py
import numpy as np
import pydicom
import SimpleITK as sitk
import torch
from scipy.ndimage import zoom
from tqdm import tqdm


def load_mha(file_path):
    image = sitk.ReadImage(file_path)
    volume = sitk.GetArrayFromImage(image)  # z, y, x
    spacing = image.GetSpacing()  # (x_spacing, y_spacing, z_spacing)
    spacing = (spacing[2], spacing[1], spacing[0])
    origin = image.GetOrigin()  # (x_origin, y_origin, z_origin)
    origin = (origin[2], origin[1], origin[0])
    direction = image.GetDirection()  # flatten 3x3 matrix
    return volume, spacing, origin, direction


def resample_image(volume, spacing, new_spacing=(1.0, 1.0, 1.0)):
    vol_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, z, y, x)
    scale_factors = torch.tensor(
        [spacing[0] / new_spacing[0], spacing[1] / new_spacing[1], spacing[2] / new_spacing[2]]
    )
    size = [int(dim * scale) for dim, scale in zip(volume.shape, scale_factors)]
    resampled = torch.nn.functional.interpolate(vol_tensor, size=size, mode="trilinear", align_corners=False)
    return resampled.squeeze().numpy()


def save_to_h5(output_path, volume, resampled, spacing, new_spacing, origin):
    with h5py.File(output_path, "w") as h5f:
        h5f.create_dataset("volume_image", data=volume)
        h5f.create_dataset("resampled_image", data=resampled)
        h5f.attrs["original_spacing"] = spacing
        h5f.attrs["resampled_spacing"] = new_spacing
        h5f.attrs["origin"] = origin


def process_mha_to_h5(mha_dir, output_dir, new_spacing=(1.0, 1.0, 1.0)):
    os.makedirs(output_dir, exist_ok=True)
    for file in tqdm(os.listdir(mha_dir)):
        if file.endswith(".mha"):
            file_path = os.path.join(mha_dir, file)
            volume, spacing, origin, direction = load_mha(file_path)
            resampled = resample_image(volume, spacing, new_spacing)
            output_path = os.path.join(output_dir, file.replace(".mha", ".h5"))
            save_to_h5(output_path, volume, resampled, spacing, new_spacing, origin)
            print(f"Saved: {output_path}")


if __name__ == "__main__":

    dicom_dir = "/team/team_blu3/lung/data/2_public/LUNA25_Original/luna25_images"
    output_dir = "/team/team_blu3/lung/data/2_public/LUNA25_resampled"
    new_spacing = (1.0, 0.67, 0.67)

    process_mha_to_h5(dicom_dir, output_dir, new_spacing)
