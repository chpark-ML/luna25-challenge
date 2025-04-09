import numpy as np
import SimpleITK as sitk


def _transform(input_image, point):
    """

    Parameters
    ----------
    input_image: SimpleITK Image
    point: array of points

    Returns
    -------
    tNumpyOrigin

    """
    return np.array(list(reversed(input_image.TransformContinuousIndexToPhysicalPoint(list(reversed(point))))))


def itk_image_to_numpy_image(input_path, mode=None):
    """

    Parameters
    ----------
    input_path: SimpleITK image

    Returns
    -------
    numpyImage: SimpleITK image to numpy image
    header: dict containing origin, spacing and transform in numpy format

    """

    if mode.lower() == 'dicom':
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(input_path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
    elif mode.lower() == 'mha':
        image = sitk.ReadImage(input_path)
    else:
        raise ValueError("Unsupported mode. Use 'mha' or 'dicom'.")

    numpy_image = sitk.GetArrayFromImage(image)  # shape: (slices, height, width)

    # Extract metadata
    origin = np.array(list(reversed(image.GetOrigin())))          # z, y, x
    spacing = np.array(list(reversed(image.GetSpacing())))        # z, y, x
    direction = np.array(image.GetDirection()).reshape(3, 3)      # 3x3 matrix
    transform_matrix = direction.dot(np.diag(spacing))            # voxel-to-world

    # Compose metadata header
    header = {
        "origin": origin,
        "spacing": spacing,
        "transform": transform_matrix
    }

    return numpy_image, header
