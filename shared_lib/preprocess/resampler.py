import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import zoom as cupy_zoom
from scipy.ndimage.interpolation import zoom


def resample(dicom_pixels, original_spacing, new_spacing, mode="scipy", order=3):
    """Resample 3-D image by CPU w/ scipy or GPU w/ cupy. cupy works fine,
    but finally remains small-gpu-context(100~300MiB. depends on env).

    Args:
        dicom_pixels : `dicom_pixels` from `DICOM_Loader.load_dicom`.
            should be shape of (axial, coronal, sagittal)
        original_spacing: `original_spacing` from `get_pixels_hu`.
        new_spacing: target spacing
        mode : 'scipy', or 'cupy'.

    Returns:
        resampled_image
    """
    assert mode in ["scipy", "cupy"]
    resize_factor, new_shape, zoom_factors = get_real_resize_factor(
        dicom_pixels, original_spacing, new_spacing, ret_zoom_factor=True, verbose=False
    )
    if mode == "scipy":
        resampled_image = zoom(dicom_pixels, resize_factor, order=order)
        return resampled_image

    elif mode == "cupy":
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        resampled_dicom = cp.asarray(dicom_pixels).astype(dtype=cp.float32)
        resampled_dicom = cupy_zoom(
            input=resampled_dicom,
            zoom=zoom_factors,
            order=order,
            mode="nearest",
            prefilter=True,
            grid_mode=False,
        )
        resampled_dicom_np = cp.asnumpy(resampled_dicom)
        del resampled_dicom
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        return resampled_dicom_np


def get_real_resize_factor(origin_image, origin_spacing, new_spacing, ret_zoom_factor=False, verbose=False):
    """Get real_resize_factor when resample from origin_spacing to new_spacing.
    영상을 resampling하는 과정에서 new_shape는 필연적으로 float값을 갖게 되는데, 이걸 단순히
    round 처리하면 오차가 발생하게 됨. 이는 CT영상이 real-world coordinate를 repr하는 방식으로
    인해 더 커지게 되는데, 이를 감안하여 계산한 resize_factor를 real_resize_factor라고 명명함.
    Args:
        origin_image (array, shape): origin_image(ndarray) or shape.
        origin_spacing (tuple, list, array): origin_spacing(z, y, x)
        new_spacing (tuple, list, array): new_spacing(z, y, x)
        ret_zoom_factor (bool):
            if True, return zoom_factor for scipy-like func.
        verbose(bool): print verbose information

    Returns:
        real_resize_factor(list): resize_factor for z, y, x axis.
        new_shape(tuple): new_image's shape.
        (Optional) ret_zoom_factor(list):
            resize factor in the order of z, y, x axis
    """
    image_shape = origin_image.shape if origin_image is not None else None

    resize_factor = [a / b for a, b in zip(origin_spacing, new_spacing)]
    new_real_shape = [((a * b) + 1) for a, b in zip([s - 1 for s in image_shape], resize_factor)]
    new_shape = tuple([int(round(a)) for a in new_real_shape])
    real_resize_factor = [(a - 1) / (b - 1) for a, b in zip(list(new_shape), list(image_shape))]

    if verbose:
        print("origin_spacing: {}".format(origin_spacing))
        print("resize_factor: {}".format(resize_factor))
        print("new_spacing: {}".format(new_spacing))
        print("real_resize_factor: {}".format(real_resize_factor))
        real_new_spacing = [a / b for a, b in zip(origin_spacing, real_resize_factor)]
        print("real_new_spacing: {}".format(real_new_spacing))
        print("current_shape: {}".format(image_shape))
        print("new_shape: {}".format(new_shape))

    if ret_zoom_factor:
        zoom_factor = [a / b for a, b in zip(new_shape, image_shape)]
        return real_resize_factor, new_shape, zoom_factor

    else:
        return real_resize_factor, new_shape


def resample_coord_reverse(resampled_coord, resampled_image, origin_spacing, resampled_spacing):
    """convert resampled_coord to dicom_coord

    Args:
        resampled_coord(list, tuple): resampled_coordinate (r_z, r_y, r_x)
        resampled_image(ndarray): resampled image or shape.
        origin_spacing(list, tuple):
            dicom's original spacing,
            (SpacingBetweenSlices, PixelSpacing, PixelSpacing).
        resampled_spacing(list, tuple): resampled spacing (r_z, r_y, r_x)

    Returns:
        dicom_coordinate
    """

    real_resize_factor, _ = get_real_resize_factor(resampled_image, resampled_spacing, origin_spacing)
    return (np.round(np.asarray(resampled_coord) * np.asarray(real_resize_factor))).astype(int)
