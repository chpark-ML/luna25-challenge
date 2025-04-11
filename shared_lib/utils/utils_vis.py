from typing import Optional, Sequence, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.colors import ListedColormap

"""
How to use `plot_3d_images`

lct/tools/plot_3d_nodule_example
lct/tools/plot_3d_volume_example

"""


def patch_extract(
    resampled_ct: np.ndarray,
    center_coord: Optional[Sequence[int]] = None,
    voxel_width: Optional[Union[int, Sequence[int]]] = 32,
    floor: bool = True,
    pad_value: Optional[Union[int, float]] = 0,
) -> np.ndarray:
    """Extract a 3D patch from the given CT volume.

    Not creating a separate copy of the patch by design.
    The `patch` may be a view of the `volume` array.

    Args:
        resampled_ct: Input volume. May also be a HDF5 memory map.
        center_coord: Center coordinate for patch extraction.
        voxel_width: Patch size.
        floor: Wheter to apply floor function when
            cropping patch. Round otherwise.
        pad_value: Value to pad external region.
            Zero-padding by default.

    Returns:
        A 3D patch extracted from the given volume.
    """
    if not isinstance(voxel_width, int):
        assert len(voxel_width) == 3
    else:
        voxel_width = (voxel_width, voxel_width, voxel_width)

    depth_ct, height_ct, width_ct = resampled_ct.shape  # Forces 3D input.
    if center_coord is None:
        center_coord = [depth_ct // 2, height_ct // 2, width_ct // 2]

    depth_center, height_center, width_center = center_coord

    depth, height, width = voxel_width

    depth_left = depth_center - depth // 2  # Left of depth axis
    height_left = height_center - height // 2  # Left of height axis
    width_left = width_center - width // 2  # Left of width axis
    depth_right = depth_left + depth
    height_right = height_left + height
    width_right = width_left + width

    # `max(x, 0)` prevents negative indexing from creating a length 0 slice.
    # Indexing with values larger than the maximum index is legal in numpy.
    padding = 0.5 if floor else 0
    patch = resampled_ct[
        int(max(0, depth_left) + padding) : int(depth_right + padding),
        int(max(0, height_left) + padding) : int(height_right + padding),
        int(max(0, width_left) + padding) : int(width_right + padding),
    ]

    if patch.shape != voxel_width:
        dp_left = int(abs(min(depth_left, 0)) + 0.5)
        hp_left = int(abs(min(height_left, 0)) + 0.5)
        wp_left = int(abs(min(width_left, 0)) + 0.5)

        dp_right = int(max(depth_right - depth_ct, 0) + 0.5)
        hp_right = int(max(height_right - height_ct, 0) + 0.5)
        wp_right = int(max(width_right - width_ct, 0) + 0.5)
        ddd, hhh, www = patch.shape

        assert (
            dp_left + ddd + dp_right == depth
        ), f"Incorrect depth - dpl:{dp_left} ddd:{ddd} dpr:{dp_right} sum:{dp_left + ddd + dp_right} dp:{depth}"
        assert (
            hp_left + hhh + hp_right == height
        ), f"Incorrect height - hpl:{hp_left} hhh:{hhh} hpr:{hp_right} sum:{hp_left + hhh + hp_right} hp:{height}"
        assert (
            wp_left + www + wp_right == width
        ), f"Incorrect width - wpl:{wp_left} www:{www} wpr:{wp_right} sum:{wp_left + www + wp_right} wp:{width}"

        pad_width = ((dp_left, dp_right), (hp_left, hp_right), (wp_left, wp_right))
        patch = np.pad(patch, pad_width=pad_width, mode="constant", constant_values=pad_value)

    assert patch.shape == voxel_width, f"{patch.shape} != {voxel_width}. Sanity check failed."
    return patch.copy()


def normalize_planes(npzarray: np.ndarray, min_hu: float = -1000.0, max_hu: float = 600.0) -> np.ndarray:
    if min_hu > max_hu:
        raise ValueError(f"minHU should be smaller than maxHU, but got {min_hu} > {max_hu}")

    npzarray = (npzarray - min_hu) / (max_hu - min_hu)
    return np.clip(npzarray, 0, 1)


def fn_imshow(
    _ax,
    image_2d: np.array,
    vmin: float,
    vmax: float,
    vmin_mask: float,
    vmax_mask: float,
    xy: tuple = None,
    patch_size: int = None,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    mask_2d: np.array = None,
):
    _ax.imshow(image_2d, cmap="gray", vmin=vmin, vmax=vmax)
    if mask_2d is not None:
        if (vmin_mask == -1) and (vmax_mask == 1):
            N = 256
            red = np.ones((N, 4))
            blue = np.ones((N, 4))
            blue[:, 0] = np.linspace(0 / 256, 1, N)
            blue[:, 1] = np.linspace(0 / 256, 1, N)
            blue[:, 2] = np.linspace(200 / 256, 1, N)
            red[:, 0] = np.linspace(1, 200 / 256, N)  # R
            red[:, 1] = np.linspace(1, 0 / 256, N)  # G
            red[:, 2] = np.linspace(1, 0 / 256, N)  # B
            red_cmp = ListedColormap(red)
            blue_cmp = ListedColormap(blue)
            newcolors2 = np.vstack((blue_cmp(np.linspace(0, 1, 128)), red_cmp(np.linspace(0, 1, 128))))
            cmap_heatmap = ListedColormap(newcolors2, name="double")
            cmap_heatmap.set_extremes()
        else:
            cmap_heatmap = plt.get_cmap("jet")

        im = _ax.imshow(mask_2d, cmap=cmap_heatmap, alpha=0.5, vmin=vmin_mask, vmax=vmax_mask)
        # _mask = (mask_2d > 0 + vmin_mask * 0.1) & (mask_2d < 0 + vmax_mask * 0.1)
        # im.set_array(np.ma.masked_where(_mask, mask_2d))

    if xy is not None and patch_size is not None:
        rect = patches.Rectangle(xy, patch_size, patch_size, linewidth=1, edgecolor="r", facecolor="none")
        _ax.add_patch(rect)

    if title:
        _ax.set_title(title)
    if xlabel:
        _ax.set_xlabel(xlabel)
    if ylabel:
        _ax.set_ylabel(ylabel)
    _ax.set_yticks([])
    _ax.set_xticks([])


def save_plot(
    input_image: np.array,
    mask_image: np.array = None,
    result_image: np.array = None,
    nodule_zyx: list = None,
    patch_size: Optional[Union[int, Sequence[int]]] = (64, 64, 64),
    figure_title: str = "",
    meta: dict = None,
    use_norm: bool = False,
    vmin: float = 0.0,
    vmax: float = 1.0,
    vmin_mask: float = 0.0,
    vmax_mask: float = 1.0,
    save_dir: str = None,
    **kwargs,
) -> None:
    """
    A function for visualization of 3D patches by taking a input image and 3D coordinates.
    It shows the multiple number of 2D axial, coronal, sagittal planes.

    case 1 : whole_image (EDA)
    case 2 : whole_image, coord (Detector, FPR)
    case 3 : whole_image, whole_mask (Lobe mask)
    case 4 : whole_image, whole_mask, coord (Nodule seg)
    case 5 : patch_image (EDA)
    case 6 : patch_image, patch_mask (Nodule seg)

    Args:
        input_image: 3D numpy array
        nodule_zyx: List of z, y, x
        figure_title: figure title
        meta: Metadata dictionary for right upper text box
        patch_size: size of patch
        use_norm: whether to normalize image
        vmin: minimum value for plotting
        vmax: maximum value for plotting
        save_dir: dir to save figure
    """
    # set plot configuration for 3D whole CT
    start_percentile = 15
    end_percentile = 95
    interval_percentile = 10

    # set plot configuration for 3D patch
    num_column = 10
    interval_between_columns = 1

    # set patch size
    if patch_size is not None:
        if not isinstance(patch_size, int):
            assert len(patch_size) == 3
        else:
            patch_size = (patch_size, patch_size, patch_size)

    # normalize image
    if use_norm:
        input_image = input_image.astype(np.float32)
        input_image = normalize_planes(input_image, min_hu=-1000.0, max_hu=600.0)
        if False:  # TODO: Not implemented.
            result_image = result_image.astype(np.float32)
            result_image = normalize_planes(result_image, min_hu=-1000.0, max_hu=600.0)

    # 1. flip z-axis for better visualization
    input_image = input_image[::-1, :, :]
    if mask_image is not None:
        mask_image = mask_image[::-1, :, :]
    if result_image is not None:
        result_image = result_image[::-1, :, :]

    if nodule_zyx:  # case 2: whole_image, coord / case 4: whole_image, whole_mask, coord
        nodule_zyx_reversed = nodule_zyx.copy()
        nodule_zyx_reversed[0] = (input_image.shape[0] - 1) - nodule_zyx_reversed[0]

        # patch extraction
        patch_img = patch_extract(input_image, center_coord=nodule_zyx_reversed, voxel_width=patch_size)
        if mask_image is not None:
            patch_mask = patch_extract(mask_image, center_coord=nodule_zyx_reversed, voxel_width=patch_size)
        if result_image is not None:
            patch_result = patch_extract(result_image, center_coord=nodule_zyx_reversed, voxel_width=patch_size)
    else:  # case1, case3, case5, case6
        patch_img = input_image

    # 2. get slice index to plot image
    slice_index = []
    input_shape = input_image.shape
    for i_axis in range(3):  # axial, coronal, sagittal
        tmp_slice_index = []
        if nodule_zyx or (input_shape == patch_size):  # (case 2, 4) or (case 5, 6)
            tmp_slice_index.append(
                np.arange(patch_size[i_axis])[
                    patch_size[i_axis] // 2
                    - (interval_between_columns * (num_column // 2)) : patch_size[i_axis] // 2
                    + (interval_between_columns * (num_column // 2))
                    + 1 : interval_between_columns
                ]
            )
        else:  # case 1, 3
            for i_percentile in np.arange(start_percentile, end_percentile, interval_percentile):
                tmp_slice_index.append(int(np.percentile(np.arange(0, input_shape[i_axis]), i_percentile)))

        slice_index.append(np.hstack(tmp_slice_index))

    # 3. get width and height
    if nodule_zyx:
        image_size_ratio_zyx = [i_shape / sum(patch_size) * 15 for i_shape in patch_size]
    else:
        image_size_ratio_zyx = [i_shape / sum(input_shape) * 15 for i_shape in input_shape]

    # heights
    heights = [
        image_size_ratio_zyx[1],
        image_size_ratio_zyx[0],
        image_size_ratio_zyx[0],
    ]  # [y, z, z]
    if nodule_zyx:
        heights.insert(0, max(0.5 * len(meta), 5))  # text, image-level view
    else:
        heights.insert(0, 0.5 * len(meta))  # [a, b, c, d]

    # width
    widths = [
        [image_size_ratio_zyx[2]] * (len(slice_index[0])),
        [image_size_ratio_zyx[2]] * (len(slice_index[0])),
        [image_size_ratio_zyx[1]] * (len(slice_index[0])),
    ]  # e.g., [[x, ..., x], [x,..., x], [y,..., y]]
    widths.insert(
        0, [image_size_ratio_zyx[2]] * (len(slice_index[0]))
    )  # e.g., [[x, ..., x], [x, ..., x], [x,..., x], [y,..., y]]
    width_max_size = np.max([sum(width) for width in widths])

    # if mask exists, add width, height
    _tmp_height = heights.copy()  # [a, b, c, d]
    _tmp_width = widths.copy()  # [[x,..., x], [x,..., x], [x,..., x], [x,..., x]]
    num_insert = int(mask_image is not None) + int(result_image is not None)
    if (mask_image is not None) or (result_image is not None):
        for i in np.arange(-3, 0, 1):
            for j in range(num_insert):
                heights.insert(i, _tmp_height[i])  # [a, b, b, c, c, d, d]
                widths.insert(i, _tmp_width[i])

    # 4. set figure configuration
    fig = plt.figure(figsize=(int(width_max_size), int(sum(heights))))  # [w, h]
    plt.rcParams.update({"font.size": 18, "font.style": "italic"})
    plt.suptitle("{}".format(figure_title), fontsize=30)
    list_gs = [
        gridspec.GridSpec(
            nrows=len(heights),
            ncols=len(widths[idx]),
            height_ratios=heights,
            width_ratios=widths[idx],
            hspace=0.15,
            wspace=0.04,
        )
        for idx, _ in enumerate(heights)
    ]

    # 5. set row name
    axis_type = ["Axial", "Coronal", "Sagittal"]
    axis_type_to_plot = list()
    for i_plane in axis_type:
        axis_type_to_plot.append(i_plane + " img")
        if mask_image is not None:
            axis_type_to_plot.append(i_plane + " mask")
        if result_image is not None:
            axis_type_to_plot.append(i_plane + " result")

    # 6. plot the first row if nodule coordinate was given (3 planes passing through the given variable "nodule_zyx")
    if nodule_zyx:
        for idx, (i_axis_type, i_coord, i_coord_rev) in enumerate(zip(axis_type, nodule_zyx, nodule_zyx_reversed)):
            _ax = fig.add_subplot(list_gs[0][0, idx])
            ylabel = "Image-level" if idx == 0 else None
            tmp_xy = nodule_zyx_reversed.copy()
            del tmp_xy[idx]
            fn_imshow(
                _ax,
                np.rollaxis(input_image, idx)[i_coord_rev],
                vmin,
                vmax,
                vmin_mask,
                vmax_mask,
                (tmp_xy[1] - patch_size[idx] // 2, tmp_xy[0] - patch_size[idx] // 2),
                patch_size[idx],
                title=i_axis_type,
                xlabel=f"{i_coord} / {input_image.shape[idx]}",
                ylabel=ylabel,
            )

    # 7. get axis-wise 2d plots
    if nodule_zyx:
        image_to_plot = patch_img
        if mask_image is not None:
            mask_to_plot = patch_mask
        if result_image is not None:
            result_to_plot = patch_result
    else:
        image_to_plot = input_image
        if mask_image is not None:
            mask_to_plot = mask_image
        if result_image is not None:
            result_to_plot = result_image

    start_row = 1
    _shape = image_to_plot.shape
    _counter_axis = -1
    for idx, i_axis_type in enumerate(axis_type_to_plot):  # row
        if "img" in i_axis_type:
            _counter_axis += 1
        for i_col, i_inter in enumerate(slice_index[_counter_axis]):  # col
            i_inter = min(max(i_inter, 0), image_to_plot.shape[_counter_axis] - 1)
            _ax = fig.add_subplot(list_gs[start_row + idx][start_row + idx, i_col])
            ylabel = i_axis_type if i_col == 0 else None
            if mask_image is not None and ("mask" in i_axis_type):
                fn_imshow(
                    _ax,
                    np.rollaxis(image_to_plot, _counter_axis)[i_inter],
                    vmin,
                    vmax,
                    vmin_mask,
                    vmax_mask,
                    None,
                    None,
                    title=None,
                    xlabel=f"{i_inter} / {_shape[_counter_axis]}",
                    ylabel=ylabel,
                    mask_2d=np.rollaxis(mask_to_plot, _counter_axis)[i_inter],
                )
            elif mask_image is not None and ("result" in i_axis_type):
                fn_imshow(
                    _ax,
                    np.rollaxis(result_to_plot, _counter_axis)[i_inter],
                    vmin,
                    vmax,
                    vmin_mask,
                    vmax_mask,
                    None,
                    None,
                    title=None,
                    xlabel=f"{i_inter} / {_shape[_counter_axis]}",
                    ylabel=ylabel,
                )
            else:
                fn_imshow(
                    _ax,
                    np.rollaxis(image_to_plot, _counter_axis)[i_inter],
                    vmin,
                    vmax,
                    vmin_mask,
                    vmax_mask,
                    None,
                    None,
                    title=None,
                    xlabel=f"{i_inter} / {_shape[_counter_axis]}",
                    ylabel=ylabel,
                )
    # font
    font = {"color": "k", "size": 20}

    # text
    _ax = fig.add_subplot(list_gs[0][0, int(len(slice_index[0]) * 0.6)])

    if meta:
        list_attr = [f"{k}: {v}" for k, v in meta.items()]
        _ax.text(
            1.0,
            0.0,
            "\n".join(list_attr),
            bbox={"facecolor": "white", "pad": 2.5},
            fontdict=font,
        )

    _ax.set_yticks([])
    _ax.set_xticks([])
    _ax.spines["right"].set_visible(False)
    _ax.spines["top"].set_visible(False)
    _ax.spines["bottom"].set_visible(False)
    _ax.spines["left"].set_visible(False)

    if save_dir:
        dpi = kwargs.get("dpi", 50)
        plt.savefig(save_dir, bbox_inches="tight", dpi=dpi)
        plt.close()
    else:
        plt.close()
        return fig
