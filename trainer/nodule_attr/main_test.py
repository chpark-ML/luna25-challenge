import logging
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from trainer.common.enums import ModelName, RunMode
from trainer.common.utils.utils import load_model

logger = logging.getLogger(__name__)
_TARGET_ATTR_TO_ANALYSIS = [
    "c_margin_logistic",
    "c_texture_logistic",
    "c_calcification_logistic",
    "c_internalStructure_logistic",
    "c_spiculation_logistic",
    "c_subtlety_logistic",
    "c_sphericity_logistic",
    "c_lobulation_logistic",
]


def fn_imshow(
    _ax,
    image_2d: np.array,
    vmin: float,
    vmax: float,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
):
    if vmin == -vmax:
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
        cmap_heatmap = plt.get_cmap("gray")

    _ax.imshow(image_2d, cmap=cmap_heatmap, vmin=vmin, vmax=vmax)

    if title:
        _ax.set_title(title)
    if xlabel:
        _ax.set_xlabel(xlabel)
    if ylabel:
        _ax.set_ylabel(ylabel)
    _ax.set_yticks([])
    _ax.set_xticks([])


def _plot_multiple_center_axis(image_to_plots: dict, save_dir=None):
    """
    e.g.,
    image_to_plost = {
        "orig": [],
        "gate" : [],
        "gan_result": [
            {"score", ...}
        ]
    }
    key : row / value : columns
    """
    row_type = [[f"{i.split('_')[1]} class evidence"] for i in _TARGET_ATTR_TO_ANALYSIS]
    row_type.insert(0, "orig")

    num_row = len(row_type) * 2 - 1
    num_column = 8

    heights = list()
    for i in range(num_row):
        if i == 0:
            heights.append(3)
        else:
            heights.append(0.5)
            heights.append(3)

    # heights.insert(0, 0.5)  # print score
    widths = [3 for _ in range(num_column)]
    widths[1] = 1

    fig = plt.figure(figsize=(int(num_column) * 1.8, int(num_row) * 2.2))  # [w, h]

    plt.rcParams.update({"font.size": 8, "font.style": "italic"})
    # plt.suptitle("{}".format(attr_to_analysis), fontsize=10)
    list_gs = [
        gridspec.GridSpec(
            nrows=len(heights),
            ncols=len(widths),
            height_ratios=heights,
            width_ratios=widths,
            hspace=0.05,
            wspace=0.04,
        )
        for idx, _ in enumerate(heights)
    ]

    _ax = fig.add_subplot(list_gs[0][0, 0])
    orig = image_to_plots["orig"]
    orig_to_print = orig.squeeze()[orig.shape[-3] // 2].detach().cpu().numpy()
    fn_imshow(_ax, orig_to_print, vmin=0, vmax=1, title="orig", xlabel=None)

    list_gate = image_to_plots["gates"]
    dict_gated_CE = image_to_plots["gated_CEs"]

    gates = list_gate
    gates_to_print = [
        F.interpolate(gate, size=orig.squeeze().size(), mode="trilinear", align_corners=True)
        .squeeze()[orig.shape[-3] // 2]
        .detach()
        .cpu()
        .numpy()
        for gate in gates
    ]

    # print gate
    for col_index, gate_to_print in enumerate(gates_to_print):
        row = 0
        row_index = row * 2
        _ax = fig.add_subplot(list_gs[row_index][row_index, 2 + col_index])
        _title = f"gate {col_index}" if row == 0 else None
        _xlabel = None
        fn_imshow(
            _ax,
            gate_to_print,
            vmin=-1,
            vmax=1,
            title=_title,
            xlabel=_xlabel,
            ylabel=None,
        )

    # plot
    for row, i_attr in enumerate(_TARGET_ATTR_TO_ANALYSIS):
        row_index = row * 2
        ces = dict_gated_CE[i_attr]
        ces_to_print = [
            F.interpolate(ce, size=orig.squeeze().size(), mode="trilinear", align_corners=True)
            .squeeze()[orig.shape[-3] // 2]
            .detach()
            .cpu()
            .numpy()
            for ce in ces
        ]

        # print CE
        for col_index, ce_to_print in enumerate(ces_to_print):
            _ax = fig.add_subplot(list_gs[row_index][row_index, 2 + col_index + len(gates_to_print)])
            _title = f"CE {col_index}" if row == 0 else None
            _xlabel = None
            _ylabel = None

            # get vmin, vmax
            _percentile = 0.1
            vmax = np.percentile(ce_to_print, 100 - _percentile)
            vmin = np.percentile(ce_to_print, _percentile)
            if np.abs(vmax) > np.abs(vmin):
                vmax = np.abs(vmax)
                vmin = -np.abs(vmax)
            else:
                vmax = np.abs(vmin)
                vmin = -np.abs(vmin)
            fn_imshow(
                _ax,
                ce_to_print,
                vmin=vmin,
                vmax=vmax,
                title=_title,
                xlabel=_xlabel,
                ylabel=_ylabel,
            )

    if save_dir:
        plt.savefig(save_dir, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.close()
    return fig


@torch.no_grad()
def get_visualization_result(config, loader, model, device):
    for idx, data in tqdm(enumerate(loader), total=len(loader)):
        if idx > 5:
            break

        # get inference results
        real_datas = data["dicom"].to(device)
        output = model[ModelName.CLASSIFIER](real_datas)
        output["orig"] = real_datas

        # plot
        save_path = f"./fig/image_{idx}"
        os.makedirs(save_path, exist_ok=True)
        save_dir = save_path + f"/result.jpg"
        a = _plot_multiple_center_axis(output, save_dir=save_dir)


def get_cpu_device():
    dict_device = {"cpu": torch.device(f"cpu")}
    return dict_device


@hydra.main(version_base="1.2", config_path="configs", config_name="config_test")
def main(config: omegaconf.DictConfig) -> None:
    # DataLoader
    run_modes = [RunMode(m) for m in config.run_modes] if "run_modes" in config else [x for x in RunMode]
    config.loader.batch_size = 1
    loaders = {
        mode: hydra.utils.instantiate(config.loader, dataset={"mode": mode}, drop_last=(mode == RunMode.TRAIN))
        for mode in run_modes
    }
    loader = loaders[RunMode.TEST]

    # load checkpoint
    model = dict()
    weight_path = os.path.join(config.base_path, config.model_path)
    model[ModelName.CLASSIFIER] = load_model(weight_path, model_config=config.model)

    # inference
    get_visualization_result(config, loader, model, device=get_cpu_device()["cpu"])
    logger.info("Done")


if __name__ == "__main__":
    main()
