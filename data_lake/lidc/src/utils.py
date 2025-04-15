import os
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from data_lake.constants import DBKey
from data_lake.dataset_handler import DatasetHandler


def fn_save_histograms(
    df, target_fields, prefix="", suffix="", fig_size=(50, 4), num_fold=None, dpi=100, save_dir: Path = None
):
    def get_axes(axes, row_idx, col_idx):
        """
        axes가 2차원 배열인지 확인하고, 1차원 배열일 경우 해당 인덱스를 반환하는 함수.
        """
        if len(axes.shape) == 1:  # axes가 1차원 배열인 경우
            return axes[col_idx]
        else:  # axes가 2차원 배열인 경우
            return axes[row_idx, col_idx]

    nrows = num_fold if num_fold else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=len(target_fields), figsize=fig_size)
    for row_idx in range(nrows):
        for col_idx, field in enumerate(target_fields):
            column_name = f"{prefix}{field}{suffix}"
            if num_fold is not None:
                _df = df[df[DBKey.FOLD] == row_idx].sort_values(by=column_name)
            else:
                _df = df.sort_values(by=column_name)
            ax = get_axes(axes, row_idx, col_idx)  # 0행 1열의 axes를 가져옵니다.
            g = sns.histplot(x=column_name, data=_df, kde=True, bins=50, ax=ax)
            g.set_xticklabels(g.get_xticklabels(), rotation=30, horizontalalignment="right")
            ax.tick_params(axis="both", labelsize="xx-large")
            ax.set_ylabel("")
            ax.set_xlim(df[column_name].min(), df[column_name].max())
            if row_idx == nrows - 1:
                ax.set_xlabel(field, size="xx-large")
            else:
                ax.set_xlabel("")

    if save_dir:
        os.makedirs(save_dir.parents[0], exist_ok=True)
        plt.savefig(save_dir, bbox_inches="tight", dpi=dpi)
        plt.close()
    else:
        plt.close()
        return fig


def fn_save_corr_heatmap(df, target_fields: list, dpi=100, save_dir: Path = None):
    fig = plt.figure(figsize=(10, 8))
    corr_matrix = df[target_fields].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, vmin=-1, vmax=1)

    if save_dir:
        os.makedirs(save_dir.parents[0], exist_ok=True)
        plt.savefig(save_dir, bbox_inches="tight", dpi=dpi)
        plt.close()
    else:
        plt.close()
        return fig


def save_hist_for_target_field(
    collection, target_fields: list, prefix="", suffix="", fig_size: tuple = (), num_fold=None, save_dir=None
):
    dataset_handler = DatasetHandler()
    image_infos = dataset_handler.fetch_documents(collection=collection, query={}, projection={})
    df = pd.DataFrame(image_infos)

    # visualization of histogram for image-level infos.
    fn_save_histograms(
        df,
        target_fields=target_fields,
        prefix=prefix,
        suffix=suffix,
        fig_size=fig_size,
        num_fold=num_fold,
        save_dir=save_dir,
    )
