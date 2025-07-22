import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymongo
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from data_lake.constants import DB_ADDRESS, TARGET_COLLECTION, TARGET_DB, DBKey
from shared_lib.tools.image_parser import extract_patch
from shared_lib.utils.utils_vis import save_plot
from trainer.common.augmentation.hu_value import DicomWindowing
from trainer.downstream.datasets.luna25 import extract_patch_dicom_space

_CLIENT = pymongo.MongoClient(DB_ADDRESS)


def get_merged_csv(best_csv, compare_csv, mode="luna25"):
    # prepare source materials
    best = pd.read_csv(best_csv)
    best = best[best["mode"] == "test"].reset_index(drop=True)

    compare = pd.read_csv(compare_csv)
    compare = compare[compare["mode"] == "test"].reset_index(drop=True)

    # get merged
    if mode == "luna25":
        merged = best[["annot_ids", "annotation", "prob_ensemble"]].merge(
            compare[["annot_ids", "prob_ensemble"]], on="annot_ids", suffixes=("_best", "_compare")
        )

    elif mode == "lidc":
        best["row_idx"] = best.index
        compare["row_idx"] = compare.index
        merged = best[["row_idx", "annotation", "prob_ensemble"]].copy()
        merged["prob_ensemble_compare"] = compare["prob_ensemble"]
        merged["prob_ensemble_best"] = best["prob_ensemble"]

    else:
        assert False

    merged["prob_diff"] = np.abs(merged["prob_ensemble_best"] - merged["prob_ensemble_compare"])

    return merged


def get_topk_diff_samples(merged, k=10, mode="luna25"):
    if "annot_ids" in merged.columns:
        if mode == "luna25":
            # luna25: annot_ids가 "숫자_숫자_숫자" 형태
            mask = merged["annot_ids"].astype(str).str.contains(r"^\d+_\d+_\d+$", na=False)
        else:
            # lidc: luna25 패턴이 아닌 모든 것
            mask = ~merged["annot_ids"].astype(str).str.contains(r"^\d+_\d+_\d+$", na=False)
        filtered = merged[mask].copy()
    else:
        filtered = merged.copy()

    return filtered.sort_values("prob_diff", ascending=False).head(k)


def fetch_db_info(annotation_ids):
    query = {"annotation_id": {"$in": annotation_ids}}
    projection = {}
    nodule_candidates = list(_CLIENT[TARGET_DB][TARGET_COLLECTION].find(query, projection))
    db_df = pd.DataFrame(nodule_candidates)

    return db_df


def visualize_samples_luna25(df):
    for _, row in tqdm(df.iterrows(), total=len(df)):
        annotation = row["annotation"]
        annot_ids = row["annot_ids"]
        save_dir = Path(os.path.join(f"luna25/{annotation}/{annot_ids}.png"))
        os.makedirs(save_dir.parents[0], exist_ok=True)
        h5_path = row[DBKey.H5_PATH_NFS]
        d_coord_zyx = row[DBKey.D_COORD_ZYX]
        origin = row[DBKey.ORIGIN]
        transform = row[DBKey.TRANSFORM]
        spacing = row[DBKey.SPACING]
        size_xy = 128
        size_z = 64
        size_mm = 50
        size_px_xy = 72
        size_px_z = 48
        patch_size = [size_z, size_xy, size_xy]
        output_shape = (size_px_z, size_px_xy, size_px_xy)
        img = extract_patch_dicom_space(h5_path, d_coord_zyx, xy_size=size_xy, z_size=size_z)
        patch = extract_patch(
            CTData=img,
            coord=tuple(np.array(patch_size) // 2),
            srcVoxelOrigin=np.array(origin),
            srcWorldMatrix=np.array(transform),
            srcVoxelSpacing=np.array(spacing),
            output_shape=output_shape,
            voxel_spacing=(size_mm / size_px_z, size_mm / size_px_xy, size_mm / size_px_xy),
            rotations=None,
            translations=None,
            coord_space_world=False,
            mode="3D",
            order=1,
        )
        patch = patch.astype(np.float32)
        HU_WINDOW = (-300, 1400)
        hu_range = (HU_WINDOW[0] - HU_WINDOW[1] // 2, HU_WINDOW[0] + HU_WINDOW[1] // 2)
        patch = DicomWindowing(hu_range=hu_range)(patch)
        patch = np.concatenate(patch, axis=0)
        patch = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)
        figure_title = f"best: {row['prob_ensemble_best']:.3f}, compare: {row['prob_ensemble_compare']:.3f}, diff: {row['prob_diff']:.3f}"
        attr = {
            "annot_ids": annot_ids,
            "annotation": annotation,
            "prob_ensemble_best": row["prob_ensemble_best"],
            "prob_ensemble_compare": row["prob_ensemble_compare"],
            "prob_diff": row["prob_diff"],
        }
        save_plot(
            input_image=patch.squeeze().detach().cpu().numpy(),
            mask_image=None,
            nodule_zyx=None,
            figure_title=figure_title,
            meta=attr,
            use_norm=False,
            vmin_mask=0.0,
            vmax_mask=1.0,
            save_dir=str(save_dir),
            dpi=60,
        )


def visualize_samples_lidc(df):
    for _, row in tqdm(df.iterrows(), total=len(df)):
        annotation = row["annotation"]
        row_idx = row["row_idx"]
        save_dir = Path(os.path.join(f"lidc/{annotation}/{row_idx}.png"))
        os.makedirs(save_dir.parents[0], exist_ok=True)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.axis("off")
        text = f"best: {row['prob_ensemble_best']:.3f}\ncompare: {row['prob_ensemble_compare']:.3f}\ndiff: {row['prob_diff']:.3f}"
        ax.text(0.5, 0.5, text, fontsize=14, ha="center", va="center")
        plt.tight_layout()
        fig.savefig(str(save_dir), dpi=60)
        plt.close(fig)


@hydra.main(version_base="1.2", config_path="configs", config_name="config_visualization_diff")
def main(config: DictConfig):
    csv_path = Path(config.csv_path)
    best_csv = csv_path / config.best_csv
    compare_csv = csv_path / config.compare_csv
    topk_num_sample = config.topk_num_sample

    for mode in ["lidc", "luna25"]:
        # prepare samples to vis
        merged = get_merged_csv(best_csv, compare_csv, mode=mode)
        topk = get_topk_diff_samples(merged, k=topk_num_sample, mode=mode)

        # visualization
        if mode == "luna25":
            # fetch info. from MongoDB and merge with keys {annot_ids, annotation_id}
            annotation_ids = topk["annot_ids"].tolist()
            db_df = fetch_db_info(annotation_ids)
            merged_with_db = topk.merge(db_df, left_on="annot_ids", right_on="annotation_id", how="left")

            # DB 정보 없는 row는 제외
            num_bef = len(merged_with_db)
            merged_with_db = merged_with_db.dropna(
                subset=[DBKey.H5_PATH_NFS, DBKey.D_COORD_ZYX, DBKey.ORIGIN, DBKey.TRANSFORM, DBKey.SPACING]
            )
            num_aft = len(merged_with_db)
            assert num_bef == num_aft

            visualize_samples_luna25(merged_with_db)
        else:
            visualize_samples_lidc(topk)


if __name__ == "__main__":
    main()
