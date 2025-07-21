import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pymongo
import torch
from tqdm import tqdm

from data_lake.constants import DB_ADDRESS, TARGET_COLLECTION, TARGET_DB, DBKey
from shared_lib.tools.image_parser import extract_patch
from shared_lib.utils.utils_vis import save_plot
from trainer.common.augmentation.hu_value import DicomWindowing
from trainer.downstream.datasets.luna25 import extract_patch_dicom_space

_CLIENT = pymongo.MongoClient(DB_ADDRESS)

def load_and_merge_csv_luna25(best_csv, compare_csv):
    best = pd.read_csv(best_csv)
    compare = pd.read_csv(compare_csv)
    best = best[best['mode'] == 'test'].reset_index(drop=True)
    compare = compare[compare['mode'] == 'test'].reset_index(drop=True)
    merged = best[['annot_ids', 'annotation', 'prob_ensemble']].merge(
        compare[['annot_ids', 'prob_ensemble']], on='annot_ids', suffixes=('_best', '_compare')
    )
    merged['prob_diff'] = np.abs(merged['prob_ensemble_best'] - merged['prob_ensemble_compare'])
    return merged

def load_and_merge_csv_lidc(best_csv, compare_csv):
    best = pd.read_csv(best_csv)
    compare = pd.read_csv(compare_csv)
    best = best[best['mode'] == 'test'].reset_index(drop=True)
    compare = compare[compare['mode'] == 'test'].reset_index(drop=True)
    best['row_idx'] = best.index
    compare['row_idx'] = compare.index
    merged = best[['row_idx', 'annotation', 'prob_ensemble']].copy()
    merged['prob_ensemble_compare'] = compare['prob_ensemble']
    merged['prob_diff'] = np.abs(merged['prob_ensemble'] - merged['prob_ensemble_compare'])
    return merged

def get_topk_diff_samples(merged, k=10, mode='luna25'):
    if 'annot_ids' in merged.columns:
        if mode == 'luna25':
            mask = merged['annot_ids'].notnull() & (merged['annot_ids'] != '')
        else:
            mask = merged['annot_ids'].isnull() | (merged['annot_ids'] == '')
        filtered = merged[mask].copy()
    else:
        filtered = merged.copy()
    return filtered.sort_values('prob_diff', ascending=False).head(k)

def fetch_db_info(annotation_ids):
    query = {"annotation_id": {"$in": annotation_ids}}
    projection = {}
    nodule_candidates = list(_CLIENT[TARGET_DB][TARGET_COLLECTION].find(query, projection))
    db_df = pd.DataFrame(nodule_candidates)
    return db_df

def visualize_samples_luna25(df, output_dir, processor, gate_levels):
    for _, row in tqdm(df.iterrows(), total=len(df)):
        annotation = row['annotation']
        annot_ids = row['annot_ids']
        row_idx = row['row_idx']
        save_dir = Path(os.path.join(output_dir, f"{annotation}/{annot_ids}.png"))
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
            "row_idx": row_idx,
            "annotation": annotation,
            "prob_ensemble_best": row['prob_ensemble_best'],
            "prob_ensemble_compare": row['prob_ensemble_compare'],
            "prob_diff": row['prob_diff'],
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

def visualize_samples_lidc(df, output_dir):
    for _, row in tqdm(df.iterrows(), total=len(df)):
        annotation = row['annotation']
        row_idx = row['row_idx']
        save_dir = Path(os.path.join(output_dir, f"{annotation}/{row_idx}.png"))
        os.makedirs(save_dir.parents[0], exist_ok=True)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.axis('off')
        text = f"best: {row['prob_ensemble_best']:.3f}\ncompare: {row['prob_ensemble_compare']:.3f}\ndiff: {row['prob_diff']:.3f}"
        ax.text(0.5, 0.5, text, fontsize=14, ha='center', va='center')
        plt.tight_layout()
        fig.savefig(str(save_dir), dpi=60)
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--best_csv', type=str, default='/team/team_blu3/lung/project/luna25/analysis/inference_results_combined/result_5_0_8rc2.csv')
    parser.add_argument('--compare_csv', type=str, default='/team/team_blu3/lung/project/luna25/analysis/inference_results_combined/result_5_0_9rc8.csv')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['luna25', 'lidc'], required=True)
    args = parser.parse_args()
    if args.mode == 'luna25':
        merged = load_and_merge_csv_luna25(args.best_csv, args.compare_csv)
        topk = get_topk_diff_samples(merged, k=10, mode='luna25')
        best = pd.read_csv(args.best_csv)
        best = best[best['mode'] == 'test'].reset_index(drop=True)
        db_info_cols = [DBKey.H5_PATH_NFS, DBKey.D_COORD_ZYX, DBKey.ORIGIN, DBKey.TRANSFORM, DBKey.SPACING]
        for col in db_info_cols:
            topk[col] = topk['annot_ids'].map(best.set_index('annot_ids')[col])
        processor = None
        gate_levels = [0, 1, 2]
        visualize_samples_luna25(topk, args.output_dir, processor, gate_levels)
    else:
        merged = load_and_merge_csv_lidc(args.best_csv, args.compare_csv)
        topk = get_topk_diff_samples(merged, k=10, mode='lidc')
        visualize_samples_lidc(topk, args.output_dir)

if __name__ == '__main__':
    main()
