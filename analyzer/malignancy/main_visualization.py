import logging
import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pymongo
from omegaconf import DictConfig
from tqdm import tqdm

from data_lake.constants import DB_ADDRESS, TARGET_COLLECTION, TARGET_DB, DBKey
from shared_lib.tools.image_parser import extract_patch
from shared_lib.utils.utils import print_config
from shared_lib.utils.utils_vis import save_plot
from trainer.downstream.datasets.luna25 import extract_patch_dicom_space

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_CLIENT = pymongo.MongoClient(DB_ADDRESS)


def _fn_save_fig(df, output_dir):
    # loop for computing individual nodule, which is corresponding to the row.
    for index, row in tqdm(df.iterrows(), total=len(df)):
        annotation = row.at["annotation"]
        annotation_id = row.at["annotation_id"]
        # make directory to save visualization results
        save_dir = Path(
            os.path.join(
                output_dir,
                f"{annotation}/{annotation_id}.png",
            )
        )
        os.makedirs(save_dir.parents[0], exist_ok=True)

        h5_path = row.at[DBKey.H5_PATH_NFS]
        d_coord_zyx = row.at[DBKey.D_COORD_ZYX]
        origin = row.at[DBKey.ORIGIN]
        transform = row.at[DBKey.TRANSFORM]
        spacing = row.at[DBKey.SPACING]

        # load input images and save the 3d plot images
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
            voxel_spacing=(
                size_mm / size_px_z,
                size_mm / size_px_xy,
                size_mm / size_px_xy,
            ),
            rotations=None,
            translations=None,
            coord_space_world=False,
            mode="3D",
            order=1,
        )
        patch = np.squeeze(patch)

        # save visualization result
        figure_title = ""
        attr = {
            "annotation_id": row.at["annotation_id"],
            "annotation": row.at["annotation"],
            "prob_ensemble": row.at["prob_ensemble"],
            "prob_variance": row.at["prob_variance"],
        }

        save_plot(
            input_image=patch,
            mask_image=None,
            nodule_zyx=None,
            figure_title=figure_title,
            meta=attr,
            use_norm=True,
            save_dir=str(save_dir),
            dpi=60,
        )


def _get_topk_interests(df, sort_criterion="prob_ensemble", top_k=20):
    # Extract model probability columns (e.g., prob_model_0, prob_model_1, ...)
    model_cols = [col for col in df.columns if col.startswith("prob_model_")]

    # Correct entropy calculation for binary classification
    binary_entropies = []
    for _, row in df[model_cols].iterrows():
        entropies = -(row.values * np.log2(row.values + 1e-10) + (1 - row.values) * np.log2(1 - row.values + 1e-10))
        binary_entropies.append(np.mean(entropies))

    df["prob_entropy"] = binary_entropies
    df["prob_variance"] = df[model_cols].var(axis=1)

    # Sort by entropy, variance, and CV for each annotation
    if sort_criterion == "prob_variance":
        sorted_positive = df[df["annotation"] == 1].sort_values(by=sort_criterion, ascending=False)
        sorted_negative = df[df["annotation"] == 0].sort_values(by=sort_criterion, ascending=False)
    else:
        # currently if sort_criterion is not "prob_variance", it considered as probability key.
        # And, target samples to visualize are set to edge cases.
        sorted_positive = df[df["annotation"] == 1].sort_values(by=sort_criterion, ascending=True)
        sorted_negative = df[df["annotation"] == 0].sort_values(by=sort_criterion, ascending=False)

    # Select top N samples
    top_positive = sorted_positive.head(top_k).copy()
    top_positive["label"] = "positive"

    top_negative = sorted_negative.head(top_k).copy()
    top_negative["label"] = "negative"

    # Merge positive and negative samples
    merged_df = pd.concat([top_positive, top_negative], ignore_index=True)

    return merged_df


@hydra.main(version_base="1.2", config_path="configs", config_name="config_visualization")
def main(config: DictConfig):
    print_config(config, resolve=True)
    result_csv_path = config.result_csv_path
    sort_criterion = config.sort_criterion

    # read result csv
    df = pd.read_csv(result_csv_path)
    df = df[df["mode"] == "test"]

    # get nodules of interest
    df_interest = _get_topk_interests(df, sort_criterion=sort_criterion)

    # Prepare query for MongoDB
    annotation_ids = df_interest["annot_ids"].tolist()
    query = {"annotation_id": {"$in": annotation_ids}}
    projection = {}  # Include all fields

    # Fetch documents from MongoDB
    nodule_candidates = list(_CLIENT[TARGET_DB][TARGET_COLLECTION].find(query, projection))

    # Convert MongoDB documents to DataFrame
    db_df = pd.DataFrame(nodule_candidates)

    # Merge the interest dataframe with MongoDB results on 'annot_ids'
    merged_df = df_interest.merge(db_df, left_on="annot_ids", right_on="annotation_id", how="left")

    # Visualization
    output_dir = f"fig_volume/{sort_criterion}"
    _fn_save_fig(merged_df, output_dir=output_dir)


if __name__ == "__main__":
    main()
