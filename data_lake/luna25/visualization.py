import argparse
import ast
import logging
import os
from pathlib import Path

import pandas as pd
import pymongo
from h5py import File
from joblib import Parallel, delayed
from tqdm import tqdm

from data_lake.constants import DB_ADDRESS, TARGET_COLLECTION, TARGET_DB, DBKey, H5DataKey
from shared_lib.tools.preprocess import patch_extract
from shared_lib.utils.utils_vis import save_plot

logger = logging.getLogger(__name__)

# datalake
_CLIENT = pymongo.MongoClient(DB_ADDRESS)
_TARGET_FIELD = []
_PROJECTION = dict()


def _fn_save_fig(df, patch_size=(48, 72, 72), output_dir=f"./fig_volume", use_r_coord=True):
    # loop for computing individual nodule, which is corresponding to the row.
    for index, row in tqdm(df.iterrows(), total=len(df)):
        series_instance_uid = row.at[DBKey.SERIES_INSTANCE_UID]
        label = row.at[DBKey.LABEL]
        if use_r_coord:
            coord = (
                row[DBKey.R_COORD_ZYX]
                if isinstance(row[DBKey.R_COORD_ZYX], list)
                else ast.literal_eval(row[DBKey.R_COORD_ZYX])
            )
            coord = [round(_coord) for _coord in coord]
        else:
            coord = (
                row[DBKey.D_COORD_ZYX]
                if isinstance(row[DBKey.D_COORD_ZYX], list)
                else ast.literal_eval(row[DBKey.D_COORD_ZYX])
            )
            coord = [round(_coord) for _coord in coord]

        file_path = row.at[DBKey.H5_PATH_NFS]

        # make directory to save visualization results
        save_dir = Path(
            os.path.join(
                output_dir,
                f"{label}/{series_instance_uid}_{coord[0]}_{coord[1]}_{coord[2]}.png",
            )
        )
        os.makedirs(save_dir.parents[0], exist_ok=True)

        # load input images and save the 3d plot images
        with File(file_path, mode="r") as hf:
            # load input
            img = patch_extract(
                hf[H5DataKey.resampled_image] if use_r_coord else hf[H5DataKey.image],
                center_coord=coord,
                voxel_width=patch_size,
                pad_value=0,
            )

            # save visualization result
            figure_title = ""
            attr = {"seriesUID": series_instance_uid, "label": label}
            save_plot(
                img,
                mask_image=None,
                nodule_zyx=None,
                patch_size=patch_size,
                figure_title=figure_title,
                meta=attr,
                use_norm=True,
                save_dir=str(save_dir),
                dpi=60,
            )


def parallel_process(df, num_jobs=1):
    chunk_size = len(df) // num_jobs
    chunks = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]

    Parallel(n_jobs=num_jobs)(delayed(_fn_save_fig)(chunk) for chunk in chunks)


def main():
    parser = argparse.ArgumentParser(description="volume visualization tool")
    parser.add_argument("--num_jobs", default=16, help="The number of jobs in parallel processing.", type=int)
    parser.add_argument("--dataset", type=str, default="luna25", choices=["luna25", "etc"])
    args = parser.parse_args()

    query = {}
    nodule_candidates = [x for x in _CLIENT[TARGET_DB][TARGET_COLLECTION].find(query, _PROJECTION)]
    df = pd.DataFrame(nodule_candidates)
    parallel_process(df, num_jobs=args.num_jobs)


if __name__ == "__main__":
    main()
