import argparse
import ast
import logging
import os
from pathlib import Path

import pandas as pd
import pymongo
from h5py import File
from joblib import Parallel, delayed

from tools.preprocess import patch_extract
from tools.visualizer import save_plot

logger = logging.getLogger(__name__)

# datalake
_CLIENT = pymongo.MongoClient(DB_ADDRESS)
_COLLECTION_NAME = ""
_TARGET_FIELD = [
    "series_instance_uid",
]
_PROJECTION = {t: 1 for t in _TARGET_FIELD}
_PROJECTION["_id"] = 0  # do not show '_id' of documents

# constants
_PATCH_SIZE = (48, 72, 72)
_HU_RANGE = (-1000, 600)

# output directory
_OUTPUT_DIR = f"./fig_volume"


def fn_save_fig(df_chunk):
    # loop for computing individual nodule, which is corresponding to the row.
    for index, row in df_chunk.iterrows():
        series_instance_uid = row.at["series_instance_uid"]
        dataset = row.at["dataset"]
        is_gt = row.at["is_gt"]
        label = row.at["confidence"]
        r_coord = (
            row["r_coord"] if isinstance(row["r_coord"], list) else ast.literal_eval(row["r_coord"])
        )
        file_path = row.at["h5_path"]

        # make directory to save visualization results
        save_dir = Path(
            os.path.join(
                _OUTPUT_DIR,
                f"{dataset}_{is_gt}/{label}/{series_instance_uid}_{r_coord[0]}_{r_coord[1]}_{r_coord[2]}.png",
            )
        )
        os.makedirs(save_dir.parents[0], exist_ok=True)

        # load input images and save the 3d plot images
        with File(file_path, mode="r") as hf:
            # load input
            img = patch_extract(
                hf["data"], center_coord=r_coord, voxel_width=_PATCH_SIZE, pad_value=0
            )

            # save visualization result
            figure_title = ""
            attr = {"series_instance_uid": series_instance_uid, "label": label}
            save_plot(
                img,
                mask_image=None,
                nodule_zyx=None,
                patch_size=_PATCH_SIZE,
                figure_title=figure_title,
                meta=attr,
                use_norm=True,
                save_dir=save_dir,
                dpi=60,
            )


def parallel_process(df, num_jobs=1):
    chunk_size = len(df) // num_jobs
    chunks = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]

    Parallel(n_jobs=num_jobs)(delayed(fn_save_fig)(chunk) for chunk in chunks)


def main():
    parser = argparse.ArgumentParser(description="volume visualization tool")
    parser.add_argument("--num_jobs", default=4, help="The number of jobs in parallel processing.", type=int)
    parser.add_argument("--dataset", type=str, default="luna25", choices=["luna25", "etc"])
    args = parser.parse_args()

    query = {"dataset": {"$in": [str(args.dataset)]}}
    nodule_candidates = [x for x in _CLIENT["lct"][_COLLECTION_NAME].find(query, _PROJECTION)]
    df = pd.DataFrame(nodule_candidates)
    parallel_process(df, num_jobs=args.num_jobs)


if __name__ == "__main__":
    main()
