import argparse
import os
from dataclasses import dataclass, fields

import h5py
import pandas as pd
import pymongo
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from data_lake.constants import (
    DB_ADDRESS,
    DEFAULT_RESAMPLED_SPACING,
    TARGET_COLLECTION,
    TARGET_DB,
    DBKey,
    H5DataKey,
    LUNA25Dir,
)
from data_lake.utils.resample_image import map_coord_to_resampled, world_to_voxel


@dataclass(frozen=True)
class ColumnKey:
    PatientID: str = "PatientID"
    SeriesInstanceUID: str = "SeriesInstanceUID"
    StudyDate: str = "StudyDate"
    AnnotationID: str = "AnnotationID"
    CoordX: str = "CoordX"
    CoordY: str = "CoordY"
    CoordZ: str = "CoordZ"
    LesionID: str = "LesionID"
    AgeAtStudy: str = "Age_at_StudyDate"
    Gender: str = "Gender"
    Label: str = "label"


@dataclass(frozen=True)
class ColumnKeyAppend:
    Fold: str = "fold"
    Coord: str = "coord"
    Origin: str = "origin"
    Spacing: str = "spacing"
    Transform: str = "transform"
    ImageShape: str = "image_shape"
    H5PathNFS: str = "h5_path_nfs"
    H5PathLocal: str = "h5_path"


def append_image_metadata(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(ColumnKey.SeriesInstanceUID)

    for series_uid, group in tqdm(grouped):
        h5_filepath_nfs = os.path.join(LUNA25Dir.output_nfs_dir, f"{series_uid}.h5")
        h5_filepath_local = os.path.join(LUNA25Dir.output_local_dir, f"{series_uid}.h5")
        with h5py.File(h5_filepath_nfs, "r") as hf:
            image_shape = hf[H5DataKey.image].shape
            origin = hf.attrs[H5DataKey.origin]
            spacing = hf.attrs[H5DataKey.spacing]
            transform = hf.attrs[H5DataKey.transform]

        for idx in group.index:
            df.at[idx, ColumnKeyAppend.H5PathNFS] = h5_filepath_nfs
            df.at[idx, ColumnKeyAppend.H5PathLocal] = h5_filepath_local
            df.at[idx, ColumnKeyAppend.ImageShape] = image_shape
            df.at[idx, ColumnKeyAppend.Origin] = origin
            df.at[idx, ColumnKeyAppend.Spacing] = spacing
            df.at[idx, ColumnKeyAppend.Transform] = transform

    return df


def split_fold(df: pd.DataFrame) -> pd.DataFrame:
    patient_df = (
        df.groupby(ColumnKey.PatientID)
        .agg(
            StudyDate=(ColumnKey.StudyDate, lambda x: x.mode()[0]),
            malignancy=(ColumnKey.Label, lambda x: (x == 0).sum()),
            benign=(ColumnKey.Label, lambda x: (x == 1).sum()),
            Age_at_StudyDate=(ColumnKey.AgeAtStudy, lambda x: x.mode()[0]),
            Gender=(ColumnKey.Gender, lambda x: x.mode()[0]),  # 최빈값(가장 많이 등장한 성별)
            z_spacing_min=(ColumnKeyAppend.Spacing, lambda x: min(v[0] for v in x)),
            z_spacing_max=(ColumnKeyAppend.Spacing, lambda x: max(v[0] for v in x)),
        )
        .reset_index()
    )

    def age_bin(age, threshold=60):
        return f"{threshold}_over" if age >= threshold else f"{threshold}_under"

    def spacing_bin(spacing, threshold=2.3):
        return f"{threshold}_over" if spacing >= threshold else f"{threshold}_under"

    patient_df["strat"] = (
        patient_df["malignancy"].astype(str)
        + "_"
        + patient_df["benign"].astype(str)
        + "_"
        + patient_df["Age_at_StudyDate"].apply(lambda x: age_bin(x, threshold=50))
        + "_"
        + patient_df["Age_at_StudyDate"].apply(lambda x: age_bin(x, threshold=60))
        + "_"
        + patient_df["Age_at_StudyDate"].apply(lambda x: age_bin(x, threshold=70))
        + "_"
        + patient_df["Age_at_StudyDate"].apply(lambda x: age_bin(x, threshold=80))
        + "_"
        + patient_df["Gender"].astype(str)
        + "_"
        + patient_df["z_spacing_max"].apply(spacing_bin)
    )

    skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
    for fold, (_, val_idx) in enumerate(skf.split(patient_df, patient_df["strat"])):
        val_patients = patient_df.iloc[val_idx][ColumnKey.PatientID]
        df.loc[df[ColumnKey.PatientID].isin(val_patients), ColumnKeyAppend.Fold] = fold

    return df


def insert_to_db(df: pd.DataFrame):
    client = pymongo.MongoClient(DB_ADDRESS)
    collection = client[TARGET_DB][TARGET_COLLECTION]

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inserting to DB"):
        try:
            w_coord = row[ColumnKeyAppend.Coord]
            origin = row[ColumnKeyAppend.Origin]
            spacing = row[ColumnKeyAppend.Spacing]
            transform = row[ColumnKeyAppend.Transform]
            image_shape = row[ColumnKeyAppend.ImageShape]
            h5_path_nfs = row[ColumnKeyAppend.H5PathNFS]
            h5_path_local = row[ColumnKeyAppend.H5PathLocal]

            d_coord = world_to_voxel(w_coord, origin, spacing)
            r_coord = map_coord_to_resampled(
                d_coord=d_coord, orig_shape=image_shape, spacing=spacing, resampled_spacing=DEFAULT_RESAMPLED_SPACING
            )

            document = {
                DBKey.PATIENT_ID: row[ColumnKey.PatientID],
                DBKey.SERIES_INSTANCE_UID: row[ColumnKey.SeriesInstanceUID],
                DBKey.ANNOTATION_ID: row[ColumnKey.AnnotationID],
                DBKey.STUDY_DATE: row[ColumnKey.StudyDate],
                DBKey.H5_PATH_NFS: h5_path_nfs,
                DBKey.H5_PATH_LOCAL: h5_path_local,
                DBKey.FOLD: row[ColumnKeyAppend.Fold],
                DBKey.LABEL: row[ColumnKey.Label],
                DBKey.AGE_AT_STUDY: row[ColumnKey.AgeAtStudy],
                DBKey.GENDER: row[ColumnKey.Gender],
                DBKey.ORIGIN: origin.tolist(),
                DBKey.TRANSFORM: transform.tolist(),
                DBKey.SPACING: spacing.tolist(),
                DBKey.RESAMPLED_SPACING: DEFAULT_RESAMPLED_SPACING,
                DBKey.W_COORD_ZYX: w_coord,
                DBKey.D_COORD_ZYX: d_coord,
                DBKey.R_COORD_ZYX: r_coord,
            }

            query = {
                "patient_id": row[ColumnKey.PatientID],
                "series_instance_uid": row[ColumnKey.SeriesInstanceUID],
                "studydate": row[ColumnKey.StudyDate],
                "w_coord_zyx": w_coord,
            }

            existing = list(collection.find(query))
            if len(existing) == 1:
                collection.update_one({"_id": existing[0]["_id"]}, {"$set": document})
            else:
                collection.insert_one(document)

        except Exception as e:
            print(f"[ERROR] Failed to insert row {row[ColumnKey.AnnotationID]}: {e}")


def insert_to_db_parallel(df: pd.DataFrame, num_jobs: int = 1):
    chunk_size = len(df) // num_jobs or 1
    chunks = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]
    Parallel(n_jobs=num_jobs)(delayed(insert_to_db)(chunk) for chunk in chunks)


def main():
    parser = argparse.ArgumentParser(description="Insert LUNA25 metadata into MongoDB.")
    parser.add_argument("--clean_documents", action="store_true", help="Clean existing documents before inserting.")
    parser.add_argument("--chunk_size", type=int, default=16, help="Number of chunks for parallel processing.")
    args = parser.parse_args()

    df = pd.read_csv(LUNA25Dir.meta_dir)
    assert len(df) == 6163

    # add columns
    df[ColumnKeyAppend.Coord] = df[[ColumnKey.CoordZ, ColumnKey.CoordY, ColumnKey.CoordX]].values.tolist()
    for col in fields(ColumnKeyAppend):
        if col.default not in df.keys():
            df[col.default] = None

    # 1. update cols
    df = append_image_metadata(df)

    # 2. split fold
    df = split_fold(df)

    # 3. insert to DB
    if args.clean_documents:
        client = pymongo.MongoClient(DB_ADDRESS)
        client[TARGET_DB][TARGET_COLLECTION].delete_many({})
        print("Deleted existing documents.")
    insert_to_db_parallel(df, num_jobs=args.chunk_size)
    print("Insertion completed successfully.")


if __name__ == "__main__":
    main()
