import argparse

import h5py
import numpy as np
import pandas as pd
import pymongo
from joblib import Parallel, delayed
from tqdm import tqdm

_VUNO_LUNG_DB = "mongodb://172.31.10.111:27017"
_TARGET_DB = "lct"
_TARGET_COLLECTION = "LUNA25-Malignancy"

def world_2_voxel(world_coordinates, origin, spacing):
    stretched_voxel_coordinates = np.absolute(np.array(world_coordinates) - np.array(origin))
    voxel_coordinates = stretched_voxel_coordinates / np.array(spacing)
    return voxel_coordinates.tolist()

def calcu_coord(D_COORD, SPACING, _RESAMPLED_SPACING):
    r_coord_zyx = [D_COORD[i] * (SPACING[i] / _RESAMPLED_SPACING[i]) for i in range(3)]
    return r_coord_zyx


def insert_to_DB(df_chunk):
    _CLIENT = pymongo.MongoClient(_VUNO_LUNG_DB)
    for index, row in tqdm(df_chunk.iterrows(), total=len(df_chunk)):
        patient_id = row.at["PatientID"]
        series_instance_uid = row.at["SeriesInstanceUID"]
        studydate = row.at["StudyDate"]
        original_nodule_coord = [row.at["CoordZ"], row.at["CoordY"], row.at["CoordX"]]
        # lesion_id = row.at["LesionID"]
        annotation_id = row.at["AnnotationID"]
        label = row.at["label"]
        age_at_study = row.at["Age_at_StudyDate"]
        gender = row.at["Gender"]
        origin = [row.at["z_origin"], row.at["y_origin"], row.at["x_origin"]]
        transform = [row.at["z_transform"], row.at["y_transform"], row.at["x_transform"]]
        origina_spacing = [row.at["z_spacing"], row.at["y_spacing"], row.at["x_spacing"]]
        dicom_path = f"/team/team_blu3/lung/data/2_public/LUNA25_Original/luna25_images/{series_instance_uid}.mha"
        resampled_h5_path = f"/team/team_blu3/lung/data/2_public/LUNA25_resampled/{series_instance_uid}.h5"
        fold = row.at["fold"]

        # get r_coord
        with h5py.File(resampled_h5_path, "r") as hf:
            dicom_pixels = hf["volume_image"]
            w_coord_zyx = original_nodule_coord
            resampled_pixels = hf["resampled_image"]
            resampled_spacing = hf.attrs["resampled_spacing"]
            original_spacing = hf.attrs["original_spacing"]
            resampled_dicom_shape = resampled_pixels.shape
            
        d_coord_zyx = world_2_voxel(w_coord_zyx, origin, original_spacing)
        r_coord_zyx = calcu_coord(d_coord_zyx, original_spacing, resampled_spacing)

        # insert a document to the collection
        dict_info = {
            "patient_id": patient_id,
            "series_instance_uid": series_instance_uid,
            "annotation_id": annotation_id,
            "studydate": studydate,
            "h5_path": resampled_h5_path,
            "fold": fold,
            "label": label,
            "age_at_study": age_at_study,
            "gender": gender,
            "origin": origin,
            "d_coord_zyx": d_coord_zyx,
            "transform": transform,
            "resampled_dicom_shape": resampled_dicom_shape,
            "original_spacing": original_spacing.tolist(),
            "resampled_spacing": resampled_spacing.tolist(),
            "w_coord_zyx": w_coord_zyx,
            "r_coord_zyx": r_coord_zyx,
        }

        query = {
            "patient_id": {"$in": [patient_id]},
            "series_instance_uid": {"$in": [series_instance_uid]},
            "studydate": {"$in": [studydate]},
            "nodule_coord": {"$in": [original_nodule_coord]},
        }
        docs = [x for x in _CLIENT[_TARGET_DB][_TARGET_COLLECTION].find(query, {})]
        collection = _CLIENT[_TARGET_DB][_TARGET_COLLECTION]
        if len(docs) == 1:
            _filter = {"_id": docs[0]["_id"]}
            newvalues = {"$set": dict_info}
            collection.update_one(_filter, newvalues)
        else:
            collection.insert_one(dict_info)


def insert_to_DB_in_parallel(df, num_jobs=1):
    # DataFrame을 청크로 나누어 처리
    chunk_size = len(df) // num_jobs
    chunks = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]

    # 각 청크를 병렬로 처리하고 결과를 합치기
    Parallel(n_jobs=num_jobs)(delayed(insert_to_DB)(chunk) for chunk in chunks)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Insert asan dataset to DB")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="../../data_eda/LUNA25_Public_Training_Development_Data_fold.csv",
        help="Path to csv file",
    )
    parser.add_argument("--clean_documents", type=bool, default=True)
    parser.add_argument("--chunk_size", type=int, default=100, help="Chunk size for parallel processing")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    print(len(df))

    if args.clean_documents:
        client = pymongo.MongoClient(_VUNO_LUNG_DB)
        col = client[_TARGET_DB][_TARGET_COLLECTION]
        col.delete_many({})

    # insert to DB
    chunk_size = args.chunk_size
    insert_to_DB_in_parallel(df, num_jobs=chunk_size)
    print("Insertion to DB is done")
