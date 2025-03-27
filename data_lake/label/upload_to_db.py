import argparse

import h5py
import numpy as np
import pandas as pd
import pymongo
from joblib import Parallel, delayed
from tqdm import tqdm

import numpy as np
import os 
import sys
from glob import glob
from sklearn.model_selection import StratifiedKFold

_VUNO_LUNG_DB = "mongodb://172.31.10.111:27017"
_TARGET_DB = "lct"
_TARGET_COLLECTION = "LUNA25-Malignancy"

def add_meta_data(df):
    
    nodule_blocks_image_dir = '/team/team_blu3/lung/data/2_public/LUNA25_Original/luna25_nodule_blocks/image/'
    nodule_blocks_metadata_dir = '/team/team_blu3/lung/data/2_public/LUNA25_Original/luna25_nodule_blocks/metadata/'
    
    df['nodule_block_image_shape'] = 0
    df['x_origin'] = 0
    df['y_origin'] = 0
    df['z_origin'] = 0
    df['x_spacing'] = 0
    df['y_spacing'] = 0
    df['z_spacing'] = 0
    df['x_transform'] = 0
    df['y_transform'] = 0
    df['z_transform'] = 0

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        
        annotat_id = row['AnnotationID']
        image = np.load(os.path.join(nodule_blocks_image_dir, f'{annotat_id}.npy'))
        metadata = np.load(os.path.join(nodule_blocks_metadata_dir, f'{annotat_id}.npy'), allow_pickle=True).item()
        image_shape = image.shape
        origin = metadata['origin']
        spacing = metadata['spacing']
        transform = metadata['transform']
        
        df.loc[idx, 'nodule_block_image_shape'] = str(image_shape)
        df.loc[idx, 'x_origin'] = origin[2]
        df.loc[idx, 'y_origin'] = origin[1]
        df.loc[idx, 'z_origin'] = origin[0]
        df.loc[idx, 'x_spacing'] = spacing[2]
        df.loc[idx, 'y_spacing'] = spacing[1]
        df.loc[idx, 'z_spacing'] = spacing[0]
        df.loc[idx, 'x_transform'] = str(transform[0])
        df.loc[idx, 'y_transform'] = str(transform[1])
        df.loc[idx, 'z_transform'] = str(transform[2])
        
    return df


def split_fold(meta_df):
    # PatientID 기준으로 그룹화
    patient_df = meta_df.groupby('PatientID').agg(
        StudyDate=('StudyDate', lambda x: x.mode()[0]),
        malignancy=('label', lambda x: (x == 0).sum()),
        benign=('label', lambda x: (x == 1).sum()),
        Age_at_StudyDate=('Age_at_StudyDate', lambda x: x.mode()[0]),
        Gender=('Gender', lambda x: x.mode()[0]),  # 최빈값(가장 많이 등장한 성별)
        z_spacing_min=('z_spacing', 'min'),
        z_spacing_max=('z_spacing', 'max')
    ).reset_index()
    patient_df['strat'] = patient_df['malignancy'].astype(str) + '_' + patient_df['benign'].astype(str)
    
    # StudyDate, label, gender, z_spacing은 골고루 분포되어 있음
    stratify = StratifiedKFold(n_splits=7, random_state=42, shuffle=True)

    random_fold = {}
    for i, (train_index, test_index) in enumerate(stratify.split(patient_df, patient_df['strat'])):
        random_fold.update({i: patient_df.iloc[test_index]['PatientID'].values})
        
    fold_df = meta_df.copy()
    for fold_num, _idx in enumerate(random_fold.values()):
        fold_df.loc[meta_df.PatientID.isin(_idx), "fold"] = str(fold_num)
        
    return fold_df


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
        "--original_csv_path",
        type=str,
        default='/team/team_blu3/lung/data/2_public/LUNA25_Original/LUNA25_Public_Training_Development_Data.csv',
        help="Path to csv file",
    )
    parser.add_argument("--clean_documents", type=bool, default=True)
    parser.add_argument("--chunk_size", type=int, default=100, help="Chunk size for parallel processing")
    args = parser.parse_args()

    df = pd.read_csv(args.original_csv_path)
    print(len(df))
    
    metda_df = add_meta_data(df)
    
    fold_df = split_fold(metda_df)

    if args.clean_documents:
        client = pymongo.MongoClient(_VUNO_LUNG_DB)
        col = client[_TARGET_DB][_TARGET_COLLECTION]
        col.delete_many({})

    # insert to DB
    chunk_size = args.chunk_size
    insert_to_DB_in_parallel(fold_df, num_jobs=chunk_size)
    print("Insertion to DB is done")
