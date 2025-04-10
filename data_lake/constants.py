DB_ADDRESS = "mongodb://172.31.10.111:27017"
TARGET_DB = "lct"
TARGET_COLLECTION = "LUNA25-Malignancy"

DEFAULT_RESAMPLED_SPACING = (1.0, 0.67, 0.67)

NUM_FOLD = 7


class DatasetKey:
    luna25 = "LUNA25"
    lidc = "LIDC"


class DataLakeKey:
    COLLECTION = "collection"
    DOC_ID = "_id"


class LUNA25Dir:
    meta_dir = "/team/team_blu3/lung/data/2_public/LUNA25_Original/LUNA25_Public_Training_Development_Data.csv"
    image_dir = "/team/team_blu3/lung/data/2_public/LUNA25_Original/luna25_images"
    output_nfs_dir = "/team/team_blu3/lung/data/2_public/LUNA25_h5"
    output_local_dir = "/nvme1/1_dataset/LUNA25_h5"


class MetaDataKey:
    origin = "origin"
    spacing = "spacing"
    transform = "transform"


class H5DataKey:
    image = "image"
    origin = "origin"
    spacing = "spacing"
    transform = "transform"
    resampled_image = "resampled_image"
    resampled_spacing = "resampled_spacing"


class DBKey:
    PATIENT_ID = "patient_id"
    SERIES_INSTANCE_UID = "series_instance_uid"
    ANNOTATION_ID = "annotation_id"
    STUDY_DATE = "studydate"
    H5_PATH_LOCAL = "h5_path"
    H5_PATH_NFS = "h5_path_nfs"
    FOLD = "fold"
    LABEL = "label"
    AGE_AT_STUDY = "age_at_study"
    GENDER = "gender"
    ORIGIN = "origin"
    TRANSFORM = "transform"
    SPACING = "spacing"
    RESAMPLED_SPACING = "resampled_spacing"
    D_COORD_ZYX = "d_coord_zyx"
    W_COORD_ZYX = "w_coord_zyx"
    R_COORD_ZYX = "r_coord_zyx"
