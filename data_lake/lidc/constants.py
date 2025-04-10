RESAMPLED_FEATURE_POSTFIX = "_resampled"
LOGISTIC_TASK_POSTFIX = "_logistic"
CLASSIFICATION_TASK_POSTFIX = "_class"
PREFIX_CONSENSUS = "c_"


class LidcKeyDict:
    LIDC_METADATA_PATH = "/lung/data/2_public/LIDC-IDRI-new/LIDC-IDRI_MetaData.csv"
    SUBJECT_ID = "Subject ID"
    SERIES_ID = "Series ID"
    MANUFACTURER = "Manufacturer"


class CollectionName:
    IMAGE = "pylidc-image"
    NODULE = "pylidc-nodule"
    CLUSTER = "pylidc-nodule-cluster"


class HFileKey:
    """Contains keys and paths used for accessing HDF5 files."""

    HFileName = "data"

    class HFilePath:
        ROOT_DIR = "/nvme1/1_dataset/lung_DB/lct"
        IMAGE_COLLECTION_PATH = f"{ROOT_DIR}/{CollectionName.IMAGE}"
        NODULE_COLLECTION_PATH = f"{ROOT_DIR}/{CollectionName.NODULE}"
        CLUSTER_COLLECTION_PATH = f"{ROOT_DIR}/{CollectionName.CLUSTER}"

    class HFileAttrName:
        DICOM_PIXELS = "dicom_pixels"
        MASK_ANNOTATION = "mask_annotation"
        DICOM_PIXELS_RESAMPLED = "dicom_pixels_resampled"
        MASK_ANNOTATION_RESAMPLED = "mask_annotation_resampled"


class ImageLevelInfo:
    STUDY_INSTANCE_UID = "study_instance_uid"
    SEREIS_INSTANCE_UID = "series_instance_uid"
    PATIENT_ID = "patient_id"
    SLICE_THICKNESS = "slice_thickness"
    SPACING_BETWEEN_SLICES = "spacing_between_slices"
    PIXEL_SPACING = "pixel_spacing"
    CONTRAST_USED = "contrast_used"
    NUM_ANNOTATION = "num_annotation"
    NUM_CLUSTER = "num_cluster"
    MANUFACTURER = "manufacturer"
    H5_FILE_PATH = "h5_file_path"


class NoduleLevelInfo:
    DICOM_SERIES_INFO = "dicom_series_info"
    ANNOTATION_ID = "annotation_id"
    DIAMETER = "diameter"
    VOLUME = "volume"
    DIAMETER_RESAMPLED = "diameter_resampled"
    VOLUME_RESAMPLED = "volume_resampled"
    BBOX_ZYX = "bbox_zyx"
    MASK_ZYX = "mask_zyx"


class ClusterLevelInfo:
    DICOM_SERIES_INFO = "dicom_series_info"
    CLUSTER_ID = "cluster_id"
    D_COORD_ZYX = "d_coord_zyx"
    R_COORD_ZYX = "r_coord_zyx"
    NUM_MASK = "num_mask"
    DIAMETER = "diameter"
    VOLUME = "volume"
    DIAMETER_RESAMPLED = "diameter_resampled"
    VOLUME_RESAMPLED = "volume_resampled"
    BBOX_ZYX = "bbox_zyx"
    MASK_ZYX = "mask_zyx"
