from trainer.common.enums import DicomMode

DB_ADDRESS = "mongodb://172.31.10.111:27017"
TARGET_DB = "lct"
INPUT_PATCH_KEY = "dicom"
SEG_ANNOTATION_KEY = "mask"
ATTR_ANNOTATION_KEY = "annot"
LOGIT_KEY = "logit"
LATENT_KEY = "latent"
SEG_LOGIT_KEY = "logit_seg"
MULTI_SCALE_LOGIT_KEY = "logits_multi_scale"
GATE_KEY = "gate_results"
GATED_LOGIT_KEY = "gated_ce_dict"


class LossKey:
    total = "total"
    cls = "cls"
    cls_dict = "cls_dict"
    entropy = "entropy"
    aux = "aux"
    seg = "seg"


# HU window (window level, window width)
DICOM_VALUE_RANGES = {
    DicomMode.BONE: (400, 1800),  # -500 ~ 1300
    DicomMode.LUNG: (-600, 1500),  # -1350 ~ 150
    DicomMode.MEDIASTINAL: (50, 350),  # -125 ~ 225
    DicomMode.PROPOSED: (-200, 1600),  # -1000 ~ 600
    DicomMode.BASE: (-300, 1400),  # -1000 ~ 400
}

HU_WINDOW = {
    "LUNG": DICOM_VALUE_RANGES[DicomMode.LUNG],
    "MEDIASTINAL": DICOM_VALUE_RANGES[DicomMode.MEDIASTINAL],
    "BONE": DICOM_VALUE_RANGES[DicomMode.BONE],
    "PROPOSED": DICOM_VALUE_RANGES[DicomMode.PROPOSED],
    "BASE": DICOM_VALUE_RANGES[DicomMode.BASE],
}
