from trainer.common.enums import DicomMode

DB_ADDRESS = "mongodb://172.31.10.111:27017"
TARGET_DB = "lct"

LOGIT_KEY = "logit"
MULTI_SCALE_LOGIT_KEY = "logits_multi_scale"
GATE_KEY = "gate_results"
GATED_LOGIT_KEY = "gated_ce_dict"

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
