from enum import Enum


class DicomMode(Enum):
    BONE = 1
    LUNG = 2
    MEDIASTINAL = 3
    PROPOSED = 4
    BASE = 5


class ThresholdMode(Enum):
    ALL = "all"
    F1 = "f1"
    YOUDEN = "youden"

    @classmethod
    def get_mode(cls, mode_str: str):
        for mode in cls:
            if mode.value == mode_str:
                return mode
        raise ValueError(f"Invalid mode: {mode_str}")


class ModelName(Enum):
    CLASSIFIER = "C"


class BitDepth(Enum):
    BIT_DEPTH_8 = "uint8"
    BIT_DEPTH_16 = "uint16"
