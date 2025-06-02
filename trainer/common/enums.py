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
    """
    This is a model name classification class designed for training pipelines that involve multiple models,
    such as those used in GAN (Generative Adversarial Network) training.

    In GAN training, both a generative model and a discriminative model are used.
    Typically, only the generative model is saved, and
    inference is performed by distinguishing between the two models in the training code.
    This class is intended to explicitly differentiate between such models in these scenarios.

    If a field value below is included in a model configuration name, it is interpreted as the corresponding model.
    For example, if the model name includes 'repr', it is considered a REPRESENTATIVE model.

    For general classification or segmentation models, it is recommended to name them as 'model_repr'.
    For GANs, use 'model_repr' for the generative model and 'model_dis' for the discriminative model.
    """

    REPRESENTATIVE = "repr"
    PATCH_LEVEL = "patch_level"
    DUAL_SCALE = "dual_scale"


class BitDepth(Enum):
    BIT_DEPTH_8 = "uint8"
    BIT_DEPTH_16 = "uint16"
