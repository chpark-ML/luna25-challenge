from enum import Enum, IntEnum


class RunMode(Enum):
    TRAIN = "train"
    VALIDATE = "val"
    TEST = "test"


class Axis3d(IntEnum):
    """Values correspond to data layout"""

    z = 0
    y = 1
    x = 2


class CTPlane(Enum):
    """Value correspond to indices in coordinates"""

    axial = 0
    coronal = 1
    sagittal = 2


class ThresholdMode(Enum):
    F1 = "f1"
    YOUDEN = "youden"


class BaseBestModelStandard(Enum):
    REPRESENTATIVE = "representative_metric"
    LAST = "last_epoch"
    AUROC = "auroc"


class NoduleType(IntEnum):
    """Values need to correspond to nodule_type classification model"""

    non_solid = 0  # same as ground glass opacity
    part_solid = 1
    solid = 2
