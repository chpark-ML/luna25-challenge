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
    GAN 모델 학습과 같이 여러 모델을 활용한 학습파이프라인을 고려한 모델 이름 구분 클래스입니다.
    GAN 학습에서는 생성 모델(generative model)과 판별 모델(discriminative model) 두 가지가 사용되며,
    일반적으로 생성 모델만 저장되고, 학습 코드에서 두 모델을 구분하여 추론을 수행합니다.
    이러한 학습 시나리오에서 두 모델을 명확히 구분하기 위해 이 클래스를 정의합니다.

    아래 필드 값이 모델 설정(config)의 이름에 포함되면 해당 모델로 간주됩니다.
    예를 들어, 'model_repr'은 'repr'을 포함하므로 REPRESENTATIVE로 인식됩니다.

    일반 분류, 분할 모델은 'model_repr'로 명명하고,
    생성 모델은 'model_repr', 판별 모델은 'model_dis' 등으로 명명하는 것을 권장합니다.
    """
    REPRESENTATIVE = "repr"


class BitDepth(Enum):
    BIT_DEPTH_8 = "uint8"
    BIT_DEPTH_16 = "uint16"
