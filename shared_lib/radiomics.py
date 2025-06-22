from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class RadiomicsFeatureKeys:
    # Shape-based features (3D morphological characteristics)
    shape: List[str] = field(
        default_factory=lambda: [
            "original_shape_Sphericity",  # 구형에 가까울수록 높은 값
            "original_shape_Elongation",  # 길쭉한 정도
            "original_shape_SurfaceVolumeRatio",  # 표면적 대비 부피, 불규칙성 판단
            "original_shape_MeshVolume",  # 삼각형 메시로 계산한 부피
            "original_shape_MajorAxisLength",  # 주축 길이
            "original_shape_MinorAxisLength",  # 부축 길이
            "original_shape_Flatness",  # 평평한 정도
            "original_shape_Maximum3DDiameter",  # 3D 최대 직경
            "original_shape_VoxelVolume",  # 복셀 기반 계산 부피
        ]
    )

    # First-order statistics (intensity distribution)
    firstorder: List[str] = field(
        default_factory=lambda: [
            "original_firstorder_Entropy",  # 강도 분포의 무질서도 (정보량)
            "original_firstorder_Skewness",  # 비대칭성 (음수: 좌측 꼬리, 양수: 우측 꼬리)
            "original_firstorder_Kurtosis",  # 분포의 뾰족함
            "original_firstorder_Mean",  # 평균 강도
            "original_firstorder_Median",  # 중앙값 강도
            # "original_firstorder_Uniformity",  # 균일도 (높을수록 값이 균일)
            "original_firstorder_Energy",  # 강도 값의 제곱합
            "original_firstorder_TotalEnergy",  # 모든 복셀의 에너지 총합
        ]
    )

    # GLCM (Gray-Level Co-occurrence Matrix) - texture features
    glcm: List[str] = field(
        default_factory=lambda: [
            # "original_glcm_Contrast",  # 픽셀 강도 차이의 지역적 변화량
            # "original_glcm_Correlation",  # 픽셀 간 선형 관계
            # "original_glcm_DifferenceEntropy",  # 차이 행렬의 엔트로피
            # "original_glcm_ClusterTendency",  # 강도 유사 픽셀의 집합 경향
            # "original_glcm_ClusterShade",  # 비대칭성, 양/음 skewness
            # "original_glcm_Id",  # 동일성 (Identity, 높은 값: 유사한 픽셀)
            # "original_glcm_Idn",  # 정규화된 동일성
            # "original_glcm_Imc1"  # 정보 상호의존도 (Information Measure of Correlation)
        ]
    )

    # GLRLM (Gray Level Run Length Matrix) - run-length texture
    glrlm: List[str] = field(
        default_factory=lambda: [
            "original_glrlm_ShortRunEmphasis",  # 짧은 연속의 강조
            "original_glrlm_LongRunEmphasis",  # 긴 연속의 강조
            "original_glrlm_RunEntropy",  # run 길이의 엔트로피
            "original_glrlm_RunLengthNonUniformity",  # run 길이 불균일도
            "original_glrlm_LongRunHighGrayLevelEmphasis",  # 긴 run + 높은 강도 강조
        ]
    )

    # GLSZM (Gray Level Size Zone Matrix) - zone-based texture
    glszm: List[str] = field(
        default_factory=lambda: [
            "original_glszm_SmallAreaEmphasis",  # 작은 영역 강조
            "original_glszm_LargeAreaEmphasis",  # 큰 영역 강조
            "original_glszm_GrayLevelNonUniformity",  # 회색 강도 불균일도
            "original_glszm_ZoneEntropy",  # zone 분포의 엔트로피
        ]
    )

    # GLDM (Gray Level Dependence Matrix) - local gray-level dependency
    gldm: List[str] = field(
        default_factory=lambda: [
            "original_gldm_SmallDependenceEmphasis",  # 짧은 의존성 강조
            "original_gldm_LargeDependenceEmphasis",  # 긴 의존성 강조
            "original_gldm_DependenceNonUniformity",  # 의존성 불균일도
            "original_gldm_DependenceEntropy",  # 의존성의 엔트로피
        ]
    )

    # NGTDM (Neighbourhood Gray-Tone Difference Matrix)
    ngtdm: List[str] = field(
        default_factory=lambda: [
            # "original_ngtdm_Busyness",  # 주변 대비의 빠른 변화
            # "original_ngtdm_Coarseness",  # 질감의 거칠기 (높을수록 덜 세밀함)
            # "original_ngtdm_Complexity",  # 회색조 차이의 복잡성
            # "original_ngtdm_Contrast",  # 회색조 대비
            # "original_ngtdm_Strength"  # 픽셀 사이 강도 차이의 강도
        ]
    )


if __name__ == "__main__":
    features = RadiomicsFeatureKeys()
    print(features.shape)
    print(features.shape[-1])

    features = RadiomicsFeatureKeys()
    total = sum(len(getattr(features, field)) for field in features.__dataclass_fields__)
    print(f"총 feature 개수: {total}")  # 43
