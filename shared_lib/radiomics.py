from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class RadiomicsFeatureKeys:
    # Shape-based features (3D morphological characteristics)
    shape: List[str] = field(
        default_factory=lambda: [
            "original_shape_Sphericity",  # Higher value means more spherical
            "original_shape_Elongation",  # Degree of elongation
            "original_shape_SurfaceVolumeRatio",  # Surface area to volume ratio, used to assess irregularity
            "original_shape_MeshVolume",  # Volume calculated by triangular mesh
            "original_shape_MajorAxisLength",  # Major axis length
            "original_shape_MinorAxisLength",  # Minor axis length
            "original_shape_Flatness",  # Degree of flatness
            "original_shape_Maximum3DDiameter",  # Maximum 3D diameter
            "original_shape_VoxelVolume",  # Volume calculated based on voxels
        ]
    )

    # First-order statistics (intensity distribution)
    firstorder: List[str] = field(
        default_factory=lambda: [
            "original_firstorder_Entropy",  # Disorder of intensity distribution (amount of information)
            "original_firstorder_Skewness",  # Asymmetry (negative: left tail, positive: right tail)
            "original_firstorder_Kurtosis",  # Peakedness of the distribution
            "original_firstorder_Mean",  # Mean intensity
            "original_firstorder_Median",  # Median intensity
            # "original_firstorder_Uniformity",  # Uniformity (higher means more uniform)
            "original_firstorder_Energy",  # Sum of squares of intensity values
            "original_firstorder_TotalEnergy",  # Total energy of all voxels
        ]
    )

    # GLCM (Gray-Level Co-occurrence Matrix) - texture features
    glcm: List[str] = field(
        default_factory=lambda: [
            # "original_glcm_Contrast",  # Local variation of pixel intensity differences
            # "original_glcm_Correlation",  # Linear relationship between pixels
            # "original_glcm_DifferenceEntropy",  # Entropy of the difference matrix
            # "original_glcm_ClusterTendency",  # Tendency of similar pixel clusters
            # "original_glcm_ClusterShade",  # Asymmetry, positive/negative skewness
            # "original_glcm_Id",  # Identity (higher value: similar pixels)
            # "original_glcm_Idn",  # Normalized identity
            # "original_glcm_Imc1"  # Information measure of correlation
        ]
    )

    # GLRLM (Gray Level Run Length Matrix) - run-length texture
    glrlm: List[str] = field(
        default_factory=lambda: [
            "original_glrlm_ShortRunEmphasis",  # Emphasis on short runs
            "original_glrlm_LongRunEmphasis",  # Emphasis on long runs
            "original_glrlm_RunEntropy",  # Entropy of run lengths
            "original_glrlm_RunLengthNonUniformity",  # Non-uniformity of run lengths
            "original_glrlm_LongRunHighGrayLevelEmphasis",  # Emphasis on long runs + high intensity
        ]
    )

    # GLSZM (Gray Level Size Zone Matrix) - zone-based texture
    glszm: List[str] = field(
        default_factory=lambda: [
            "original_glszm_SmallAreaEmphasis",  # Emphasis on small areas
            "original_glszm_LargeAreaEmphasis",  # Emphasis on large areas
            "original_glszm_GrayLevelNonUniformity",  # Non-uniformity of gray levels
            "original_glszm_ZoneEntropy",  # Entropy of zone distribution
        ]
    )

    # GLDM (Gray Level Dependence Matrix) - local gray-level dependency
    gldm: List[str] = field(
        default_factory=lambda: [
            "original_gldm_SmallDependenceEmphasis",  # Emphasis on short dependence
            "original_gldm_LargeDependenceEmphasis",  # Emphasis on long dependence
            "original_gldm_DependenceNonUniformity",  # Non-uniformity of dependence
            "original_gldm_DependenceEntropy",  # Entropy of dependence
        ]
    )

    # NGTDM (Neighbourhood Gray-Tone Difference Matrix)
    ngtdm: List[str] = field(
        default_factory=lambda: [
            # "original_ngtdm_Busyness",  # Rapid change in contrast
            # "original_ngtdm_Coarseness",  # Roughness of texture (higher means less detailed)
            # "original_ngtdm_Complexity",  # Complexity of gray-level differences
            # "original_ngtdm_Contrast",  # Contrast of gray-level differences
            # "original_ngtdm_Strength"  # Strength of intensity differences between pixels
        ]
    )


if __name__ == "__main__":
    features = RadiomicsFeatureKeys()
    print(features.shape)
    print(features.shape[-1])

    features = RadiomicsFeatureKeys()
    total = sum(len(getattr(features, field)) for field in features.__dataclass_fields__)
    print(f"Total feature count: {total}")  # 43
