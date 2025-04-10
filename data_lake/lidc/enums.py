from enum import Enum


class NoduleAttribute(Enum):
    SUBTLETY = "subtlety"
    MARGIN = "margin"
    SPHERICITY = "sphericity"
    CALCIFICATION = "calcification"
    TEXTURE = "texture"
    INTERNAL_STRUCTURE = "internalStructure"
    LOBULATION = "lobulation"
    SPICULATION = "spiculation"
    MALIGNANCY = "malignancy"


class NoduleAttributeCluster(Enum):
    SUBTLETY = "c_subtlety_logistic"
    MARGIN = "c_margin_logistic"
    SPHERICITY = "c_sphericity_logistic"
    CALCIFICATION = "c_calcification_logistic"
    TEXTURE = "c_texture_logistic"
    INTERNAL_STRUCTURE = "c_internalStructure_logistic"
    LOBULATION = "c_lobulation_logistic"
    SPICULATION = "c_spiculation_logistic"
    MALIGNANCY = "c_malignancy_logistic"
