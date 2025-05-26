from typing import NamedTuple

import torch


class ModelOutput(NamedTuple):
    c_subtlety_logistic: torch.Tensor
    c_sphericity_logistic: torch.Tensor
    c_lobulation_logistic: torch.Tensor
    c_spiculation_logistic: torch.Tensor
    c_margin_logistic: torch.Tensor
    c_texture_logistic: torch.Tensor
    c_calcification_logistic: torch.Tensor
    c_internalStructure_logistic: torch.Tensor
    c_malignancy_logistic: torch.Tensor
    logit_seg: torch.Tensor
