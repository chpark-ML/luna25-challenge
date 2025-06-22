from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from shared_lib.model_output import ModelOutputCls, ModelOutputClsSeg
from shared_lib.processor.base_processor import BaseProcessor


class NoduleAttrProcessor(BaseProcessor):
    """
    Loads a chest CT scan, and predicts the malignancy around a nodule
    """

    def __init__(
        self, models=None, mode="3D", device=torch.device("cuda:0"), suppress_logs=False, do_segmentation=True
    ):
        super().__init__(models=models, mode=mode, device=device, suppress_logs=suppress_logs)
        self.do_segmentation = do_segmentation  # whether model predict segmentation mask
        self.return_segmentation_mask = True  # whether processor return segmentation mask if available

    def predict(self, numpy_image, header, coord):
        """
        1. nodule attr, nodule segmentation will be returned
        2. radiomic feature extraction using pyradiomic
        3. return  nodule attr, radiomic features
        """
        pass

    def predict_ensemble_result(self, patch_image: torch.Tensor) -> dict:
        """
        1. Perform inference using multiple models (nodule attr, segmentation)
        2. Apply sigmoid and average outputs across models
        3. Optionally extract radiomic features
        4. Return averaged nodule attributes and radiomic features
        """
        patch_image = patch_image.to(self.device)  # (B, 1, w, h ,d)

        if self.do_segmentation and self.return_segmentation_mask:
            keys = ModelOutputClsSeg._fields
        else:
            keys = ModelOutputCls._fields

        model_outputs = list()
        for model_name, model in self.models.items():
            output = model.get_prediction(patch_image)

            # Convert to namedtuple
            if self.do_segmentation:
                output = ModelOutputClsSeg(*output)
            else:
                output = ModelOutputCls(*output)

            # Apply sigmoid and convert to dict
            output_dict = {key: torch.sigmoid(getattr(output, key)) for key in keys}
            model_outputs.append(output_dict)

        # Assumes all outputs are logistic regression probabilities,
        # where the second dimension (dim=1) corresponds to the model ensemble axis.
        # Each output tensor is either for classification: (B, 1),
        # or for segmentation: (B, 1, W, H, D).
        ensemble_output = {
            key: torch.stack([output[key] for output in model_outputs], dim=1).mean(dim=1) for key in keys
        }

        return ensemble_output  # (B, 1) or (B, 1, w, h, d)

    def inference(self, loader, mode, sanity_check=False) -> Dict[str, np.ndarray]:
        list_results = list()
        list_annot_ids = list()

        for data in tqdm(loader):
            # prediction
            patch_image = data["image"].to(self.device)

            # annotation id
            annot_ids = data["ID"]

            # inference
            result = self.predict_ensemble_result(patch_image)  # (B, 1) or (B, 1, w, h, d)

            list_results.append(result)
            list_annot_ids.extend(annot_ids)
            # sanity check
            if sanity_check:
                break

        # Combine batches
        if self.do_segmentation and self.return_segmentation_mask:
            keys = ModelOutputClsSeg._fields
        else:
            keys = ModelOutputCls._fields

        stacked_probs = {
            key: torch.vstack([result[key] for result in list_results]).detach().cpu().numpy() for key in keys
        }

        return stacked_probs  # dict[str, np.ndarray] of shape (N, 1) or (N, 1, w, h, d)
