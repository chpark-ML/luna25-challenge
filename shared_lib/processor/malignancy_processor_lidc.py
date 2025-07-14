import json
import os

import numpy as np
import torch
from tqdm import tqdm

from shared_lib.processor.base_processor import BaseProcessor
from trainer.common.constants import ATTR_ANNOTATION_KEY


class MalignancyProcessorLIDC(BaseProcessor):
    """
    Loads a chest CT scan, and predicts the malignancy around a nodule
    """

    def __init__(
        self,
        models=None,
        mode="3D",
        device=torch.device("cuda:0"),
        suppress_logs=False,
        is_lr_weight=False,
        lr_weights_path=None,
    ):
        super().__init__(models=models, mode=mode, device=device, suppress_logs=suppress_logs)

        # Initialize LR weights
        self.is_lr_weight = is_lr_weight
        self.lr_weights = None
        self.lr_intercept = None

        if self.is_lr_weight:
            if lr_weights_path and os.path.exists(lr_weights_path):
                self.load_lr_weights(lr_weights_path)
            else:
                print("=" * 100)
                print(
                    f"Warning: is_lr_weight is True but weights file not found at {lr_weights_path}. Using simple averaging."
                )
                print("=" * 100)
                self.is_lr_weight = False

    def load_lr_weights(self, weights_path: str):
        """Load logistic regression weights from JSON file"""
        with open(weights_path, "r") as f:
            weights_data = json.load(f)

        self.lr_weights = np.array(weights_data["weights"])
        self.lr_intercept = weights_data["intercept"]

        if not self.suppress_logs:
            print(f"Loaded LR weights from {weights_path}")
            print(f"Weights: {self.lr_weights}")
            print(f"Intercept: {self.lr_intercept}")

    def apply_lr_weights(self, probs_array):
        """Apply logistic regression weights to model probabilities"""
        if self.lr_weights is None or self.lr_intercept is None:
            return np.mean(probs_array, axis=0)

        # Apply linear combination
        linear_combination = np.dot(probs_array.T, self.lr_weights) + self.lr_intercept

        # Apply sigmoid to get final probability
        final_probs = 1 / (1 + np.exp(-linear_combination))

        return final_probs

    def predict(self, numpy_image, header, coord):
        """
        Perform model inference on the given input image and coordinate.
        """
        patch = self.prepare_patch(numpy_image, header, coord, self.mode)

        probs = list()
        for model_name, model in self.models.items():
            logits = model.get_prediction(patch)
            logits = logits.data.cpu().numpy()
            probs.append(torch.sigmoid(torch.from_numpy(logits)).numpy())

        probs = np.stack(probs, axis=0)  # shape: (num_models, ...)

        # Apply LR weights if available, otherwise use simple averaging
        if self.is_lr_weight:
            mean_probs = self.apply_lr_weights(probs)
        else:
            mean_probs = np.mean(probs, axis=0)

        return mean_probs

    def inference(self, loader, mode, sanity_check=False):
        list_probs = list()
        dict_probs = {model_name: [] for model_name in self.models.keys()}
        list_annots = list()
        list_annot_ids = list()

        for batch_idx, data in enumerate(tqdm(loader)):
            # prediction
            patch_image = data["image"].to(self.device)

            # annotation
            attributes = data[ATTR_ANNOTATION_KEY]
            malignancy_tensor = attributes["c_malignancy_logistic"]
            if isinstance(malignancy_tensor, torch.Tensor):
                annot = malignancy_tensor.to(self.device)
            else:
                annot = torch.tensor(malignancy_tensor).to(self.device).float()

            # batch dimension 확인 및 조정
            if len(annot.shape) == 1:
                annot = annot.unsqueeze(-1)  # (B, 1) 형태로 만들기

            annot_ids = data["doc_id"]

            # inference (model-wise)
            batch_probs = list()
            for model_name, model in self.models.items():
                logits = model.get_prediction(patch_image)  # (B, 1)
                prob = torch.sigmoid(logits)  # (B, 1)

                # Save per-model probabilities
                dict_probs[model_name].append(prob)

                # Aggregate for overall averaging
                batch_probs.append(prob)

            # Mean across models or apply LR weights
            batch_probs = torch.stack(batch_probs)
            if self.is_lr_weight:
                # Convert to numpy for LR weights application
                batch_probs_np = batch_probs.cpu().numpy()  # (num_models, B, 1)
                mean_probs_np = self.apply_lr_weights(batch_probs_np.squeeze(-1))  # (B,)
                mean_probs = torch.from_numpy(mean_probs_np).unsqueeze(-1).to(self.device)  # (B, 1)
            else:
                mean_probs = torch.mean(batch_probs, dim=0)  # (B, 1)

            list_probs.append(mean_probs)
            list_annots.append(annot)
            list_annot_ids.extend(annot_ids)

            if sanity_check:
                break

        # Combine batches
        probs = torch.vstack(list_probs)
        annots = torch.vstack(list_annots)

        # Convert to numpy
        overall_probs = probs.squeeze().cpu().numpy()
        overall_annots = annots.squeeze().cpu().numpy()

        # Convert dict_probs to numpy
        dict_probs = {k: torch.vstack(v).squeeze().cpu().numpy() for k, v in dict_probs.items()}

        return overall_probs, overall_annots, list_annot_ids, dict_probs
