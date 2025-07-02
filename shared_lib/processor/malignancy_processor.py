import json
import os

import numpy as np
import torch
from tqdm import tqdm

from shared_lib.processor.base_processor import BaseProcessor

LOGIT_KEY = "logit"
GATE_KEY = "gate_results"
GATED_LOGIT_KEY = "gated_ce_dict"


class MalignancyProcessor(BaseProcessor):
    """
    Loads a chest CT scan, and predicts the malignancy around a nodule
    """

    def __init__(self, models=None, mode="3D", device=torch.device("cuda:0"), suppress_logs=False,
                 lr_weights_path=None):
        super().__init__(models=models, mode=mode, device=device, suppress_logs=suppress_logs)

        # Load logistic regression weights if provided
        self.lr_weights = None
        self.lr_intercept = None
        if lr_weights_path and os.path.exists(lr_weights_path):
            self.load_lr_weights(lr_weights_path)

    def load_lr_weights(self, weights_path: str):
        """Load logistic regression weights from JSON file"""
        try:
            with open(weights_path, 'r') as f:
                weights_dict = json.load(f)

            self.lr_weights = weights_dict['weights']
            self.lr_intercept = weights_dict['intercept']
            self.model_order = weights_dict.get('model_order', [])

            if not self.suppress_logs:
                print(f"Loaded LR weights from: {weights_path}")
                print(f"Weights: {self.lr_weights}")
                print(f"Intercept: {self.lr_intercept}")

        except Exception as e:
            if not self.suppress_logs:
                print(f"Failed to load LR weights from {weights_path}: {e}")
            self.lr_weights = None
            self.lr_intercept = None

    def predict(self, numpy_image, header, coord, size_mm=None):
        """
        Perform model inference on the given input image and coordinate.
        """
        if not isinstance(size_mm, list):
            size_mm = [size_mm]

        tta_by_size = []
        for size in size_mm:
            patch = self.prepare_patch(numpy_image, header, coord, self.mode, size_mm=size)
            probs = list()
            for model_name, model in self.models.items():
                logits = model.get_prediction(patch)
                logits = logits.data.cpu().numpy()
                probs.append(torch.sigmoid(torch.from_numpy(logits)).numpy())

            probs = np.stack(probs, axis=0)  # shape: (num_models, ...)
            mean_prob = np.mean(probs, axis=0)  # ensemble result

            tta_by_size.append(mean_prob)

        # Combine predictions over different sizes (TTA)
        final_prediction = np.mean(tta_by_size, axis=0)  # shape: (...)

        return final_prediction

    def get_class_evidence(self, patch):
        """
        Perform model inference using multiple models on a single input patch.
        Aggregates outputs including logits, gate activations, and gated logits.
        """
        gate_levels = [0, 1, 2]

        # Initialize containers for model ensemble aggregation
        logits_all_models = []
        gates_all_models = {i: [] for i in gate_levels}
        gated_logits_all_models = {i: [] for i in gate_levels}

        for model in self.models.values():
            target_attr = model.model.classifier.target_attr_downstream
            outputs = model.get_intermediate_results(patch)

            # Extract and store sigmoid-activated logits
            logits = outputs[LOGIT_KEY][target_attr]
            logits_np = torch.sigmoid(logits).detach().cpu().numpy()
            logits_all_models.append(logits_np)

            # Extract gate features and gated logits at each level
            for i in gate_levels:
                gate_np = outputs[GATE_KEY][i].detach().cpu().numpy()
                gated_logit_np = outputs[GATED_LOGIT_KEY][target_attr][i].detach().cpu().numpy()

                gates_all_models[i].append(gate_np)
                gated_logits_all_models[i].append(gated_logit_np)

        # Final aggregation over models
        final_results = {
            LOGIT_KEY: np.mean(np.stack(logits_all_models, axis=0), axis=0),
            GATE_KEY: {},
            GATED_LOGIT_KEY: {}
        }

        for i in gate_levels:
            final_results[GATE_KEY][i] = np.mean(np.stack(gates_all_models[i], axis=0), axis=0)
            final_results[GATED_LOGIT_KEY][i] = np.mean(np.stack(gated_logits_all_models[i], axis=0), axis=0)

        return final_results

    @staticmethod
    def interpolate_and_center_crop_5d(image, scale_factor, target_size):
        B, C, W, H, D = image.shape
        tgt_W, tgt_H, tgt_D = target_size

        new_W = int(W * scale_factor)
        new_H = int(H * scale_factor)
        new_D = int(D * scale_factor)

        x = torch.nn.functional.interpolate(
            image,
            size=(new_W, new_H, new_D),
            mode='trilinear',
            align_corners=True
        )

        # Determine padding or cropping
        diff_W = tgt_W - new_W
        diff_H = tgt_H - new_H
        diff_D = tgt_D - new_D

        if diff_W > 0 or diff_H > 0 or diff_D > 0:
            # Pad if any dimension is smaller
            pad = [
                max(diff_D // 2, 0), max(diff_D - diff_D // 2, 0),
                max(diff_H // 2, 0), max(diff_H - diff_H // 2, 0),
                max(diff_W // 2, 0), max(diff_W - diff_W // 2, 0),
            ]
            x = torch.nn.functional.pad(x, pad, mode='constant', value=0)

        if x.shape[2] > tgt_W or x.shape[3] > tgt_H or x.shape[4] > tgt_D:
            # Crop if any dimension is larger
            start_w = (x.shape[2] - tgt_W) // 2
            start_h = (x.shape[3] - tgt_H) // 2
            start_d = (x.shape[4] - tgt_D) // 2
            x = x[:, :,
                start_w:start_w + tgt_W,
                start_h:start_h + tgt_H,
                start_d:start_d + tgt_D]

        return x

    def inference(self, loader, mode, do_tta_by_size=False, sanity_check=False):
        list_probs = list()
        dict_probs = {model_name: [] for model_name in self.models.keys()}
        list_annots = list()
        list_annot_ids = list()

        for data in tqdm(loader):
            # prediction
            patch_image = data["image"].to(self.device)
            _, _, W, H, D = patch_image.shape
            target_size = (W, H, D)

            # annotation
            annot = data["label"].to(self.device).float()
            annot_ids = data["ID"]

            # inference (model-wise)
            batch_probs = list()
            for model_name, model in self.models.items():
                if do_tta_by_size:
                    probs_by_size = []
                    for scale_factor in [0.8, 1.0, 1.2]:
                        resized_image = self.interpolate_and_center_crop_5d(patch_image, scale_factor, target_size)
                        logits = model.get_prediction(resized_image)  # (B, 1)
                        probs_by_size.append(torch.sigmoid(logits))
                    prob = torch.mean(torch.stack(probs_by_size, dim=0), dim=0)  # (B, 1)
                else:
                    logits = model.get_prediction(patch_image)  # (B, 1)
                    prob = torch.sigmoid(logits)  # (B, 1)

                # Save per-model probabilities
                dict_probs[model_name].append(prob)

                # Collect probabilities
                batch_probs.append(prob)

            # Stack probabilities
            batch_probs = torch.stack(batch_probs)  # (num_models, B, 1)

            # Apply logistic regression weights if loaded, otherwise use simple mean
            if self.lr_weights is not None and self.lr_intercept is not None:
                # Convert probabilities back to logits for proper logistic regression ensemble
                batch_logits = torch.log(batch_probs / (1 - batch_probs + 1e-7))  # (num_models, B, 1)

                # Use loaded LR weights
                weights = torch.tensor(self.lr_weights, device=batch_logits.device)
                mean_logits = torch.sum(batch_logits * weights[:, None, None], dim=0)  # (B, 1)

                # Add loaded intercept
                intercept_tensor = torch.tensor(self.lr_intercept, device=mean_logits.device)
                mean_logits += intercept_tensor

                # Convert back to probabilities
                mean_probs = torch.sigmoid(mean_logits)  # (B, 1)
            else:
                # Fallback to simple mean if no LR weights loaded
                mean_probs = torch.mean(batch_probs, dim=0)  # (B, 1)

            list_probs.append(mean_probs)
            list_annots.append(annot)
            list_annot_ids.extend(annot_ids)
            # sanity check
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
