import numpy as np
import torch
from tqdm import tqdm

from shared_lib.processor.base_processor import BaseProcessor


class MalignancyProcessor(BaseProcessor):
    """
    Loads a chest CT scan, and predicts the malignancy around a nodule
    """

    def __init__(self, models=None, mode="3D", device=torch.device("cuda:0"), suppress_logs=False):
        super().__init__(models=models, mode=mode, device=device, suppress_logs=suppress_logs)

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

                # Aggregate for overall averaging
                batch_probs.append(prob)

            # Mean across models
            batch_probs = torch.stack(batch_probs)  # (num_models, B, 1)
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
