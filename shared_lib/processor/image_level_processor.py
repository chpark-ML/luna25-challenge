import numpy as np
import torch
from tqdm import tqdm

from shared_lib.processor.base_processor import BaseProcessor


class ImageLevelProcessor(BaseProcessor):
    """
    Loads a chest CT scan, and predicts the malignancy around a nodule using image_level models
    """

    def __init__(self, models=None, mode="3D", device=torch.device("cuda:0"), suppress_logs=False):
        super().__init__(models=models, mode=mode, device=device, suppress_logs=suppress_logs)

    def predict(self, numpy_image, header, coord):
        """
        Perform model inference on the given input image and coordinate.
        For image_level models, we need both patch and large image inputs.
        """
        # Prepare patch input (smaller patch)
        patch_image = self.prepare_patch(numpy_image, header, coord, self.mode)
        
        # Prepare large image input (larger context)
        # For now, we'll use the same patch but with different size parameters
        # This might need to be adjusted based on your specific image_level dataset implementation
        image_large = self.prepare_patch(numpy_image, header, coord, self.mode)

        probs = list()
        for model_name, model in self.models.items():
            logits = model.get_prediction(patch_image, image_large)
            logits = logits.data.cpu().numpy()
            probs.append(torch.sigmoid(torch.from_numpy(logits)).numpy())

        probs = np.stack(probs, axis=0)  # shape: (num_models, ...)
        mean_probs = np.mean(probs, axis=0)

        return mean_probs

    def inference(self, loader, mode, sanity_check=False):
        list_probs = list()
        dict_probs = {model_name: [] for model_name in self.models.keys()}
        list_annots = list()
        list_annot_ids = list()

        for data in tqdm(loader):
            # prediction - image_level models expect both patch_image and image_large
            patch_image = data["image"].to(self.device)  # patch input
            image_large = data["image_large"].to(self.device)  # large image input

            # annotation
            annot = data["label"].to(self.device).float()
            annot_ids = data["ID"]

            # inference (model-wise)
            batch_probs = list()
            for model_name, model in self.models.items():
                logits = model.get_prediction(patch_image, image_large)  # (B, 1)
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
