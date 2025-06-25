import numpy as np
import torch
from tqdm import tqdm

from shared_lib.processor.base_processor import BaseProcessor
from trainer.common.constants import ATTR_ANNOTATION_KEY


class MalignancyProcessorLIDC(BaseProcessor):
    """
    Loads a chest CT scan, and predicts the malignancy around a nodule
    """

    def __init__(self, models=None, mode="3D", device=torch.device("cuda:0"), suppress_logs=False):
        super().__init__(models=models, mode=mode, device=device, suppress_logs=suppress_logs)

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

            # Mean across models
            batch_probs = torch.stack(batch_probs)
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
