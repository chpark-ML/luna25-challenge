import torch
from tqdm import tqdm

from shared_lib.processor.base_processor import BaseProcessor
from shared_lib.model_output import ModelOutputCls, ModelOutputClsSeg


class NoduleAttrProcessor(BaseProcessor):
    """
    Loads a chest CT scan, and predicts the malignancy around a nodule
    """

    def __init__(self, models=None, mode="3D", device=torch.device("cuda:0"), suppress_logs=False,
                 do_segmentation=True):
        super().__init__(models=models, mode=mode, device=device, suppress_logs=suppress_logs)
        self.do_segmentation = do_segmentation

    def predict(self, numpy_image, header, coord):
        """
        1. nodule attr, nodule segmentation will be returned
        2. radiomic feature extraction using pyradiomic
        3. return  nodule attr, radiomic features
        """
        pass

    def predict_given_patch(self, patch_image: torch.Tensor) -> dict:
        """
        1. Perform inference using multiple models (nodule attr, segmentation)
        2. Apply sigmoid and average outputs across models
        3. Optionally extract radiomic features
        4. Return averaged nodule attributes and radiomic features
        """
        patch_image = patch_image.to(self.device)  # (B, 1, w, h ,d)

        if self.do_segmentation:
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
            output_dict = {
                key: torch.sigmoid(getattr(output, key)) for key in keys
            }
            model_outputs.append(output_dict)

        # Assumes all outputs are logistic regression probabilities,
        # where the second dimension (dim=1) corresponds to the model ensemble axis.
        # Each output tensor is either for classification: (B, 1),
        # or for segmentation: (B, 1, W, H, D).
        ensemble_output = {
            key: torch.stack([output[key] for output in model_outputs], dim=1).mean(dim=1)
            for key in keys
        }

        return ensemble_output

    # def inference(self, loader, mode, sanity_check=False):
    #     list_probs = list()
    #     dict_probs = {model_name: [] for model_name in self.models.keys()}
    #     list_annots = list()
    #     list_annot_ids = list()
    #
    #     for data in tqdm(loader):
    #         # prediction
    #         patch_image = data["image"].to(self.device)
    #
    #         # annotation
    #         annot = data["label"].to(self.device).float()
    #         annot_ids = data["ID"]
    #
    #         # inference (model-wise)
    #         batch_probs = list()
    #         for model_name, model in self.models.items():
    #             logits = model.get_prediction(patch_image)  # (B, 1)
    #             print(logits)
    #             prob = torch.sigmoid(logits)  # (B, 1)
    #
    #             # Save per-model probabilities
    #             dict_probs[model_name].append(prob)
    #
    #             # Aggregate for overall averaging
    #             batch_probs.append(prob)
    #
    #         # Mean across models
    #         batch_probs = torch.stack(batch_probs)  # (num_models, B, 1)
    #         mean_probs = torch.mean(batch_probs, dim=0)  # (B, 1)
    #
    #         list_probs.append(mean_probs)
    #         list_annots.append(annot)
    #         list_annot_ids.extend(annot_ids)
    #         # sanity check
    #         if sanity_check:
    #             break
    #
    #     # Combine batches
    #     probs = torch.vstack(list_probs)
    #     annots = torch.vstack(list_annots)
    #
    #     # Convert to numpy
    #     overall_probs = probs.squeeze().cpu().numpy()
    #     overall_annots = annots.squeeze().cpu().numpy()
    #
    #     # Convert dict_probs to numpy
    #     dict_probs = {k: torch.vstack(v).squeeze().cpu().numpy() for k, v in dict_probs.items()}
    #
    #     return overall_probs, overall_annots, list_annot_ids, dict_probs
