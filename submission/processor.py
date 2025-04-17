"""
Inference script for predicting malignancy of lung nodules
"""

import logging
import os

import dataloader
import numpy as np
import torch

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)


# define processor
class MalignancyProcessor:
    """
    Loads a chest CT scan, and predicts the malignancy around a nodule
    """

    def __init__(self, models=None, mode="2D", suppress_logs=False):
        self.device = torch.device("cuda:0")
        self.size_px_xy = 72
        self.size_px_z = 48
        self.size_mm = 50
        self.order = 1

        self.suppress_logs = suppress_logs
        if not self.suppress_logs:
            logging.info("Initializing the deep learning system")

        self.mode = mode
        self.model_root = "/opt/app/resources/"

        self.models = dict()
        for model_name, model in models.items():
            ckpt = torch.load(os.path.join(self.model_root, model_name, "model.pth"), map_location=self.device)
            model.load_state_dict(ckpt["model"])
            model.eval()
            self.models[model_name] = model.to(self.device)

    def define_inputs(self, image, header, coords):
        self.image = image
        self.header = header
        self.coords = coords

    def extract_patch(self, coord, output_shape, mode):
        patch = dataloader.extract_patch(
            CTData=self.image,
            coord=coord,
            srcVoxelOrigin=self.header["origin"],
            srcWorldMatrix=self.header["transform"],
            srcVoxelSpacing=self.header["spacing"],
            output_shape=output_shape,
            voxel_spacing=(
                self.size_mm / self.size_px_z,
                self.size_mm / self.size_px_xy,
                self.size_mm / self.size_px_xy,
            ),
            coord_space_world=True,
            mode=mode,
            order=self.order,
        )

        # ensure same datatype...
        patch = patch.astype(np.float32)

        # clip and scale...
        patch = dataloader.clip_and_scale(patch)
        return patch

    def _prepare_input(self, mode):
        if not self.suppress_logs:
            logging.info("Processing in " + mode)

        assert mode == "3D"
        output_shape = [self.size_px_z, self.size_px_xy, self.size_px_xy]

        nodules = []
        for _coord in self.coords:
            patch = self.extract_patch(_coord, output_shape, mode=mode)
            nodules.append(patch)

        nodules = np.array(nodules)
        nodules = torch.from_numpy(nodules).to(self.device)

        return nodules

    def predict(self):
        nodules = self._prepare_input(self.mode)

        probs = list()
        for model_name, model in self.models.items():
            logits = model(nodules)
            logits = logits.data.cpu().numpy()
            probs.append(torch.sigmoid(torch.from_numpy(logits)).numpy())

        probs = np.stack(probs, axis=0)  # shape: (num_models, ...)
        mean_probs = np.mean(probs, axis=0)

        return mean_probs
