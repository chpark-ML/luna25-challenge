import logging
from pathlib import Path
from typing import Dict, List

import dataloader
import numpy as np
import SimpleITK
import torch

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)


def transform(input_image, point):
    """

    Parameters
    ----------
    input_image: SimpleITK Image
    point: array of points

    Returns
    -------
    tNumpyOrigin

    """
    return np.array(list(reversed(input_image.TransformContinuousIndexToPhysicalPoint(list(reversed(point))))))


def itk_image_to_numpy_image(input_image):
    """

    Parameters
    ----------
    input_image: SimpleITK image

    Returns
    -------
    numpyImage: SimpleITK image to numpy image
    header: dict containing origin, spacing and transform in numpy format

    """

    numpyImage = SimpleITK.GetArrayFromImage(input_image)
    numpyOrigin = np.array(list(reversed(input_image.GetOrigin())))
    numpySpacing = np.array(list(reversed(input_image.GetSpacing())))

    # get numpyTransform
    tNumpyOrigin = transform(input_image, np.zeros((numpyImage.ndim,)))
    tNumpyMatrixComponents = [None] * numpyImage.ndim
    for i in range(numpyImage.ndim):
        v = [0] * numpyImage.ndim
        v[i] = 1
        tNumpyMatrixComponents[i] = transform(input_image, v) - tNumpyOrigin
    numpyTransform = np.vstack(tNumpyMatrixComponents).dot(np.diag(1 / numpySpacing))

    # define necessary image metadata in header
    header = {
        "origin": numpyOrigin,
        "spacing": numpySpacing,
        "transform": numpyTransform,
    }

    return numpyImage, header


# define processor
class MalignancyProcessor:
    """
    Loads a chest CT scan, and predicts the malignancy around a nodule
    """

    def __init__(self, config_models=None, mode="3D", device=torch.device("cuda:0"), suppress_logs=False):
        self.device = device
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
        for model_indicator, config_model in config_models.items():
            model_path = Path(config_model.root_path) / config_model.exp_name / config_model.file_name
            self.models[model_indicator] = torch.jit.load(model_path, map_location=self.device).to(self.device).eval()

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
            # print(f"[INFO] Input device: {nodules.device}")
            # print(f"[INFO] Model device: {next(model.parameters()).device}")
            logits = model(nodules)
            logits = logits.data.cpu().numpy()
            probs.append(torch.sigmoid(torch.from_numpy(logits)).numpy())

        probs = np.stack(probs, axis=0)  # shape: (num_models, ...)
        mean_probs = np.mean(probs, axis=0)

        return mean_probs


class NoduleProcessor:
    def __init__(self, config_models, mode="3D"):
        """
        Parameters
        ----------
        ct_image_file: Path to the CT image file
        nodule_locations: Dictionary containing nodule coordinates and annotationIDs
        clinical_information: Dictionary containing clinical information (Age and Gender)
        mode: 2D or 3D
        """
        self.mode = mode
        self.processor = MalignancyProcessor(config_models=config_models, mode=mode, suppress_logs=True)

    def predict(self, input_image: SimpleITK.Image, coords: np.array) -> List:
        """

        Parameters
        ----------
        input_image: SimpleITK Image
        coords: numpy array with list of nodule coordinates in /input/nodule-locations.json

        Returns
        -------
        malignancy risk of the nodules provided in /input/nodule-locations.json
        """

        numpy_image, header = itk_image_to_numpy_image(input_image)

        malignancy_risks = []
        for i in range(len(coords)):
            self.processor.define_inputs(numpy_image, header, [coords[i]])
            malignancy_risk = self.processor.predict()
            malignancy_risk = np.array(malignancy_risk).reshape(-1)[0]
            malignancy_risks.append(malignancy_risk)

        malignancy_risks = np.array(malignancy_risks)

        malignancy_risks = list(malignancy_risks)

        return malignancy_risks

    @staticmethod
    def load_inputs(ct_image_file, nodule_locations):
        # load image
        print(f"Reading {ct_image_file}")
        image = SimpleITK.ReadImage(str(ct_image_file))

        annotation_ids = [p["name"] for p in nodule_locations["points"]]
        coords = np.array([p["point"] for p in nodule_locations["points"]])
        coords = np.flip(coords, axis=1)  # reverse to [z, y, x] format

        return image, coords, annotation_ids

    def process(self, ct_image_file, nodule_locations, clinical_information=None) -> Dict:
        """
        Load CT scan(s) and nodule coordinates, predict malignancy risk and write the outputs
        Returns
        -------
        None
        """
        image, coords, annotation_ids = self.load_inputs(ct_image_file, nodule_locations)
        output = self.predict(image, coords)

        assert len(output) == len(annotation_ids), "Number of outputs should match number of inputs"
        results = {
            "name": "Points of interest",
            "type": "Multiple points",
            "points": [],
            "version": {"major": 1, "minor": 0},
        }

        # Populate the "points" section dynamically
        coords = np.flip(coords, axis=1)
        for i in range(len(annotation_ids)):
            results["points"].append(
                {"name": annotation_ids[i], "point": coords[i].tolist(), "probability": float(output[i])}
            )
        return results
