import logging
from typing import Dict, List

import hydra
import numpy as np
import SimpleITK

from shared_lib.utils.utils import set_seed

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


class ImageProcessor:
    def __init__(self, config):
        """
        Parameters
        ----------
        ct_image_file: Path to the CT image file
        nodule_locations: Dictionary containing nodule coordinates and annotationIDs
        clinical_information: Dictionary containing clinical information (Age and Gender)
        mode: 2D or 3D
        """
        set_seed()
        self.mode = config.mode
        models = dict()
        for model_indicator, config_model in config.models.items():
            models[model_indicator] = hydra.utils.instantiate(config_model)
        self.malignancy_processor = hydra.utils.instantiate(config.processor, models=models)
        self.do_tta_by_size = config.do_tta_by_size

    def _predict(self, input_image: SimpleITK.Image, coords: np.array, clinical_information) -> List:
        """

        Parameters
        ----------
        input_image: SimpleITK Image
        coords: numpy array with list of nodule coordinates in /input/nodule-locations.json

        Returns
        -------
        malignancy risk of the nodules provided in /input/nodule-locations.json
        """

        # image loader
        numpy_image, header = itk_image_to_numpy_image(input_image)

        # prepare image-level features, e.g., lobe seg mask, image-level feature representation, etc.

        # predict malignancy risk
        malignancy_risks = []
        for i in range(len(coords)):
            if self.do_tta_by_size:
                malignancy_risk = self.malignancy_processor.predict(numpy_image, header, coords[i], size_mm=[40, 50, 60])
            else:
                malignancy_risk = self.malignancy_processor.predict(numpy_image, header, coords[i])
            malignancy_risk = np.array(malignancy_risk).reshape(-1)[0]
            malignancy_risks.append(malignancy_risk)
        malignancy_risks = np.array(malignancy_risks)
        malignancy_risks = list(malignancy_risks)

        return malignancy_risks

    @staticmethod
    def _load_inputs(ct_image_file, nodule_locations):
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
        # prepare image and coords
        image, coords, annotation_ids = self._load_inputs(ct_image_file, nodule_locations)
        # get malignancy risk score
        output = self._predict(image, coords, clinical_information)

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
