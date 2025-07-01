"""
Inference script for predicting malignancy of lung nodules
"""

import json
from glob import glob
from pathlib import Path

import numpy as np
import torch
from input_processor import ImageProcessor

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("/opt/app/resources")


def _load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def _write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def _load_image_path(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.tif")) + glob(str(location / "*.tiff")) + glob(str(location / "*.mha"))

    assert len(input_files) == 1, "Please upload only one .mha file per job for grand-challenge.org"

    result = input_files[0]

    return result


def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch version: {torch.version.cuda}")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


def apply_logistic_regression(predictions, weights, intercept):
    """Apply pre-trained logistic regression weights to ensemble predictions"""
    # predictions: array of shape (n_samples, n_models)
    # weights: array of shape (n_models,)
    # intercept: float
    return 1 / (1 + np.exp(-(np.dot(predictions, weights) + intercept)))


def run(config):
    # Read the inputs
    input_nodule_locations = _load_json_file(location=INPUT_PATH / "nodule-locations.json")
    input_clinical_information = _load_json_file(location=INPUT_PATH / "clinical-information-lung-ct.json")
    input_chest_ct = _load_image_path(location=INPUT_PATH / "images/chest-ct")

    # Validate access to GPU
    _show_torch_cuda_info()

    # Run your algorithm here
    processor = ImageProcessor(config=config)
    malignancy_risks = processor.process(input_chest_ct, input_nodule_locations, input_clinical_information)

    # If ensemble weights are provided in config, apply logistic regression
    if hasattr(config, 'ensemble') and config.ensemble.use_logistic_regression:
        raw_predictions = np.array([pred['probability'] for pred in predictions['points']])
        ensemble_predictions = apply_logistic_regression(
            raw_predictions,
            config.ensemble.weights,
            config.ensemble.intercept
        )

        # Update predictions with ensemble results
        for i, point in enumerate(malignancy_risks['points']):
            point['probability'] = float(ensemble_predictions[i])

    # Save your output
    _write_json_file(
        location=OUTPUT_PATH / "lung-nodule-malginancy-likelihoods.json",
        content=malignancy_risks,
    )
    print(f"Completed writing output to {OUTPUT_PATH}")
    print(f"Output: {malignancy_risks}")
    return 0
