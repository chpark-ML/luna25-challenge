"""
Inference script for predicting malignancy of lung nodules
"""

import json
from glob import glob
from pathlib import Path

import torch
from processor import NoduleProcessor

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


def run(config_models, mode="3D"):
    # Read the inputs
    input_nodule_locations = _load_json_file(location=INPUT_PATH / "nodule-locations.json")
    input_clinical_information = _load_json_file(location=INPUT_PATH / "clinical-information-lung-ct.json")
    input_chest_ct = _load_image_path(location=INPUT_PATH / "images/chest-ct")

    # Validate access to GPU
    _show_torch_cuda_info()

    # Run your algorithm here
    processor = NoduleProcessor(
        config_models=config_models,
        mode=mode,
    )
    malignancy_risks = processor.process(input_chest_ct, input_nodule_locations, input_clinical_information)

    # Save your output
    _write_json_file(
        location=OUTPUT_PATH / "lung-nodule-malginancy-likelihoods.json",
        content=malignancy_risks,
    )
    print(f"Completed writing output to {OUTPUT_PATH}")
    print(f"Output: {malignancy_risks}")
    return 0
