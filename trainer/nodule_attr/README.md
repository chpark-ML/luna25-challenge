# Trainer - Nodule Attribute Module

This directory contains scripts and configurations for training and evaluating nodule attribute prediction models in the LUNA25 Challenge. It supports flexible experimentation with various model architectures, datasets, and training strategies for nodule attribute tasks.

## Directory Structure

```
nodule_attr/
  configs/      # Configuration files for models, loaders, trainers, etc.
  datasets/     # Dataset classes for LIDC and nodule attribute tasks
  scripts/      # Shell scripts for training, cross-validation, and ablation studies
  train.py      # Main training script for nodule attribute models
  convert_to_torchscript.py # Script to export models to TorchScript
  attr_main_test.py # Test script for nodule attribute pipeline
  main.py       # Entry point for training
```

---

## Main Components

- **configs/**: YAML configuration files for models, data loaders, optimizers, schedulers, and trainers.
- **datasets/**: Dataset classes for LIDC and nodule attribute tasks, supporting flexible data loading and augmentation.
- **scripts/**: Shell scripts for running experiments, cross-validation, and ablation studies.
- **train.py**: Main script for training nodule attribute models.
- **convert_to_torchscript.py**: Exports trained models to TorchScript format for deployment.
- **attr_main_test.py**: Test script for validating the nodule attribute pipeline.
- **main.py**: Entry point for launching training jobs.

---

## Typical Workflow
1. **Configure**: Edit YAML files in `configs/` to set up your experiment (model, dataset, augmentation, optimizer, etc.).
2. **Train**: Run `train.py` or `main.py` to start model training.
3. **Cross-Validation & Ablation**: Use scripts in `scripts/` for cross-validation and ablation studies.
4. **Export**: Use `convert_to_torchscript.py` to export trained models for inference.

---

## Requirements
- Python 3.7+
- PyTorch, omegaconf, pandas, numpy, scikit-learn, tqdm, and other dependencies as specified in the project root.

---

## Usage Example

```bash
# Example: Train a nodule attribute model
cd trainer/nodule_attr
python train.py --config configs/config.yaml

# Example: Export a trained model to TorchScript
python convert_to_torchscript.py
```

---

## Contact
For questions or issues, please refer to the main project repository or contact the maintainers. 