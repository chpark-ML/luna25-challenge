# Trainer - Common Module

This directory contains core components and utilities for model training in the LUNA25 Challenge. It provides shared modules for model architectures, loss functions, data augmentation, dataset handling, and training utilities.

## Directory Structure

```
common/
  configs/        # Default configuration files
  criterions/     # Loss functions (e.g., focal loss, classification loss)
  models/         # Model architectures (3D/2D UNet, ResNeSt, Swin, etc.)
  datasets/       # Dataset classes for LIDC, LUNA25, and combined datasets
  augmentation/   # Data augmentation modules
  train.py        # Main training script
  utils.py        # Utility functions
  enums.py        # Enum definitions
  constants.py    # Shared constants
  experiment_tool.py # Experiment management and logging
  scheduler_tool.py  # Learning rate scheduler utilities
  criterion.py    # Loss function wrapper
  sampler.py      # Data sampler
  model_config.sh # Shell script for model config management
```

---

## Main Components

- **configs/**: YAML configuration files for training and experiments.
- **criterions/**: Implements various loss functions, including focal loss and classification loss.
- **models/**: Contains model architectures such as 3D/2D UNet, ResNeSt, Swin, and custom modules.
- **datasets/**: Dataset classes for LIDC, LUNA25, and combined datasets, supporting flexible data loading.
- **augmentation/**: Data augmentation modules for cropping, flipping, rotation, rescaling, and more.
- **train.py**: Main training script supporting multi-GPU, mixed precision, and advanced logging.
- **experiment_tool.py**: Utilities for experiment tracking, logging, and reproducibility.
- **scheduler_tool.py**: Learning rate scheduler utilities.
- **sampler.py**: Custom data samplers for balanced training.

---

## Typical Workflow
1. **Configure**: Edit YAML files in `configs/` to set up your experiment (model, dataset, augmentation, optimizer, etc.).
2. **Train**: Run `train.py` to start model training.
3. **Experiment Management**: Use `experiment_tool.py` for logging and tracking experiments.

---

## Requirements
- Python 3.7+
- PyTorch, omegaconf, pandas, numpy, scikit-learn, tqdm, and other dependencies as specified in the project root.

---

## Usage Example

```bash
# Example: Train a model with default config
cd trainer/common
python train.py --config configs/config.yaml
```

---

## Contact
For questions or issues, please refer to the main project repository or contact the maintainers. 