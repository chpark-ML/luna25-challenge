# Analyzer Module

This directory contains analysis pipelines and tools for the LUNA25 Challenge, focusing on two main areas:

- **Malignancy Analysis**: Ensemble modeling, inference, failure analysis, and visualization for malignancy prediction models.
- **Nodule Attribute Analysis**: Inference and evaluation of nodule attribute prediction models, including radiomics feature extraction.

## Directory Structure

```
analyzer/
  malignancy/      # Malignancy prediction analysis and visualization
  nodule_attr/     # Nodule attribute prediction and analysis
```

---

## 1. Malignancy Module (`malignancy/`)

### Purpose
Provides scripts and tools for:
- Running inference with trained malignancy models
- Performing ensemble analysis and model weight optimization
- Analyzing and visualizing model failures
- Generating visualizations for model predictions

### Main Scripts
- `main_inference.py` / `main_inference_lidc.py` / `main_inference_combined.py`: Run inference on test/validation datasets using various model configurations.
- `main_model_weight.py`: Optimize and analyze ensemble weights for multiple models using validation/test results.
- `failure_analysis.py`: Perform comprehensive failure analysis on model predictions.
- `failure_case_visualization.py`: Visualize and summarize failure cases.
- `main_visualization.py` / `main_visualization_diff.py`: Generate visualizations for model predictions and compare differences.
- `update_model_config.py`: Update model configuration files with optimized weights.
- `analyzer_malignancy_inference_test.py`: Test script for inference pipeline.
- `analysis.ipynb`: Jupyter notebook for interactive analysis.

### Typical Workflow
1. **Inference**: Run `main_inference.py` or related scripts to generate prediction results.
2. **Ensemble Analysis**: Use `main_model_weight.py` to optimize ensemble weights and compare methods.
3. **Failure Analysis**: Use `failure_analysis.py` and `failure_case_visualization.py` to identify and visualize failure cases.
4. **Visualization**: Use `main_visualization.py` to generate visual summaries of predictions.

---

## 2. Nodule Attribute Module (`nodule_attr/`)

### Purpose
Provides scripts and tools for:
- Running inference for nodule attribute prediction models
- Extracting radiomics features from predicted masks
- Evaluating and analyzing attribute prediction performance

### Main Scripts
- `main_inference.py`: Run inference and extract nodule attributes and radiomics features.
- `main_eval.py`: Evaluate model predictions and compute metrics.
- `eval_result_analysis.ipynb`: Jupyter notebook for detailed result analysis.
- `qualitative_analysis.ipynb`: Jupyter notebook for qualitative review of predictions.
- `scripts/`: Contains shell scripts for running and debugging inference pipelines.

### Typical Workflow
1. **Inference**: Run `main_inference.py` to predict nodule attributes and extract radiomics features.
2. **Evaluation**: Use `main_eval.py` to evaluate predictions and compute metrics.
3. **Analysis**: Use the provided notebooks for in-depth analysis and visualization.

---

## Configuration

Both modules use Hydra and YAML configuration files (in the `configs/` subdirectories) to specify model, processor, and data loader settings. Adjust these configs to match your experiment setup.

---

## Requirements
- Python 3.7+
- PyTorch, Hydra, pandas, numpy, scikit-learn, matplotlib, seaborn, SimpleITK, PyRadiomics, and other dependencies as specified in the project root.

---

## Usage Example

```bash
# Example: Run malignancy model inference
cd analyzer/malignancy
python main_inference.py

# Example: Run nodule attribute inference
cd analyzer/nodule_attr
python main_inference.py
```

---

## Contact
For questions or issues, please refer to the main project repository or contact the maintainers. 