# LUNA25 Challenge

Site: https://luna25.grand-challenge.org/

The LUNA25 challenge is a grand challenge designed to evaluate the diagnostic performance of AI algorithms and radiologists in lung nodule malignancy risk estimation in screening CT.

---

## Dataset
- 2120 patients (National Lung Cancer Screening Trial (NLST) between 2002 and 2004 in one of the 33 centers in the United States)
- 4069 low-dose chest CT scans
  - 555 annotated malignant nodules
  - 5608 annotated benign nodules

---

## Project Structure

```
.
├── analyzer/      # Analysis pipelines for malignancy and nodule attribute models
├── data_lake/     # Data processing, database upload, and EDA scripts
├── docker/        # Dockerfiles and requirements for reproducible environments
├── shared_lib/    # Shared utilities, model runners, processors, and tools
├── submission/    # Inference and packaging scripts for challenge submission
├── test/          # Centralized pytest-based test suite
├── trainer/       # Model training code (common, downstream, nodule_attr)
├── scripts/       # Shell scripts for automation and utilities
├── pyproject.toml # Project configuration and dependencies
├── makefile       # Build and workflow automation
└── README.md      # Project overview (this file)
```

---

## Main Components

- **data_lake/**: Data ingestion, preprocessing, database upload, and EDA notebooks for LIDC and LUNA25 datasets.
- **analyzer/**: Analysis pipelines for malignancy and nodule attribute models, including ensemble, failure analysis, and visualization.
- **trainer/**: Model training code, including shared modules, downstream tasks, and nodule attribute models.
- **shared_lib/**: Shared utilities, model runners, processors, and tools used across the project.
- **submission/**: Inference and packaging scripts for generating challenge submissions.
- **test/**: Centralized directory for all pytest-based tests, including unit and integration tests.
- **docker/**: Dockerfiles and requirements for reproducible environments.
- **scripts/**: Shell scripts for automation and workflow support.

---

## Testing

All pytest-based tests are organized in the `test/` directory. To run all tests:

```bash
pytest test/
```

---

## Getting Started

1. **Install dependencies** (see `pyproject.toml` or Docker setup)
2. **Prepare data** using scripts in `data_lake/`
3. **Train models** using scripts in `trainer/`
4. **Analyze results** using `analyzer/`
5. **Run tests** using `pytest test/`
6. **Package and submit** using `submission/`

---

## Contact
For questions or issues, please refer to the main project repository or contact the maintainers.
 
