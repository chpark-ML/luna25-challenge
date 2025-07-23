# Data Lake Module

This directory contains all data processing, management, and utility scripts for the LUNA25 Challenge, including dataset construction, preprocessing, database upload, and exploratory data analysis (EDA).

## Directory Structure

```
data_lake/
  lidc/        # LIDC-IDRI dataset processing and management
  luna25/      # LUNA25 dataset processing and management
  utils/       # Utility scripts for data conversion and resampling
  notebooks/   # Jupyter notebooks for EDA and data processing
  assets/      # CSV files and other data assets
  constants.py # Shared constants for data handling
  dataset_handler.py # Unified interface for dataset access and DB operations
```

---

## 1. LIDC Submodule (`lidc/`)

- **Purpose**: Scripts for constructing, processing, and uploading the LIDC-IDRI dataset using the [pylidc library](https://pylidc.github.io/).
- **Key Scripts**:
  - `src/upload_to_db.py`: Uploads consensus bounding boxes and masks to MongoDB.
  - `src/mask_processing.py`: Converts 3D volumes and segmentation annotations to training data.
  - `src/prepare_train.py`: Splits data into folds and prepares training/validation sets.
  - `src/prepare_attributes.py`: Calculates consensus features for nodule attributes.
  - `src/visualization.py`: Visualizes mask annotations and paired images.
- **Reference**: See `lidc/README.md` for detailed schema and motivation.

---

## 2. LUNA25 Submodule (`luna25/`)

- **Purpose**: Scripts for processing and uploading the LUNA25 dataset, including metadata handling and visualization.
- **Key Scripts**:
  - `upload_to_db.py`: Inserts LUNA25 metadata and processed data into MongoDB.
  - `resample.py`: Resamples images and masks to standard spacing.
  - `visualization.py`: Visualizes nodule volumes and saves 3D plots.
  - `nfs_to_local.sh`: Utility for copying data from NFS to local storage.

---

## 3. Utilities (`utils/`)

- **Purpose**: Helper scripts for data conversion and resampling.
- **Key Scripts**:
  - `itk_to_npy.py`: Converts ITK images to NumPy arrays.
  - `resample_image.py`: Resamples images to a target spacing.
  - `client.py`: MongoDB client utility.

---

## 4. Notebooks (`notebooks/`)

- **Purpose**: Jupyter notebooks for exploratory data analysis and data processing.
- **Key Notebooks**:
  - `EDA.ipynb`: General EDA for LUNA25 Challenge data.
  - `EDA_DB.ipynb`: EDA focused on database contents and label/gender distribution.
  - `EDA_features.ipynb`: Analysis of predicted nodule attributes and radiomics features.
  - `split_folds.ipynb`: Demonstrates k-fold splitting with class balance.

---

## 5. Assets (`assets/`)

- **Purpose**: Contains CSV files with metadata, patient information, and fold splits for the LUNA25 dataset.
- **Files**:
  - `LUNA25_Public_Training_Development_Data_with_metadata.csv`
  - `LUNA25_Public_Training_Development_Data_fold.csv`
  - `LUNA25_Public_Training_Development_Data_Patient.csv`

---

## 6. Shared Components

- `constants.py`: Centralized constants for DB, file paths, and data keys.
- `dataset_handler.py`: Unified interface for accessing datasets and performing DB operations.

---

## Typical Workflow
1. **Preprocessing**: Use scripts in `lidc/src/` and `luna25/` to preprocess and resample images/masks.
2. **Database Upload**: Run `upload_to_db.py` scripts to insert processed data and metadata into MongoDB.
3. **Visualization**: Use visualization scripts to generate 3D plots of nodules and masks.
4. **EDA**: Explore and analyze data using the provided Jupyter notebooks.

---

## Requirements
- Python 3.7+
- pandas, numpy, h5py, pymongo, scikit-learn, matplotlib, seaborn, tqdm, joblib, SimpleITK, and other dependencies as specified in the project root.

---

## Usage Example

```bash
# Example: Upload LIDC data to MongoDB
cd data_lake/lidc/src
python upload_to_db.py

# Example: Upload LUNA25 data to MongoDB
cd data_lake/luna25
python upload_to_db.py

# Example: Run EDA notebook
cd data_lake/notebooks
jupyter notebook EDA.ipynb
```

---

## Contact
For questions or issues, please refer to the main project repository or contact the maintainers. 