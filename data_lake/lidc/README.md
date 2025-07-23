# LIDC-IDRI database

Codebase for constructing LIDC-IDRI data based on [pylidc library](https://pylidc.github.io/).

## Motivation

Previously, to construct a dataset for nodule segmentation model training, annotations provided by four raters were treated as independent samples. However, this approach causes the model to be trained on different annotation masks for the same input image, making the model's results highly dependent on randomness. To address this and to intuitively conduct various experiments for model performance improvement, it was deemed necessary to reflect the consensus of segmentation masks that are physically considered identical.

## pylidc Tutorial
#### 1. ```src/pylidc_tutorial/pylidc_tutorial.ipynb```
- Notebook to understand the classes, functions, and variables provided by pylidc

## Schema
#### 1. ``` src/upload_to_db.py``` 
- Organizes scans and annotations provided by pylidc and uploads them to MongoDB
#### 2. ``` src/mask_processing.py``` 
- Stores masks saved in Bounding Box format in the same space as dicom images to build training/validation datasets
#### 3. ``` src/resampling_image.py``` 
- Preprocessing (image resampling)
#### 4. ``` src/prepare_train.py```
- Splits folds by patient and updates nodule coordinates, etc. in MongoDB (lct/pylidc-nodule-cluster)
#### 5. ``` src/prepare_attributes.py```
- Calculates consensus features of nodule attributes and updates them in MongoDB (lct/pylidc-nodule-cluster)
#### 6. ``` src/visualization.py```
- Visualization of mask annotation paired images


## 7-Fold Splitting
``` src/prepare_train.py ```
1. Image-level attributes
* slice thickness
* spacing between slices, pixel spacing
* contrast
* num of nodule cluster
* manufacturer, model

2. nodule-level attributes ([reference](https://pylidc.github.io/_modules/pylidc/Annotation.html#Annotation))
* **(categorical)**
    internalStructure, calcification
* **(continuous)**
    margin, texture
* **(shape attribution)**
    subtlety, sphericity, lobulation, spiculation, malignancy, diameter, volume