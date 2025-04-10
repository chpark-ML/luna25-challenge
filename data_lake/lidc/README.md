# LIDC-IDRI database

Codebase for constructing LIDC-IDRI data based on [pylidc library](https://pylidc.github.io/).

## Motivation

기존 nodule segmentation model 학습을 위한 데이터셋 구축은 4명의 rater에 의해서 제공되는 annotation을 독립적인 sample로 간주하였지만, 
해당 방법은 동일한 입력 이미지에 대해서 상이한 annotation mask로부터 모델을 학습하게 하고, 학습된 모델의 결과는 랜덤성에 상당히 의존하게 된다.
이를 해소하고 모델 성능 향상을 위한 다양한 실험을 직관적이게 하기 위해서는 
물리적으로 동일하다고 판단되는 segmentation mask의 consensus가 반영될 필요가 있다고 판단하였다.

## pylidc Tutorial
#### 1. ```src/pylidc_tutorial/pylidc_tutorial.ipynb```
- pylidc에서 제공하는 클래스, 함수 및 변수에 대해서 이해하기 위한 notebook 

## Schema
#### 1. ``` src/upload_to_db.py``` 
- pylidc에서 제공하는 scan 및 annotation을 정리하고 MongoDB에 upload
#### 2. ``` src/mask_processing.py``` 
- Bounding Box 형태로 저장된 mask를 dicom image와 같은 space에 저장하여 학습/검증 데이터셋을 구축
#### 3. ``` src/resampling_image.py``` 
- Preprocessing (image resampling)
#### 4. ``` src/prepare_train.py```
- Patient 단위로 fold를 나누고, 노듈의 좌표 등을 MongoDB(lct/pylidc-nodule-cluster)에 업데이트
#### 5. ``` src/prepare_attributes.py```
- 노듈 attributes의 consensus feature를 계산하고, MongoDB(lct/pylidc-nodule-cluster)에 업데이트
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