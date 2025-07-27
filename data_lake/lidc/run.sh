cd /opt/challenge/data_lake/lidc/src

python3 upload_to_db.py         # mongoDB updates, hfile (dicom_pixels)
python3 mask_processing.py      # hfile (mask annotation)
python3 resampling_image.py     # hfile (dicom_pixels_resampled, mask_annotation_resampled)
python3 prepare_train.py        # mongoDB updates (fold, r_coord_zyx), histogram, correlation heatmap
python3 prepare_attributes.py   # mongoDB updates (consensus features), histogram
