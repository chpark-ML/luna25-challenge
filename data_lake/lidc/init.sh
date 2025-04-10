#!/bin/bash

dicom_path="/team/team_blu3/lung/data/2_public/LIDC-IDRI-new/volumes/manifest-1600709154662/LIDC-IDRI"
config_file="$HOME/.pylidcrc"
echo "[dicom]" > $config_file
echo "path = $dicom_path" >> $config_file
echo "warn = True" >> $config_file
cat $config_file
chmod +x $config_file
