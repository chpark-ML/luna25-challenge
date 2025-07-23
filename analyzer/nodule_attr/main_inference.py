import logging

import hydra
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from omegaconf import DictConfig
from radiomics import featureextractor
from tqdm import tqdm

from data_lake.constants import DataLakeKey
from data_lake.dataset_handler import DatasetHandler
from shared_lib.enums import RunMode
from shared_lib.radiomics import RadiomicsFeatureKeys
from shared_lib.utils.utils import print_config, set_seed
from trainer.downstream.datasets.constants import DataLoaderKeys

# disable PyRadiomics logger
for logger_name in ["radiomics", "radiomics.featureextractor", "radiomics.glcm"]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL + 1)  # Ignore all log levels
    logger.propagate = False  # Block propagation to parent logger
    for handler in logger.handlers:
        logger.removeHandler(handler)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="configs", config_name="config_inference")
def main(config: DictConfig):
    # print config
    print_config(config, resolve=True)

    # set seed
    set_seed()

    # runner
    models = dict()
    for model_indicator, config_model in config.models.items():
        models[model_indicator] = hydra.utils.instantiate(config_model)
    processor = hydra.utils.instantiate(config.processor, models=models)

    # run modes
    run_modes = [RunMode(m) for m in config.run_modes] if "run_modes" in config else [x for x in RunMode]

    # loader
    loaders = {
        mode: hydra.utils.instantiate(
            config.loader,
            dataset={"mode": mode},
            drop_last=(mode == RunMode.TRAIN),
            shuffle=(mode == RunMode.TRAIN),
        )
        for mode in run_modes
    }

    # inference and save nodule attributes and radiomic features
    for mode in run_modes:
        print(f"Mode: {mode}")

        # get inference results
        all_features = []
        for data in tqdm(loaders[mode]):
            # annotation id
            col_ids = data[DataLoaderKeys.COLLECTION_ID]
            doc_ids = data[DataLoaderKeys.DOC_ID]

            # prediction
            patch_image = data[DataLoaderKeys.IMAGE]

            # inference
            pred_results = processor.predict_ensemble_result(patch_image)  # (B, 1) or (B, 1, w, h, d)
            pred_results = {
                key: val.squeeze(1).detach().cpu().numpy() if isinstance(val, torch.Tensor) and val.dim() > 1 else val
                for key, val in pred_results.items()
            }

            # batch samples
            images = patch_image.squeeze(1).numpy()  # (B, w, h, d)
            masks = pred_results[config.mask_key]  # (B, w, h, d)
            attr_lists = [pred_results[key] for key in config.target_attr_total]  # nodule attributes
            for values in zip(col_ids, doc_ids, images, masks, *attr_lists):
                col_id, doc_id, image, mask, *attrs = values
                updated_cols = [attr for attr in config.target_attr_total]

                # nodule attr features
                _features = {
                    DataLakeKey.COLLECTION: col_id,
                    DataLakeKey.DOC_ID: doc_id,
                    **{key: value for key, value in zip(config.target_attr_total, attrs)},
                }

                # calculate radiomics
                if config.calcu_radiomics:
                    image_np = image
                    radiomics = None
                    for threshold in [0.5, 0.3, 0.1, 0.05, 0.01]:
                        mask_np = (mask > threshold).astype(np.uint8)
                        if mask_np.sum() <= 1:
                            continue
                        try:
                            image_sitk = sitk.GetImageFromArray(image_np)
                            mask_sitk = sitk.GetImageFromArray(mask_np)

                            spacing = [1.0, 0.67, 0.67]
                            image_sitk.SetSpacing(spacing[::-1])
                            mask_sitk.SetSpacing(spacing[::-1])

                            # PyRadiomics extractor initialization (YAML config is possible)
                            extractor = featureextractor.RadiomicsFeatureExtractor()

                            # Calculate radiomics
                            radiomics = extractor.execute(image_sitk, mask_sitk)
                            break
                        except ValueError as e:
                            logger.info(f"Threshold {threshold} failed: {e}")
                            continue

                    # convert to dict and update feature dict
                    features = RadiomicsFeatureKeys()
                    for group_name in features.__dataclass_fields__:
                        group_keys = getattr(features, group_name)
                        for group_key in group_keys:
                            value = radiomics[group_key] if radiomics is not None else 0.0
                            _features[group_key] = float(value.item()) if hasattr(value, "item") else float(value)
                            updated_cols.append(group_key)

                # update docs
                if config.update_docs:
                    df = pd.DataFrame([_features])
                    DatasetHandler().update_existing_docs(df, updated_cols, field_prefix="pred")

                if config.sanity_check:
                    break

            if config.sanity_check:
                break


if __name__ == "__main__":
    main()
