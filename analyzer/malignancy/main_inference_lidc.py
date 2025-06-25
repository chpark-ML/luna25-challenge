import logging

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from data_lake.constants import DataLakeKey
from data_lake.dataset_handler import DatasetHandler
from shared_lib.enums import RunMode
from shared_lib.utils.utils import print_config, set_seed
from trainer.downstream.datasets.luna25 import DataLoaderKeys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="configs", config_name="config_inference_lidc")
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

    # inference and save nodule attributes
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
            attr_lists = [pred_results[key] for key in config.target_attr_total]  # nodule attributes
            for values in zip(col_ids, doc_ids, *attr_lists):
                col_id, doc_id, *attrs = values
                updated_cols = [attr for attr in config.target_attr_total]

                # nodule attr features
                _features = {
                    DataLakeKey.COLLECTION: col_id,
                    DataLakeKey.DOC_ID: doc_id,
                    **{key: value for key, value in zip(config.target_attr_total, attrs)},
                }

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
