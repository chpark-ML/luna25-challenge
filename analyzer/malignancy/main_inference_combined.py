import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn import metrics

from shared_lib.enums import RunMode
from shared_lib.utils.utils import print_config, set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="configs", config_name="config_inference_combined")
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
    loaders = dict()
    for mode in run_modes:
        dataset_cfg = config.loader.dataset
        patched_datasets_cfg = [OmegaConf.merge(ds_cfg, {"mode": mode}) for ds_cfg in dataset_cfg.datasets]
        combined_dataset = hydra.utils.instantiate(dataset_cfg, datasets=patched_datasets_cfg)
        loaders[mode] = hydra.utils.instantiate(
            config.loader,
            dataset=combined_dataset,
            drop_last=(mode == RunMode.TRAIN),
            shuffle=(mode == RunMode.TRAIN),
        )

    # Initialize an empty list to store all results
    all_results = []

    # get inference results
    for mode in run_modes:
        print(f"Mode: {mode}")
        probs, annots, annot_ids, dict_probs = processor.inference(
            loaders[mode], mode=mode, do_tta_by_size=config.do_tta_by_size, sanity_check=config.sanity_check
        )

        # get auroc score
        auroc = metrics.roc_auc_score(annots, probs)
        print(f"Auroc score: {auroc}")

        # Prepare DataFrame data
        df_data = {
            "mode": [mode.value] * len(annot_ids),
            "annot_ids": annot_ids,
            "annotation": annots,
            "prob_ensemble": probs,
        }
        for model_name, model_probs in dict_probs.items():
            df_data[f"prob_{model_name}"] = model_probs

        # Append to the combined results list
        all_results.append(pd.DataFrame(df_data))

    # Combine all results into a single DataFrame
    combined_df = pd.concat(all_results, ignore_index=True)

    # Save to a single CSV file
    df_path = Path(f"result_{config.run_name}.csv")
    combined_df.to_csv(df_path, index=False)
    print(f"Results saved to: {df_path}")

    return 0


if __name__ == "__main__":
    main()
