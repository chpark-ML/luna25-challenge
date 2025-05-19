import logging
import os
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn import metrics

from shared_lib.enums import RunMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="configs", config_name="config_inference")
def main(config: DictConfig):
    # runner
    models = dict()
    for model_indicator, config_model in config.models.items():
        models[model_indicator] = hydra.utils.instantiate(config_model)
    malignancy_processor = hydra.utils.instantiate(config.processor, models=models)

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

    # get inference results
    probs, annots, dict_probs = malignancy_processor.inference(loaders[RunMode.TEST], sanity_check=config.sanity_check)

    # get auroc score
    auroc = metrics.roc_auc_score(annots, probs)
    print(f"Auroc score: {auroc}")

    # Prepare DataFrame
    df_data = {
        "annotation": annots,
        "prob_ensemble": probs,
    }
    for model_name, model_probs in dict_probs.items():
        df_data[f"prob_{model_name}"] = model_probs
    df = pd.DataFrame(df_data)

    # Save to DataFrame
    output_dir = Path(config.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    df_path = output_dir / f"result_{config.run_name}_auroc{auroc:.4f}.csv"
    df.to_csv(df_path, index=False)
    print(f"Results saved to: {df_path}")


if __name__ == "__main__":
    main()
