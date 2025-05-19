import logging

import hydra
from omegaconf import DictConfig

from shared_lib.enums import RunMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="configs", config_name="config_inference")
def main(config: DictConfig):
    # instantiate model, data_loader, and runner
    models = dict()
    for model_indicator, config_model in config.models.items():
        models[model_indicator] = hydra.utils.instantiate(config_model)
    malignancy_processor = hydra.utils.instantiate(config.processor, models=models)
    run_modes = [RunMode(m) for m in config.run_modes] if "run_modes" in config else [x for x in RunMode]

    loaders = {
        mode: hydra.utils.instantiate(
            config.loader,
            dataset={"mode": mode},
            drop_last=(mode == RunMode.TRAIN),
            shuffle=(mode == RunMode.TRAIN),
        )
        for mode in run_modes
    }


if __name__ == "__main__":
    main()
