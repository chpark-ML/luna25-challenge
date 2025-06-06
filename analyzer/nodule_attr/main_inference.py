import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from shared_lib.enums import RunMode
from shared_lib.utils.utils import print_config, set_seed

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
        dict_probs = processor.inference(
            loaders[mode], mode=mode, sanity_check=config.sanity_check
        )
        breakpoint()


if __name__ == "__main__":
    main()
