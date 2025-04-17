import hydra
from omegaconf import DictConfig

from inference import run

@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def main(config: DictConfig):
    models = dict()
    for model_indicator, config_model in config.model.items():
        models[model_indicator] = hydra.utils.instantiate(config_model)

    return run(models=models, mode=config.mode)


if __name__ == "__main__":
    raise SystemExit(main())
