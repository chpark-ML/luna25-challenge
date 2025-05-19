import hydra
from inference import run
from omegaconf import DictConfig


@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def main(config: DictConfig):
    return run(config=config)


if __name__ == "__main__":
    raise SystemExit(main())
