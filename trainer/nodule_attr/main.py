import logging

import hydra
import omegaconf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def main(config: omegaconf.DictConfig) -> object:
    from trainer.common.train import train

    logger.info("Training Nodule Classification Model.")

    return train(config)


if __name__ == "__main__":
    main()
