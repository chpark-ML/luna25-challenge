import logging
import os

_THIS_DIR = os.path.dirname(os.path.realpath(__file__))


def setup_logger(save_path=None):
    # init root logger
    # If a specific library (e.g., albumentations) has added handlers to the root logger, remove them and apply new settings
    root_logger = logging.getLogger("")
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # set up logging to file - see previous section for more details
    save_path = save_path if save_path is not None else os.path.join(_THIS_DIR, "default.log")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
        handlers=[logging.FileHandler(save_path), logging.StreamHandler()],
    )
