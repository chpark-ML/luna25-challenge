import logging
import os

_THIS_DIR = os.path.dirname(os.path.realpath(__file__))


def setup_logger(save_path=None):
    # init root logger
    # 특정 라이브러리(e.g., albumentations)가 root 로거에 핸들러를 추가한 경우, 이를 제거하고 새로운 설정을 적용
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
