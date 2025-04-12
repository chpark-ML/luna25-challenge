import logging
import os

import GPUtil
import pytest
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session", autouse=True)
def setenv():
    device_ids = GPUtil.getAvailable(order="memory", limit=1, maxLoad=0.5, maxMemory=0.5)
    if device_ids:
        logger.info(f"\nAvailable device to use: {device_ids}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_ids[0])
        logger.info(f"Set environtment variable 'CUDA_VISIBLE_DEVICES': {device_ids[0]}")
    else:
        logger.warning("There's no available GPU at this moment.")

    # torch script 모델의 GPU mode 에서 deterministic 연산 결과를 보장하기 위한 환경 변수 추가.
    # torch script 모델의 CPU mode 에서는 필요하지 않음.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Make execution deterministic.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
