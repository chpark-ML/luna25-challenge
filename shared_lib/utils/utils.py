import logging
import os
import random
import re
import socket
import warnings
from pathlib import Path
from typing import Union

import GPUtil
import numpy as np
import rich.syntax
import rich.tree
import torch
from omegaconf import DictConfig, OmegaConf
from rich.style import Style
from torch import Tensor, nn as nn


# Automatically detect project root by looking for a specific marker file or folder
def find_project_root(marker: str = "docker") -> Path:
    """Find the project root by looking for a marker (e.g., .git directory)."""
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Project root not found. '{marker}' not present in any parent directories.")


# Define project root and config file path
PROJECT_ROOT = find_project_root()
_DEFAULT_CONFIG_FILE = PROJECT_ROOT / "trainer" / "common" / "configs" / "config.yaml"

logger = logging.getLogger(__name__)


def get_host_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]


def get_torch_device_string(gpus):
    """
    Get torch device string based on GPU config.
    Args:
        gpus: one of ['auto', 'None', ','.join(\\d+)] otherwise an error will be raised.
            'auto': Select the first available GPU. If there is no available GPU, CPU setting will return.
            'None': 'cpu' is returned.
            0[,1,2,3]: a list of 'cuda:\\d' is returned.
    Returns:
        device string value that is to call torch.device(str) or a list of device strings in case multiple values are
         set.
    """
    if gpus == "auto":
        # GPU 사양에 따라 maxLoad, maxMemory의 적정값이 바뀔 수 있음.
        # GPUtil계열 package는 신형 GPU가 나올때마다 호환성 이슈가 빈번히 발생: 3090, 3080, RTX TITAN 세 군데에서 작동함,
        # 향후 새로운 GPU가 추가되면 (40xx?) 에러가 날 수 있음
        device_ids = GPUtil.getAvailable(order="memory", limit=1, maxLoad=0.5, maxMemory=0.5)
        return f"cuda:{device_ids[0]}" if device_ids else "cpu"
    elif isinstance(gpus, int):
        return f"cuda:{gpus}"
    elif gpus is None or gpus in ("None", "cpu"):
        return "cpu"
    elif re.match("^\\d(,\\d)*$", gpus):
        return [f"cuda:{num}" for num in gpus.split(",")]
    else:
        raise ValueError(f"Valid values for gpus is 'auto', 'None', or comma separated digits. invalid: {gpus}")


def print_config(config: DictConfig, resolve: bool = True) -> None:
    """
    Prints the content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """
    style = Style(color="white", bgcolor="black")
    tree = rich.tree.Tree(":gear: CONFIG", style=style, guide_style=style)

    for field in config.keys():
        branch = tree.add(field, style=style, guide_style=style)
        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    rich.print(tree)


def set_config(config: OmegaConf, default_config_path: str = _DEFAULT_CONFIG_FILE) -> OmegaConf:
    """
    Applies optional utilities, controlled by main config file:
    - disabling warnings
    - set debug-friendly mode
    - forcing debug friendly configuration
    - forcing multi-gpu friendly configuration

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        default_config_path (str): path of the default config to base on.
    """
    if default_config_path:
        config = OmegaConf.merge(OmegaConf.load(default_config_path), config)

    # Enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # Disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        logger.info("Disabling python warnings <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        logger.info("Running in debug mode <config.debug=True>")
        config.trainer.fast_dev_run = True
        config.experiment_tool.enable = False

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        if config.trainer.get("gpus"):
            # If env var is set, use the one that was chosen, and set the gpu number to 0
            device_id = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            config.trainer.gpus = 0 if device_id else _get_available_gpu()

    # if auto is set for trainer's gpu, assign it here
    if config.trainer.get("gpus") == "auto":
        config.trainer.gpus = _get_available_gpu()

    return config


def get_device(device_idx: int = None):
    device_ids: list = GPUtil.getAvailable(order="memory", limit=1, maxLoad=0.8, maxMemory=0.8, includeNan=False)
    assert len(device_ids) > 0, "There is no available GPU."
    dict_device = {"cpu": torch.device(f"cpu"),
                   "cuda": torch.device(f"cuda:{device_idx}") if device_idx is not None else torch.device(
                       f"cuda:{int(device_ids[0])}")}
    return dict_device


def _get_available_gpu():
    device_ids = GPUtil.getAvailable(order="memory", limit=1, maxLoad=0.5, maxMemory=0.5)
    return device_ids[0] if device_ids else "cpu"


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_torch_model(model: nn.Module, model_path: str) -> torch.nn.Module:
    """Loads checkpoint from directory"""
    assert os.path.exists(model_path)
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        # keys of checkpoint, (epoch, model, optimizer, scaler)
        model.load_state_dict(checkpoint["model"], strict=True)
    return model


@torch.no_grad()
def get_inference_result(
        fn: Union[nn.Module, torch.jit._script.RecursiveScriptModule],
        sample: Tensor,
        device: torch.device,
) -> torch.Tensor:
    if isinstance(fn, nn.Module):
        fn.to(device).eval()
    result = fn(sample.to(device))
    return result
