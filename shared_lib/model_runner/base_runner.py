import os
from abc import ABC, abstractmethod
from typing import Union

import hydra
import torch
from omegaconf import OmegaConf

from shared_lib.utils.utils import get_device


class ModelBaseTorchscript(ABC):
    """
    Base abstract class for all the models that use Torchscript
    """

    def __init__(self, checkpoint_path=None, device=None):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.checkpoint_path = checkpoint_path
        self._build_model(training=False)

    def _build_model(self, training=False) -> None:
        self.model = torch.jit.load(self.checkpoint_path, map_location=self.device)
        if not training:
            self.model.eval()

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: Union[str, torch.device]):
        if device == "cpu":
            self._device = torch.device("cpu")
        elif device == "cuda":
            self._device = get_device()[device]
        elif isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
            # Handle numeric or numeric string
            device_index = int(device)
            self._device = torch.device(f"cuda:{device_index}")
        elif isinstance(device, torch.device):  # reallocate existing device
            self._device = device
        else:
            assert False, "Device setup is not supported."

    @abstractmethod
    def get_prediction(self, input_tensor):
        """
        Abstract method for getting predictions.
        Must be implemented in subclasses.
        """
        pass


class ModelBaseTorchCheckpoint(ABC):
    """
    Base abstract class for all the models that use Torchscript
    """

    def __init__(self, config_path=None, checkpoint_path=None, device=None):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

        self._build_model(training=False)

    def _build_model(self, training=False) -> None:
        # init
        # TODO: model.model_repr is hardcoded; consider making it configurable via a variable.
        model_config = OmegaConf.load(self.config_path).model.model_repr
        if hasattr(model_config, "return_downstream_logit"):
            model_config.return_downstream_logit = False
        if hasattr(model_config, "return_named_tuple"):
            model_config.return_named_tuple = False
        self.model = hydra.utils.instantiate(model_config)
        model_dict = self.model.state_dict()

        # load pretrained
        assert os.path.exists(self.checkpoint_path), f"{self.checkpoint_path} doesn't exists"
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in checkpoint["model"].items() if k in model_dict}

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)

        # 3. load the new state dict
        self.model.load_state_dict(model_dict)
        self.model.to(self.device)

        if not training:
            self.model.eval()

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: Union[str, torch.device]):
        if device == "cpu":
            self._device = torch.device("cpu")
        elif device == "cuda":
            self._device = get_device()[device]
        elif isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
            # Handle numeric or numeric string
            device_index = int(device)
            self._device = torch.device(f"cuda:{device_index}")
        elif isinstance(device, torch.device):  # reallocate existing device
            self._device = device
        else:
            assert False, "Device setup is not supported."

    @abstractmethod
    def get_prediction(self, input_tensor):
        """
        Abstract method for getting predictions.
        Must be implemented in subclasses.
        """
        pass
