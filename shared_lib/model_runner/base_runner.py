from abc import ABC, abstractmethod
from typing import Union

import torch

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
