import io
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence, Union

import analyzer.common.model_base as model_base
import numpy as np
import torch

from shared_lib.utils.utils import get_device


def get_absolute_path(file_path):
    return str(Path(file_path).resolve())


class MalignancyModel:
    def __init__(self, device: Union[str, torch.device] = None, root_path=None, checkpoint_path=None, model_stem=None):
        assert device == "cpu" or isinstance(device, torch.device)
        self.device = device
        self.root_path = root_path
        if checkpoint_path is None:
            model_name = f"{self.device.type}_{model_stem}.pt.enc"
            default_checkpoint_path = os.path.join(model_base.WEIGHTS_DIR, model_stem, model_name)
            checkpoint_path = default_checkpoint_path

        super().__init__(device=device, checkpoint_path=checkpoint_path)
        self._build_model(training=False)

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: Union[str, torch.device]):
        if isinstance(device, str):  # create new device
            self._device = get_device()[device]
        elif isinstance(device, torch.device):  # reallocate existing device
            self._device = device
        else:
            assert False, "Device setup is not supported."

    @classmethod
    def _get_pt_path(self, checkpoint) -> str:
        return get_absolute_path(os.path.join(self.root_path, checkpoint))

    def _build_model(self, training=False) -> None:
        buff = torch.load(self.checkpoint_path, map_location=self.device)
        self.model = torch.jit.load(buff, map_location=self.device)
        if not training:
            self.model.eval()

    @torch.no_grad()
    def get_prediction(self, input):
        results = self.model(input)

        return results
