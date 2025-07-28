from pathlib import Path
from typing import Union

import torch

from shared_lib.model_output import ModelOutputCls, ModelOutputClsSeg
from shared_lib.model_runner.base_runner import ModelBaseTorchscript


class NoduleAttrRunner(ModelBaseTorchscript):
    def __init__(self, root_path, exp_name, file_name, device: Union[str, torch.device] = None):
        checkpoint_path = Path(root_path) / exp_name / file_name
        super().__init__(checkpoint_path=checkpoint_path, device=device)

    @torch.no_grad()
    def get_prediction(self, input_tensor) -> Union[ModelOutputCls, ModelOutputClsSeg]:
        """
        Perform inference with no gradient tracking.
        Args:
            input_tensor (torch.Tensor): Input data tensor.
        Returns:
            torch.Tensor: Model output.
        """
        input_tensor = input_tensor.to(self.device)
        output = self.model(input_tensor)

        return output
