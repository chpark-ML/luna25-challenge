from pathlib import Path
from typing import Union

import torch

from shared_lib.model_runner.base_runner import ModelBaseTorchscript, ModelBaseTorchCheckpoint

LOGIT_KEY = "logit"


class MalignancyRunner(ModelBaseTorchscript):
    def __init__(self, root_path, exp_name, file_name, device: Union[str, torch.device] = None):
        checkpoint_path = Path(root_path) / exp_name / file_name
        super().__init__(checkpoint_path=checkpoint_path, device=device)

    @torch.no_grad()
    def get_prediction(self, input_tensor) -> torch.Tensor:
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


class MalignancyRunnerCheckpoint(ModelBaseTorchCheckpoint):
    def __init__(self, root_path, exp_name, file_name, device: Union[str, torch.device] = None):
        checkpoint_path = Path(root_path) / exp_name / file_name
        config_path = Path(root_path) / exp_name / ".hydra/config.yaml"
        super().__init__(config_path=config_path, checkpoint_path=checkpoint_path, device=device)

    @torch.no_grad()
    def get_prediction(self, input_tensor) -> torch.Tensor:
        """
        Perform inference with no gradient tracking.
        Args:
            input_tensor (torch.Tensor): Input data tensor.
        Returns:
            torch.Tensor: Model output.
        """

        input_tensor = input_tensor.to(self.device)
        output = self.model(input_tensor)

        return output[LOGIT_KEY][self.model.classifier.target_attr_downstream]

    @torch.no_grad()
    def get_intermediate_results(self, input_tensor) -> torch.Tensor:
        """
        Perform inference with no gradient tracking.
        Args:
            input_tensor (torch.Tensor): Input data tensor.
        Returns:
            torch.Tensor: Model output.
        """

        input_tensor = input_tensor.to(self.device)
        output = self.model(input_tensor)
        # output[LOGIT_KEY][self.model.classifier.target_attr_downstream]  # (B, 1)
        # output[GATE_KEY][0]  # (B, 1, 24, 36, 36)
        # output[GATE_KEY][1]  # (B, 1, 12, 18, 18)
        # output[GATE_KEY][2]  # (B, 1, 6, 9, 9)
        # output[GATED_LOGIT_KEY][self.model.classifier.target_attr_downstream][0]  # (B, 1, 24, 36, 36)
        # output[GATED_LOGIT_KEY][self.model.classifier.target_attr_downstream][1]  # (B, 1, 12, 18, 18)
        # output[GATED_LOGIT_KEY][self.model.classifier.target_attr_downstream][2]  # (B, 1, 6, 9, 9)

        return output
