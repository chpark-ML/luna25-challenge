import logging
import os
from glob import glob
from pathlib import Path

import hydra
import omegaconf
import torch

from shared_lib.utils.utils import get_device, get_torch_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_BEST_CHECKPOINT_NAME = "model"
_LOSS_CHECKPOINT_POSTFIX = "loss"
_AUROC_CHECKPOINT_POSTFIX = "auroc"
_FINAL_CHECKPOINT_POSTFIX = "final"
_WEIGHT_SUFFIX = ".pt.enc"
_REPRESENTATIVE_MODEL_NAME = "model_repr"
_ABSOLUTE_TOLERANCE = 1e-01
_RELATIVE_TOLERANCE = 1e-05

_THIS_DIR = os.path.dirname(os.path.realpath(__file__))


def main() -> None:
    logger.info("Encrypt weight file and export TorchScript.")

    list_prefix = [_THIS_DIR + f"/outputs/default/cv_fine_val_fold{fold_index}" for fold_index in range(6)]
    print(list_prefix)

    for prefix in list_prefix:
        path_to_load_weights = glob(prefix + "/*.pth")

        # Skip interrupted or invalid checkpoints
        path_to_load_weights = [p for p in path_to_load_weights if "keyboard" not in p]

        for path_to_load_weight in path_to_load_weights:
            path_to_load_weight = Path(path_to_load_weight)
            ckpt_name = path_to_load_weight.stem

            # Load model config
            cfg = omegaconf.OmegaConf.load(prefix + "/.hydra/config.yaml")
            cfg_model = cfg.model[_REPRESENTATIVE_MODEL_NAME]
            if hasattr(cfg_model.classifier, "return_logit"):
                # If the model has a return_logit attribute, set it to True
                cfg_model.classifier.return_logit = True

            # Dummy input for tracing
            sample = torch.rand((1, 1, 48, 72, 72), dtype=torch.float32)

            # Load model and weights
            model = hydra.utils.instantiate(cfg_model, _recursive_=True)
            model = get_torch_model(model, path_to_load_weight)
            model.eval()

            # Get available devices (cpu/gpu)
            dict_device = get_device(device_idx=0)

            for device_type, device in dict_device.items():
                if device is None:
                    continue

                sample_device = sample.to(device)
                model_device = model.to(device)

                # Determine TorchScript file name
                if _LOSS_CHECKPOINT_POSTFIX in ckpt_name:
                    torchscript_name = f"{device.type}_model_loss.ts"
                elif _AUROC_CHECKPOINT_POSTFIX in ckpt_name:
                    torchscript_name = f"{device.type}_model_auroc.ts"
                elif _FINAL_CHECKPOINT_POSTFIX in ckpt_name:
                    torchscript_name = f"{device.type}_model_final.ts"
                else:
                    continue  # skip unknown checkpoint

                torchscript_path = os.path.join(prefix, torchscript_name)

                with torch.no_grad():
                    # Trace the model
                    traced_model = torch.jit.trace(model_device, sample_device)

                    # Save TorchScript model
                    traced_model.save(torchscript_path)
                    logger.info(f"TorchScript model saved: {torchscript_path}")

                    # Load back and run inference for consistency check
                    original_output = model_device(sample_device)
                    loaded_model = torch.jit.load(torchscript_path, map_location=device)
                    loaded_model.eval()
                    reloaded_output = loaded_model(sample_device)

                    # Compare original and reloaded outputs
                    if not torch.allclose(
                        original_output, reloaded_output, rtol=_RELATIVE_TOLERANCE, atol=_ABSOLUTE_TOLERANCE
                    ):
                        logger.warning(f"[WARN] Mismatch in outputs for: {torchscript_path}")
                        logger.warning(f"Max abs diff: {(original_output - reloaded_output).abs().max().item():.5f}")
                    else:
                        logger.info(f"Output check passed for: {torchscript_path}")


if __name__ == "__main__":
    main()
