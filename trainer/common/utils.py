import io
import os
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from cryptography.fernet import Fernet
from omegaconf import DictConfig, OmegaConf

ENCRYPTION_KEY = ""


def freeze_layers(model, do_freeze_all: bool, target_attr_to_train: list):
    if do_freeze_all:
        for name, param in model.named_parameters():
            param.requires_grad = False

    for name, param in model.named_parameters():
        if any(j_attr in name for j_attr in target_attr_to_train):
            param.requires_grad = True


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


def save_encrypted_checkpoint(
    trace: torch.jit._script.RecursiveScriptModule,
    key: bytes,
    path_to_save: str = "",
    file_name: str = "project.pt.enc",
) -> None:
    # CPU trace
    buff = io.BytesIO()
    torch.jit.save(trace, buff)
    buff.seek(0)
    enc = Fernet(key).encrypt(buff.read())

    # save encrypted trace_fn
    file_path = os.path.join(path_to_save, file_name)
    torch.save(enc, file_path)


def get_decrypted_model(
    key: bytes, path_to_load: str = "", file_name: str = ""
) -> torch.jit._script.RecursiveScriptModule:
    # Decryption using the model key.
    file_path = os.path.join(path_to_load, file_name)
    buff = io.BytesIO(Fernet(key).decrypt(torch.load(file_path)))
    trace = torch.jit.load(buff)

    return trace


def load_model(weight_path, model_config=None):
    if Path(weight_path).suffix == ".enc":
        fn = get_decrypted_model(
            key=ENCRYPTION_KEY,
            path_to_load=str(Path(weight_path).parents[0]),
            file_name=str(Path(weight_path).name),
        )
    elif Path(weight_path).suffix in [".pt", ".pth"]:  # product
        cfg = OmegaConf.load(model_config) if isinstance(model_config, str) else model_config
        fn = hydra.utils.instantiate(cfg)
        fn = get_torch_model(fn, weight_path)
    else:
        assert False, "Wrong weight path"

    return fn
