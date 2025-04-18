import importlib
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type, TypeVar, Union

import hydra
import numpy as np
import omegaconf
import optuna
import torch
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp

from shared_lib.enums import RunMode
from shared_lib.utils.utils import get_torch_device_string, print_config, set_config
from trainer.common.experiment_tool import load_logging_tool
from trainer.common.sampler import make_weights_for_balanced_classes
from trainer.common.scheduler_tool import SchedulerTool

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="Trainer")


def get_loaders(config):
    # Data Loaders
    logger.info(f"Instantiating dataloader <{config.loader._target_}>")
    run_modes = [RunMode(m) for m in config.run_modes] if "run_modes" in config else [x for x in RunMode]

    loaders = dict()
    for mode in run_modes:
        if mode == RunMode.TRAIN:
            dataset = hydra.utils.instantiate(config.loader.dataset, mode=mode)
            _sampler = None
            if dataset.use_weighted_sampler:
                train_df = dataset.dataset
                weights = make_weights_for_balanced_classes(train_df.label.values)
                weights = torch.DoubleTensor(weights)
                _sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_df))

            loaders[mode] = hydra.utils.instantiate(
                config.loader,
                dataset={"mode": mode},
                sampler=_sampler,
                drop_last=True,
                shuffle=False,
            )

        else:
            loaders[mode] = hydra.utils.instantiate(
                config.loader,
                dataset={"mode": mode},
                drop_last=False,
                shuffle=False,
            )

    return loaders


def get_trainer(config, loaders, logging_tool, optuna_trial=None):
    # Trainers
    module_name, _, trainer_class = config.trainer._target_.rpartition(".")
    module = importlib.import_module(module_name)
    class_ = getattr(module, trainer_class)
    trainer = class_.instantiate_trainer(config, loaders, logging_tool, optuna_trial=optuna_trial)

    return trainer


@dataclass
class Metrics(ABC):
    """
    dataclass for storing evaluation metrics.
    should define a representative metric for evaluating the intermediate result.
    """

    @abstractmethod
    def get_representative_metric(self):
        """
        Returns: float type evaluation metric
        """


class Trainer(ABC):
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        logging_tool,
        gpus,
        fast_dev_run,
        max_epoch,
        log_every_n_steps=1,
        test_epoch_start=0,
        resume_from_checkpoint=False,
        benchmark=False,
        deterministic=True,
        fine_tune_info=None,
        early_stop_patience: int = None,
        use_amp: bool = True,
        optuna_trial: optuna.Trial = None,
    ):
        self.model = model
        if not hasattr(self, "repr_model_name"):
            self.repr_model_name = None
        self.dict_threshold = dict()
        self.path_best_model = None
        self.epoch_best_model = 0
        self.optimizer = optimizer
        if isinstance(scheduler, omegaconf.DictConfig):
            self.scheduler = dict()
            for key, value in scheduler.items():
                self.scheduler[key] = SchedulerTool(value)
        else:
            self.scheduler = SchedulerTool(scheduler)
        self.criterion = criterion
        self.logging_tool = logging_tool
        self.use_amp = use_amp
        self.scaler = amp.GradScaler()

        self.epoch = 0
        self.resume_epoch = 0

        # Training configurations
        self.max_epoch = max_epoch
        self.fast_dev_run = fast_dev_run
        self.log_every_n_steps = log_every_n_steps
        self.test_epoch_start = test_epoch_start
        self.resume_from_checkpoint = resume_from_checkpoint

        # This constant is used in fit function for counting epochs for early stop functionality.
        self.early_stop_patience = early_stop_patience if early_stop_patience else max_epoch
        self.optuna_trial = optuna_trial

        torch_device = get_torch_device_string(gpus)
        if torch_device.startswith("cuda"):
            cudnn.benchmark = benchmark
            cudnn.deterministic = deterministic
        logger.info(f"Using torch device: {torch_device}")
        self.device = torch.device(torch_device)

        if isinstance(model, (omegaconf.DictConfig, dict)):
            self.model = {key: sub_model.to(self.device) for key, sub_model in model.items()}
        else:
            self.model = model.to(self.device)

        # Load pretrained encoder
        self.fine_tune_info = fine_tune_info
        if fine_tune_info.pretrained_weight_path:
            self.load_pretrained_weight(fine_tune_info.pretrained_weight_path)

    @classmethod
    @abstractmethod
    def instantiate_trainer(cls: Type[T], config: omegaconf.DictConfig, loaders, logging_tool) -> T:
        """
        Concrete methods should hydrate Trainer and return it.
        Args:
            config: configs.
            loaders: data loader
            logging_tool

        Returns: Trainer object
        """

    @staticmethod
    def set_seed(seed):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True)
        # If the above line is uncommented, we get the following RuntimeError:
        #  max_pool3d_with_indices_backward_cuda does not have a deterministic implementation
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
        try:
            import cupy as cp

            cp.random.seed(seed)
        except ImportError:
            pass

    def load_pretrained_weight(self, pretrained_weight_path, model_name=None):
        model_name = model_name or self.repr_model_name
        assert os.path.exists(pretrained_weight_path), f"{pretrained_weight_path} doesn't exists"
        checkpoint = torch.load(pretrained_weight_path, map_location=self.device)
        if model_name is not None:
            model_dict = self.model[model_name].state_dict()
        else:
            model_dict = self.model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in checkpoint["model"].items() if k in model_dict}

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)

        # 3. load the new state dict
        if model_name is not None:
            self.model[model_name].load_state_dict(model_dict)
        else:
            self.model.load_state_dict(model_dict)
        logger.info(f"Weight loaded from {pretrained_weight_path}")

    def load_checkpoint(self, model_path, model_name=None):
        """Loads checkpoint from directory.

        Args:
            model_path (str): Path to the checkpoint file.
            model_name (optional): Specific model name to load.
        """
        assert os.path.exists(model_path), f"Checkpoint file not found: {model_path}"
        model_name = model_name or self.repr_model_name

        checkpoint = torch.load(model_path, map_location=self.device)

        if model_name is not None and isinstance(self.model, (omegaconf.DictConfig, dict)):
            assert model_name in self.model, f"Model name {model_name} not found in self.model"
            self.model[model_name].load_state_dict(checkpoint["model"], strict=True)
        else:
            self.model.load_state_dict(checkpoint["model"], strict=True)

        logger.info(f"Model loaded from {model_path}")

        # Load optimizer state if present
        if self.optimizer is not None and "optimizer" in checkpoint:
            if isinstance(self.optimizer, (omegaconf.DictConfig, dict)) and model_name is not None:
                assert model_name in self.optimizer, f"Optimizer for {model_name} not found"
                self.optimizer[model_name].load_state_dict(checkpoint["optimizer"])
            else:
                self.optimizer.load_state_dict(checkpoint["optimizer"])

        # Load scaler state if present
        if hasattr(self, "scaler") and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        # Load additional threshold values if present
        if hasattr(self, "dict_threshold"):
            for i_key in checkpoint.keys():
                if "threshold_" in i_key:
                    self.dict_threshold[i_key] = checkpoint[i_key]

    def save_checkpoint(self, model_path, model_name=None, thresholds=None):
        model_name = model_name or self.repr_model_name
        checkpoint = {
            "epoch": self.epoch,
            "model": (
                self.model[model_name].state_dict()
                if model_name and isinstance(self.model, (omegaconf.DictConfig, dict))
                else self.model.state_dict()
            ),
            "optimizer": (
                self.optimizer[model_name].state_dict()
                if model_name and isinstance(self.optimizer, (omegaconf.DictConfig, dict))
                else self.optimizer.state_dict()
            ),
        }

        if self.use_amp:
            checkpoint["scaler"] = self.scaler.state_dict()

        if thresholds is not None:
            for key, value in thresholds.items():
                checkpoint[key] = value

        torch.save(checkpoint, model_path)

    @abstractmethod
    def train_epoch(self, epoch, loader):
        """Training for one epoch and return a metrics object."""
        if isinstance(self.model, dict):
            for k, v in self.model.items():
                v.train()
        else:
            self.model.train()

    @abstractmethod
    def validate_epoch(self, epoch, loader):
        """Validation for one epoch and return metrics object"""
        if isinstance(self.model, dict):
            for k, v in self.model.items():
                v.eval()
        else:
            self.model.eval()

    @abstractmethod
    def test_epoch(self, epoch, loader, export_results=False):
        """Test the epoch and return metrics object"""
        if isinstance(self.model, dict):
            for k, v in self.model.items():
                v.eval()
        else:
            self.model.eval()

    @abstractmethod
    def optimizing_metric(self, metrics) -> float:
        """Return the metrics to optimize. This can be used by Optuna when run with --multirun"""

    @abstractmethod
    def get_lr(self):
        """LR"""

    @abstractmethod
    def get_initial_model_metric(self) -> object:
        """Define what to collect"""

    @abstractmethod
    def save_best_metrics(
        self,
        val_metrics: Union[object, dict],
        best_model_metrics: Union[object, dict],
        epoch,
    ) -> (object, bool):
        """Save best metrics and return best metrics and whether it better metrics was found"""

    def run_epoch(self, epoch, loaders, best_model_metrics: object):
        self.epoch = epoch

        # Train for one epoch
        start = time.time()
        train_metrics = self.train_epoch(epoch, loaders[RunMode.TRAIN])
        self.log_lr(
            self.get_lr(),
            epoch,
            log_prefix=f"[{epoch}/{self.max_epoch}]",
            mlflow_log_prefix="EPOCH",
        )
        self.log_metrics(
            RunMode.TRAIN.value,
            epoch,
            train_metrics,
            log_prefix=f"[{epoch}/{self.max_epoch}]",
            mlflow_log_prefix="EPOCH",
            duration=time.time() - start,
        )

        # Validation for one epoch
        val_metrics = self.validate_epoch(epoch, loaders[RunMode.VALIDATE])
        self.log_metrics(RunMode.VALIDATE.value, epoch, val_metrics, mlflow_log_prefix="EPOCH")

        if self.optuna_trial is not None:
            # Report intermediate objective value.
            self.optuna_trial.report(val_metrics.get_representative_metric(), epoch)

            # Handle pruning based on the intermediate value.
            if self.optuna_trial.should_prune():
                raise optuna.TrialPruned()

        # Test if possible
        if RunMode.TEST in loaders and epoch >= self.test_epoch_start:
            test_metrics = self.test_epoch(epoch, loaders[RunMode.TEST])
            self.log_metrics(RunMode.TEST.value, epoch, test_metrics, mlflow_log_prefix="EPOCH")

        # Save model and return metrics
        best_metrics, found_better = self.save_best_metrics(val_metrics, best_model_metrics, epoch)
        if found_better:
            self.log_metrics("best", epoch, best_metrics)

        return best_metrics, found_better

    def fit(self, loaders: dict):
        """
        Fit and make the model.
        Args:
            loaders: a dictionary of data loaders keyed by RunMode.
        Returns:
            Metric
        """
        logger.info(f"Size of datasets {dict((mode, len(loader.dataset)) for mode, loader in loaders.items())}")
        if self.resume_from_checkpoint:
            self.load_checkpoint(self.path_best_model)

        self.logging_tool.log_param("save_dir", os.getcwd())

        # Reset the counters
        self.epoch = 0
        self.resume_epoch = 0

        # Training loop
        patience = 0
        best_model_metrics = self.get_initial_model_metric()
        for epoch in range(
            self.resume_epoch,
            self.resume_epoch + 2 if self.fast_dev_run else self.max_epoch,
        ):
            best_model_metrics, found_better = self.run_epoch(
                epoch,
                loaders,
                best_model_metrics=best_model_metrics,
            )
            # Early Stop if patience reaches threshold.
            patience = 0 if found_better else patience + 1
            if patience >= self.early_stop_patience:
                logger.info(f"Met the early stop condition. Stopping at epoch # {epoch}.")
                break

        return best_model_metrics

    def test(self, loaders):
        # Test the checkpoints
        if os.path.exists(self.path_best_model):
            self.load_checkpoint(self.path_best_model)
        else:
            logger.info("The best model path has never been updated, initial model has been used for testing.")

        if RunMode.VALIDATE in loaders:
            best_model_test_metrics = self.test_epoch(
                self.epoch_best_model, loaders[RunMode.VALIDATE], export_results=True
            )
            self.log_metrics("checkpoint_val", None, best_model_test_metrics)

        if RunMode.TEST in loaders:
            best_model_test_metrics = self.test_epoch(self.epoch_best_model, loaders[RunMode.TEST], export_results=True)
            self.log_metrics("checkpoint_test", None, best_model_test_metrics)

    def log_metrics(
        self,
        run_mode_str: str,
        step,
        metrics: object,
        log_prefix="",
        mlflow_log_prefix="",
        duration=None,
    ):
        """Log the metrics to logger and to mlflow if mlflow is used. Metrics could be None if learning isn't
        performed for the epoch."""
        self.logging_tool.log_metrics(
            run_mode_str=run_mode_str,
            step=step,
            metrics=metrics,
            log_prefix=log_prefix,
            mlflow_log_prefix=mlflow_log_prefix,
            duration=duration,
        )

    def log_lr(self, lr: float, step, log_prefix="", mlflow_log_prefix="", duration=None):
        """Log the learning rate to logger and to mlflow if mlflow is used."""
        self.logging_tool.log_lr(lr, step, log_prefix, mlflow_log_prefix, duration)


def train(config: omegaconf.DictConfig, optuna_trial=None) -> object:
    """
    Train code that takes a config and returns the final metric.
    Args:
        config: Config that contains everything needed for training.
        optuna_trial (optuna.Trial, optional): Optuna trial object for hyperparameter optimization.

    Returns:
        Metric object defined in the deriving trainer module.
    """
    # combine to default configuration
    config: omegaconf.DictConfig = set_config(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        print_config(config, resolve=True)

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        Trainer.set_seed(config.seed)

    # Get dataset loaders
    loaders = get_loaders(config)

    # Init logging tool
    logging_tool = load_logging_tool(config=config)

    # Init trainer object
    trainer = get_trainer(config, loaders, logging_tool, optuna_trial=optuna_trial)

    # Run model training
    best_model_metrics, best_model_test_metrics = None, None
    try:
        # train
        if RunMode.TRAIN in loaders:
            best_model_metrics = trainer.fit(loaders)
        else:
            if hasattr(config, "path_best_model"):
                trainer.path_best_model = config.path_best_model

        # test
        trainer.test(loaders)

        # logging representative model
        repr_model_name = getattr(trainer, "repr_model_name", None)
        if isinstance(repr_model_name, list):
            for model_name in repr_model_name:
                logging_tool.log_model(trainer.model[model_name], model_name)
        elif repr_model_name is None:
            logging_tool.log_model(trainer.model, "model")
        else:
            logging_tool.log_model(trainer.model[repr_model_name], "model")

    except KeyboardInterrupt:
        # Save intermediate output on keyboard interrupt
        model_path = "keyboard-interrupted-final.pth"
        trainer.save_checkpoint(model_path)
        logging_tool.raise_keyboardinterrupt()

    except optuna.TrialPruned:
        if logging_tool:
            logging_tool.end_run()
        raise optuna.TrialPruned()

    except Exception as e:
        logger.error(f"Error while training: {e}")
        logger.exception(e)
        if logging_tool:
            logging_tool.raise_error(error=e)

    if logging_tool:
        logging_tool.end_run()

    optimizing_metric = trainer.optimizing_metric(best_model_metrics) if best_model_metrics else None
    logger.info(f"Optimizing metrics is {optimizing_metric}")

    return optimizing_metric
