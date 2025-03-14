import json
import logging
import random
import time
from dataclasses import dataclass

import hydra
import numpy as np
import omegaconf
import torch
import tqdm
from sklearn import metrics

import projects.common.train as comm_train
from projects.common.enums import ModelName, RunMode, ThresholdMode
from projects.downstream.datasets.luna25 import DataKeys

logger = logging.getLogger(__name__)


@dataclass
class Metrics(comm_train.Metrics):
    loss: float = np.inf

    def __str__(self):
        return f"loss_{self.loss:.4f}"

    def get_representative_metric(self):
        """
        Returns: float type evaluation metric
        """
        return self.loss


def _check_any_nan(arr):
    if torch.any(torch.isnan(arr)):
        import pdb

        pdb.set_trace()


class Trainer(comm_train.Trainer):
    """Trainer to train model"""

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        thresholding_mode_representative,
        thresholding_mode,
        grad_clip_max_norm,
        **kwargs,
    ) -> None:
        self.repr_model_name = ModelName.CLASSIFIER
        super().__init__(model, optimizer, scheduler, criterion, **kwargs)
        self.thresholding_mode_representative = ThresholdMode.get_mode(thresholding_mode_representative)
        self.thresholding_mode = ThresholdMode.get_mode(thresholding_mode)
        self.grad_clip_max_norm = grad_clip_max_norm

    @classmethod
    def instantiate_trainer(
        cls, config: omegaconf.DictConfig, loaders, logging_tool, optuna_trial=None
    ) -> comm_train.Trainer:
        # Init model
        models = dict()
        if isinstance(config.model, (omegaconf.DictConfig, dict)):
            for model_indicator, config_model in config.model.items():
                logger.info(f"Instantiating model <{config.model[model_indicator]['_target_']}>")
                for model_name in ModelName:
                    if model_name.value in model_indicator:
                        models[model_name] = hydra.utils.instantiate(config_model)
        else:
            raise NotImplementedError

        optimizers = None
        schedulers = None
        if RunMode.TRAIN in loaders:
            # Set steps per epoch for the scheduler if specified
            if isinstance(config.scheduler, (omegaconf.DictConfig, dict)):
                for scheduler_indicator, config_scheduler in config.scheduler.items():
                    if "steps_per_epoch" in config_scheduler:
                        config.scheduler[scheduler_indicator]["steps_per_epoch"] = len(loaders[RunMode.TRAIN])
            else:
                raise NotImplementedError

            # Init optimizer, scheduler
            if isinstance(config.model, (omegaconf.DictConfig, dict)):
                optimizers = dict()
                schedulers = dict()
                for optim_indicator, config_optim in config.optim.items():
                    for model_name in ModelName:
                        if model_name.value in optim_indicator:
                            optimizers[model_name] = hydra.utils.instantiate(
                                config_optim, models[model_name].parameters()
                            )

                for scheduler_indicator, config_scheduler in config.scheduler.items():
                    for model_name in ModelName:
                        if model_name.value in scheduler_indicator:
                            schedulers[model_name] = hydra.utils.instantiate(
                                config_scheduler, optimizer=optimizers[model_name]
                            )
            else:
                raise NotImplementedError
                # optimizers = hydra.utils.instantiate(config.optim, models.parameters())
                # schedulers = hydra.utils.instantiate(config.scheduler, optimizer=optimizers)

        # Set criterion, and its alpha based on training data
        criterion = hydra.utils.instantiate(config.criterion)

        # Init trainer
        logger.info(f"Instantiating trainer <{config.trainer._target_}>")
        return hydra.utils.instantiate(
            config.trainer,
            model=models,
            optimizer=optimizers,
            scheduler=schedulers,
            criterion=criterion,
            logging_tool=logging_tool,
            grad_clip_max_norm=config.trainer.grad_clip_max_norm,
            thresholding_mode_representative=config.trainer.thresholding_mode_representative,
            thresholding_mode=config.trainer.thresholding_mode,
            use_amp=config.trainer.use_amp,
            optuna_trial=optuna_trial,
        )

    def train_epoch(self, epoch, loader) -> Metrics:
        super().train_epoch(epoch, loader)
        train_losses = []
        start = time.time()

        for i, data in enumerate(loader):
            global_step = epoch * len(loader) + i + 1
            self.optimizer[ModelName.CLASSIFIER].zero_grad()
            patch_image = data[DataKeys.IMAGE].to(self.device)
            _check_any_nan(patch_image)
            annot = data[DataKeys.LABEL].to(self.device)

            # forward propagation
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                output = self.model[ModelName.CLASSIFIER](patch_image)
                loss = self.criterion(output, annot, is_logit=True, is_logistic=True)
                train_losses.append(loss.detach())

            # set trace for checking nan values
            if torch.any(torch.isnan(loss)):
                import pdb

                pdb.set_trace()
                is_param_nan = torch.stack(
                    [torch.isnan(p).any() for p in self.model[ModelName.CLASSIFIER].parameters()]
                ).any()
                continue

            # Copy model parameters before backward pass and optimization step
            if self.fast_dev_run:
                param_before = {
                    name: param.clone() for name, param in self.model[ModelName.CLASSIFIER].named_parameters()
                }

            # Backpropagation
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer[ModelName.CLASSIFIER])
                torch.nn.utils.clip_grad_norm_(
                    self.model[ModelName.CLASSIFIER].parameters(), max_norm=self.grad_clip_max_norm
                )
                self.scaler.step(self.optimizer[ModelName.CLASSIFIER])
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model[ModelName.CLASSIFIER].parameters(), max_norm=self.grad_clip_max_norm
                )
                self.optimizer[ModelName.CLASSIFIER].step()

            # Check if any parameter has changed
            if self.fast_dev_run:
                for name, param in self.model[ModelName.CLASSIFIER].named_parameters():
                    if not torch.equal(param_before[name], param):
                        print(f"Parameter '{name}' has changed.")
                    else:
                        print(f"Parameter '{name}' remains unchanged.")

            if global_step % self.log_every_n_steps == 0:
                batch_time = time.time() - start
                self.log_metrics(
                    RunMode.TRAIN.value,
                    global_step,
                    Metrics(loss.detach()),
                    log_prefix=f"[{epoch}/{self.max_epoch}] [{i}/{len(loader)}]",
                    mlflow_log_prefix="STEP",
                    duration=batch_time,
                )
                start = time.time()

            if self.fast_dev_run:
                # Runs 1 train batch and program ends if 'fast_dev_run' set to 'True'
                break
            self.scheduler[ModelName.CLASSIFIER].step("step")
        self.scheduler[ModelName.CLASSIFIER].step("epoch")

        train_loss = torch.stack(train_losses).sum().item()
        return Metrics(train_loss / len(loader))

    def optimizing_metric(self, metrics: Metrics):
        return metrics.loss

    def get_lr(self):
        return self.optimizer[ModelName.CLASSIFIER].param_groups[0]["lr"]

    def get_initial_model_metric(self):
        return Metrics()

    def set_threshold(self, probs, annots, mode=ThresholdMode.YOUDEN):
        assert mode == ThresholdMode.YOUDEN or mode == ThresholdMode.F1 or mode == ThresholdMode.ALL
        dict_threshold = dict()

        best_f1 = -1
        y_true = (annots.detach().cpu().numpy() > 0.5) * 1.0
        if mode == ThresholdMode.YOUDEN or mode == ThresholdMode.ALL:
            # calculate roc curves
            fpr, tpr, thresholds = metrics.roc_curve(y_true, probs.detach().cpu().numpy())
            J = tpr - fpr
            ix = max(np.argmax(J), 1)
            best_thresh = thresholds[ix]
            dict_threshold[f"threshold_{ThresholdMode.YOUDEN.value}"] = best_thresh

        if mode == ThresholdMode.F1 or mode == ThresholdMode.ALL:
            for threshold in np.arange(0.0, 1.0, 0.01):
                y_pred = (probs.detach().cpu().numpy() > threshold).astype(int)
                f1 = metrics.f1_score(y_true, y_pred)

                if f1 > best_f1:
                    best_f1 = f1
                    dict_threshold[f"threshold_{ThresholdMode.F1.value}"] = threshold

        self.dict_threshold = dict_threshold

    def get_metrics(self, logits, probs, annots):
        losses = self.criterion(logits, annots, is_logit=True, is_logistic=True)
        result_dict = self.get_binary_classification_metrics(
            probs,
            annots,
            threshold=self.dict_threshold,
            threshold_mode=self.thresholding_mode_representative,
        )

        return losses.detach(), result_dict

    def get_binary_classification_metrics(
        self,
        prob: torch.Tensor,
        annot: torch.Tensor,
        threshold: dict,
        threshold_mode: ThresholdMode = ThresholdMode.YOUDEN,
    ):
        assert type(prob) == type(annot)
        result_dict = dict()
        _annot = (annot.squeeze().cpu().numpy() > 0.5) * 1.0
        _pred = prob.squeeze().cpu().numpy() > threshold[f"threshold_{threshold_mode.value}"]
        _prob = prob.squeeze().cpu().numpy()
        result_dict[f"acc"] = metrics.accuracy_score(_annot, _pred)
        try:
            result_dict[f"auroc"] = metrics.roc_auc_score(_annot, _prob)
        except ValueError:  # in the case when only one class exists, AUROC can not be calculated. (in fast_dev_run)
            pass
        result_dict[f"f1"] = metrics.f1_score(_annot, _pred)

        return result_dict

    def save_best_metrics(self, val_metrics: Metrics, best_metrics: Metrics, epoch) -> (object, bool):
        found_better = False
        if val_metrics.loss < best_metrics.loss:
            found_better = True
            model_path = f"model.pth"
            logger.info(
                f"loss improved from {best_metrics.loss:4f} to {val_metrics.loss:4f}, " f"saving model to {model_path}."
            )

            best_metrics = val_metrics
            self.path_best_model = model_path
            self.epoch_best_model = epoch
            self.save_checkpoint(model_path, thresholds=self.dict_threshold)

        return best_metrics, found_better

    def _inference(self, loader):
        list_logits = []
        list_annots = []

        for data in tqdm.tqdm(loader):
            # prediction
            patch_image = data[DataKeys.IMAGE].to(self.device)
            _check_any_nan(patch_image)

            # annotation
            annot = data[DataKeys.LABEL].to(self.device)

            # inference
            logits = self.model[ModelName.CLASSIFIER](patch_image)

            list_logits.append(logits)
            list_annots.append(annot)

            if self.fast_dev_run:
                # Runs 1 train batch and program ends if 'fast_dev_run' set to 'True'
                # TODO: runs 'n(int)' train batches otherwise.
                # FIXME: progress bar does not update when 'fast_dev_run==True'
                break

        logits = torch.vstack(list_logits)
        probs = torch.sigmoid(logits)
        annots = torch.vstack(list_annots)

        return logits, probs, annots

    @torch.no_grad()
    def validate_epoch(self, epoch, loader) -> Metrics:
        super().validate_epoch(epoch, loader)
        logits, probs, annots = self._inference(loader)
        self.set_threshold(probs, annots, mode=self.thresholding_mode)
        loss, dict_metrics = self.get_metrics(logits, probs, annots)

        self.scheduler[ModelName.CLASSIFIER].step("epoch_val", loss)

        return Metrics(loss)

    @torch.no_grad()
    def test_epoch(self, epoch, loader, export_results=False) -> Metrics:
        super().test_epoch(epoch, loader)
        logits, probs, annots = self._inference(loader)
        loss, dict_metrics = self.get_metrics(logits, probs, annots)

        return Metrics(loss)
