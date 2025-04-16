import json
import logging
import os
import random
import time
from dataclasses import dataclass

import hydra
import numpy as np
import omegaconf
import torch
import tqdm
from sklearn import metrics

import trainer.common.train as comm_train
from data_lake.lidc.constants import LOGISTIC_TASK_POSTFIX, RESAMPLED_FEATURE_POSTFIX
from shared_lib.enums import RunMode
from trainer.common.constants import ANNOTATION_KEY, INPUT_PATCH_KEY, LOGIT_KEY, LossKey
from trainer.common.enums import ModelName, ThresholdMode
from trainer.common.utils import freeze_layers

logger = logging.getLogger(__name__)


@dataclass
class Metrics(comm_train.Metrics):
    loss: float = np.inf
    multi_label_losses: dict = None
    multi_label_metrics: dict = None
    multi_label_threshold: dict = None

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


def get_binary_classification_metrics(
    logit: dict, prob: dict, annot: dict, threshold: dict, threshold_mode: ThresholdMode = ThresholdMode.YOUDEN
):
    assert type(prob) == type(annot)
    result_dict = dict()
    target_attr_total = list(prob.keys())
    for i_attr in target_attr_total:
        if LOGISTIC_TASK_POSTFIX in i_attr:
            _annot = (annot[i_attr].squeeze().cpu().numpy() > 0.5) * 1.0
            _pred = prob[i_attr].squeeze().cpu().numpy() > threshold[f"threshold_{threshold_mode.value}_{i_attr}"]
            _prob = prob[i_attr].squeeze().cpu().numpy()
            result_dict[f"acc_{i_attr}"] = metrics.accuracy_score(_annot, _pred)
            try:
                result_dict[f"auroc_{i_attr}"] = metrics.roc_auc_score(_annot, _prob)
            except ValueError:  # in the case when only one class exists, AUROC can not be calculated. (in fast_dev_run)
                pass
            result_dict[f"f1_{i_attr}"] = metrics.f1_score(_annot, _pred)
        elif RESAMPLED_FEATURE_POSTFIX in i_attr:
            result_dict[f"MAE_{i_attr}"] = np.mean(
                np.abs(logit[i_attr].squeeze().cpu().numpy() - annot[i_attr].squeeze().cpu().numpy())
            )

    return result_dict


class Trainer(comm_train.Trainer):
    """Trainer to train model"""

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        remove_ambiguous_in_val_test,
        lower_bound_ambiguous_label,
        upper_bound_ambiguous_label,
        thresholding_mode_representative,
        thresholding_mode,
        target_attr_total,
        target_attr_to_train,
        target_attr_downstream,
        grad_clip_max_norm,
        **kwargs,
    ) -> None:
        self.repr_model_name = ModelName.CLASSIFIER
        super().__init__(model, optimizer, scheduler, criterion, **kwargs)
        self.remove_ambiguous_in_val_test = remove_ambiguous_in_val_test
        self.lower_bound_ambiguous_label = lower_bound_ambiguous_label
        self.upper_bound_ambiguous_label = upper_bound_ambiguous_label
        self.thresholding_mode_representative = ThresholdMode.get_mode(thresholding_mode_representative)
        self.thresholding_mode = ThresholdMode.get_mode(thresholding_mode)

        self.target_attr_total = target_attr_total
        self.target_attr_to_train = target_attr_to_train
        self.target_attr_downstream = target_attr_downstream

        self.grad_clip_max_norm = grad_clip_max_norm

        if self.fine_tune_info["enable"]:
            logger.info(f"freeze layers func is called.")
            freeze_layers(model[self.repr_model_name], self.fine_tune_info["freeze_encoder"], self.target_attr_to_train)

    @classmethod
    def instantiate_trainer(
        cls, config: omegaconf.DictConfig, loaders, logging_tool, optuna_trial=None
    ) -> comm_train.Trainer:
        # Init model
        if isinstance(config.model, (omegaconf.DictConfig, dict)):
            models = dict()
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
            # Perform random balanced sampling if needed
            for mode in [RunMode.TRAIN, RunMode.VALIDATE]:
                if mode in loaders and loaders[mode].dataset.do_random_balanced_sampling:
                    loaders[mode].dataset.random_balanced_sampling()

            # Set steps per epoch for the scheduler if specified
            if isinstance(config.scheduler, (omegaconf.DictConfig, dict)):
                for scheduler_indicator, config_scheduler in config.scheduler.items():
                    if hasattr(config_scheduler, "steps_per_epoch"):
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

        # Set criterion, and its alpha based on training data
        criterion = hydra.utils.instantiate(config.criterion)
        if RunMode.TRAIN in loaders:
            for crit in [criterion.cls_criterion, criterion.aux_criterion]:
                crit.set_criterion_alpha(train_df=loaders[RunMode.TRAIN].dataset.meta_df)

        # Init trainer
        logger.info(f"Instantiating trainer <{config.trainer._target_}>")
        return hydra.utils.instantiate(
            config.trainer,
            model=models,
            optimizer=optimizers,
            scheduler=schedulers,
            criterion=criterion,
            logging_tool=logging_tool,
            target_attr_total=config.trainer.target_attr_total,
            target_attr_to_train=config.trainer.target_attr_to_train,
            target_attr_downstream=config.trainer.target_attr_downstream,
            grad_clip_max_norm=config.trainer.grad_clip_max_norm,
            fine_tune_info=config.trainer.fine_tune_info,
            thresholding_mode=config.trainer.thresholding_mode,
            use_amp=config.trainer.use_amp,
            optuna_trial=optuna_trial,
        )

    def train_epoch(self, epoch, loader) -> Metrics:
        super().train_epoch(epoch, loader)
        train_losses = []
        start = time.time()

        if loader.dataset.do_random_balanced_sampling:
            loader.dataset.random_balanced_sampling()

        for i, data in enumerate(loader):
            global_step = epoch * len(loader) + i + 1
            self.optimizer[ModelName.CLASSIFIER].zero_grad()
            dicom = data[INPUT_PATCH_KEY].to(self.device)
            _check_any_nan(dicom)
            annots = dict()
            for key, value in data[ANNOTATION_KEY].items():
                _annot = value.to(self.device).float()
                _check_any_nan(_annot)
                annots[key] = torch.unsqueeze(_annot, dim=1)

            # forward propagation
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                output = self.model[ModelName.CLASSIFIER](dicom)
                dict_loss = self.criterion(output, annots, mask=None, is_logit=True, is_logistic=True)
                loss = dict_loss[LossKey.total]
                loss_cls = dict_loss[LossKey.cls]
                loss_cls_dict = dict_loss[LossKey.cls_dict]
                train_losses.append(loss_cls.detach())

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
                    Metrics(loss_cls.detach(), loss_cls_dict, None),
                    log_prefix=f"[{epoch}/{self.max_epoch}] [{i}/{len(loader)}]",
                    mlflow_log_prefix="STEP",
                    duration=batch_time,
                )
                start = time.time()

            if self.fast_dev_run:
                # Runs 1 train batch and program ends if 'fast_dev_run' set to 'True'
                # TODO: runs 'n(int)' train batches otherwise.
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

    def get_samples_to_validate(self, dict_logits, dict_probs, dict_annots):
        target_attr_downstream = self.target_attr_downstream
        if not isinstance(target_attr_downstream, str):
            raise NotImplementedError

        if LOGISTIC_TASK_POSTFIX in target_attr_downstream:
            # samples where annotation label is not ambiguous, 0.5.
            index_to_be_validated = (dict_annots[target_attr_downstream][:, 0] < self.lower_bound_ambiguous_label) | (
                dict_annots[target_attr_downstream][:, 0] > self.upper_bound_ambiguous_label
            )
            assert index_to_be_validated.sum() != 0, f"{target_attr_downstream} has no sample for validation"
            dict_logits[target_attr_downstream] = dict_logits[target_attr_downstream][index_to_be_validated]
            dict_probs[target_attr_downstream] = dict_probs[target_attr_downstream][index_to_be_validated]
            dict_annots[target_attr_downstream] = dict_annots[target_attr_downstream][index_to_be_validated]
        elif RESAMPLED_FEATURE_POSTFIX in target_attr_downstream:
            pass
        return dict_logits, dict_probs, dict_annots

    def set_threshold(self, dict_probs, dict_annots, mode=ThresholdMode.YOUDEN):
        assert mode == ThresholdMode.YOUDEN or mode == ThresholdMode.F1 or mode == ThresholdMode.ALL
        dict_threshold = dict()
        for i_attr in self.target_attr_to_train:
            best_f1 = -1

            y_true = (dict_annots[i_attr].detach().cpu().numpy() > 0.5) * 1.0
            if LOGISTIC_TASK_POSTFIX in i_attr:
                if mode == ThresholdMode.YOUDEN or mode == ThresholdMode.ALL:
                    # calculate roc curves
                    fpr, tpr, thresholds = metrics.roc_curve(
                        y_true,
                        dict_probs[i_attr].detach().cpu().numpy(),
                    )

                    # get the best threshold
                    J = tpr - fpr
                    ix = max(np.argmax(J), 1)
                    best_thresh = thresholds[ix]
                    dict_threshold[f"threshold_{ThresholdMode.YOUDEN.value}_{i_attr}"] = best_thresh

                if mode == ThresholdMode.F1 or mode == ThresholdMode.ALL:
                    # 다양한 threshold 값을 시도해보고 가장 높은 F1 스코어를 찾습니다.
                    for threshold in np.arange(0.0, 1.0, 0.01):
                        y_pred = (dict_probs[i_attr].detach().cpu().numpy() > threshold).astype(int)
                        # y_true = dict_annots[i_attr].detach().cpu().numpy()
                        f1 = metrics.f1_score(y_true, y_pred)

                        if f1 > best_f1:
                            best_f1 = f1
                            dict_threshold[f"threshold_{ThresholdMode.F1.value}_{i_attr}"] = threshold

        self.dict_threshold = dict_threshold

    def get_metrics(self, dict_logits, dict_probs, dict_annots):
        outputs = self.criterion({LOGIT_KEY: dict_logits}, dict_annots, mask=None, is_logit=True, is_logistic=True)
        losses = outputs[LossKey.cls]
        dict_losses = outputs[LossKey.cls_dict]
        result_dict = get_binary_classification_metrics(
            dict_logits,
            dict_probs,
            dict_annots,
            self.dict_threshold,
            threshold_mode=self.thresholding_mode_representative,
        )
        return losses.detach(), dict_losses, result_dict

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
            dicom = data[INPUT_PATCH_KEY].to(self.device)
            _check_any_nan(dicom)
            output = self.model[ModelName.CLASSIFIER](dicom)
            logits = output[LOGIT_KEY]

            # annotation
            annots = dict()
            for key in self.target_attr_total:
                value = data[ANNOTATION_KEY][key]
                annots[key] = torch.unsqueeze(value.to(self.device).float(), dim=1)

            list_logits.append(logits)
            list_annots.append(annots)

            if self.fast_dev_run:
                # Runs 1 train batch and program ends if 'fast_dev_run' set to 'True'
                # TODO: runs 'n(int)' train batches otherwise.
                # FIXME: progress bar does not update when 'fast_dev_run==True'
                break

        dict_logits = {
            key: torch.vstack([(i_logits[key]) for i_logits in list_logits]) for key in self.target_attr_to_train
        }
        dict_probs = {
            key: torch.vstack([torch.sigmoid(i_logits[key]) for i_logits in list_logits])
            for key in self.target_attr_to_train
        }
        dict_annots = {key: torch.vstack([i_annots[key] for i_annots in list_annots]) for key in self.target_attr_total}

        if self.remove_ambiguous_in_val_test:
            dict_logits, dict_probs, dict_annots = self.get_samples_to_validate(dict_logits, dict_probs, dict_annots)

        return dict_logits, dict_probs, dict_annots

    @torch.no_grad()
    def validate_epoch(self, epoch, val_loader) -> Metrics:
        super().validate_epoch(epoch, val_loader)
        dict_logits, dict_probs, dict_annots = self._inference(val_loader)
        self.set_threshold(dict_probs, dict_annots, mode=self.thresholding_mode)
        loss, dict_loss, dict_metrics = self.get_metrics(dict_logits, dict_probs, dict_annots)
        self.scheduler[ModelName.CLASSIFIER].step("epoch_val", loss)

        return Metrics(loss, dict_loss, dict_metrics, self.dict_threshold)

    @torch.no_grad()
    def test_epoch(self, epoch, loader, export_results=False) -> Metrics:
        super().test_epoch(epoch, loader)
        dict_logits, dict_probs, dict_annots = self._inference(loader)
        loss, dict_loss, dict_metrics = self.get_metrics(dict_logits, dict_probs, dict_annots)

        if export_results:
            # prepare results to save
            results = {
                "logits": {k: v.tolist() for k, v in dict_logits.items()},  # Convert tensors to lists
                "probs": {k: v.tolist() for k, v in dict_probs.items()},
                "annotations": {k: v.tolist() for k, v in dict_annots.items()},
                "threshold": {k: float(v) for k, v in self.dict_threshold.items()},
            }

            # save to json
            json_filename = f"results_{loader.dataset.mode.value}.json"
            with open(json_filename, "w") as f:
                json.dump(results, f, indent=4)

        return Metrics(loss, dict_loss, dict_metrics, self.dict_threshold)
