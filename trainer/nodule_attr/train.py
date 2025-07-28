import json
import logging
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
from shared_lib.enums import BaseBestModelStandard, RunMode
from trainer.common.constants import (
    ATTR_ANNOTATION_KEY,
    INPUT_PATCH_KEY,
    LOGIT_KEY,
    SEG_ANNOTATION_KEY,
    SEG_LOGIT_KEY,
    LossKey,
)
from trainer.common.enums import ModelName, ThresholdMode
from trainer.common.utils import freeze_layers

logger = logging.getLogger(__name__)


@dataclass
class Metrics(comm_train.Metrics):
    loss_total: float = np.inf

    loss_cls: float = np.inf
    cls_losses: dict = None
    cls_metrics: dict = None

    loss_seg: float = np.inf
    seg_metrics: dict = None

    thresholds: dict = None

    def __str__(self):
        loss_cls_str = f"{self.loss_cls:.4f}" if self.loss_cls is not None else "None"
        loss_seg_str = f"{self.loss_seg:.4f}" if self.loss_seg is not None else "None"
        return f"loss_cls_{loss_cls_str}_loss_seg_{loss_seg_str}"

    def get_representative_metric(self):
        """
        Returns: float type evaluation metric
        """
        loss_cls = self.loss_cls if self.loss_cls is not None else 0.0
        loss_seg = self.loss_seg if self.loss_seg is not None else 0.0
        return loss_cls + loss_seg


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


def compute_dice(
    input_array,
    target_array,
    epsilon=1e-6,
    weight=None,
    squared=True,
    per_channel=False,
    reduction="mean",  # "mean" or "none"
):
    """
    Computes Dice coefficient between input and target NumPy arrays.

    Args:
        input_array (np.ndarray): (N, C, ...) predicted values (float or binary)
        target_array (np.ndarray): (N, C, ...) ground truth (binary)
        epsilon (float): small constant to avoid division by zero
        weight (np.ndarray, optional): (C,) or (C, 1), per-channel weight
        squared (bool): whether to square input and target in denominator
        per_channel (bool): if True, return per-channel dice (N, C) or (C,)
        reduction (str): "mean" (default) or "none"

    Returns:
        float or np.ndarray: Dice score
    """
    assert input_array.shape == target_array.shape, "'input_array' and 'target_array' must have the same shape"
    N, C = input_array.shape[:2]
    input_flat = input_array.reshape(N, C, -1)
    target_flat = target_array.astype(np.float32).reshape(N, C, -1)

    intersect = np.sum(input_flat * target_flat, axis=-1)  # shape: (N, C)

    if squared:
        input_sum = np.sum(input_flat**2, axis=-1)
        target_sum = np.sum(target_flat**2, axis=-1)
    else:
        input_sum = np.sum(input_flat, axis=-1)
        target_sum = np.sum(target_flat, axis=-1)

    denominator = input_sum + target_sum
    dice = (2.0 * intersect) / (np.maximum(denominator, epsilon))  # shape: (N, C)

    if weight is not None:
        weight = weight.reshape(1, -1)  # shape: (1, C)
        dice = dice * weight

    if per_channel:
        if reduction == "mean":
            return dice.mean(axis=0)  # (C,)
        elif reduction == "none":
            return dice  # (N, C)
        else:
            raise ValueError(f"Invalid reduction mode: {reduction}")
    else:
        return dice.mean()


class Trainer(comm_train.Trainer):
    """Trainer to train model"""

    def __init__(
        self,
        model,
        ema,
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
        do_segmentation,
        grad_clip_max_norm,
        **kwargs,
    ) -> None:
        self.repr_model_name = ModelName.REPRESENTATIVE
        super().__init__(model, ema, optimizer, scheduler, criterion, **kwargs)
        self.remove_ambiguous_in_val_test = remove_ambiguous_in_val_test
        self.lower_bound_ambiguous_label = lower_bound_ambiguous_label
        self.upper_bound_ambiguous_label = upper_bound_ambiguous_label
        self.thresholding_mode_representative = ThresholdMode.get_mode(thresholding_mode_representative)
        self.thresholding_mode = ThresholdMode.get_mode(thresholding_mode)

        self.target_attr_total = target_attr_total
        self.target_attr_to_train = target_attr_to_train
        self.target_attr_downstream = target_attr_downstream

        self.do_segmentation = do_segmentation

        self.grad_clip_max_norm = grad_clip_max_norm

        if self.fine_tune_info["enable"]:
            logger.info(f"freeze layers func is called.")
            freeze_layers(model[self.repr_model_name], self.fine_tune_info["freeze_encoder"], self.target_attr_to_train)

    @classmethod
    def instantiate_trainer(
        cls, config: omegaconf.DictConfig, loaders, logging_tool, ema=None, optuna_trial=None
    ) -> comm_train.Trainer:
        # Init model
        if isinstance(config.model, (omegaconf.DictConfig, dict)):
            models = dict()
            for model_indicator, config_model in config.model.items():
                logger.info(f"Instantiating model <{config.model[model_indicator]['_target_']}>")
                for model_name in ModelName:
                    if model_name.value in model_indicator:
                        models[model_name] = hydra.utils.instantiate(config_model)
                        models[model_name] = models[model_name].float()  # change model to float32
        else:
            raise NotImplementedError

        # Set ema
        if ema is not None:
            ema.register(models[ModelName.REPRESENTATIVE])

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
            ema=ema,
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
            self.optimizer[ModelName.REPRESENTATIVE].zero_grad()
            dicom = data[INPUT_PATCH_KEY].to(self.device).float()  # Convert input to float32
            _check_any_nan(dicom)

            # annots
            seg_annot = data[SEG_ANNOTATION_KEY].to(self.device) if self.do_segmentation else None
            annots = dict()
            for key, value in data[ATTR_ANNOTATION_KEY].items():
                _annot = value.to(self.device).float()  # Convert annotation to float32
                _check_any_nan(_annot)
                annots[key] = torch.unsqueeze(_annot, dim=1)  # (B, 1)

            # forward propagation
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                output = self.model[ModelName.REPRESENTATIVE](dicom)
                dict_loss = self.criterion(
                    output,
                    annots,
                    seg_annot,
                    epoch=epoch,
                    total_epoch=self.max_epoch - 1,
                    attr_mask=None,
                    is_logit=True,
                    is_logistic=True,
                )
                loss_total = dict_loss[LossKey.total]

            # attributes
            loss_cls = dict_loss[LossKey.cls].detach()
            loss_cls_dict = dict_loss[LossKey.cls_dict]

            # segmentation
            if self.do_segmentation:
                loss_seg = dict_loss[LossKey.seg].detach()
                train_losses.append(loss_cls + loss_seg)
            else:
                train_losses.append(loss_cls)

            # set trace for checking nan values
            if torch.any(torch.isnan(loss_total)):
                import pdb

                pdb.set_trace()
                is_param_nan = torch.stack(
                    [torch.isnan(p).any() for p in self.model[ModelName.REPRESENTATIVE].parameters()]
                ).any()
                continue

            # Copy model parameters before backward pass and optimization step
            if self.fast_dev_run:
                param_before = {
                    name: param.clone() for name, param in self.model[ModelName.REPRESENTATIVE].named_parameters()
                }

            # Backpropagation
            if self.use_amp:
                self.scaler.scale(loss_total).backward()
                self.scaler.unscale_(self.optimizer[ModelName.REPRESENTATIVE])
                torch.nn.utils.clip_grad_norm_(
                    self.model[ModelName.REPRESENTATIVE].parameters(), max_norm=self.grad_clip_max_norm
                )
                self.scaler.step(self.optimizer[ModelName.REPRESENTATIVE])
                self.scaler.update()
            else:
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model[ModelName.REPRESENTATIVE].parameters(), max_norm=self.grad_clip_max_norm
                )
                self.optimizer[ModelName.REPRESENTATIVE].step()

            # Update EMA model
            if self.ema:
                self.ema.update(self.model[ModelName.REPRESENTATIVE])

            # Check if any parameter has changed
            if self.fast_dev_run:
                for name, param in self.model[ModelName.REPRESENTATIVE].named_parameters():
                    if not torch.equal(param_before[name], param):
                        print(f"Parameter '{name}' has changed.")
                    else:
                        print(f"Parameter '{name}' remains unchanged.")

            if global_step % self.log_every_n_steps == 0:
                batch_time = time.time() - start
                self.log_metrics(
                    RunMode.TRAIN.value,
                    global_step,
                    Metrics(
                        loss_cls + (loss_seg if self.do_segmentation else 0.0),  # total
                        loss_cls,
                        loss_cls_dict,
                        {},  # cls
                        loss_seg if self.do_segmentation else None,
                        {},  # seg
                        {},
                    ),  # thresholds
                    log_prefix=f"[{epoch}/{self.max_epoch}] [{i}/{len(loader)}]",
                    mlflow_log_prefix="STEP",
                    duration=batch_time,
                )
                start = time.time()

            if self.fast_dev_run:
                # Runs 1 train batch and program ends if 'fast_dev_run' set to 'True'
                # TODO: runs 'n(int)' train batches otherwise.
                break
            self.scheduler[ModelName.REPRESENTATIVE].step("step")
        self.scheduler[ModelName.REPRESENTATIVE].step("epoch")

        train_loss = torch.stack(train_losses).sum().item()
        return Metrics(train_loss / len(loader))

    def optimizing_metric(self, metrics: Metrics):
        return metrics.loss_cls

    def get_lr(self):
        return self.optimizer[ModelName.REPRESENTATIVE].param_groups[0]["lr"]

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

    def set_threshold(self, dict_probs, dict_annots, seg_probs, seg_annots, mode=ThresholdMode.YOUDEN):
        assert mode == ThresholdMode.YOUDEN or mode == ThresholdMode.F1 or mode == ThresholdMode.ALL
        dict_threshold = dict()

        # attributes
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
                    # Try various threshold values and find the one with the highest F1 score.
                    for threshold in np.arange(0.0, 1.0, 0.01):
                        y_pred = (dict_probs[i_attr].detach().cpu().numpy() > threshold).astype(int)
                        # y_true = dict_annots[i_attr].detach().cpu().numpy()
                        f1 = metrics.f1_score(y_true, y_pred)

                        if f1 > best_f1:
                            best_f1 = f1
                            dict_threshold[f"threshold_{ThresholdMode.F1.value}_{i_attr}"] = threshold
        # segmentation
        if self.do_segmentation:
            y_true = (seg_annots.detach().cpu().numpy() > 0.5).astype(np.uint8)
            probs = seg_probs.detach().cpu().numpy()

            best_dice = -1
            for threshold in np.arange(0.0, 1.0, 0.01):
                y_pred = (probs > threshold).astype(int)
                dice = compute_dice(y_pred, y_true, squared=True)
                if dice > best_dice:
                    best_dice = dice
                    dict_threshold[f"threshold_dice_seg"] = threshold

        self.dict_threshold = dict_threshold

    def get_metrics(self, dict_logits, dict_probs, dict_annots, seg_logits=None, seg_annots=None):
        # attributes
        logits = {LOGIT_KEY: dict_logits}
        if self.do_segmentation:
            logits[SEG_LOGIT_KEY] = seg_logits
        outputs = self.criterion(logits, dict_annots, seg_annots, attr_mask=None, is_logit=True, is_logistic=True)

        loss_cls = outputs[LossKey.cls].detach()
        dict_losses = outputs[LossKey.cls_dict]
        result_dict = get_binary_classification_metrics(
            dict_logits,
            dict_probs,
            dict_annots,
            self.dict_threshold,
            threshold_mode=self.thresholding_mode_representative,
        )

        # segmentation
        loss_seg = 0.0
        result_dice = dict()
        if self.do_segmentation:
            loss_seg = outputs[LossKey.seg].detach()

            y_true = (seg_annots.detach().cpu().numpy() > 0.5).astype(np.uint8)
            seg_probs = torch.sigmoid(seg_logits)
            y_pred = (seg_probs.detach().cpu().numpy() > self.dict_threshold["threshold_dice_seg"]).astype(np.uint8)

            dice_l1 = compute_dice(y_pred, y_true, squared=False)
            dice_l2 = compute_dice(y_pred, y_true, squared=True)
            result_dice["dice_l1"] = dice_l1
            result_dice["dice_l2"] = dice_l2

        return loss_cls, dict_losses, result_dict, loss_seg, result_dice

    def save_best_metrics(self, val_metrics: Metrics, best_metrics: Metrics, epoch) -> (object, bool):
        found_better = False
        if val_metrics.loss_cls < best_metrics.loss_cls:
            found_better = True
            model_path = f"model_loss.pth"
            logger.info(
                f"loss improved from {best_metrics.loss_cls:4f} to {val_metrics.loss_cls:4f}, "
                f"saving model to {model_path}."
            )

            best_metrics = val_metrics
            self.path_best_model[BaseBestModelStandard.REPRESENTATIVE] = model_path
            self.epoch_best_model[BaseBestModelStandard.REPRESENTATIVE] = epoch
            self.save_checkpoint(model_path, thresholds=self.dict_threshold)

        if epoch == self.max_epoch - 1:  # the given `epoch` is in the range of (0, self.max_epoch)
            found_better = True
            model_path = f"model_final.pth"
            logger.info(f"saving model to {model_path}.")
            self.path_best_model[BaseBestModelStandard.LAST] = model_path
            self.epoch_best_model[BaseBestModelStandard.LAST] = epoch
            self.threshold_best_model[BaseBestModelStandard.LAST] = self.dict_threshold
            self.save_checkpoint(model_path, thresholds=self.dict_threshold)

        return best_metrics, found_better

    def _inference(self, loader):
        list_logits = []
        list_annots = []
        list_seg_logits = []
        list_seg_annots = []

        for data in tqdm.tqdm(loader):
            # prediction
            dicom = data[INPUT_PATCH_KEY].to(self.device).float()  # Convert input to float32
            _check_any_nan(dicom)
            output = self.model[ModelName.REPRESENTATIVE](dicom)
            logits = output[LOGIT_KEY]

            if self.do_segmentation:
                seg_logits = output[SEG_LOGIT_KEY]

            # annotation
            seg_annot = data[SEG_ANNOTATION_KEY].to(self.device) if self.do_segmentation else None
            annots = dict()
            for key in self.target_attr_total:
                value = data[ATTR_ANNOTATION_KEY][key]
                annots[key] = torch.unsqueeze(value.to(self.device).float(), dim=1)  # Convert annotation to float32

            list_logits.append(logits)
            list_annots.append(annots)

            if self.do_segmentation:
                list_seg_logits.append(seg_logits)
                list_seg_annots.append(seg_annot)

            if self.fast_dev_run:
                # Runs 1 train batch and program ends if 'fast_dev_run' set to 'True'
                # TODO: runs 'n(int)' train batches otherwise.
                # FIXME: progress bar does not update when 'fast_dev_run==True'
                break

        # attributes
        dict_logits = {
            key: torch.vstack([(i_logits[key]) for i_logits in list_logits]) for key in self.target_attr_to_train
        }
        dict_probs = {
            key: torch.vstack([torch.sigmoid(i_logits[key]) for i_logits in list_logits])
            for key in self.target_attr_to_train
        }
        dict_annots = {key: torch.vstack([i_annots[key] for i_annots in list_annots]) for key in self.target_attr_total}

        # segmentation
        seg_logits = None
        seg_annots = None
        if self.do_segmentation:
            seg_logits = torch.vstack(list_seg_logits)
            seg_annots = torch.vstack(list_seg_annots)

        if self.remove_ambiguous_in_val_test:
            dict_logits, dict_probs, dict_annots = self.get_samples_to_validate(dict_logits, dict_probs, dict_annots)

        return dict_logits, dict_probs, dict_annots, seg_logits, seg_annots

    @torch.no_grad()
    def validate_epoch(self, epoch, val_loader) -> Metrics:
        super().validate_epoch(epoch, val_loader)
        dict_logits, dict_probs, dict_annots, seg_logits, seg_annots = self._inference(val_loader)
        seg_probs = torch.sigmoid(seg_logits) if self.do_segmentation else None
        self.set_threshold(dict_probs, dict_annots, seg_probs, seg_annots, mode=self.thresholding_mode)

        # attributes, segmentation
        loss_cls, dict_loss, dict_metrics, loss_seg, result_dice = self.get_metrics(
            dict_logits, dict_probs, dict_annots, seg_logits, seg_annots
        )

        # set scheduler input
        scheduler_criteria = None
        if isinstance(self.scheduler[ModelName.REPRESENTATIVE].scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if self.scheduler[ModelName.REPRESENTATIVE].scheduler.mode == "min":
                scheduler_criteria = loss_cls
            else:
                scheduler_criteria = dict_metrics[f"auroc_{self.target_attr_downstream}"]

        # update scheduler
        self.scheduler[ModelName.REPRESENTATIVE].step("epoch_val", scheduler_criteria)

        return Metrics(
            loss_cls + loss_seg,  # total
            loss_cls,
            dict_loss,
            dict_metrics,  # cls
            loss_seg,
            result_dice,  # seg
            self.dict_threshold,
        )  # thresholds

    @torch.no_grad()
    def test_epoch(self, epoch, loader, export_results=False) -> Metrics:
        super().test_epoch(epoch, loader)
        dict_logits, dict_probs, dict_annots, seg_logits, seg_annots = self._inference(loader)

        # attributes, segmentation
        loss_cls, dict_loss, dict_metrics, loss_seg, result_dice = self.get_metrics(
            dict_logits, dict_probs, dict_annots, seg_logits, seg_annots
        )

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

        return Metrics(
            loss_cls + loss_seg,  # total
            loss_cls,
            dict_loss,
            dict_metrics,  # cls
            loss_seg,
            result_dice,  # seg
            self.dict_threshold,
        )  # thresholds
