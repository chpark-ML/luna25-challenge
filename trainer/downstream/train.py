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
from trainer.downstream.datasets.constants import DataLoaderKeys

logger = logging.getLogger(__name__)


@dataclass
class Metrics(comm_train.Metrics):
    loss: float = np.inf
    eval_metrics: dict = None
    threshold: dict = None

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
        ema,
        optimizer,
        scheduler,
        criterion,
        thresholding_mode_representative,
        thresholding_mode,
        grad_clip_max_norm,
        target_attr_total,
        target_attr_to_train,
        target_attr_downstream,
        **kwargs,
    ) -> None:
        self.repr_model_name = ModelName.REPRESENTATIVE
        super().__init__(model, ema, optimizer, scheduler, criterion, **kwargs)
        self.thresholding_mode_representative = ThresholdMode.get_mode(thresholding_mode_representative)
        self.thresholding_mode = ThresholdMode.get_mode(thresholding_mode)
        self.grad_clip_max_norm = grad_clip_max_norm
        self.target_attr_downstream = target_attr_downstream

    @classmethod
    def instantiate_trainer(
        cls, config: omegaconf.DictConfig, loaders, logging_tool, ema=None, optuna_trial=None
    ) -> comm_train.Trainer:
        # Init model
        models = dict()
        if isinstance(config.model, (omegaconf.DictConfig, dict)):
            for model_indicator, config_model in config.model.items():
                logger.info(f"Instantiating model <{config.model[model_indicator]['_target_']}>")
                for model_name in ModelName:
                    if model_name.value in model_indicator:
                        models[model_name] = hydra.utils.instantiate(config_model)
                        models[model_name] = models[model_name].float()  # change model to float32
        else:
            raise NotImplementedError

        optimizers = None
        schedulers = None
        if RunMode.TRAIN in loaders:
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
        total_epoch = self.max_epoch - 1

        for i, data in enumerate(loader):
            global_step = epoch * len(loader) + i + 1
            self.optimizer[ModelName.REPRESENTATIVE].zero_grad()
            patch_image = data[DataLoaderKeys.IMAGE].to(self.device)
            _check_any_nan(patch_image)

            annot = data[DataLoaderKeys.LABEL].to(self.device).float()  # (B, 1)
            annots = {self.target_attr_downstream: annot}

            # forward propagation
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                output = self.model[ModelName.REPRESENTATIVE](patch_image)
                dict_loss = self.criterion(
                    output,
                    annots,
                    seg_annot=None,
                    epoch=epoch,
                    total_epoch=total_epoch,
                    attr_mask=None,
                    is_logit=True,
                    is_logistic=True,
                )

                loss = dict_loss[LossKey.total]
                train_losses.append(loss.detach())

            # set trace for checking nan values
            if torch.any(torch.isnan(loss)):
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
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer[ModelName.REPRESENTATIVE])
                torch.nn.utils.clip_grad_norm_(
                    self.model[ModelName.REPRESENTATIVE].parameters(), max_norm=self.grad_clip_max_norm
                )
                self.scaler.step(self.optimizer[ModelName.REPRESENTATIVE])
                self.scaler.update()
            else:
                loss.backward()
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
                    Metrics(loss.detach(), {}, {}),
                    log_prefix=f"[{epoch}/{self.max_epoch}] [{i}/{len(loader)}]",
                    mlflow_log_prefix="STEP",
                    duration=batch_time,
                )
                start = time.time()

            if self.fast_dev_run:
                # Runs 1 train batch and program ends if 'fast_dev_run' set to 'True'
                break
            self.scheduler[ModelName.REPRESENTATIVE].step("step")
        self.scheduler[ModelName.REPRESENTATIVE].step("epoch")

        train_loss = torch.stack(train_losses).sum().item()
        return Metrics(train_loss / len(loader), {}, {})

    def optimizing_metric(self, metrics: Metrics):
        return metrics.loss

    def get_lr(self):
        return self.optimizer[ModelName.REPRESENTATIVE].param_groups[0]["lr"]

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
        outputs = {LOGIT_KEY: {self.target_attr_downstream: logits}}
        annots_dict = {self.target_attr_downstream: annots}
        losses = self.criterion(outputs, annots_dict)[LossKey.cls]
        result_dict = self.get_binary_classification_metrics(
            probs,
            annots,
            threshold=self.dict_threshold,
            threshold_mode=self.thresholding_mode_representative,
        )

        return losses.detach(), result_dict

    @staticmethod
    def get_binary_classification_metrics(
        prob: torch.Tensor,
        annot: torch.Tensor,
        threshold: dict,
        threshold_mode: ThresholdMode = ThresholdMode.YOUDEN,
    ):
        assert type(prob) == type(annot)
        result_dict = dict()

        _annot = (annot.squeeze().cpu().numpy() > 0.5) * 1.0
        _prob = prob.squeeze().cpu().numpy()
        _pred = _prob > threshold[f"threshold_{threshold_mode.value}"]

        result_dict["acc"] = metrics.accuracy_score(_annot, _pred)
        try:
            result_dict["auroc"] = metrics.roc_auc_score(_annot, _prob)
        except ValueError:
            pass
        result_dict["f1"] = metrics.f1_score(_annot, _pred)

        cm = metrics.confusion_matrix(_annot, _pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        elif cm.shape == (1, 1):  # Only one class present, can't compute TN/FP/FN/TP
            tn = fp = fn = tp = 0
            if _annot[0] == 1:
                tp = cm[0, 0]
            else:
                tn = cm[0, 0]
        elif cm.shape == (1, 2):  # Only positive class in ground truth
            tn = fp = 0
            fn, tp = cm[0]
        elif cm.shape == (2, 1):  # Only negative class in ground truth
            fn = tp = 0
            tn, fp = cm[:, 0]
        else:
            raise ValueError(f"Unexpected confusion matrix shape: {cm.shape}")

        # Sensitivity (Recall)
        result_dict["sen"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Specificity
        result_dict["spe"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Precision
        result_dict["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Kappa
        result_dict["kappa"] = metrics.cohen_kappa_score(_annot, _pred)

        # Sensitivity at Specificity = 95%
        fpr, tpr, thresholds = metrics.roc_curve(_annot, _prob)
        specificity = 1 - fpr
        try:
            sen_at_spe_95 = max(tpr[specificity >= 0.95]) if any(specificity >= 0.95) else 0.0
        except:
            sen_at_spe_95 = 0.0
        result_dict["sen_at_spe95"] = sen_at_spe_95

        # Specificity at Sensitivity = 95%
        try:
            spe_at_sen_95 = max(specificity[tpr >= 0.95]) if any(tpr >= 0.95) else 0.0
        except:
            spe_at_sen_95 = 0.0
        result_dict["spe_at_sen95"] = spe_at_sen_95

        return result_dict

    def save_best_metrics(self, val_metrics: Metrics, best_metrics: Metrics, epoch) -> (object, bool):
        found_better = False

        best_metric = self.best_metrics.get(BaseBestModelStandard.REPRESENTATIVE, None)
        val_loss = getattr(val_metrics, "loss", None)
        best_loss = getattr(best_metric, "loss", None)
        if best_loss is None or (val_loss is not None and val_loss < best_loss):
            found_better = True
            model_path = f"model_loss.pth"
            logger.info(f"loss improved to {val_metrics.loss:4f}, " f"saving model to {model_path}.")
            self.best_metrics[BaseBestModelStandard.REPRESENTATIVE] = val_metrics
            self.path_best_model[BaseBestModelStandard.REPRESENTATIVE] = model_path
            self.epoch_best_model[BaseBestModelStandard.REPRESENTATIVE] = epoch
            self.threshold_best_model[BaseBestModelStandard.REPRESENTATIVE] = self.dict_threshold
            self.save_checkpoint(model_path, thresholds=self.dict_threshold)

        best_metric = self.best_metrics.get(BaseBestModelStandard.AUROC, None)
        val_auroc = val_metrics.eval_metrics.get("auroc", None)
        best_auroc = best_metric.eval_metrics.get("auroc", None) if best_metric is not None else None
        if best_auroc is None or (val_auroc is not None and val_auroc > best_auroc):
            model_path = f"model_auroc.pth"
            logger.info(f"AUROC improved to {val_metrics.eval_metrics['auroc']:4f}, " f"saving model to {model_path}.")
            self.best_metrics[BaseBestModelStandard.AUROC] = val_metrics
            self.path_best_model[BaseBestModelStandard.AUROC] = model_path
            self.epoch_best_model[BaseBestModelStandard.AUROC] = epoch
            self.threshold_best_model[BaseBestModelStandard.AUROC] = self.dict_threshold
            self.save_checkpoint(model_path, thresholds=self.dict_threshold)

        if epoch == self.max_epoch - 1:  # the given `epoch` is in the range of (0, self.max_epoch)
            model_path = f"model_final.pth"
            logger.info(f"saving model to {model_path}.")
            self.path_best_model[BaseBestModelStandard.LAST] = model_path
            self.epoch_best_model[BaseBestModelStandard.LAST] = epoch
            self.threshold_best_model[BaseBestModelStandard.LAST] = self.dict_threshold
            self.save_checkpoint(model_path, thresholds=self.dict_threshold)

        best_metrics = self.best_metrics[BaseBestModelStandard.REPRESENTATIVE]

        return best_metrics, found_better

    def _inference(self, loader):
        list_logits = []
        list_annots = []

        for data in tqdm.tqdm(loader):
            # prediction
            patch_image = data[DataLoaderKeys.IMAGE].to(self.device)
            _check_any_nan(patch_image)

            # annotation
            annot = data[DataLoaderKeys.LABEL].to(self.device).float()

            # inference
            outputs = self.model[ModelName.REPRESENTATIVE](patch_image)
            logits = outputs[LOGIT_KEY][self.target_attr_downstream]
            logits = logits.view(-1, 1)  # considering the prediction tensor can be either (B,) or (B, 1)

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

        # set scheduler input
        scheduler_criteria = None
        if isinstance(self.scheduler[ModelName.REPRESENTATIVE].scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if self.scheduler[ModelName.REPRESENTATIVE].scheduler.mode == "min":
                scheduler_criteria = loss
            else:
                scheduler_criteria = dict_metrics[f"auroc"]
        self.scheduler[ModelName.REPRESENTATIVE].step("epoch_val", scheduler_criteria)

        return Metrics(loss, dict_metrics, self.dict_threshold)

    @torch.no_grad()
    def test_epoch(self, epoch, loader, export_results=False) -> Metrics:
        super().test_epoch(epoch, loader)
        logits, probs, annots = self._inference(loader)
        loss, dict_metrics = self.get_metrics(logits, probs, annots)

        return Metrics(loss, dict_metrics, self.dict_threshold)
