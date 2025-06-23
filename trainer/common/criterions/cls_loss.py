import torch
import torch.nn as nn

from data_lake.lidc.constants import LOGISTIC_TASK_POSTFIX, RESAMPLED_FEATURE_POSTFIX
from trainer.common.criterions.focal_loss import FocalLoss


def smooth_one_hot(targets, n_classes, smoothing=0.0):
    assert 0 <= smoothing < 1
    with torch.no_grad():
        targets = (
            torch.empty(size=(targets.size(0), n_classes))
            .cuda()
            .fill_(smoothing / (n_classes - 1))
            .scatter_(1, targets.long().data.unsqueeze(-1), 1 - smoothing)
        )
    return targets[:, -1]


class ClsLoss(nn.Module):
    def __init__(
        self,
        gamma=0,
        use_alpha=True,
        use_threshold=False,
        smoothing=0.0,
        loss_weight: float = 1.0,
        target_attr_total: list = None,
        target_attr_to_train: list = None,
        dict_threshold: dict = None,
        target_threshold_gte: float = None,
    ):
        super(ClsLoss, self).__init__()
        self.gamma = gamma
        self.use_alpha = use_alpha
        self.use_threshold = use_threshold
        self.smoothing = smoothing
        self.loss_weight = loss_weight
        self.target_attr_total = target_attr_total
        self.target_attr_to_train = target_attr_to_train
        self.dict_threshold = dict_threshold
        self.target_threshold_gte = target_threshold_gte

        self.dict_criterion = dict()
        self.threshold_mode = "youden"
        for i_attr in self.target_attr_total:
            if LOGISTIC_TASK_POSTFIX in i_attr:
                alpha = None
                self.dict_criterion[i_attr] = FocalLoss(
                    gamma=self.gamma,
                    alpha=alpha,
                    smoothing=self.smoothing,
                    target_threshold_gte=self.target_threshold_gte,
                )
            elif RESAMPLED_FEATURE_POSTFIX in i_attr:
                self.dict_criterion[i_attr] = nn.SmoothL1Loss(reduction="none")

    def set_criterion_alpha(self, train_df):
        for i_attr in self.target_attr_total:
            if LOGISTIC_TASK_POSTFIX in i_attr:
                alpha = (
                    ((train_df[i_attr] > 0.5).sum()) / len(train_df[i_attr][train_df[i_attr] != 0.5])
                    if self.use_alpha
                    else None
                )
                self.dict_criterion[i_attr] = FocalLoss(
                    gamma=self.gamma,
                    alpha=alpha,
                    smoothing=self.smoothing,
                    target_threshold_gte=self.target_threshold_gte,
                )
            elif RESAMPLED_FEATURE_POSTFIX in i_attr:
                self.dict_criterion[i_attr] = nn.SmoothL1Loss(reduction="none")

    def forward(self, pred, annot, epoch=None, total_epoch=None, mask=None, is_logit=True, is_logistic=True):
        dict_loss = {}
        list_loss = []
        target_attr = self.target_attr_to_train

        # Apply threshold if required
        if self.use_threshold:
            for i_attr, i_annot in annot.items():
                threshold_key = f"threshold_{self.threshold_mode}_{i_attr}"
                if threshold_key in (self.dict_threshold or {}):
                    annot[i_attr] = (i_annot > self.dict_threshold[threshold_key]) * 1.0

        # calculate loss weight
        if epoch is not None and total_epoch is not None:
            weight = self.loss_weight * (1 - (epoch / total_epoch))
        else:
            weight = self.loss_weight

        for i_attr in target_attr:
            criterion = self.dict_criterion[i_attr]
            prediction = pred[i_attr]
            annotation = annot[i_attr]

            if LOGISTIC_TASK_POSTFIX in i_attr:
                loss = criterion(prediction, annotation, is_logit=is_logit, is_logistic=is_logistic)  # (B, 1)
            elif RESAMPLED_FEATURE_POSTFIX in i_attr:
                loss = criterion(prediction, annotation)

            if mask:
                _mask = mask[i_attr]
                loss = (loss * _mask).sum()
                loss = (loss / _mask.sum()) if _mask.sum() != 0 else loss
            else:
                loss = loss.mean()

            loss = loss * weight

            list_loss.append(loss)
            dict_loss[f"loss_{i_attr}"] = loss.detach()

        return dict_loss, sum(list_loss)
