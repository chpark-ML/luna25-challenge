import torch.nn as nn

from projects.common.criterions.focal_loss import FocalLoss


class ClsLoss(nn.Module):
    def __init__(self, gamma=0, alpha=0.5):
        super(ClsLoss, self).__init__()
        self.criterion = FocalLoss(gamma=gamma, alpha=alpha)

    def forward(self, pred, annot, is_logit=True, is_logistic=True):
        loss = self.criterion(pred, annot, is_logit=is_logit, is_logistic=is_logistic).mean()

        return loss
