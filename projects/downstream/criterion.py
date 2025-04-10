import torch.nn as nn


class ClsLoss(nn.Module):
    def __init__(self, cls_criterion):
        super(ClsLoss, self).__init__()
        self.cls_criterion = cls_criterion

    def forward(self, pred, annot, is_logit=True, is_logistic=True):
        loss = self.cls_criterion(pred, annot, is_logit=is_logit, is_logistic=is_logistic).mean()

        return loss
