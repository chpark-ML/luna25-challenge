import torch.nn as nn


class ClsLoss(nn.Module):
    def __init__(self, criterion):
        super(ClsLoss, self).__init__()
        self.criterion = criterion

    def forward(self, pred, annot, is_logit=True, is_logistic=True):
        loss = self.criterion(pred, annot, is_logit=is_logit, is_logistic=is_logistic).mean()

        return loss
