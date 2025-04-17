import omegaconf
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, smoothing=0.0, target_threshold_gte=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # can be considered to either the positive annotation ratio or the weight for negatives
        self.smoothing = smoothing
        self.target_threshold_gte = target_threshold_gte
        self.eps = torch.finfo(torch.float32).eps
        if isinstance(alpha, omegaconf.listconfig.ListConfig):
            alpha = omegaconf.OmegaConf.to_container(alpha, resolve=True)
            self.alpha = torch.Tensor(alpha)
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        elif isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)

    def forward(self, input, target, is_logit=True, is_logistic=True):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # (B, C, N)
            input = input.transpose(1, 2)  # (B, N, C)
            input = input.contiguous().view(-1, input.size(2))  # (B * N, C)
        if target.dim() > 2:
            target = target.view(target.size(0), target.size(1), -1)  # (B, C, N)
            target = target.transpose(1, 2)  # (B, N, C)
            target = target.contiguous().view(-1, target.size(2))  # (B * N, C)
        target = target.view(-1, 1)  # (B * N, C)

        # (B * N, 1) # smooth target을 고려해서 round 연산
        if isinstance(self.target_threshold_gte, float):
            target_int = (target >= self.target_threshold_gte).long()
            target = target_int
        else:
            target_int = target.round().data.long()

        # get probability
        if is_logistic:
            pt = torch.sigmoid(input) if is_logit else input
            pt = torch.cat([1 - pt, pt], dim=1) + self.eps  # (B * N, 2)

        else:
            pt = torch.softmax(input, dim=1) if is_logit else input  # (B * N, C)

        # get log-likelihood
        logpt = torch.log(pt)  # (B * N, C)

        # get target
        if isinstance(self.smoothing, float):
            if is_logistic:
                smoothed_target = torch.cat([1 - target, target], dim=1)
                smoothed_target = smoothed_target * (1 - self.smoothing) + self.smoothing / 2
            else:
                # multi-class target assumed to be [B, num_classes]
                smoothed_target = target * (1 - self.smoothing) + self.smoothing / target.size(1)
        else:
            smoothed_target = torch.cat([1 - target, target], dim=1)  # (B * N, C)

        # Focal loss
        loss = -1 * ((1 - pt) ** self.gamma * logpt * smoothed_target).sum(dim=1, keepdims=True)  # (B * N, 1)

        # alpha
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)  # (C,)
            at = self.alpha.gather(0, target_int.view(-1))
            loss = loss * at  # Apply alpha

        return loss  # (B * N)
