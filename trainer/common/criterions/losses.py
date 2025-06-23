import torch
import torch.nn as nn


class GDLLoss(nn.Module):
    def __init__(self, pNorm=1, loss_weight=1.0):
        super(GDLLoss, self).__init__()
        self.convX = nn.Conv3d(1, 1, kernel_size=(1, 1, 2), stride=1, padding=(0, 0, 1), bias=False)
        self.convY = nn.Conv3d(1, 1, kernel_size=(1, 2, 1), stride=1, padding=(0, 1, 0), bias=False)
        self.convZ = nn.Conv3d(1, 1, kernel_size=(2, 1, 1), stride=1, padding=(1, 0, 0), bias=False)

        filterX = torch.FloatTensor([[[[[1, -1]]]]])  # 1x1x1x2
        filterY = torch.FloatTensor([[[[[1], [-1]]]]])  # 1x1x2x1
        filterZ = torch.FloatTensor([[[[[1]], [[-1]]]]])  # 1x2x1x1

        self.convX.weight = torch.nn.Parameter(filterX, requires_grad=False)
        self.convY.weight = torch.nn.Parameter(filterY, requires_grad=False)
        self.convZ.weight = torch.nn.Parameter(filterZ, requires_grad=False)
        self.pNorm = pNorm
        self.loss_weight = loss_weight

    def forward(self, pred):
        assert pred.dim() == 5
        gt = torch.zeros_like(pred)
        pred_dx = torch.abs(self.convX.to(pred.device)(pred))
        pred_dy = torch.abs(self.convY.to(pred.device)(pred))
        pred_dz = torch.abs(self.convZ.to(pred.device)(pred))

        gt_dx = torch.abs(self.convX.to(pred.device)(gt))
        gt_dy = torch.abs(self.convY.to(pred.device)(gt))
        gt_dz = torch.abs(self.convZ.to(pred.device)(gt))

        grad_diff_x = torch.abs(gt_dx - pred_dx)
        grad_diff_y = torch.abs(gt_dy - pred_dy)
        grad_diff_z = torch.abs(gt_dz - pred_dz)

        mat_loss_x = grad_diff_x**self.pNorm
        mat_loss_y = grad_diff_y**self.pNorm  # Batch x Channel x width x height
        mat_loss_z = grad_diff_z**self.pNorm

        mean_loss = (torch.sum(mat_loss_x) + torch.sum(mat_loss_y) + torch.sum(mat_loss_z)) / torch.flatten(pred).size(
            0
        )

        return mean_loss * self.loss_weight


class EntropyLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(EntropyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = 1e-06

    def forward(self, probs, epoch=None, total_epoch=None):
        if epoch is not None and total_epoch is not None:
            weight = self.loss_weight * (1 - (epoch / total_epoch))
        else:
            weight = self.loss_weight

        term1 = probs * torch.log(probs + self.eps)
        term2 = (1 - probs) * torch.log(1 - probs + self.eps)
        entropy = -(term1 + term2)
        loss = -1 * entropy.mean(dim=(2, 3, 4)).squeeze()  # (B,)

        return loss.mean() * weight
