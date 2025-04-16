import torch.nn as nn

from trainer.common.constants import GATE_KEY, LOGIT_KEY, MULTI_SCALE_LOGIT_KEY, LossKey


class AttrLoss(nn.Module):
    def __init__(self, cls_criterion, entropy_criterion, aux_criterion):
        super(AttrLoss, self).__init__()
        self.cls_criterion = cls_criterion
        self.entropy_criterion = entropy_criterion
        self.aux_criterion = aux_criterion

    def forward(self, outputs, annot, mask=None, is_logit=True, is_logistic=True):
        losses = list()

        # cls loss
        cls_loss = None
        dict_loss = None
        if LOGIT_KEY in outputs:
            logits = outputs[LOGIT_KEY]  # dict
            dict_loss, cls_loss = self.cls_criterion(
                logits, annot, mask=mask, is_logit=is_logit, is_logistic=is_logistic
            )
            losses.append(cls_loss)

        # entropy loss for gate
        entropy_losses = list()
        if GATE_KEY in outputs:
            # list of tensors, e.g., [(B, 1, 24, 36, 36), (B, 1, 12, 18, 18), (B, 1, 6, 9 ,9)]
            gates = outputs[GATE_KEY]
            for gate in gates:  # loop for scale
                entropy_loss = self.entropy_criterion(gate)
                entropy_losses.append(entropy_loss)
                losses.append(entropy_loss)

        # auxiliary loss
        aux_losses = list()
        if MULTI_SCALE_LOGIT_KEY in outputs:
            ms_logits = outputs[MULTI_SCALE_LOGIT_KEY]  # list of dicts
            for aux_logits in ms_logits:  # loop for scale
                _, aux_loss = self.aux_criterion(
                    aux_logits, annot, mask=mask, is_logit=is_logit, is_logistic=is_logistic
                )
                aux_losses.append(aux_loss)
                losses.append(aux_loss)

        return {
            LossKey.total: sum(losses),
            LossKey.cls: cls_loss,
            LossKey.cls_dict: dict_loss,
            LossKey.entropy: sum(entropy_losses),
            LossKey.aux: sum(aux_losses),
        }
