import torch
import torch.nn as nn

from trainer.common.constants import GATE_KEY, GATED_LOGIT_KEY, LOGIT_KEY, MULTI_SCALE_LOGIT_KEY
from trainer.common.models.modules.gate import GateBlock


class Classifier(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        drop_prob,
        target_attr_total,
        target_attr_to_train,
        target_attr_downstream,
        return_logit,
    ):
        super(Classifier, self).__init__()
        self.target_attr_total = target_attr_total
        self.target_attr_to_train = target_attr_to_train
        self.target_attr_downstream = target_attr_downstream
        self.return_logit = return_logit

        self.classifiers = nn.ModuleDict(
            {
                i_attr: nn.Sequential(
                    nn.Dropout(p=drop_prob),
                    nn.Linear(in_planes, out_planes, bias=True),
                )
                for i_attr in target_attr_total
            }
        )

    def forward(self, inputs: list) -> dict:
        x = inputs[-1]  # (B, 192, 6, 9, 9)

        # Loop for attributions
        logits = dict()
        for i_attr in self.target_attr_to_train:
            x = nn.AvgPool3d(x.size()[-3:])(x)  # (B, 192, 1, 1, 1)
            f_flatten = torch.flatten(x, start_dim=1)  # (B, 192)
            logits[i_attr] = self.classifiers[i_attr](f_flatten)  # (B, 1)

        if self.return_logit:
            return logits[self.target_attr_downstream]

        return {LOGIT_KEY: logits}


class DualScaleClassifier(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim,
        drop_prob,
        target_attr_total,
        target_attr_to_train,
        target_attr_downstream,
    ):
        super().__init__()

        self.target_attr_total = target_attr_total
        self.target_attr_to_train = target_attr_to_train
        self.target_attr_downstream = target_attr_downstream

        # Feature fusion layers for each attribute
        self.fusion_layers = nn.ModuleDict(
            {
                i_attr: nn.Sequential(
                    nn.Linear(feature_dim * 2, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(drop_prob),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.BatchNorm1d(hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(drop_prob),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for i_attr in target_attr_total
            }
        )

    def forward(self, patch_features, image_features):
        # Concatenate features along feature dimension
        combined_features = torch.cat([patch_features, image_features], dim=1)  # (B, 2C)

        # Get predictions for each attribute
        logits = dict()
        for i_attr in self.target_attr_to_train:
            logits[i_attr] = self.fusion_layers[i_attr](combined_features)

        return {LOGIT_KEY: logits}


class MultiScaleAttnClassifier(nn.Module):
    def __init__(
        self,
        f_maps,
        num_levels,
        num_features,
        drop_prob,
        pyramid_channels,
        num_fpn_layers,
        target_attr_total,
        target_attr_to_train,
        target_attr_downstream,
        use_gate,
        use_coord,
        use_fusion,
    ):
        super(MultiScaleAttnClassifier, self).__init__()

        if isinstance(f_maps, int):
            f_maps = [f_maps * 2**k for k in range(num_levels)]  # e.g., [24, 48, 96, 192]
        self.num_features = num_features

        self.target_attr_total = target_attr_total
        self.target_attr_to_train = target_attr_to_train
        self.target_attr_downstream = target_attr_downstream
        self.use_gate = use_gate
        self.use_coord = use_coord
        self.use_fusion = use_fusion

        # f1, f2, f3 > g1, g2, g3_attr
        self.gate_block = (
            GateBlock(
                f_maps[-self.num_features :],
                pyramid_channels,
                num_fpn_layers,
                drop_prob=drop_prob,
                use_coord=use_coord,
                use_fusion=use_fusion,
                target_attr_total=target_attr_total,
            )
            if self.use_gate
            else None
        )

        # f1, f2, f3 > e1, e2, e3_attr
        self.classifiers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        i_attr: nn.Sequential(
                            nn.Dropout3d(p=drop_prob),
                            nn.Conv3d(f_map, 1, kernel_size=1, bias=True),
                        )
                        for i_attr in target_attr_total
                    }
                )
                for f_map in f_maps[-self.num_features :]
            ]
        )

    def forward(self, inputs: list) -> dict:
        f_maps = inputs[-self.num_features :]  # (B, 48, 24, 36, 36), (B, 96, 12, 18, 18), (B, 192, 6, 9, 9)

        # get gate results
        gate_results = list()
        if self.use_gate:
            gate_results = self.gate_block(f_maps)
            gates_flatten = [torch.flatten(i_gate, start_dim=1) for i_gate in gate_results]
            gates_total = torch.cat(gates_flatten, dim=1).sum(1, keepdim=True)

        else:
            gates_total = 0
            for x in f_maps:
                gates_total += torch.prod(torch.tensor(x.size()[-3:]))

        # Loop for attributions
        logits = dict()
        logits_multi_scale = [dict() for _ in range(self.num_features)]
        gated_ce_dict = dict()
        for i_attr in self.target_attr_to_train:
            # init
            gated_ce_dict[i_attr] = list()
            logits[i_attr] = None

            # Loop for multi-scale
            _logits = list()
            _gates = list()
            for idx_fmap, (x, classifier) in enumerate(zip(f_maps, self.classifiers)):
                # local class evidence
                if self.use_gate:
                    CE = classifier[i_attr](x)
                    CE = CE * gate_results[idx_fmap]
                    total_interest = torch.sum(gate_results[idx_fmap], dim=(2, 3, 4), keepdim=True)
                else:
                    CE = classifier[i_attr](x)
                    total_interest = torch.prod(torch.tensor(x.size()[-3:]))

                logits_multi_scale[idx_fmap][i_attr] = torch.sum(CE, dim=(2, 3, 4), keepdim=True) / total_interest
                gated_ce_dict[i_attr].append(CE)

                # spatial aggregation for local class evidence
                _logit = torch.sum(CE, dim=(2, 3, 4), keepdim=True)
                _logits.append(torch.flatten(_logit, start_dim=1))

            # aggregate probs provided from multi-scale
            logits[i_attr] = torch.cat(_logits, dim=1).sum(1, keepdim=True) / gates_total

        return {
            LOGIT_KEY: logits,
            MULTI_SCALE_LOGIT_KEY: logits_multi_scale,
            GATE_KEY: gate_results,
            GATED_LOGIT_KEY: gated_ce_dict,
        }
