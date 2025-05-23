# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
from typing import Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from monai.networks.blocks import MLPBlock as Mlp, PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from torch.nn import LayerNorm

from trainer.common.models.swin_unetr import SwinTransformer
from trainer.common.constants import LOGIT_KEY, MULTI_SCALE_LOGIT_KEY, GATE_KEY, GATED_LOGIT_KEY

rearrange, _ = optional_import("einops", name="rearrange")


class swin_classifier(nn.Module):
    """
    Swin UNETR based classifier adapted from segmentation model
    """

    def __init__(
        self,
        img_size: tuple = (32, 224, 224),  # D, H, W
        in_channels: int = 3,
        num_classes: int = 1,
        feature_size: int = 48,
        depths: tuple = (2, 2, 2, 2),
        num_heads: tuple = (3, 6, 12, 24),
        norm_name: str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        classifier: nn.Module = None,
    ) -> None:
        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        if not (spatial_dims == 2 or spatial_dims == 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        self.normalize = normalize
        self.num_classes = num_classes
        self.spatial_dims = spatial_dims

        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
        )

        self.avg_pool = nn.AdaptiveAvgPool3d(1) if spatial_dims == 3 else nn.AdaptiveAvgPool2d(1)

        self.classifier = classifier

    def forward(self, x_in):
        hidden_states_out = self.swinViT(x_in, self.normalize)
        
        if self.classifier is not None:
            result = self.classifier(hidden_states_out)
            return result
        else:
            x = hidden_states_out[-1]
            if self.spatial_dims == 3:
                x = self.avg_pool(x)
            else:
                x = self.avg_pool(x) 
                
            x = torch.flatten(x, 1)
            return {LOGIT_KEY: x}

    def load_from(self, weights):
        with torch.no_grad():
            self.swinViT.patch_embed.proj.weight.copy_(weights["state_dict"]["module.patch_embed.proj.weight"])
            self.swinViT.patch_embed.proj.bias.copy_(weights["state_dict"]["module.patch_embed.proj.bias"])
            for bname, block in self.swinViT.layers1[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers1")
            self.swinViT.layers1[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.reduction.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers2[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers2")
            self.swinViT.layers2[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.reduction.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers3[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers3")
            self.swinViT.layers3[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.reduction.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers4[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers4")
            self.swinViT.layers4[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.reduction.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.bias"]
            )


if __name__ == "__main__":
    # Example usage
    model = swin_classifier(
        img_size=(32, 32, 32),  # D, H, W
        in_channels=1,            # RGB input
        num_classes=1,            # Binary classification
        feature_size=16,          # Base feature size
        depths=(2, 2, 2, 2),      # Number of blocks in each stage
        num_heads=(2, 4, 8, 16), # Number of attention heads in each stage
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        spatial_dims=3,
    ).cuda()

    # Test with a random input
    x = torch.randn(1, 1, 32, 32, 32).cuda()  # B, C, D, H, W
    with torch.no_grad():
        y = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output keys: {y.keys()}")
        print(f"Logits shape: {y[LOGIT_KEY].shape}")  # Should be (1,) for binary classification

    del model, x, y
    torch.cuda.empty_cache()
