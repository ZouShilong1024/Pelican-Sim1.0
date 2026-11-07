# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Action-video VACE branch: a parallel stream of RGB DiT ``Block`` copies."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from cosmos_predict2._src.predict2.networks.minimal_v4_dit import Block, PatchEmbed


def _unwrap_checkpoint_module(module: nn.Module) -> nn.Module:
    """Unwrap ``ptd_checkpoint_wrapper`` modules."""
    while hasattr(module, "_checkpoint_wrapped_module"):
        module = module._checkpoint_wrapped_module  # type: ignore[assignment]
    return module


def _safe_zero_(param: torch.Tensor) -> None:
    """``nn.init.zeros_`` that also works on DTensor."""
    with torch.no_grad():
        param.zero_()


def _safe_eye_(param: torch.Tensor) -> None:
    """``nn.init.eye_`` that also works on DTensor."""
    with torch.no_grad():
        try:
            from torch.distributed.tensor import DTensor, distribute_tensor
        except Exception:  # noqa: BLE001
            DTensor = ()  # type: ignore[assignment]
            distribute_tensor = None  # type: ignore[assignment]

        if DTensor and isinstance(param, DTensor) and distribute_tensor is not None:
            eye = torch.eye(param.size(0), param.size(1), dtype=param.dtype, device=param.device)
            param.copy_(distribute_tensor(eye, device_mesh=param.device_mesh, placements=param.placements))
        else:
            nn.init.eye_(param)


class ActionVideoVaceBlock(Block):
    """VACE context block = RGB ``Block`` + skip projections.

    Block 0 computes ``c = before_proj(c) + x``; every block emits ``after_proj(c)`` as a hint.
    ``after_proj`` is zero-initialized (no effect at init), ``before_proj`` identity-initialized.
    """

    def __init__(self, *args, block_id: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.x_dim, self.x_dim)
        self.after_proj = nn.Linear(self.x_dim, self.x_dim)

    def init_weights(self) -> None:
        super().init_weights()
        _safe_zero_(self.after_proj.weight)
        _safe_zero_(self.after_proj.bias)
        if hasattr(self, "before_proj"):
            _safe_eye_(self.before_proj.weight)
            if self.before_proj.bias is not None:
                _safe_zero_(self.before_proj.bias)

    def forward(
        self,
        c: torch.Tensor,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self.block_id == 0:
            c = self.before_proj(c) + x_B_T_H_W_D
            all_c: List[torch.Tensor] = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)

        c = super().forward(
            c,
            emb_B_T_D,
            crossattn_emb,
            rope_emb_L_1_1_D=rope_emb_L_1_1_D,
            adaln_lora_B_T_3D=adaln_lora_B_T_3D,
            extra_per_block_pos_emb=extra_per_block_pos_emb,
            **kwargs,
        )
        all_c += [self.after_proj(c), c]
        return torch.stack(all_c)


class ActionVideoVaceModel(nn.Module):
    """Patch-embeds the action-video latents and runs ``len(vace_layers)`` VACE blocks.

    Returns one hint per entry of ``vace_layers`` (the main DiT blocks that receive them).
    """

    def __init__(
        self,
        vace_layers: Sequence[int],
        vace_in_dim: int = 16,
        patch_spatial: int = 2,
        patch_temporal: int = 1,
        model_channels: int = 2048,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        context_dim: int = 1024,
        use_adaln_lora: bool = True,
        adaln_lora_dim: int = 256,
        backend: str = "transformer_engine",
        use_wan_fp32_strategy: bool = False,
        image_context_dim: Optional[int] = None,
        scale: float = 1.0,
    ):
        super().__init__()
        self.vace_layers = tuple(int(i) for i in vace_layers)
        self.scale = scale
        self.vace_patch_embedding = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=vace_in_dim,
            out_channels=model_channels,
        )
        self.vace_blocks = nn.ModuleList(
            [
                ActionVideoVaceBlock(
                    x_dim=model_channels,
                    context_dim=context_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                    backend=backend,
                    image_context_dim=image_context_dim,
                    use_wan_fp32_strategy=use_wan_fp32_strategy,
                    block_id=i,
                )
                for i in range(len(self.vace_layers))
            ]
        )
        self.init_weights()

    def init_weights(self) -> None:
        self.vace_patch_embedding.init_weights()
        for block in self.vace_blocks:
            _unwrap_checkpoint_module(block).init_weights()

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        vace_context: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        if vace_context.dim() == 4:
            vace_context = vace_context.unsqueeze(0)

        c = self.vace_patch_embedding(vace_context)  # [B, T', H', W', D]
        B, T, H, W, D = x_B_T_H_W_D.shape
        if c.shape[1:4] != (T, H, W):
            c_nchw = rearrange(c, "b t h w d -> b d t h w")
            c_nchw = F.interpolate(c_nchw.float(), size=(T, H, W), mode="trilinear", align_corners=False)
            c = rearrange(c_nchw.to(dtype=c.dtype), "b d t h w -> b t h w d")

        for block in self.vace_blocks:
            c = block(
                c,
                x_B_T_H_W_D,
                emb_B_T_D,
                crossattn_emb,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_per_block_pos_emb,
            )
        return list(torch.unbind(c))[:-1]
