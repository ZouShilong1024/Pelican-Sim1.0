# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pelican-Sim 1.0 net: 2B action-inject DiT with 1 shared + 8 routed experts (top-2)."""

from __future__ import annotations

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.action.networks.action_inject_moe_minimal_v1_lvg_dit import (
    ActionInjectMoEMinimalV1LVGDiT,
)
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import SACConfig

COSMOS_V1_2B_ACTION_INJECT_MOE_E8: LazyDict = L(ActionInjectMoEMinimalV1LVGDiT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="minimal_a2a",
    extra_per_block_abs_pos_emb=False,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=1.0,
    sac_config=SACConfig(),
    # Action value: scale-shift on odd blocks. Action video: VACE hints on even blocks.
    action_dim=28,
    action_intermediate_dim=1024,
    action_chunk_size=4,
    action_inject_layers=list(range(1, 28, 2)),
    action_video_inject_layers=list(range(0, 28, 2)),
    action_video_in_channels=16,
    vace_action_scale=1.0,
    num_experts=8,
    num_experts_per_tok=2,
    moe_score_func="sigmoid",
    moe_norm_topk_prob=True,
    moe_route_scale=2.5,
    moe_aux_loss_coeff=0.01,
)


def register_net():
    ConfigStore.instance().store(
        group="net",
        package="model.config.net",
        name="cosmos_v1_2B_action_inject_moe_e8",
        node=COSMOS_V1_2B_ACTION_INJECT_MOE_E8,
    )
