# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimizer for action-inject fine-tuning: action modules at ``lr``, backbone at ``lr * backbone_lr_scale``."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import ListConfig
from torch import nn

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.action.networks.action_inject_minimal_v1_lvg_dit import is_action_param_name
from cosmos_predict2._src.predict2.utils.fused_adam_dtensor import FusedAdam


def get_action_inject_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    backbone_lr_scale: float = 0.1,
    optim_type: str = "fusedadam",
    **kwargs: Any,
) -> torch.optim.Optimizer:
    action_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (action_params if is_action_param_name(name) else backbone_params).append(param)

    param_groups = [
        {"params": action_params, "lr": lr, "weight_decay": weight_decay, "name": "action_modules"},
        {"params": backbone_params, "lr": lr * backbone_lr_scale, "weight_decay": weight_decay, "name": "backbone"},
    ]
    log.critical(
        f"action_inject optimizer: action={sum(p.numel() for p in action_params):,} @ lr={lr:.2e}, "
        f"backbone={sum(p.numel() for p in backbone_params):,} @ lr={lr * backbone_lr_scale:.2e}"
    )

    opt_cls = {"adamw": torch.optim.AdamW, "fusedadam": FusedAdam}[optim_type]
    kwargs = {k: list(v) if isinstance(v, ListConfig) else v for k, v in kwargs.items()}
    return opt_cls(param_groups, **kwargs)
