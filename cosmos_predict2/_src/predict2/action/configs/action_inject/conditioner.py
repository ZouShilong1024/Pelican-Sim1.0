# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conditioner that forwards action_state / action_delta / action_video_latents to the net."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import (
    Video2WorldCondition,
    Video2WorldConditioner,
)
from cosmos_predict2._src.predict2.conditioner import BooleanFlag, ReMapkey, TextAttr


@dataclass(frozen=True)
class ActionInjectCondition(Video2WorldCondition):
    """Video2World condition + OminiEWM-style action fields."""

    action_state: Optional[torch.Tensor] = None
    action_delta: Optional[torch.Tensor] = None
    action_video_latents: Optional[torch.Tensor] = None
    # Absolute action sequence [B, T, D]; the net derives state / delta from it when those are absent.
    action: Optional[torch.Tensor] = None


class ActionInjectConditioner(Video2WorldConditioner):
    """Like Video2WorldConditioner, but copies the optional action keys from the batch."""

    OPTIONAL_ACTION_KEYS = ("action_state", "action_delta", "action_video_latents", "action")
    # Clone numeric action tensors so later in-place ops cannot alias into them.
    CLONE_ACTION_KEYS = ("action_state", "action_delta", "action")

    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> ActionInjectCondition:
        output = super()._forward(batch, override_dropout_rate)
        for key in self.OPTIONAL_ACTION_KEYS:
            if key not in batch or batch[key] is None:
                continue
            value = batch[key]
            if key in self.CLONE_ACTION_KEYS and isinstance(value, torch.Tensor):
                value = value.contiguous().clone()
            output[key] = value
        return ActionInjectCondition(**output)


_SHARED_CONFIG = dict(
    fps=L(ReMapkey)(
        input_key="fps",
        output_key="fps",
        dropout_rate=0.0,
        dtype=None,
    ),
    padding_mask=L(ReMapkey)(
        input_key="padding_mask",
        output_key="padding_mask",
        dropout_rate=0.0,
        dtype=None,
    ),
    text=L(TextAttr)(
        input_key=["t5_text_embeddings"],
        dropout_rate=0.2,
        use_empty_string=False,
    ),
    use_video_condition=L(BooleanFlag)(
        input_key="fps",
        output_key="use_video_condition",
        dropout_rate=0.2,
    ),
)

ActionInjectConditionerConfig: LazyDict = L(ActionInjectConditioner)(
    **_SHARED_CONFIG,
)


def register_conditioner():
    cs = ConfigStore.instance()
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="action_inject_video_conditioner",
        node=ActionInjectConditionerConfig,
    )
