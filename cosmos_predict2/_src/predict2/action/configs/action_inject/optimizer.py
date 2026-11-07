# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import PLACEHOLDER, LazyDict
from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.action.utils.action_inject_optim import get_action_inject_optimizer

ActionInjectFusedAdamWConfig: LazyDict = L(get_action_inject_optimizer)(
    model=PLACEHOLDER,
    lr=1e-4,
    weight_decay=0.1,
    backbone_lr_scale=0.1,
    betas=[0.9, 0.99],
    optim_type="fusedadam",
    eps=1e-8,
    master_weights=True,
    capturable=True,
)


def register_optimizer():
    ConfigStore.instance().store(
        group="optimizer", package="optimizer", name="action_inject_fusedadamw", node=ActionInjectFusedAdamWConfig
    )
