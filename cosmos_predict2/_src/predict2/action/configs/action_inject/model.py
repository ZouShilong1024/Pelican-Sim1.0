# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.action.models.action_inject_moe_video2world_rectified_flow_model import (
    ActionInjectMoEVideo2WorldModelRectifiedFlow,
    ActionInjectMoEVideo2WorldModelRectifiedFlowConfig,
)

FSDP_RECTIFIED_FLOW_MOE_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(ActionInjectMoEVideo2WorldModelRectifiedFlow)(
        config=ActionInjectMoEVideo2WorldModelRectifiedFlowConfig(
            fsdp_shard_size=8,
            state_t=24,
        ),
        _recursive_=False,
    ),
)


def register_model():
    ConfigStore.instance().store(
        group="model",
        package="_global_",
        name="action_inject_moe_video2world_fsdp_rectified_flow",
        node=FSDP_RECTIFIED_FLOW_MOE_CONFIG,
    )
