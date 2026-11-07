# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference experiment for the Pelican-Sim 1.0 checkpoint (MoE 1 shared + 8 routed, top-2)."""

from __future__ import annotations

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict

PARENT_EXPERIMENT = (
    "/experiment/Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-"
    "Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only"
)

ac_inject_moe_shared_dense_e8_rectified_flow_2b = LazyDict(
    dict(
        defaults=[
            PARENT_EXPERIMENT,
            {"override /model": "action_inject_moe_video2world_fsdp_rectified_flow"},
            {"override /conditioner": "action_inject_video_conditioner"},
            {"override /data_train": "mock"},
            {"override /data_val": "mock"},
            {"override /net": "cosmos_v1_2B_action_inject_moe_e8"},
            "_self_",
        ],
        job=dict(
            project="cosmos_predict2_action_inject",
            group="cosmos_predict_v2p5",
            name="2b_action_inject_moe_shared_dense_e8k2_from_28d_all",
        ),
        checkpoint=dict(
            load_training_state=False,
            strict_resume=False,
            load_from_object_store=dict(enabled=False),
            save_to_object_store=dict(enabled=False),
        ),
        model_parallel=dict(context_parallel_size=1),
        model=dict(
            config=dict(
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                conditional_frames_probs=None,
                state_t=24,
            ),
        ),
    ),
    flags={"allow_objects": True},
)

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="ac_inject_moe_shared_dense_e8_rectified_flow_2b",
    node=ac_inject_moe_shared_dense_e8_rectified_flow_2b,
)
