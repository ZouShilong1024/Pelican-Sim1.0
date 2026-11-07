# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fine-tune Pelican-Sim on your own episodes (e.g. RoboDojo), full-parameter.

Starts from the weights of ``FT_INIT_CKPT`` (a DCP ``iter_*`` dir; ``finetune.sh`` converts a
``model_ema_bf16.pt``). Optimizer, scheduler and iteration start fresh. All parameters train:
action modules at ``FT_LR`` and the rest at ``0.1 * FT_LR``, with a 100-step warmup.

Two experiments:

    ac_inject_moe_finetune_2b     single view (head camera), 93-frame windows (state_t 24)
    ac_inject_moe_finetune_mv_2b  RoboDojo multi-view: head + both wrists packed onto one 480x640
                                  canvas (``compose_multiview``), ``FT_MV_NUM_FRAMES``-frame windows
                                  (13 = one policy chunk, state_t 4); each sample draws its stride
                                  from ``FT_MV_STRIDES`` with weights ``FT_MV_STRIDE_PROBS``

Env (read at import; ``finetune.sh`` sets them):

    FT_INIT_CKPT        DCP iter dir to start from
    FT_EPISODES_JSONL   episode manifest (``scripts/prepare_robodojo.py``; ``--multiview`` for _mv)
    FT_STRIDE           model stride for episodes without ``temporal_stride`` (2)
    FT_LR               action-module LR (2**-15)
    FT_MAX_ITER         (10000)
    FT_SAVE_ITER        (1000)
    FT_SAMPLE_EVERY     EMA sample video every N iters (1000)
    FT_MV_NUM_FRAMES    multi-view window, 4k+1 pixel frames (13)
    FT_MV_STRIDES       multi-view stride choices (1,2,4 = 25 / 12.5 / 6.25 fps)
    FT_MV_STRIDE_PROBS  their weights (0.5,0.3,0.2)
"""

from __future__ import annotations

import os
from copy import deepcopy

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.callbacks.every_n_draw_sample_video import EveryNDrawSampleVideo
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_episode_action import EpisodeActionDataset
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_real_source_world_action import (
    build_action_delta_scaler,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_robodojo_action import MULTIVIEW_VIEWS
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import (
    get_generic_dataloader,
    get_sampler,
)
from cosmos_predict2.experiments.base.action_inject_moe import PARENT_EXPERIMENT

FT_INIT_CKPT = os.environ.get("FT_INIT_CKPT", "")
FT_EPISODES_JSONL = os.environ.get("FT_EPISODES_JSONL", "")
FT_STRIDE = int(os.environ.get("FT_STRIDE", "2"))
FT_LR = float(os.environ.get("FT_LR", str(2**-15)))
FT_MAX_ITER = int(os.environ.get("FT_MAX_ITER", "10000"))
FT_SAVE_ITER = int(os.environ.get("FT_SAVE_ITER", "1000"))
FT_SAMPLE_EVERY = int(os.environ.get("FT_SAMPLE_EVERY", "1000"))
FT_MV_NUM_FRAMES = int(os.environ.get("FT_MV_NUM_FRAMES", "13"))
FT_MV_STRIDES = [int(s) for s in os.environ.get("FT_MV_STRIDES", "1,2,4").split(",")]
FT_MV_STRIDE_PROBS = [float(p) for p in os.environ.get("FT_MV_STRIDE_PROBS", "0.5,0.3,0.2").split(",")]
if (FT_MV_NUM_FRAMES - 1) % 4:
    raise ValueError(f"FT_MV_NUM_FRAMES must be 4k+1 (Wan VAE temporal compression), got {FT_MV_NUM_FRAMES}")


def _episodes(start_from_first_frame_prob: float, **kwargs):
    return L(EpisodeActionDataset)(
        episodes_jsonl=FT_EPISODES_JSONL,
        num_frames=93,
        video_size=(480, 640),
        verify_files=True,
        target_fps=16.0,
        start_from_first_frame_prob=start_from_first_frame_prob,
        include_gripper=True,
        action_delta_scaler=build_action_delta_scaler(include_gripper=True, include_lift=False),
        temporal_stride=FT_STRIDE,
        min_temporal_stride=1,
        **kwargs,
    )


def _dataloader(dataset):
    return L(get_generic_dataloader)(
        dataset=dataset,
        sampler=L(get_sampler)(dataset=dataset),
        batch_size=1,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )


def _sample_callback(dataset):
    return L(EveryNDrawSampleVideo)(
        every_n=FT_SAMPLE_EVERY,
        is_ema=True,
        guidance=[3.0],
        save_s3=False,
        do_x0_prediction=False,
        dataset=dataset,
    )


dataset_train = _episodes(0.3)

ac_inject_moe_finetune_2b = LazyDict(
    dict(
        defaults=[
            PARENT_EXPERIMENT,
            {"override /model": "action_inject_moe_video2world_fsdp_rectified_flow"},
            {"override /optimizer": "action_inject_fusedadamw"},
            {"override /conditioner": "action_inject_video_conditioner"},
            {"override /data_train": "mock"},
            {"override /data_val": "mock"},
            {"override /net": "cosmos_v1_2B_action_inject_moe_e8"},
            "_self_",
        ],
        job=dict(project="pelican_sim", group="finetune", name="pelican_sim_finetune"),
        upload_reproducible_setup=False,
        optimizer=dict(lr=FT_LR, weight_decay=0.1, backbone_lr_scale=0.1),
        # 100-step warmup, linear decay to 0.4x at FT_MAX_ITER, then hold.
        scheduler=dict(
            f_start=[1e-6, 0.4],
            f_max=[1.0, 0.4],
            f_min=[0.4, 0.4],
            warm_up_steps=[100, 0],
            cycle_lengths=[FT_MAX_ITER, 1_000_000_000],
        ),
        checkpoint=dict(
            save_iter=FT_SAVE_ITER,
            load_path=FT_INIT_CKPT,
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
        trainer=dict(
            max_iter=FT_MAX_ITER,
            logging_iter=50,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                iter_speed=dict(hit_thres=200),
                device_monitor=dict(every_n=100),
                dataloader_speed=dict(every_n=100),
                every_n_sample_reg=dict(every_n=10**12),
                every_n_sample_ema=_sample_callback(_episodes(1.0)),
            ),
        ),
        dataloader_train=_dataloader(dataset_train),
    ),
    flags={"allow_objects": True},
)

dataset_train_mv = _episodes(
    0.3,
    temporal_stride_choices=FT_MV_STRIDES,
    temporal_stride_probs=FT_MV_STRIDE_PROBS,
    views=MULTIVIEW_VIEWS,
)
dataset_train_mv.num_frames = FT_MV_NUM_FRAMES
# Fixed smallest-stride clips from frame 0, so successive sample videos stay comparable.
dataset_sample_mv = _episodes(1.0, views=MULTIVIEW_VIEWS)
dataset_sample_mv.num_frames = FT_MV_NUM_FRAMES
dataset_sample_mv.temporal_stride = min(FT_MV_STRIDES)
dataset_sample_mv.min_temporal_stride = min(FT_MV_STRIDES)

ac_inject_moe_finetune_mv_2b = deepcopy(ac_inject_moe_finetune_2b)
ac_inject_moe_finetune_mv_2b["job"]["name"] = "pelican_sim_finetune_mv"
ac_inject_moe_finetune_mv_2b["model"]["config"]["state_t"] = 1 + (FT_MV_NUM_FRAMES - 1) // 4
ac_inject_moe_finetune_mv_2b["dataloader_train"] = _dataloader(dataset_train_mv)
ac_inject_moe_finetune_mv_2b["trainer"]["callbacks"]["every_n_sample_ema"] = _sample_callback(dataset_sample_mv)

cs = ConfigStore.instance()
cs.store(group="experiment", package="_global_", name="ac_inject_moe_finetune_2b", node=ac_inject_moe_finetune_2b)
cs.store(group="experiment", package="_global_", name="ac_inject_moe_finetune_mv_2b", node=ac_inject_moe_finetune_mv_2b)
