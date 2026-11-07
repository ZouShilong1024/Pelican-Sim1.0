# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EveryNDrawSample variant that samples fixed clips of one dataset and saves mp4s.

Instead of the training batch + a 3-frame jpg grid, every data-parallel rank
generates one fixed clip from ``dataset`` and writes, under
``<job>/<name>/iter_XXXXXXXXX/``::

    {tag}_rank{RR}_{scene}_{episode}_pred.mp4      generated video (per guidance: _g{G})
    {tag}_rank{RR}_{scene}_{episode}_gt.mp4        source clip
    {tag}_rank{RR}_{scene}_{episode}_compare.mp4   gt | pred side by side

Clips are fixed across sampling events (same episode, same window), so the
videos of different iterations are directly comparable. Pass a dataset built
with ``start_from_first_frame_prob=1.0`` to make the window deterministic.
"""

from __future__ import annotations

import os
from collections import defaultdict
from contextlib import nullcontext
from functools import partial
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
import wandb
from megatron.core import parallel_state
from torch.utils.data import default_collate

from cosmos_predict2._src.imaginaire.utils import distributed, log, misc
from cosmos_predict2._src.imaginaire.utils.parallel_state_helper import is_tp_cp_pp_rank0
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_predict2._src.predict2.callbacks.every_n_draw_sample import EveryNDrawSample


def _to01(x: torch.Tensor) -> torch.Tensor:
    return (x.float().clamp(-1, 1) + 1.0) / 2.0


class EveryNDrawSampleVideo(EveryNDrawSample):
    """Args (in addition to ``EveryNDrawSample``):

    dataset: Map-style dataset returning training-format items (``video``,
        ``ai_caption``, ``fps``, action keys, ...). Samples must expose
        ``task_name`` / ``video_path`` (Robotwin-style ``samples`` list).
    sample_indices: Explicit dataset indices; rank ``r`` uses
        ``sample_indices[r % len]``. Default: one episode per rank from
        evenly spaced scenes (middle episode of each scene).
    seed: numpy seed used while loading an item (restored afterwards).
    """

    def __init__(
        self,
        *args,
        dataset: Any = None,
        sample_indices: Optional[Sequence[int]] = None,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if dataset is None:
            raise ValueError("EveryNDrawSampleVideo requires a dataset")
        self.dataset = dataset
        self.sample_indices = list(sample_indices) if sample_indices else None
        self.seed = int(seed)
        self._index: Optional[int] = None

    def on_train_start(self, model, iteration: int = 0) -> None:
        super().on_train_start(model, iteration)
        os.makedirs(self.local_dir, exist_ok=True)
        if parallel_state.is_initialized():
            dp_size = parallel_state.get_data_parallel_world_size()
        else:
            dp_size = distributed.get_world_size()
        self._index = self._pick_index(self.data_parallel_id, dp_size)
        s = self.dataset.samples[self._index]
        log.info(
            f"{self.name}: dp_rank={self.data_parallel_id} samples #{self._index} "
            f"{getattr(s, 'task_name', '')} {os.path.basename(s.video_path)}",
            rank0_only=False,
        )

    def _pick_index(self, dp_rank: int, dp_size: int) -> int:
        if self.sample_indices:
            return int(self.sample_indices[dp_rank % len(self.sample_indices)]) % len(self.dataset)
        by_scene: dict[str, list[int]] = defaultdict(list)
        for i, s in enumerate(self.dataset.samples):
            by_scene[str(getattr(s, "task_name", ""))].append(i)
        scenes = sorted(by_scene)
        scene = scenes[(dp_rank * len(scenes)) // max(dp_size, 1) % len(scenes)]
        idxs = by_scene[scene]
        return idxs[len(idxs) // 2]

    def _load_batch(self, model) -> dict:
        state = np.random.get_state()
        np.random.seed(self.seed)
        try:
            item = self.dataset[self._index]
        finally:
            np.random.set_state(state)
        batch = misc.to(default_collate([item]), device="cuda")
        # Training batches get this cast from LowPrecisionCallback; a float32 padding_mask
        # would otherwise promote the patch-embed input to float32 against bf16 weights.
        precision = getattr(model, "precision", None)
        if precision is not None:
            for k, v in batch.items():
                if torch.is_tensor(v) and torch.is_floating_point(v):
                    batch[k] = v.to(dtype=precision)
        return batch

    @torch.no_grad()
    def every_n_impl(self, trainer, model, data_batch, output_batch, loss, iteration):
        if self.is_ema:
            if not model.config.ema.enabled:
                return
            context = partial(model.ema_scope, "every_n_sampling")
        else:
            context = nullcontext

        batch = self._load_batch(model)
        with context():
            compare_fp = self.sample_video(model, batch, iteration)
            dist.barrier()

        if wandb.run and self.rank == 0 and compare_fp:
            tag = "ema" if self.is_ema else "reg"
            wandb.log(
                {f"{self.name}/{tag}_compare": wandb.Video(compare_fp)},
                step=iteration,
            )
        torch.cuda.empty_cache()

    @misc.timer("EveryNDrawSampleVideo: sample")
    def sample_video(self, model, data_batch: dict, iteration: int) -> Optional[str]:
        tag = "ema" if self.is_ema else "reg"
        text_encoder_config = getattr(model.config, "text_encoder_config", None)
        if text_encoder_config is not None and text_encoder_config.compute_online:
            text_embeddings = model.text_encoder.compute_text_embeddings_online(data_batch, model.input_caption_key)
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(
                text_embeddings.shape[0], text_embeddings.shape[1], device="cuda"
            )

        raw_data, x0, _ = model.get_data_and_condition(data_batch)
        preds = []
        for guidance in self.guidance:
            sample = model.generate_samples_from_batch(
                data_batch,
                guidance=guidance,
                state_shape=x0.shape[1:],
                n_sample=x0.shape[0],
                num_steps=self.num_sampling_step,
                is_negative_prompt=False,
            )
            if hasattr(model, "decode"):
                sample = model.decode(sample)
            preds.append(_to01(sample[0]).cpu())

        if not is_tp_cp_pp_rank0():
            return None

        fps = float(torch.as_tensor(data_batch["fps"]).flatten()[0])
        s = self.dataset.samples[self._index]
        scene = str(getattr(s, "task_name", "")) or "sample"
        episode = os.path.splitext(os.path.basename(s.video_path))[0]
        out_dir = os.path.join(self.local_dir, f"iter_{iteration:09d}")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.join(out_dir, f"{tag}_rank{self.data_parallel_id:02d}_{scene}_{episode}")

        gt = _to01(raw_data[0]).cpu()
        save_img_or_video(gt, f"{base}_gt", fps=fps)
        multi = len(preds) > 1
        for guidance, pred in zip(self.guidance, preds):
            save_img_or_video(pred, f"{base}_pred_g{guidance:g}" if multi else f"{base}_pred", fps=fps)
        save_img_or_video(torch.cat([gt, *preds], dim=-1), f"{base}_compare", fps=fps)
        return f"{base}_compare.mp4"
