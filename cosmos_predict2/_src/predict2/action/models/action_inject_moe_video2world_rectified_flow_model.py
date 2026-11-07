# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video2World rectified-flow model for the action-inject MoE DiT.

On top of ``Video2WorldModelRectifiedFlow`` this model
  - VAE-encodes ``action_video`` into ``action_video_latents`` before training / sampling,
  - builds a LambdaLR that drives every optimizer param group (action modules + backbone),
  - adds the MoE load-balance loss to the rectified-flow loss.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import attrs
import hydra
import torch
from torch import Tensor

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.lazy_config import instantiate as lazy_instantiate
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import Video2WorldCondition
from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import (
    Video2WorldModelRectifiedFlow,
    Video2WorldModelRectifiedFlowConfig,
)


@attrs.define(slots=False)
class ActionInjectMoEVideo2WorldModelRectifiedFlowConfig(Video2WorldModelRectifiedFlowConfig):
    pass


class ActionInjectMoEVideo2WorldModelRectifiedFlow(Video2WorldModelRectifiedFlow):
    def _maybe_encode_action_video(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if data_batch.get("action_video_latents") is not None or data_batch.get("action_video") is None:
            return data_batch
        action_video = data_batch["action_video"]
        if action_video.dtype == torch.uint8:
            action_video = action_video.to(**self.tensor_kwargs) / 127.5 - 1.0
        else:
            action_video = action_video.to(**self.tensor_kwargs)
        with torch.no_grad():
            data_batch["action_video_latents"] = self.encode(action_video)
        return data_batch

    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[Tensor, Tensor, Video2WorldCondition]:
        return super().get_data_and_condition(self._maybe_encode_action_video(data_batch))

    def get_velocity_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:
        return super().get_velocity_fn_from_batch(
            self._maybe_encode_action_video(data_batch), guidance=guidance, is_negative_prompt=is_negative_prompt
        )

    def init_optimizer_scheduler(
        self, optimizer_config: LazyDict, scheduler_config: LazyDict
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        optimizer = lazy_instantiate(optimizer_config, model=self.net)
        net_scheduler = hydra.utils.instantiate(scheduler_config)
        net_scheduler.model = self
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=[net_scheduler.schedule] * len(optimizer.param_groups)
        )
        return optimizer, scheduler

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        output, loss = super().training_step(data_batch, iteration)
        output = dict(output)
        output["rf_loss"] = loss.detach()
        aux = self.net.pop_moe_aux_loss()
        if aux is not None:
            aux = aux.to(device=loss.device, dtype=loss.dtype)
            output["moe_aux_loss"] = aux.detach()
            loss = loss + aux
            if iteration % 50 == 0:
                log.info(
                    f"[MoE] iter={iteration} rf_loss={output['rf_loss'].float().item():.5f} "
                    f"moe_aux={aux.detach().float().item():.5f}"
                )
        return output, loss
