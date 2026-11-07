# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video2World DiT with action-value and action-video conditioning.

- Action value: ``ActionEncoder`` embeds (state, deltas); blocks in ``action_inject_layers``
  apply a scale-shift after self-attention.
- Action video: the VAE latents of the rendered skeleton video run through a parallel stream of
  DiT-block copies (VACE); its hints are added after the blocks in ``action_video_inject_layers``.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.action.networks.action_encoder import ActionEncoder
from cosmos_predict2._src.predict2.action.networks.action_video_modules import ActionVideoVaceModel
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import Block, MiniTrainDIT, SACConfig, VideoSize

ACTION_PARAM_NAME_PARTS: Tuple[str, ...] = (
    "action_encoder",
    "action_proj_layer1",
    "action_proj_layer2",
    "action_injector",
    "action_scale",
    "action_shift",
    "vace_action",
)


def is_action_param_name(name: str) -> bool:
    return any(part in name for part in ACTION_PARAM_NAME_PARTS)


class ActionInjectBlock(Block):
    """DiT block with an optional action scale-shift between self- and cross-attention.

    Scale / shift layers are zero-initialized, so a new block equals the pretrained one.
    """

    def __init__(self, *args, has_action_injection: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_action_injection = has_action_injection
        if has_action_injection:
            dim = self.x_dim
            self.action_injector_layer1 = nn.Linear(dim, dim)
            self.action_injector_layer2 = nn.Linear(dim, dim)
            self.action_scale_layer = nn.Linear(dim, dim)
            self.action_shift_layer = nn.Linear(dim, dim)
            self._init_action_injection_weights()

    def _init_action_injection_weights(self) -> None:
        if not self.has_action_injection:
            return
        nn.init.xavier_uniform_(self.action_injector_layer1.weight)
        nn.init.zeros_(self.action_injector_layer1.bias)
        nn.init.xavier_uniform_(self.action_injector_layer2.weight)
        nn.init.zeros_(self.action_injector_layer2.bias)
        nn.init.zeros_(self.action_scale_layer.weight)
        nn.init.zeros_(self.action_scale_layer.bias)
        nn.init.zeros_(self.action_shift_layer.weight)
        nn.init.zeros_(self.action_shift_layer.bias)

    def init_weights(self) -> None:
        # Block.init_weights re-initializes every Linear child, including scale / shift.
        super().init_weights()
        self._init_action_injection_weights()

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
        action_emb_B_T_H_W_D: Optional[torch.Tensor] = None,
        kv_cache_cfg=None,
    ) -> torch.Tensor:
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            lora = adaln_lora_B_T_3D if self.use_adaln_lora else 0
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                self.adaln_modulation_self_attn(emb_B_T_D) + lora
            ).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                self.adaln_modulation_cross_attn(emb_B_T_D) + lora
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (self.adaln_modulation_mlp(emb_B_T_D) + lora).chunk(
                3, dim=-1
            )

        def _b_t_1_1_d(t: torch.Tensor) -> torch.Tensor:
            return rearrange(t, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)

        def _modulate(_x, _norm_layer, _scale, _shift):
            return _norm_layer(_x) * (1 + _b_t_1_1_d(_scale)) + _b_t_1_1_d(_shift)

        B, T, H, W, D = x_B_T_H_W_D.shape
        video_size = VideoSize(T=T, H=H, W=W)
        if self.cp_size is not None and self.cp_size > 1:
            video_size = VideoSize(T=T * self.cp_size, H=H, W=W)

        normalized_x = _modulate(x_B_T_H_W_D, self.layer_norm_self_attn, scale_self_attn_B_T_D, shift_self_attn_B_T_D)
        result = rearrange(
            self.self_attn(
                rearrange(normalized_x, "b t h w d -> b (t h w) d"),
                None,
                rope_emb=rope_emb_L_1_1_D,
                video_size=video_size,
                kv_cache_cfg=kv_cache_cfg,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + _b_t_1_1_d(gate_self_attn_B_T_D) * result

        if self.has_action_injection and action_emb_B_T_H_W_D is not None:
            action_hidden = (
                self.action_injector_layer2(F.silu(self.action_injector_layer1(action_emb_B_T_H_W_D)))
                + action_emb_B_T_H_W_D
            )
            x_B_T_H_W_D = (1.0 + self.action_scale_layer(action_hidden)) * x_B_T_H_W_D + self.action_shift_layer(
                action_hidden
            )

        normalized_x = _modulate(
            x_B_T_H_W_D, self.layer_norm_cross_attn, scale_cross_attn_B_T_D, shift_cross_attn_B_T_D
        )
        result = rearrange(
            self.cross_attn(
                rearrange(normalized_x, "b t h w d -> b (t h w) d"),
                crossattn_emb,
                rope_emb=rope_emb_L_1_1_D,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = result * _b_t_1_1_d(gate_cross_attn_B_T_D) + x_B_T_H_W_D

        normalized_x = _modulate(x_B_T_H_W_D, self.layer_norm_mlp, scale_mlp_B_T_D, shift_mlp_B_T_D)
        return x_B_T_H_W_D + _b_t_1_1_d(gate_mlp_B_T_D) * self.mlp(normalized_x)


class ActionInjectMinimalV1LVGDiT(MiniTrainDIT):
    """Video2World DiT (with condition-mask channel) plus action-value and action-video branches."""

    def __init__(
        self,
        *args,
        timestep_scale: float = 1.0,
        action_dim: int = 28,
        action_intermediate_dim: Optional[int] = 1024,
        action_chunk_size: int = 4,
        action_inject_layers: Sequence[int] = tuple(range(1, 28, 2)),
        action_video_in_channels: int = 16,
        action_video_inject_layers: Sequence[int] = tuple(range(0, 28, 2)),
        vace_action_scale: float = 1.0,
        **kwargs,
    ):
        assert "in_channels" in kwargs, "in_channels must be provided"
        kwargs["in_channels"] += 1  # condition mask channel (same as MinimalV1LVGDiT)

        self.timestep_scale = timestep_scale
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.action_inject_layers = sorted(int(i) for i in action_inject_layers)
        self.action_video_inject_layers = sorted(int(i) for i in action_video_inject_layers)
        self.vace_action_scale = vace_action_scale

        sac_config = kwargs.get("sac_config", SACConfig())
        crossattn_emb_channels = kwargs.get("crossattn_emb_channels", 1024)
        mlp_ratio = kwargs.get("mlp_ratio", 4.0)
        super().__init__(*args, **kwargs)

        inject_set = set(self.action_inject_layers)
        self.blocks = nn.ModuleList(
            [
                ActionInjectBlock(
                    x_dim=self.model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=self.use_adaln_lora,
                    adaln_lora_dim=self.adaln_lora_dim,
                    backend=self.atten_backend,
                    image_context_dim=None if self.extra_image_context_dim is None else self.model_channels,
                    use_wan_fp32_strategy=self.use_wan_fp32_strategy,
                    has_action_injection=i in inject_set,
                )
                for i in range(self.num_blocks)
            ]
        )
        for block in self.blocks:
            block.init_weights()
        self.enable_selective_checkpoint(sac_config, self.blocks)

        self.action_encoder = ActionEncoder(
            action_dim=action_dim,
            hidden_dim=self.model_channels,
            intermediate_dim=action_intermediate_dim,
            chunk_size=action_chunk_size,
        )
        self.action_proj_layer1 = nn.Linear(self.model_channels, self.model_channels)
        self.action_proj_layer2 = nn.Linear(self.model_channels, self.model_channels)
        self._init_action_proj()

        self.vace_layers_mapping = {layer: n for n, layer in enumerate(self.action_video_inject_layers)}
        self.vace_action = ActionVideoVaceModel(
            vace_layers=self.action_video_inject_layers,
            vace_in_dim=action_video_in_channels,
            patch_spatial=self.patch_spatial,
            patch_temporal=self.patch_temporal,
            model_channels=self.model_channels,
            num_heads=self.num_heads,
            mlp_ratio=mlp_ratio,
            context_dim=crossattn_emb_channels,
            use_adaln_lora=self.use_adaln_lora,
            adaln_lora_dim=self.adaln_lora_dim,
            backend=self.atten_backend,
            use_wan_fp32_strategy=self.use_wan_fp32_strategy,
            image_context_dim=None if self.extra_image_context_dim is None else self.model_channels,
            scale=vace_action_scale,
        )
        self.enable_selective_checkpoint(sac_config, self.vace_action.vace_blocks)

        log.info(
            f"ActionInjectMinimalV1LVGDiT: action_dim={action_dim}, chunk_size={action_chunk_size}, "
            f"action_inject_layers={self.action_inject_layers}, "
            f"action_video_inject_layers={self.action_video_inject_layers}"
        )

    def _init_action_proj(self) -> None:
        for layer in (self.action_proj_layer1, self.action_proj_layer2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def init_weights(self):
        """Also covers the action modules (needed after ``to_empty()``)."""
        super().init_weights()
        # MiniTrainDIT.__init__ calls this before the action modules exist.
        if getattr(self, "action_encoder", None) is None:
            return
        self.action_encoder.init_weights()
        self._init_action_proj()
        self.vace_action.init_weights()

    @staticmethod
    def _resolve_action_value_inputs(
        action_state: Optional[torch.Tensor],
        action_delta: Optional[torch.Tensor],
        action: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Batch the dataset's ``action_state [B, D]`` / ``action_delta [B, T-1, D]``.

        Missing fields are derived from the absolute ``action [B, T, D]``; given ones are kept.
        """
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action_state is None:
                action_state = action[:, 0, :]
            if action_delta is None:
                action_delta = (
                    action[:, 1:, :] - action[:, :-1, :]
                    if action.shape[1] > 1
                    else torch.zeros_like(action[:, :1, :])
                )
        if action_state is not None and action_state.ndim == 1:
            action_state = action_state.unsqueeze(0)
        if action_delta is not None and action_delta.ndim == 2:
            action_delta = action_delta.unsqueeze(0)
        if action_state is not None and action_delta is not None:
            assert action_state.ndim == 2 and action_delta.ndim == 3, (action_state.shape, action_delta.shape)
            assert action_state.shape[0] == action_delta.shape[0], (action_state.shape, action_delta.shape)
            assert action_state.shape[-1] == action_delta.shape[-1], (action_state.shape, action_delta.shape)
            action_state, action_delta = action_state.contiguous(), action_delta.contiguous()
        return action_state, action_delta

    def encode_action(self, action_state: torch.Tensor, action_delta: torch.Tensor) -> torch.Tensor:
        weight_dtype = self.action_encoder.state_encoder[0].weight.dtype
        return self.action_encoder(action_state.to(dtype=weight_dtype), action_delta.to(dtype=weight_dtype))

    def prepare_action_emb(self, action_emb_seq: torch.Tensor, num_frames: int, spatial_h: int, spatial_w: int):
        """Resample to the latent frame count, residual MLP, broadcast → ``[B, T, H, W, D]``."""
        B, T_action = action_emb_seq.shape[:2]
        if T_action != num_frames:
            indices = torch.linspace(0, T_action - 1, num_frames, device=action_emb_seq.device).long()
            action_emb_seq = action_emb_seq[:, indices, :]
        action_emb = self.action_proj_layer2(F.silu(self.action_proj_layer1(action_emb_seq))) + action_emb_seq
        return action_emb[:, :, None, None, :].expand(B, num_frames, spatial_h, spatial_w, self.model_channels)

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        img_context_emb: Optional[torch.Tensor] = None,
        action_state: Optional[torch.Tensor] = None,
        action_delta: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        action_video_latents: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if data_type == DataType.VIDEO:
            cond_mask = condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)
        else:
            cond_mask = torch.zeros_like(x_B_C_T_H_W[:, :1])
        x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, cond_mask], dim=1)
        timesteps_B_T = timesteps_B_T * self.timestep_scale

        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = self.prepare_embedded_sequence(
            x_B_C_T_H_W, fps=fps, padding_mask=padding_mask
        )
        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)
        context_input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = crossattn_emb
        if img_context_emb is not None:
            context_input = (crossattn_emb, self.img_context_proj(img_context_emb))

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)
        self.affline_emb = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb

        B, T, H, W, D = x_B_T_H_W_D.shape

        action_emb_B_T_H_W_D = None
        action_state, action_delta = self._resolve_action_value_inputs(action_state, action_delta, action)
        if action_state is not None and action_delta is not None:
            action_emb_B_T_H_W_D = self.prepare_action_emb(self.encode_action(action_state, action_delta), T, H, W)

        vace_hints = None
        if action_video_latents is not None:
            vace_hints = self.vace_action(
                x_B_T_H_W_D,
                action_video_latents,
                emb_B_T_D=t_embedding_B_T_D,
                crossattn_emb=context_input,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb,
            )

        for i, block in enumerate(self.blocks):
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                context_input,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb,
                action_emb_B_T_H_W_D=action_emb_B_T_H_W_D,
            )
            if vace_hints is not None and i in self.vace_layers_mapping:
                x_B_T_H_W_D = x_B_T_H_W_D + vace_hints[self.vace_layers_mapping[i]] * self.vace_action.scale

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        return self.unpatchify(x_B_T_H_W_O)

    def fully_shard(self, mesh, **fsdp_kwargs):
        super().fully_shard(mesh, **fsdp_kwargs)
        from torch.distributed._composable.fsdp import fully_shard

        for module in (self.action_encoder, self.action_proj_layer1, self.action_proj_layer2, self.vace_action):
            fully_shard(module, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)
