# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Action-inject DiT whose RGB-block FFNs are shared + routed MoE layers.

Each RGB DiT block ``mlp`` is ``shared_expert(x) + sum_k gate_k * routed_expert_k(x)``
with a token-choice top-k router. VACE blocks keep their dense FFN.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.action.networks.action_inject_minimal_v1_lvg_dit import (
    ActionInjectMinimalV1LVGDiT,
)
from cosmos_predict2._src.predict2.action.networks.action_video_modules import (
    _unwrap_checkpoint_module,
)
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import GPT2FeedForward


class GPT2Expert(nn.Module):
    """Linear → GELU → Linear, same as ``GPT2FeedForward``."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)
        self._dim = d_model
        self._hidden_dim = d_ff

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._dim)
        torch.nn.init.trunc_normal_(self.layer1.weight, std=std, a=-3 * std, b=3 * std)
        std2 = 1.0 / math.sqrt(self._hidden_dim)
        torch.nn.init.trunc_normal_(self.layer2.weight, std=std2, a=-3 * std2, b=3 * std2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.activation(self.layer1(x)))


class TokenChoiceTopKRouter(nn.Module):
    """Token-choice top-k router; gates are the selected scores, renormalized and scaled."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        score_func: str = "sigmoid",
        norm_topk_prob: bool = True,
        route_scale: float = 1.0,
    ):
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"top_k={top_k} invalid for num_experts={num_experts}")
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.norm_topk_prob = norm_topk_prob
        self.route_scale = route_scale
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        # Part of the checkpoint format; added to the scores for expert selection only.
        self.register_buffer("e_score_correction_bias", torch.zeros(num_experts), persistent=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.e_score_correction_bias.zero_()

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """tokens ``[N, H]`` → top indices ``[N, K]``, gates ``[N, K]``, all scores ``[N, E]``."""
        logits = F.linear(tokens.float(), self.weight.float())
        scores = logits.sigmoid() if self.score_func == "sigmoid" else F.softmax(logits, dim=-1)
        bias = self.e_score_correction_bias.to(device=scores.device, dtype=scores.dtype)
        top_indices = torch.topk(scores + bias.unsqueeze(0), k=self.top_k, dim=-1, sorted=False)[1]
        top_scores = scores.gather(1, top_indices)
        if self.top_k > 1 and self.norm_topk_prob:
            top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20)
        top_scores = top_scores * self.route_scale
        return top_indices, top_scores.to(tokens.dtype), scores


class SparseMoEGPT2FeedForward(nn.Module):
    """Drop-in replacement for ``GPT2FeedForward``: one shared expert + routed experts."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        score_func: str = "sigmoid",
        norm_topk_prob: bool = True,
        route_scale: float = 1.0,
        aux_loss_coeff: float = 0.01,
        moe_capacity_factor: float = 2.0,
    ):
        super().__init__()
        self._dim = d_model
        self._hidden_dim = d_ff
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.aux_loss_coeff = float(aux_loss_coeff)
        self.moe_capacity_factor = float(moe_capacity_factor)
        # Per-expert microbatch for the training dispatch; caps GELU activation memory.
        self.moe_expert_chunk_size = 2048

        self.router = TokenChoiceTopKRouter(
            d_model,
            num_experts,
            num_experts_per_tok,
            score_func=score_func,
            norm_topk_prob=norm_topk_prob,
            route_scale=route_scale,
        )
        self.experts = nn.ModuleList([GPT2Expert(d_model, d_ff) for _ in range(num_experts)])
        self.shared_experts = GPT2Expert(d_model, d_ff)
        # Aliases kept so checkpoints also carry ``mlp.layer{1,2}.weight`` (= shared expert).
        self.layer1 = self.shared_experts.layer1
        self.layer2 = self.shared_experts.layer2

        self._last_aux_loss: Optional[torch.Tensor] = None

    def init_weights(self) -> None:
        self.router.reset_parameters()
        for expert in self.experts:
            expert.init_weights()
        self.shared_experts.init_weights()

    @staticmethod
    def _load_balance_aux(dense_scores: torch.Tensor, top_indices: torch.Tensor, num_experts: int) -> torch.Tensor:
        """Switch-style ``E * sum_i f_i * P_i`` (1 when balanced) on per-token normalized scores."""
        ones = torch.ones_like(top_indices, dtype=dense_scores.dtype)
        counts = torch.zeros(num_experts, device=dense_scores.device, dtype=dense_scores.dtype)
        counts.scatter_add_(0, top_indices.reshape(-1), ones.reshape(-1))
        f = counts / (top_indices.numel() + 1e-20)
        probs = dense_scores / (dense_scores.sum(dim=-1, keepdim=True) + 1e-20)
        return num_experts * (f * probs.mean(dim=0)).sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        tokens = x.reshape(-1, self._dim)
        top_indices, top_scores, dense_scores = self.router(tokens)
        if torch.is_grad_enabled():
            self._last_aux_loss = self._load_balance_aux(dense_scores, top_indices, self.num_experts)
            # Fixed per-expert shapes so activation-checkpoint recompute matches the forward.
            routed = self._forward_capacity_dispatch(tokens, top_indices, top_scores)
        else:
            routed = self._forward_dynamic_dispatch(tokens, top_indices, top_scores)
        return (self.shared_experts(tokens) + routed).view(orig_shape)

    def _forward_capacity_dispatch(
        self, tokens: torch.Tensor, top_indices: torch.Tensor, top_scores: torch.Tensor
    ) -> torch.Tensor:
        """Each expert runs a ``[capacity, H]`` batch; overflow drops the lowest gates."""
        n_tokens = tokens.shape[0]
        slots = top_indices.numel()
        capacity = int(math.ceil(slots / self.num_experts * self.moe_capacity_factor))
        capacity = max(1, min(slots, capacity))

        out = tokens.new_zeros(n_tokens, self._dim)
        flat_idx = top_indices.reshape(-1)
        flat_score = top_scores.reshape(-1)
        token_ids = torch.arange(n_tokens, device=tokens.device).unsqueeze(1).expand_as(top_indices).reshape(-1)
        neg = flat_score.new_tensor(-1.0e9)
        chunk = int(self.moe_expert_chunk_size)
        for e in range(self.num_experts):
            priority = torch.where(flat_idx == e, flat_score, neg)
            chosen_scores, chosen_pos = torch.topk(priority, k=capacity, largest=True, sorted=False)
            chosen_tokens = token_ids[chosen_pos]
            valid = (chosen_scores > -1.0e8).to(dtype=tokens.dtype).unsqueeze(-1)
            gate = chosen_scores.to(dtype=tokens.dtype).unsqueeze(-1) * valid
            for s in range(0, capacity, chunk):
                sl = slice(s, min(s + chunk, capacity))
                out.index_add_(0, chosen_tokens[sl], self.experts[e](tokens[chosen_tokens[sl]]) * gate[sl])
        return out

    def _forward_dynamic_dispatch(
        self, tokens: torch.Tensor, top_indices: torch.Tensor, top_scores: torch.Tensor
    ) -> torch.Tensor:
        """Gather / scatter with variable per-expert batch sizes (inference)."""
        n_tokens = tokens.shape[0]
        out = tokens.new_zeros(n_tokens, self._dim)
        flat_idx = top_indices.reshape(-1)
        flat_score = top_scores.reshape(-1)
        token_ids = torch.arange(n_tokens, device=tokens.device).unsqueeze(1).expand_as(top_indices).reshape(-1)
        for e in range(self.num_experts):
            mask = flat_idx == e
            if not mask.any():
                continue
            sel_tokens = token_ids[mask]
            out.index_add_(0, sel_tokens, self.experts[e](tokens[sel_tokens]) * flat_score[mask].unsqueeze(-1))
        return out

    def pop_aux_loss(self) -> Optional[torch.Tensor]:
        loss = self._last_aux_loss
        self._last_aux_loss = None
        return loss


def collect_moe_aux_loss(module: nn.Module) -> Optional[torch.Tensor]:
    """``aux_loss_coeff * mean over layers`` of the last forward's load-balance loss."""
    pieces: list[torch.Tensor] = []
    coeff = 0.0
    for m in module.modules():
        if isinstance(m, SparseMoEGPT2FeedForward):
            piece = m.pop_aux_loss()
            if piece is not None:
                pieces.append(piece.reshape(()))
                coeff = m.aux_loss_coeff
    if not pieces or coeff <= 0:
        return None
    return torch.stack(pieces).mean() * coeff


class ActionInjectMoEMinimalV1LVGDiT(ActionInjectMinimalV1LVGDiT):
    """``ActionInjectMinimalV1LVGDiT`` with MoE FFNs in the RGB blocks listed in ``moe_layers``."""

    def __init__(
        self,
        *args,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        moe_score_func: str = "sigmoid",
        moe_norm_topk_prob: bool = True,
        moe_route_scale: float = 1.0,
        moe_aux_loss_coeff: float = 0.01,
        moe_layers: Optional[Sequence[int]] = None,
        moe_capacity_factor: float = 2.0,
        **kwargs,
    ):
        self.num_experts = int(num_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.moe_score_func = moe_score_func
        self.moe_norm_topk_prob = bool(moe_norm_topk_prob)
        self.moe_route_scale = float(moe_route_scale)
        self.moe_aux_loss_coeff = float(moe_aux_loss_coeff)
        self.moe_capacity_factor = float(moe_capacity_factor)
        # Set before super().__init__, which calls init_weights() on this subclass.
        num_blocks = int(kwargs.get("num_blocks", 28))
        self.moe_layers = list(range(num_blocks)) if moe_layers is None else [int(i) for i in moe_layers]
        super().__init__(*args, **kwargs)
        self._replace_rgb_mlps_with_moe()
        log.info(
            f"ActionInjectMoEMinimalV1LVGDiT: experts={self.num_experts} (+1 shared), "
            f"top_k={self.num_experts_per_tok}, moe_layers={self.moe_layers}"
        )

    def _replace_rgb_mlps_with_moe(self) -> None:
        # Blocks are wrapped by ptd_checkpoint_wrapper; replace inside the wrapper so state-dict
        # keys stay ``blocks.{i}.mlp.*``.
        for i in self.moe_layers:
            raw_block = _unwrap_checkpoint_module(self.blocks[i])
            old_mlp = raw_block.mlp
            if isinstance(old_mlp, SparseMoEGPT2FeedForward):
                continue
            assert isinstance(old_mlp, GPT2FeedForward), type(old_mlp)
            raw_block.mlp = SparseMoEGPT2FeedForward(
                d_model=old_mlp._dim,
                d_ff=old_mlp._hidden_dim,
                num_experts=self.num_experts,
                num_experts_per_tok=self.num_experts_per_tok,
                score_func=self.moe_score_func,
                norm_topk_prob=self.moe_norm_topk_prob,
                route_scale=self.moe_route_scale,
                aux_loss_coeff=self.moe_aux_loss_coeff,
                moe_capacity_factor=self.moe_capacity_factor,
            )

    def init_weights(self):
        super().init_weights()
        for i in self.moe_layers if hasattr(self, "blocks") else []:
            mlp = _unwrap_checkpoint_module(self.blocks[i]).mlp
            if isinstance(mlp, SparseMoEGPT2FeedForward):
                mlp.init_weights()

    def pop_moe_aux_loss(self) -> Optional[torch.Tensor]:
        return collect_moe_aux_loss(self)
