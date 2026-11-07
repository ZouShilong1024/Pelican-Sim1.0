# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ported from OminiEWM ActionEncoder (Ctrl-World style chunk concatenation).

"""Action value encoder: state[0] + delta chunks → latent-aligned embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn


class ActionEncoder(nn.Module):
    """Encode absolute state + per-frame deltas into DiT-aligned action tokens.

    Output layout matches VAE temporal compression::

        state_token ↔ latent_0
        chunk_k     ↔ latent_{k+1}  (driven by delta[chunk_size*k : chunk_size*(k+1)])

    Args:
        action_dim: Raw joint / action dimension (e.g. 7 or 14).
        hidden_dim: Output embedding dim (must match DiT ``model_channels``).
        intermediate_dim: MLP hidden size (default ``hidden_dim * 4``).
        chunk_size: Deltas per chunk (default 4, matching Wan VAE temporal stride).
    """

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int,
        intermediate_dim: int | None = None,
        chunk_size: int = 4,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.intermediate_dim = intermediate_dim if intermediate_dim is not None else hidden_dim * 4

        self.state_encoder = nn.Sequential(
            nn.Linear(action_dim, self.intermediate_dim),
            nn.SiLU(),
            nn.Linear(self.intermediate_dim, hidden_dim),
        )
        self.chunk_encoder = nn.Sequential(
            nn.Linear(action_dim * chunk_size, self.intermediate_dim),
            nn.SiLU(),
            nn.Linear(self.intermediate_dim, hidden_dim),
        )

        self.init_weights()

    def init_weights(self) -> None:
        """(Re)initialize after ``Module.to_empty()``."""
        for mod in (self.state_encoder, self.chunk_encoder):
            nn.init.kaiming_normal_(mod[0].weight, mode="fan_in", nonlinearity="relu")
            if mod[0].bias is not None:
                nn.init.zeros_(mod[0].bias)
            nn.init.kaiming_normal_(mod[2].weight, mode="fan_in", nonlinearity="relu")
            if mod[2].bias is not None:
                nn.init.zeros_(mod[2].bias)

    def forward(self, action_state: torch.Tensor, action_delta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_state: ``[B, action_dim]`` absolute state at frame 0.
            action_delta: ``[B, T-1, action_dim]`` per-frame deltas.

        Returns:
            ``action_emb`` of shape ``[B, T_latent, hidden_dim]`` where
            ``T_latent = 1 + ceil((T-1) / chunk_size)``.
        """
        B, num_deltas, D = action_delta.shape

        state_token = self.state_encoder(action_state).unsqueeze(1)  # [B, 1, H]

        pad_len = (self.chunk_size - (num_deltas % self.chunk_size)) % self.chunk_size
        if pad_len > 0:
            padding = action_delta[:, -1:, :].expand(B, pad_len, D)
            action_delta_padded = torch.cat([action_delta, padding], dim=1)
        else:
            action_delta_padded = action_delta

        num_chunks = action_delta_padded.shape[1] // self.chunk_size
        chunks = action_delta_padded.view(B, num_chunks, D * self.chunk_size)
        chunk_tokens = self.chunk_encoder(chunks)

        return torch.cat([state_token, chunk_tokens], dim=1)
