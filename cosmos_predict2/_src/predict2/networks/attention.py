# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# From Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

# Description:
# Single point of entry for all generic attention ops (self and cross attention), that tries to
# deliver the best performance possible given any use case (GPU and environment).
#
# Preferred backends by architecture (when packages are installed):
#   - Hopper (SM90):        Flash Attention 3 > Flash Attention 4 > cuDNN SDPA
#   - Blackwell DC (SM100): Flash Attention 4 > cuDNN SDPA
#   - Blackwell RTX/SM120:  Flash Attention 4 > Flash Attention 2 > cuDNN / xformers
#   - Ampere / Ada:         Flash Attention 2 (via SDPA) > cuDNN > xformers
#   - Pre-Ampere:           memory-efficient (xformers) FMHA only
#
# Override with env: COSMOS_ATTENTION_BACKEND=auto|fa4|fa3|cudnn|fa2|efficient
#
# Relevant compute capabilities (SmXXX = major*10 + minor):
#
# | GPU / category | Arch   |
# |================|========|
# | A100 / A800    | SM80   |
# | Ampere RTX     | SM86   |
# | Ada Lovelace   | SM89   |
# | H20 / H100/H200| SM90   |
# | B200 / GB200   | SM100  |
# | Blackwell RTX  | SM103+ |
# | GeForce 50     | SM120  |

from __future__ import annotations

import os
from functools import partial
from typing import Optional

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    from flash_attn_3.flash_attn_interface import flash_attn_func as _flash_attn_3_func

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    _flash_attn_3_func = None
    FLASH_ATTN_3_AVAILABLE = False

try:
    # FlashAttention-4 (CuTeDSL): pip install flash-attn-4  (use [cu13] on CUDA 13 / B200)
    from flash_attn.cute import flash_attn_func as _flash_attn_4_func

    FLASH_ATTN_4_AVAILABLE = True
except ModuleNotFoundError:
    _flash_attn_4_func = None
    FLASH_ATTN_4_AVAILABLE = False

_LOGGED_BACKEND: Optional[str] = None


def get_device_cc(device) -> int:
    """
    Returns the compute capability of a given torch device if it's a CUDA device, otherwise returns 0.

    Args:
        device: torch device.

    Returns:
        device_cc (int): compute capability in the SmXXX format (i.e. 90 for Hopper, 100 for B200).
    """
    if torch.cuda.is_available() and torch.version.cuda and device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        return major * 10 + minor
    return 0


def is_hopper(cc: int) -> bool:
    return 90 <= cc < 100


def is_blackwell(cc: int) -> bool:
    # Datacenter B200/GB200 (SM100), Blackwell RTX (SM103+), GeForce 50 (SM120)
    return cc >= 100


def _normalize_backend_choice(choice: str) -> str:
    choice = choice.strip().lower()
    aliases = {
        "auto": "auto",
        "fa4": "fa4",
        "flash4": "fa4",
        "flash_attn_4": "fa4",
        "flash-attn-4": "fa4",
        "fa3": "fa3",
        "flash3": "fa3",
        "flash_attn_3": "fa3",
        "cudnn": "cudnn",
        "fa2": "fa2",
        "flash2": "fa2",
        "flash_attn": "fa2",
        "flash_attention": "fa2",
        "efficient": "efficient",
        "xformers": "efficient",
        "mem_efficient": "efficient",
    }
    if choice not in aliases:
        raise ValueError(
            f"Invalid COSMOS_ATTENTION_BACKEND={choice!r}. "
            f"Expected one of: {sorted(set(aliases.values()))}"
        )
    return aliases[choice]


def select_attention_backend(compute_cap: int, dtype: torch.dtype, override: Optional[str] = None) -> str:
    """Pick the best available attention backend name for this device/dtype."""
    is_half = dtype in (torch.bfloat16, torch.float16)
    choice = _normalize_backend_choice(override or os.environ.get("COSMOS_ATTENTION_BACKEND", "auto"))

    if choice != "auto":
        if choice == "fa4" and not FLASH_ATTN_4_AVAILABLE:
            raise RuntimeError("COSMOS_ATTENTION_BACKEND=fa4 but flash_attn.cute is not installed. "
                               "Install with: pip install 'flash-attn-4[cu13]'  # CUDA 13 / B200")
        if choice == "fa3" and not FLASH_ATTN_3_AVAILABLE:
            raise RuntimeError("COSMOS_ATTENTION_BACKEND=fa3 but flash_attn_3 is not installed.")
        if choice in ("fa3", "fa4", "fa2", "cudnn") and not is_half:
            raise RuntimeError(f"Backend {choice} requires bf16/fp16, got {dtype}.")
        return choice

    if not is_half:
        return "efficient"

    # Blackwell datacenter / RTX: FA4 is currently the fastest open-source kernel
    # (~1.3x vs cuDNN 9.13 on B200 per FA4 paper). Fall back to cuDNN FMHA.
    if is_blackwell(compute_cap):
        if FLASH_ATTN_4_AVAILABLE:
            return "fa4"
        return "cudnn"

    # Hopper: prefer FA3 (Hopper-specialized), then FA4 (also supports SM90), then cuDNN.
    if is_hopper(compute_cap):
        if FLASH_ATTN_3_AVAILABLE:
            return "fa3"
        if FLASH_ATTN_4_AVAILABLE:
            return "fa4"
        return "cudnn"

    # Ampere / Ada and other CUDA GPUs with half precision.
    if compute_cap >= 80:
        return "fa2"
    return "efficient"


def describe_attention_backend(device: Optional[torch.device] = None) -> dict:
    """Return a small diagnostic dict (useful for check scripts / logs)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cc = get_device_cc(device)
    backend = select_attention_backend(cc, torch.bfloat16)
    return {
        "device": str(device),
        "compute_capability": cc,
        "is_hopper": is_hopper(cc),
        "is_blackwell": is_blackwell(cc),
        "flash_attn_3_available": FLASH_ATTN_3_AVAILABLE,
        "flash_attn_4_available": FLASH_ATTN_4_AVAILABLE,
        "selected_backend": backend,
        "override": os.environ.get("COSMOS_ATTENTION_BACKEND", "auto"),
    }


def _log_backend_once(backend: str, compute_cap: int) -> None:
    global _LOGGED_BACKEND
    if _LOGGED_BACKEND == backend:
        return
    _LOGGED_BACKEND = backend
    try:
        from cosmos_predict2._src.imaginaire.utils import log

        log.info(
            f"Attention backend selected: {backend} (SM{compute_cap}, "
            f"FA3={FLASH_ATTN_3_AVAILABLE}, FA4={FLASH_ATTN_4_AVAILABLE}, "
            f"override={os.environ.get('COSMOS_ATTENTION_BACKEND', 'auto')})"
        )
    except Exception:
        print(
            f"[cosmos-predict2.5] Attention backend selected: {backend} "
            f"(SM{compute_cap}, FA3={FLASH_ATTN_3_AVAILABLE}, FA4={FLASH_ATTN_4_AVAILABLE})"
        )


def _run_fa4(q, k, v, softmax_scale, causal, deterministic):
    # FA4 FlashAttnFunc always returns (out, lse); layout is (B, S, H, D).
    out = _flash_attn_4_func(
        q=q,
        k=k,
        v=v,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )
    return out[0] if isinstance(out, tuple) else out


def _run_fa3(q, k, v, softmax_scale, causal, deterministic):
    return _flash_attn_3_func(
        q=q,
        k=k,
        v=v,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )[0]


def _run_sdpa(q, k, v, dropout_p, softmax_scale, causal, backends, best_backend):
    try:
        sdpa_kernel(backends=backends, set_priority_order=True)
        sdpa_kernel_ = partial(sdpa_kernel, set_priority_order=True)
    except TypeError:
        sdpa_kernel_ = sdpa_kernel
        backends = [best_backend]

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    with sdpa_kernel_(backends=backends):
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=causal,
            dropout_p=dropout_p,
            scale=softmax_scale,
        )

    return out.transpose(1, 2).contiguous()


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    deterministic=False,
    dtype=torch.bfloat16,
):
    del q_lens, k_lens  # variable-length paths are not used by the DiT minimal_a2a backend

    supported_dtypes = [torch.bfloat16, torch.float16, torch.float32]
    compute_cap = get_device_cc(q.device)

    if dtype not in supported_dtypes:
        raise NotImplementedError(f"{dtype=} is not supported.")

    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)

    if q_scale is not None:
        q = q * q_scale

    backend = select_attention_backend(compute_cap, dtype)
    _log_backend_once(backend, compute_cap)

    if backend == "fa4":
        return _run_fa4(q, k, v, softmax_scale, causal, deterministic)
    if backend == "fa3":
        return _run_fa3(q, k, v, softmax_scale, causal, deterministic)

    if deterministic and backend != "fa3":
        # Keep historical behavior: deterministic path was only validated for FA3.
        raise NotImplementedError(
            "Deterministic mode in attention is only supported when Flash Attention 3 is available "
            "and selected (Hopper + flash_attn_3)."
        )

    if backend == "cudnn":
        backends = [
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
        ]
        best = SDPBackend.CUDNN_ATTENTION
    elif backend == "fa2":
        backends = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
        ]
        best = SDPBackend.FLASH_ATTENTION if compute_cap >= 80 else SDPBackend.EFFICIENT_ATTENTION
    else:
        backends = [SDPBackend.EFFICIENT_ATTENTION]
        best = SDPBackend.EFFICIENT_ATTENTION

    return _run_sdpa(q, k, v, dropout_p, softmax_scale, causal, backends, best)
