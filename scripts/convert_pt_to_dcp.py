#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Turn a consolidated ``model_ema_bf16.pt`` (``net.*`` keys) into a DCP checkpoint dir that
``checkpoint.load_path`` can start a training run from.

The trainer only loads DCP: a ``.pt`` given as ``load_path`` is skipped and the run would start
from the base weights. The EMA weights fill both the live model (``net.*``, bf16) and the EMA
copy (``net_ema.*``, fp32), the dtypes the trainer saves. TransformerEngine ``_extra_state``
entries are dropped; the loader skips them anyway.

    python scripts/convert_pt_to_dcp.py checkpoints/Pelican-Sim-1.0/model_ema_bf16.pt \
        checkpoints/Pelican-Sim-1.0/dcp_model_ema_bf16/iter_000000000
    # optional: check keys / shapes against a checkpoint saved by the trainer
    python scripts/convert_pt_to_dcp.py model_ema_bf16.pt out_dir --reference <job>/checkpoints/iter_X/model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter


def build_state_dict(pt_path: Path) -> dict[str, torch.Tensor]:
    src = torch.load(pt_path, map_location="cpu", weights_only=False)
    out: dict[str, torch.Tensor] = {}
    for key, value in src.items():
        if not key.startswith("net.") or "_extra_state" in key or not isinstance(value, torch.Tensor):
            continue
        name = key[len("net.") :]
        if value.is_floating_point():
            out[f"net.{name}"] = value.to(torch.bfloat16).contiguous()
            out[f"net_ema.{name}"] = value.to(torch.float32).contiguous()
        else:
            out[f"net.{name}"] = value.contiguous()
            out[f"net_ema.{name}"] = value.clone().contiguous()
    if not out:
        raise ValueError(f"no net.* tensors in {pt_path}")
    return out


def check_against(state_dict: dict[str, torch.Tensor], reference: Path) -> None:
    ref = FileSystemReader(str(reference)).read_metadata().state_dict_metadata
    ref = {k: v for k, v in ref.items() if "_extra_state" not in k}
    missing = sorted(set(ref) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(ref))
    mismatched = [
        k
        for k in set(ref) & set(state_dict)
        if tuple(ref[k].size) != tuple(state_dict[k].shape) or ref[k].properties.dtype != state_dict[k].dtype
    ]
    print(f"reference {reference}: {len(ref)} keys, missing={len(missing)}, unexpected={len(unexpected)}, "
          f"shape/dtype mismatch={len(mismatched)}")
    for name, keys in (("missing", missing), ("unexpected", unexpected), ("mismatch", mismatched)):
        for k in keys[:10]:
            print(f"  {name}: {k}")
    if missing or unexpected or mismatched:
        raise SystemExit("converted checkpoint does not match the reference")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pt", type=Path, help="consolidated model_ema_bf16.pt / model_ema_fp32.pt")
    ap.add_argument("out_dir", type=Path, help="output iter dir; weights go to <out_dir>/model")
    ap.add_argument("--reference", type=Path, help="DCP model dir saved by the trainer, to verify keys")
    ap.add_argument("--check-only", action="store_true", help="verify against --reference, write nothing")
    args = ap.parse_args()

    state_dict = build_state_dict(args.pt)
    n_params = sum(v.numel() for k, v in state_dict.items() if k.startswith("net."))
    print(f"{args.pt}: {len(state_dict) // 2} tensors, {n_params / 1e9:.2f}B elements")
    if args.reference:
        check_against(state_dict, args.reference)
    if args.check_only:
        return

    model_dir = args.out_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    dcp.save(state_dict, storage_writer=FileSystemWriter(str(model_dir), thread_count=8), no_dist=True)
    print(f"wrote DCP checkpoint {model_dir}")


if __name__ == "__main__":
    main()
