#!/usr/bin/env python
"""Upload the Pelican-Sim 1.0 release (checkpoint + model card) to the Hugging Face Hub.

Usage::

    hf auth login   # token with write access
    python scripts/upload_to_hf.py --ckpt /path/to/iter_XXXXXXXXX/model_ema_bf16.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True, help="Consolidated EMA bf16 checkpoint.")
    p.add_argument("--repo-id", default="ShilongZou/Pelican-Sim1.0")
    p.add_argument("--private", action="store_true")
    p.add_argument("--card-only", action="store_true", help="Only refresh README.md and assets.")
    args = p.parse_args()

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    api.upload_file(
        path_or_fileobj=str(ROOT / "assets" / "teaser.png"),
        path_in_repo="assets/teaser.png",
        repo_id=args.repo_id,
        commit_message="Add teaser figure",
    )
    api.upload_file(
        path_or_fileobj=str(ROOT / "release" / "hf" / "README.md"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        commit_message="Add model card",
    )
    if args.card_only:
        return

    ckpt = args.ckpt.resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    api.upload_file(
        path_or_fileobj=str(ckpt),
        path_in_repo="model_ema_bf16.pt",
        repo_id=args.repo_id,
        commit_message="Add Pelican-Sim 1.0 checkpoint (MoE E8/K2, EMA bf16)",
    )
    print(f"Uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
