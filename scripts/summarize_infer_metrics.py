"""Summarize PSNR / SSIM from ``inference_results.jsonl`` (written with ``--metrics``)."""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np


def main(path: str) -> None:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    rows = [r for r in rows if r.get("metrics")]
    if not rows:
        print(f"no metrics in {path}")
        return
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r.get("task_name") or r.get("source") or "all"].append(r["metrics"])
    print(f"{'task':40s} {'n':>3s} {'PSNR':>8s} {'SSIM':>7s}")
    for task in sorted(by_task):
        ms = by_task[task]
        print(f"{task:40s} {len(ms):3d} {np.mean([m['psnr'] for m in ms]):8.3f} {np.mean([m['ssim'] for m in ms]):7.4f}")
    allm = [r["metrics"] for r in rows]
    print(f"{'ALL':40s} {len(allm):3d} {np.mean([m['psnr'] for m in allm]):8.3f} {np.mean([m['ssim'] for m in allm]):7.4f}")
    per_view: dict[str, list[float]] = defaultdict(list)
    for m in allm:
        for view, value in (m.get("psnr_per_view") or {}).items():
            per_view[view].append(value)
    for view, values in per_view.items():
        print(f"{'  PSNR ' + view:40s} {len(values):3d} {np.mean(values):8.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
