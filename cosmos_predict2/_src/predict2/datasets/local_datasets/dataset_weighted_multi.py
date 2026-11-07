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

"""Weighted mixture of multiple Video2World-compatible datasets."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from torch.utils.data import Dataset

from cosmos_predict2._src.imaginaire.utils import log


class WeightedMultiDataset(Dataset):
    """Mix several datasets with user-specified sampling ratios.

    On each ``__getitem__(index)`` call:

    1. Choose a source dataset with probability proportional to ``weights``.
    2. Choose a random sample *inside* that source (uniform).

    The sampler index only seeds the RNG, so ``DistributedSampler`` still
    yields diverse clips across ranks / epoch steps, while the mixture ratio
    stays independent of each source's raw size.

    Args:
        datasets: One or more ``Dataset`` instances (same output contract).
        weights: Non-negative mixing weights (need not sum to 1). ``None`` → equal.
        names: Optional labels for logging / debugging (len == len(datasets)).
        epoch_size: Virtual length exposed to the DataLoader. ``None`` → sum of
            source lengths.
        seed: Base seed mixed with ``index`` for reproducible source/local picks.
        attach_source_name: If True, add ``dataset_name`` to each sample dict.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        weights: Optional[Sequence[float]] = None,
        names: Optional[Sequence[str]] = None,
        epoch_size: Optional[int] = None,
        seed: int = 0,
        attach_source_name: bool = False,
    ) -> None:
        super().__init__()
        if not datasets:
            raise ValueError("WeightedMultiDataset requires at least one dataset")

        self.datasets: list[Dataset] = list(datasets)
        self.num_sources = len(self.datasets)
        self.seed = int(seed)
        self.attach_source_name = bool(attach_source_name)

        if names is None:
            self.names = [f"dataset_{i}" for i in range(self.num_sources)]
        else:
            if len(names) != self.num_sources:
                raise ValueError(f"names length {len(names)} != datasets length {self.num_sources}")
            self.names = [str(n) for n in names]

        for i, ds in enumerate(self.datasets):
            if len(ds) == 0:
                raise ValueError(f"Source dataset {self.names[i]!r} is empty")

        if weights is None:
            raw = [1.0] * self.num_sources
        else:
            if len(weights) != self.num_sources:
                raise ValueError(f"weights length {len(weights)} != datasets length {self.num_sources}")
            raw = [float(w) for w in weights]
            if any(w < 0 for w in raw):
                raise ValueError(f"weights must be non-negative, got {raw}")
            if sum(raw) <= 0:
                raise ValueError(f"weights must sum to a positive value, got {raw}")

        total = float(sum(raw))
        self.weights = raw
        self.probs = [w / total for w in raw]

        if epoch_size is None:
            self.epoch_size = int(sum(len(ds) for ds in self.datasets))
        else:
            self.epoch_size = int(epoch_size)
            if self.epoch_size <= 0:
                raise ValueError(f"epoch_size must be positive, got {self.epoch_size}")

        # Helpful for debugging / scripts that expect video_paths on the root dataset.
        self.video_paths: list[str] = []
        for ds in self.datasets:
            paths = getattr(ds, "video_paths", None)
            if paths:
                self.video_paths.extend(list(paths))

        mix_desc = ", ".join(
            f"{self.names[i]}={self.probs[i]:.3f} (n={len(self.datasets[i])})"
            for i in range(self.num_sources)
        )
        log.info(
            f"WeightedMultiDataset: epoch_size={self.epoch_size}, sources=[{mix_desc}]"
        )

    def __str__(self) -> str:
        parts = [f"{self.names[i]}:{len(self.datasets[i])}" for i in range(self.num_sources)]
        return f"WeightedMultiDataset(epoch_size={self.epoch_size}, [{', '.join(parts)}])"

    def __len__(self) -> int:
        return self.epoch_size

    def pick(self, index: int) -> tuple[int, int]:
        """Deterministically map ``index`` → ``(source_idx, local_idx)``."""
        # Avoid correlating consecutive indices too strongly.
        rng = np.random.RandomState((self.seed + int(index) * 1_000_003) % (2**32))
        source_idx = int(rng.choice(self.num_sources, p=self.probs))
        local_idx = int(rng.randint(0, len(self.datasets[source_idx])))
        return source_idx, local_idx

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_idx, local_idx = self.pick(index)
        sample = self.datasets[source_idx][local_idx]
        if self.attach_source_name:
            if not isinstance(sample, dict):
                raise TypeError(
                    f"attach_source_name requires dict samples, got {type(sample)}"
                )
            # Shallow copy so we do not mutate cached / shared dicts.
            sample = dict(sample)
            sample["dataset_name"] = self.names[source_idx]
            sample["dataset_index"] = source_idx
            sample["source_local_index"] = local_idx
        return sample
