
from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import Dataset

from src.data_utils import load_split as _load_split_raw, load_mivia_test


def load_dataset_split(
    split: str,
    subset_fraction: float = 1.0,
    max_samples: int = 0,
    seed: int = 42,
    source: str = "local",
) -> Dataset:
    if source == "mivia":
        ds = load_mivia_test()
    else:
        ds = _load_split_raw(split)

    effective_n = len(ds)
    if subset_fraction < 1.0:
        effective_n = max(1, int(len(ds) * subset_fraction))
    if 0 < max_samples < effective_n:
        effective_n = max_samples

    if effective_n < len(ds):
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(ds)), effective_n))
        ds = ds.select(indices)

    return ds


def save_index_manifest(
    run_dir: Path,
    ds: Dataset,
    split: str,
    seed: int,
) -> Path:
    manifest = {
        "split": split,
        "seed": seed,
        "n_samples": len(ds),
        "indices": list(range(len(ds))),
    }
    path = run_dir / "index_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path
