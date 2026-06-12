from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BASE = PROJECT_ROOT / "configs" / "base.yaml"

VALID_APPROACHES = {"base", "sft", "skill"}
VALID_STRATEGIES = {"zero_shot", "few_shot", "cot", "skill", "skill_few_shot"}


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


@dataclass
class ExperimentConfig:
    # Identity
    phase: str = "base_eval"
    run_tag: str = ""

    # Model
    model_name: str = ""
    models: List[str] = field(default_factory=list)

    # Approach & strategy
    approach: str = "base"
    prompt_strategy: str = "zero_shot"
    prompt_strategies: List[str] = field(default_factory=list)
    few_shot_k: int = 3
    few_shot_counts: List[int] = field(default_factory=lambda: [1, 3, 5])

    # Inference
    max_seq_length: int = 2048
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False

    # Quantization
    quantization: str = "4bit"
    compute_dtype: str = "bfloat16"
    device_map: str = "auto"

    # Dataset
    dataset_split: str = "test"
    subset_fraction: float = 1.0
    test_subset_fraction: float = 1.0
    max_samples: int = 0
    limit: int = 0

    # Skill
    skill_path: str = ""
    skill_max_iterations: int = 20

    # Paths
    output_dir: str = "results/runs"
    models_dir: str = "models"
    data_dir: str = "data"

    # Resume / caching
    resume: bool = False
    save_every_n: int = 10

    # Storage
    use_gcs: bool = False
    gcs_bucket: str = ""
    gcs_prefix: str = ""

    # Control
    seed: int = 42
    save_predictions: bool = True
    save_raw_outputs: bool = True

    def effective_limit(self) -> int:
        if self.limit > 0:
            return self.limit
        if self.max_samples > 0:
            return self.max_samples
        return 0

    def config_hash(self) -> str:
        blob = json.dumps({
            "model_name": self.model_name,
            "approach": self.approach,
            "prompt_strategy": self.prompt_strategy,
            "few_shot_k": self.few_shot_k,
            "max_seq_length": self.max_seq_length,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
            "quantization": self.quantization,
            "skill_path": self.skill_path,
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for k in self.__dataclass_fields__:
            d[k] = getattr(self, k)
        return d


def load_config(
    phase_path: str | Path,
    base_path: str | Path | None = None,
    overrides: Dict[str, Any] | None = None,
) -> ExperimentConfig:
    base_path = Path(base_path) if base_path else DEFAULT_BASE
    phase_path = Path(phase_path)

    base_dict = _load_yaml(base_path) if base_path.exists() else {}
    phase_dict = _load_yaml(phase_path) if phase_path.exists() else {}
    merged = _deep_merge(base_dict, phase_dict)

    if overrides:
        merged = _deep_merge(merged, overrides)

    cfg = ExperimentConfig()
    for key, value in merged.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    _validate(cfg)
    return cfg


def _validate(cfg: ExperimentConfig) -> None:
    if cfg.approach and cfg.approach not in VALID_APPROACHES:
        raise ValueError(
            f"Invalid approach '{cfg.approach}'. Must be one of {VALID_APPROACHES}"
        )
    if cfg.prompt_strategy and cfg.prompt_strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Invalid prompt_strategy '{cfg.prompt_strategy}'. "
            f"Must be one of {VALID_STRATEGIES}"
        )
    if cfg.subset_fraction <= 0.0 or cfg.subset_fraction > 1.0:
        raise ValueError("subset_fraction must be in (0, 1]")
