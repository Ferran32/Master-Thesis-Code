from __future__ import annotations

import datetime
import platform
import subprocess
import time
from typing import Any, Dict, Optional

from src.pipeline.config_loader import ExperimentConfig


def _get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _get_environment_info() -> Dict[str, str]:
    info: Dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda or ""
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_mb"] = str(
                torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            )
    except ImportError:
        pass
    return info


def generate_run_id(cfg: ExperimentConfig) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model = cfg.model_name or "multi"
    parts = [cfg.phase, model, cfg.prompt_strategy, ts]
    return "_".join(p for p in parts if p)


def build_run_meta(
    cfg: ExperimentConfig,
    run_id: str,
    elapsed_seconds: float = 0.0,
    n_samples: int = 0,
    n_errors: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "phase": cfg.phase,
        "model": cfg.model_name,
        "approach": cfg.approach,
        "strategy": cfg.prompt_strategy,
        "split": cfg.dataset_split,
        "n_samples": n_samples,
        "n_errors": n_errors,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "git_commit": _get_git_hash(),
        "environment": _get_environment_info(),
        "config_hash": cfg.config_hash(),
    }
    if extra:
        meta.update(extra)
    return meta


class RunTimer:
    def __init__(self):
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time

    @property
    def elapsed_so_far(self) -> float:
        return time.time() - self.start_time if self.start_time else 0.0
