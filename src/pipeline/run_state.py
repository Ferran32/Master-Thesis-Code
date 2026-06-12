from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Set


def load_completed_indices(predictions_path: Path) -> Set[int]:
    completed: Set[int] = set()
    if not predictions_path.exists():
        return completed
    with open(predictions_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                completed.add(row["idx"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed

def append_prediction(predictions_path: Path, row: Dict[str, Any]) -> None:
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(predictions_path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
