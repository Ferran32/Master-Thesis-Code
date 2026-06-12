from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_REGISTRY = Path("results") / "skill_versions.jsonl"


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def register_version(
    skill_path: str,
    skill_content: str,
    metrics: Dict[str, float],
    notes: str = "",
    registry_path: Path = DEFAULT_REGISTRY,
) -> Dict[str, Any]:
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_registry(registry_path)
    version_num = len(existing) + 1

    entry = {
        "version_id": f"v{version_num}",
        "content_hash": _content_hash(skill_content),
        "content_path": skill_path,
        "timestamp": datetime.datetime.now().isoformat(),
        "metrics": metrics,
        "notes": notes,
    }

    with open(registry_path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entry


def load_registry(registry_path: Path = DEFAULT_REGISTRY) -> List[Dict[str, Any]]:
    if not registry_path.exists():
        return []
    entries = []
    with open(registry_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def get_best_version(
    metric: str = "f1",
    registry_path: Path = DEFAULT_REGISTRY,
) -> Optional[Dict[str, Any]]:
    entries = load_registry(registry_path)
    if not entries:
        return None
    return max(entries, key=lambda e: e.get("metrics", {}).get(metric, 0))


def compare_versions(
    registry_path: Path = DEFAULT_REGISTRY,
) -> str:
    entries = load_registry(registry_path)
    if not entries:
        return "No skill versions registered yet."

    lines = [
        f"{'Version':<10} {'Hash':<14} {'F1':>8} {'F2':>8} {'P':>8} {'R':>8} {'Timestamp':<22} Notes",
        "-" * 100,
    ]
    for e in entries:
        m = e.get("metrics", {})
        lines.append(
            f"{e['version_id']:<10} "
            f"{e['content_hash']:<14} "
            f"{m.get('f1', 0):>8.4f} "
            f"{m.get('f2', 0):>8.4f} "
            f"{m.get('precision', 0):>8.4f} "
            f"{m.get('recall', 0):>8.4f} "
            f"{e.get('timestamp', '')[:19]:<22} "
            f"{e.get('notes', '')}"
        )

    best = get_best_version("f1", registry_path)
    if best:
        lines.append(f"\nBest by F1: {best['version_id']} (F1={best['metrics'].get('f1', 0):.4f})")

    return "\n".join(lines)
