"""One-off provenance script: reconstructs the vuln/fixed pair_id for each of
the 872 MVFSC rows by re-running the exact same expansion loop as
src.data_utils.load_mivia_test() against the raw ferran-solanes/MIVIA HF
dataset, but keeping the (pair_id, half) that loop normally discards.

Requires network access + HF_TOKEN (the private MIVIA dataset) -- this is a
data-fetch, not model inference/training. Already run once; its output is
checked into analysis/statistics/data/mvfsc_pairs.json so the rest of
the analysis (near_duplicates.py) never needs to re-run this or touch the
network. Row i of the output aligns 1:1 with row i of data/splits/mivia_test
and with `idx` in every predictions.jsonl file (verified byte-for-byte in
Step 0: 0/872 code_sha1 or label mismatches).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def main() -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        env_path = REPO_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip()
    assert token, "HF_TOKEN required (private MIVIA dataset)"

    raw = load_dataset("ferran-solanes/MIVIA", split="test", token=token)
    seen = set()
    records = []
    for pair_idx, row in enumerate(raw):
        vuln_code = (row.get("vulnerable_code") or "").strip()
        fixed_code = (row.get("fixed_code") or "").strip()
        for code, label, half in [(vuln_code, 1, "vuln"), (fixed_code, 0, "fixed")]:
            if not code:
                continue
            sha = _sha1(code)
            if sha in seen:
                continue
            seen.add(sha)
            records.append({"code_sha1": sha, "pair_id": pair_idx, "half": half})

    out = [{"idx": i, **r} for i, r in enumerate(records)]
    out_path = Path(__file__).resolve().parent / "data" / "mvfsc_pairs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    main()
