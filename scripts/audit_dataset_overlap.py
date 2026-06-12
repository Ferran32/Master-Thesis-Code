from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import load_from_disk


def load_sets(path: Path) -> tuple[set[str], set[str]]:
    ds = load_from_disk(str(path))
    projects = set(ds["project"])
    snippets = set(ds["code_sha1"])
    return projects, snippets


def pct(shared: int, total: int) -> str:
    return f"{100 * shared / total:.1f}%" if total else "N/A"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", metavar="DATASET_PATH")
    args = parser.parse_args()

    paths = [Path(p) for p in args.datasets]
    for p in paths:
        if not p.exists():
            print(f"[error] Not found: {p}", file=sys.stderr)
            sys.exit(1)

    print("[load] Loading datasets …")
    loaded: dict[str, tuple[set[str], set[str]]] = {}
    for p in paths:
        projects, snippets = load_sets(p)
        loaded[p.name] = (projects, snippets)
        print(f"  {p.name}: {len(snippets)} snippets, {len(projects)} projects")

    print()
    names = list(loaded.keys())
    for a, b in itertools.combinations(names, 2):
        proj_a, snip_a = loaded[a]
        proj_b, snip_b = loaded[b]

        shared_proj = proj_a & proj_b
        shared_snip = snip_a & snip_b

        print(f"=== {a} vs {b} ===")
        print(
            f"  Projects:  {a}={len(proj_a)}  {b}={len(proj_b)}"
            f"  shared={len(shared_proj)}"
            f"  ({pct(len(shared_proj), len(proj_b))} of {b})"
        )
        print(
            f"  Snippets:  {a}={len(snip_a)}  {b}={len(snip_b)}"
            f"  shared={len(shared_snip)}"
            f"  ({pct(len(shared_snip), len(snip_b))} of {b})"
        )
        if shared_proj:
            names_str = ", ".join(sorted(shared_proj))
            print(f"  Shared projects: {names_str}")
        print()


if __name__ == "__main__":
    main()
