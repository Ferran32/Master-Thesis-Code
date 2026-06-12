from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk

from src.config import (
    BIGVUL_HF,
    CVEFIXES_HF,
    MEGAVUL_HF,
    MIVIA_HF,
    RAW_DATASETS_DIR,
    SEED,
    SPLITS_DIR,
    TRAIN_RATIO,
)

COMMENT_RE = re.compile(
    r"""
    //.*?$                           |
    /\*.*?\*/                        |
    ^\s*\#\s*(include|define|if|ifdef|ifndef|elif|else|endif).*?$
    """,
    re.MULTILINE | re.DOTALL | re.VERBOSE,
)
WS_RE = re.compile(r"\s+")


def _normalize(code: str) -> str:
    if not code:
        return ""
    code = COMMENT_RE.sub("", code)
    return WS_RE.sub(" ", code).strip()


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _load_or_cache_raw(hf_id: str, name: str, split: str = "train") -> Dataset:
    cache_dir = RAW_DATASETS_DIR / name
    if cache_dir.exists() and any(cache_dir.iterdir()):
        print(f"[data_utils] Loading cached {name} from {cache_dir}")
        return load_from_disk(str(cache_dir))

    print(f"[data_utils] Downloading {name} from {hf_id} …")
    ds = load_dataset(hf_id, split=split)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(cache_dir))
    print(f"[data_utils] Cached {name} ({len(ds)} rows) → {cache_dir}")
    return ds


def _pick(cols: list, candidates: list) -> Optional[str]:
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def _load_raw_with_metadata() -> pd.DataFrame:
    records: List[Dict[str, str]] = []

    bigvul = _load_or_cache_raw(BIGVUL_HF, "bigvul")
    for row in bigvul:
        vuln_n = _normalize(row.get("func_before") or "")
        fix_n = _normalize(row.get("func_after") or "")
        if not vuln_n or not fix_n or vuln_n == fix_n:
            continue
        records.append({
            "source": "bigvul",
            "project": (row.get("project") or "unknown").strip(),
            "cve_id": (row.get("CVE ID") or ""),
            "cwe_id": (row.get("CWE ID") or ""),
            "vuln_code": vuln_n,
            "fixed_code": fix_n,
            "pair_sha1": _sha1(f"{vuln_n}\n<SEP>\n{fix_n}"),
        })
    print(f"[data_utils] BigVul: {len(records)} valid pairs")

    megavul = _load_or_cache_raw(MEGAVUL_HF, "megavul")
    n_before = len(records)
    for row in megavul:
        vuln_n = _normalize(row.get("vulnerable_code") or "")
        fix_n = _normalize(row.get("fixed_code") or "")
        if not vuln_n or not fix_n or vuln_n == fix_n:
            continue
        repo = (row.get("repo_url") or "unknown").strip()
        project = repo.rstrip("/").split("/")[-1] if "/" in repo else repo
        records.append({
            "source": "megavul",
            "project": project,
            "cve_id": (row.get("cve_id") or ""),
            "cwe_id": (row.get("cwe_id") or ""),
            "vuln_code": vuln_n,
            "fixed_code": fix_n,
            "pair_sha1": _sha1(f"{vuln_n}\n<SEP>\n{fix_n}"),
        })
    print(f"[data_utils] MegaVul: {len(records) - n_before} valid pairs")

    cvefixes = _load_or_cache_raw(CVEFIXES_HF, "cvefixes")
    n_before = len(records)
    for row in cvefixes:
        vuln_n = _normalize(row.get("vulnerable_code") or "")
        fix_n = _normalize(row.get("fixed_code") or "")
        if not vuln_n or not fix_n or vuln_n == fix_n:
            continue
        repo = (row.get("repo_url") or "unknown").strip()
        project = repo.rstrip("/").split("/")[-1] if "/" in repo else repo
        records.append({
            "source": "cvefixes",
            "project": project,
            "cve_id": (row.get("cve_id") or ""),
            "cwe_id": (row.get("cwe_id") or ""),
            "vuln_code": vuln_n,
            "fixed_code": fix_n,
            "pair_sha1": _sha1(f"{vuln_n}\n<SEP>\n{fix_n}"),
        })
    print(f"[data_utils] CVEfixes: {len(records) - n_before} valid pairs")

    df = pd.DataFrame(records)
    print(f"[data_utils] Total records: {len(df)}")

    before = len(df)
    df = df.drop_duplicates(subset=["pair_sha1"], keep="first")
    print(f"[data_utils] Global dedup: {before} → {len(df)} rows")
    return df


def _split_by_project(
    df: pd.DataFrame,
    test_project_frac: float = 0.10,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    projects = sorted(df["project"].unique().tolist())
    rng.shuffle(projects)

    n_test_projects = max(1, int(len(projects) * test_project_frac))
    test_projects: Set[str] = set(projects[:n_test_projects])

    mask = df["project"].isin(test_projects)
    test_df = df[mask].copy()
    main_df = df[~mask].copy()

    print(
        f"[data_utils] Project-disjoint split: "
        f"{len(test_projects)} test projects ({len(test_df)} rows), "
        f"{len(projects) - len(test_projects)} main projects ({len(main_df)} rows)"
    )
    return main_df, test_df


def _train_val_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_train = int(len(df) * train_ratio)
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:].copy()
    print(f"[data_utils] Train/Val split: {len(train_df)} train, {len(val_df)} val")
    return train_df, val_df


def _df_to_text_label(df: pd.DataFrame) -> Dataset:
    seen: Set[str] = set()
    records = []
    for _, row in df.iterrows():
        for code, label in [(row["vuln_code"], 1), (row["fixed_code"], 0)]:
            sha = _sha1(code)
            if sha in seen:
                continue
            seen.add(sha)
            records.append({
                "text": code,
                "label": label,
                "code_sha1": sha,
                "cve_id": row.get("cve_id", ""),
                "cwe_id": row.get("cwe_id", ""),
                "project": row.get("project", ""),
            })
    return Dataset.from_pandas(pd.DataFrame(records))


def _build_few_shot_pool(
    df: pd.DataFrame,
    max_examples: int = 200,
    seed: int = SEED,
) -> List[Tuple[str, Dict[str, Any]]]:
    rng = random.Random(seed)
    vuln_rows = df[df["vuln_code"].notna()].to_dict("records")
    rng.shuffle(vuln_rows)

    pool: List[Tuple[str, Dict[str, Any]]] = []
    half = max_examples // 2

    for row in vuln_rows[:half]:
        label_json: Dict[str, Any] = {
            "vulnerable": True,
            "explanation": "This code contains a security vulnerability.",
        }
        cwe = row.get("cwe_id", "")
        if cwe:
            label_json["cwe_id"] = cwe
        pool.append((row["vuln_code"], label_json))
        pool.append((row["fixed_code"], {"vulnerable": False}))

    rng.shuffle(pool)
    return pool[:max_examples]


def _load_mvfsc_norm_sha1s() -> Optional[Set[str]]:
    import os

    cache_path = SPLITS_DIR / "mivia_test"
    if not cache_path.exists() and not os.getenv("HF_TOKEN"):
        print(
            "[data_utils] WARNING: MVFSC cache not found and HF_TOKEN not set. "
            "Skipping MVFSC overlap removal. Re-run with HF_TOKEN to apply it."
        )
        return None

    ds = load_mivia_test()
    sha1s: Set[str] = set()
    for row in ds:
        norm = _normalize(row["text"])
        if norm:
            sha1s.add(_sha1(norm))
    print(f"[data_utils] MVFSC normalised SHA1s loaded: {len(sha1s)}")
    return sha1s


def prepare_splits(force: bool = False) -> Dict[str, Path]:
    paths = {
        "train": SPLITS_DIR / "train",
        "val": SPLITS_DIR / "val",
        "test": SPLITS_DIR / "test",
        "few_shot_pool": SPLITS_DIR / "few_shot_pool.json",
    }

    if not force and all(p.exists() for p in paths.values()):
        print("[data_utils] Splits already exist. Use force=True to regenerate.")
        return paths

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_raw_with_metadata()
    main_df, test_df = _split_by_project(df)
    train_df, val_df = _train_val_split(main_df)

    train_ds = _df_to_text_label(train_df)
    val_ds = _df_to_text_label(val_df)
    test_ds = _df_to_text_label(test_df)

    test_sha1s: Set[str] = set(test_ds["code_sha1"])
    n_train_before, n_val_before = len(train_ds), len(val_ds)
    train_ds = train_ds.filter(lambda r: r["code_sha1"] not in test_sha1s)
    val_ds   = val_ds.filter(  lambda r: r["code_sha1"] not in test_sha1s)
    print(
        f"[data_utils] Post-split dedup (code_sha1 ∩ test): "
        f"train {n_train_before}→{len(train_ds)} (-{n_train_before-len(train_ds)}), "
        f"val {n_val_before}→{len(val_ds)} (-{n_val_before-len(val_ds)})"
    )
    train_sha1s: Set[str] = set(train_ds["code_sha1"])
    n_val_before2 = len(val_ds)
    val_ds = val_ds.filter(lambda r: r["code_sha1"] not in train_sha1s)
    print(
        f"[data_utils] Post-split dedup (code_sha1 ∩ train): "
        f"val {n_val_before2}→{len(val_ds)} (-{n_val_before2-len(val_ds)})"
    )

    mvfsc_sha1s = _load_mvfsc_norm_sha1s()
    if mvfsc_sha1s is not None:
        n_train_before3, n_val_before3 = len(train_ds), len(val_ds)
        train_ds = train_ds.filter(lambda r: r["code_sha1"] not in mvfsc_sha1s)
        val_ds   = val_ds.filter(  lambda r: r["code_sha1"] not in mvfsc_sha1s)
        print(
            f"[data_utils] MVFSC overlap removal: "
            f"train {n_train_before3}→{len(train_ds)} (-{n_train_before3-len(train_ds)}), "
            f"val {n_val_before3}→{len(val_ds)} (-{n_val_before3-len(val_ds)})"
        )

    train_ds.save_to_disk(str(paths["train"]))
    val_ds.save_to_disk(str(paths["val"]))
    test_ds.save_to_disk(str(paths["test"]))
    print(f"[data_utils] Saved: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    pool = _build_few_shot_pool(train_df)
    with open(paths["few_shot_pool"], "w") as f:
        json.dump(pool, f, indent=2)
    print(f"[data_utils] Few-shot pool: {len(pool)} examples")

    return paths


def load_split(name: str) -> Dataset:
    path = SPLITS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Split '{name}' not found at {path}. Run prepare_splits() first."
        )
    return load_from_disk(str(path))


def load_few_shot_pool() -> List[Tuple[str, Dict[str, Any]]]:
    path = SPLITS_DIR / "few_shot_pool.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Few-shot pool not found at {path}. Run prepare_splits() first."
        )
    with open(path) as f:
        raw = json.load(f)
    return [(code, label) for code, label in raw]


def load_mivia_test(force: bool = False) -> Dataset:
    import os

    cache_path = SPLITS_DIR / "mivia_test"

    if not force and cache_path.exists():
        print(f"[data_utils] Loading cached MIVIA split from {cache_path}")
        return load_from_disk(str(cache_path))

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required to access the private MIVIA dataset. "
            "Set it in your .env file or as an environment variable."
        )

    print(f"[data_utils] Downloading MIVIA test set from {MIVIA_HF} …")
    raw = load_dataset(MIVIA_HF, split="test", token=token)
    print(f"[data_utils] MIVIA raw pairs: {len(raw)}")

    seen: Set[str] = set()
    records: List[Dict[str, Any]] = []
    for row in raw:
        vuln_code = (row.get("vulnerable_code") or "").strip()
        fixed_code = (row.get("fixed_code") or "").strip()
        cwe_id = row.get("cwe_id", "")
        project = row.get("project", "")

        for code, label in [(vuln_code, 1), (fixed_code, 0)]:
            if not code:
                continue
            sha = _sha1(code)
            if sha in seen:
                continue
            seen.add(sha)
            records.append({
                "text": code, "label": label, "code_sha1": sha,
                "cve_id": "", "cwe_id": cwe_id, "project": project,
            })

    ds = Dataset.from_pandas(pd.DataFrame(records))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(cache_path))
    print(f"[data_utils] MIVIA expanded: {len(ds)} rows → cached at {cache_path}")
    return ds


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare project-disjoint dataset splits")
    parser.add_argument("--force", action="store_true", help="Regenerate even if splits exist")
    args = parser.parse_args()

    paths = prepare_splits(force=args.force)
    print("\nSaved splits:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
