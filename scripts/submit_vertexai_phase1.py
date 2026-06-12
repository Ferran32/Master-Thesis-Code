#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

from google.cloud import aiplatform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODELS, machine_spec_for

# GCP target is read from the environment so no project or bucket name is committed.
# Required: GCP_PROJECT, GCS_RESULTS_BUCKET. Optional: GCP_REGION (defaults to europe-west4).
PROJECT_ID = os.environ.get("GCP_PROJECT", "")
LOCATION = os.environ.get("GCP_REGION", "europe-west4")
IMAGE_URI = f"{LOCATION}-docker.pkg.dev/{PROJECT_ID}/docker-image/base-eval:latest"
QWEN36_IMAGE_URI = f"{LOCATION}-docker.pkg.dev/{PROJECT_ID}/docker-image/qwen36:latest"
GCS_BUCKET = os.environ.get("GCS_RESULTS_BUCKET", "")

ALL_MODELS: List[str] = list(MODELS.keys())


def image_for(model_name: str) -> str:
    return QWEN36_IMAGE_URI if model_name == "qwen3.6-27b" else IMAGE_URI


STRATEGY_VARIANTS = [
    ("zero_shot", 0, "zero-shot"),
    ("few_shot",  1, "few-shot-k1"),
    ("few_shot",  3, "few-shot-k3"),
    ("cot",       0, "cot"),
]


def submit_jobs(models: List[str], hf_token: str = "", strategies: List[str] | None = None) -> None:
    aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=f"gs://{GCS_BUCKET}")

    variants = STRATEGY_VARIANTS
    if strategies:
        variants = [(s, k, sfx) for s, k, sfx in STRATEGY_VARIANTS if sfx in strategies or s in strategies]

    total = 0
    for model_name in models:
        for strategy, k, suffix in variants:
            env = [
                {"name": "MODEL_NAME", "value": model_name},
                {"name": "GCS_BUCKET", "value": GCS_BUCKET},
                {"name": "GCS_PREFIX", "value": "base-eval"},
                {"name": "PROMPT_STRATEGY", "value": strategy},
                {"name": "FEW_SHOT_K", "value": str(k)},
            ]
            if hf_token:
                env.append({"name": "HF_TOKEN", "value": hf_token})

            display_name = f"base-eval-{model_name}-{suffix}"
            job = aiplatform.CustomJob(
                display_name=display_name,
                worker_pool_specs=[{
                    "machine_spec": machine_spec_for(model_name),
                    "replica_count": 1,
                    "container_spec": {
                        "image_uri": image_for(model_name),
                        "command": ["python3", "run_job_base_eval.py"],
                        "env": env,
                    },
                }],
            )
            job.submit()
            print(f"Submitted: {display_name:45s} → {job.resource_name}")
            total += 1

    print(f"\nAll {total} job(s) submitted ({len(models)} model(s) × {len(variants)} strategies).")
    print("\nMonitor at:")
    print(f"  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
    print(f"\nRetrieve results after completion:")
    print(f"  gsutil -m cp -r gs://{GCS_BUCKET}/base-eval/ results/runs/base-eval/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit Vertex AI Phase 1 job")
    parser.add_argument(
        "--models", nargs="+", default=ALL_MODELS,
        help="Models to evaluate. Defaults to all 4 models.",
    )
    hf_token = os.getenv("HF_TOKEN")
    parser.add_argument("--hf-token", default=hf_token, help="HuggingFace token")
    parser.add_argument(
        "--strategies", nargs="+", default=None,
        help="Subset of strategies to submit (e.g. few_shot few-shot-k1 few-shot-k3). Defaults to all.",
    )
    args = parser.parse_args()

    if not PROJECT_ID or not GCS_BUCKET:
        sys.exit(
            "Set GCP_PROJECT and GCS_RESULTS_BUCKET environment variables before submitting "
            "(see README). GCP_REGION is optional and defaults to europe-west4."
        )

    if not args.hf_token:
        print("ERROR: HF_TOKEN is not set")
        sys.exit(1)

    models = args.models
    if not models:
        models = ALL_MODELS

    submit_jobs(models, hf_token=args.hf_token, strategies=args.strategies)


if __name__ == "__main__":
    main()
