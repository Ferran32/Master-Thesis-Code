#!/usr/bin/env python3
"""Submit a one-off model load + single-generation smoke test on Vertex AI.

Validates that a model/image actually loads and generates before spending GPU
hours on a full evaluation job. Defaults to qwen3.6-27b on its A100 image.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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


def image_for(model_name: str) -> str:
    return QWEN36_IMAGE_URI if model_name == "qwen3.6-27b" else IMAGE_URI


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a Vertex AI model smoke test")
    parser.add_argument("--model", default="qwen3.6-27b", choices=list(MODELS.keys()))
    parser.add_argument("--max-seq-length", default="2048")
    hf_token = os.getenv("HF_TOKEN")
    parser.add_argument("--hf-token", default=hf_token, help="Defaults to HF_TOKEN env var.")
    args = parser.parse_args()

    if not PROJECT_ID or not GCS_BUCKET:
        sys.exit(
            "Set GCP_PROJECT and GCS_RESULTS_BUCKET environment variables before submitting "
            "(see README). GCP_REGION is optional and defaults to europe-west4."
        )

    aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=f"gs://{GCS_BUCKET}")

    env = [
        {"name": "MODEL_NAME", "value": args.model},
        {"name": "MAX_SEQ_LENGTH", "value": str(args.max_seq_length)},
    ]
    if args.hf_token:
        env.append({"name": "HF_TOKEN", "value": args.hf_token})

    job = aiplatform.CustomJob(
        display_name=f"smoketest-{args.model}",
        worker_pool_specs=[{
            "machine_spec": machine_spec_for(args.model),
            "replica_count": 1,
            "container_spec": {
                "image_uri": image_for(args.model),
                "command": ["python3", "run_job_smoketest.py"],
                "env": env,
            },
        }],
    )
    job.submit()
    print(f"Submitted: smoketest-{args.model} → {job.resource_name}")
    print("\nMonitor at:")
    print(f"  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")


if __name__ == "__main__":
    main()
