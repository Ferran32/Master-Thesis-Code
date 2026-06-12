#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
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

def extract_skill_version(skill_path: Path) -> str:
    return skill_path.stem.split("_")[-1]


def upload_skill_to_gcs(skill_path: Path) -> str:
    gcs_path = f"skills/{skill_path.name}"
    dest = f"gs://{GCS_BUCKET}/{gcs_path}"
    print(f"Uploading skill to {dest}")
    subprocess.run(["gsutil", "cp", str(skill_path), dest], check=True)
    return gcs_path


def submit_jobs(models: List[str], skill_path: Path, skill_gcs_path: str,
                hf_token: str = "", diagnose: bool = False, test_set: bool = False) -> None:
    aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=f"gs://{GCS_BUCKET}")

    for model_name in models:
        env = [
            {"name": "MODEL_NAME", "value": model_name},
            {"name": "GCS_BUCKET", "value": GCS_BUCKET},
            {"name": "SKILL_GCS_PATH", "value": skill_gcs_path},
            {"name": "GCS_PREFIX", "value": "skill"},
            {"name": "DIAGNOSE", "value": "true" if diagnose else "false"},
        ]
        if test_set:
            env.append({"name": "DATASET_SPLIT", "value": "mivia_test"})
            env.append({"name": "LIMIT", "value": "0"})
            env.append({"name": "SKILL_VERSION_SUFFIX", "value": "_test"})
        if hf_token:
            env.append({"name": "HF_TOKEN", "value": hf_token})

        skill_version = extract_skill_version(skill_path)
        job = aiplatform.CustomJob(
            display_name=f"skill-{skill_version}-{model_name}",
            worker_pool_specs=[{
                "machine_spec": machine_spec_for(model_name),
                "replica_count": 1,
                "container_spec": {
                    "image_uri": image_for(model_name),
                    "command": ["python3", "run_job_skill.py"],
                    "env": env,
                },
            }],
        )
        job.submit()
        print(f"Submitted: {model_name:30s} → {job.resource_name}")

    print(f"\nAll {len(models)} job(s) submitted.")
    print("\nMonitor at:")
    print(f"  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
    print(f"\nRetrieve results after completion:")
    print(f"  gsutil -m cp -r gs://{GCS_BUCKET}/skill/ results/")
    if diagnose:
        skill_ver = extract_skill_version(skill_path)

        print(f"\nView diagnostic reports:")
        for m in models:
            print(f"  gsutil cat gs://{GCS_BUCKET}/skill/{skill_ver}/{m}/diagnostic.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit Vertex AI Phase 4 (skill) jobs")
    parser.add_argument("--skill", required=True, help="Local path to SKILL.md file")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, metavar="MODEL")
    hf_token = os.getenv("HF_TOKEN")
    parser.add_argument("--hf-token", default=hf_token, help="HuggingFace token. Defaults to HF_TOKEN environment variable.")
    parser.add_argument("--diagnose", action="store_true", help="Generate diagnostic report")
    parser.add_argument("--test-set", action="store_true", help="Evaluate on the MVFSC test set instead of validation set")
    args = parser.parse_args()

    if not PROJECT_ID or not GCS_BUCKET:
        sys.exit(
            "Set GCP_PROJECT and GCS_RESULTS_BUCKET environment variables before submitting "
            "(see README). GCP_REGION is optional and defaults to europe-west4."
        )

    skill_path = Path(args.skill)
    if not skill_path.exists():
        print(f"ERROR: Skill file not found: {skill_path}")
        sys.exit(1)

    invalid = [m for m in args.models if m not in ALL_MODELS]
    if invalid:
        print(f"ERROR: Unknown model(s): {invalid}")
        sys.exit(1)

    skill_gcs_path = upload_skill_to_gcs(skill_path)
    submit_jobs(args.models, skill_path, skill_gcs_path, hf_token=args.hf_token,
                diagnose=args.diagnose, test_set=args.test_set)


if __name__ == "__main__":
    main()
