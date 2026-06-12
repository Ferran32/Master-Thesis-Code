#!/usr/bin/env python3
"""Submit Vertex AI combined-method jobs (few-shot + skill, SFT + few-shot, triple).

Three combos, each run for k=1 and k=3:
  - skill_few_shot_base : skill + few-shot on the base (un-tuned) models
  - sft_few_shot        : few-shot prompt applied to the fine-tuned adapters
  - triple              : skill + few-shot applied to the fine-tuned adapters

The SFT combos reuse the existing adapters in gs://.../sft/{model}/adapter/final_adapter
(no retraining). All jobs evaluate on the MVFSC test set by default.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

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

ALL_COMBOS = ["skill_few_shot_base", "sft_few_shot", "triple"]
FEW_SHOT_KS = [1, 3]

# Skill + k=3 prompts are long; raise the input window to avoid truncation.
COMBO_MAX_SEQ = "8192"


def upload_skill_to_gcs(skill_path: Path) -> str:
    gcs_path = f"skills/{skill_path.name}"
    dest = f"gs://{GCS_BUCKET}/{gcs_path}"
    print(f"Uploading skill to {dest}")
    subprocess.run(["gsutil", "cp", str(skill_path), dest], check=True)
    return gcs_path


def _common_env(model_name: str, dataset_split: str, limit: int, hf_token: str) -> List[Dict[str, str]]:
    env = [
        {"name": "MODEL_NAME", "value": model_name},
        {"name": "GCS_BUCKET", "value": GCS_BUCKET},
        {"name": "DATASET_SPLIT", "value": dataset_split},
        {"name": "LIMIT", "value": str(limit)},
    ]
    if hf_token:
        env.append({"name": "HF_TOKEN", "value": hf_token})
    return env


def build_job_spec(combo: str, model_name: str, k: int, skill_gcs_path: str,
                   dataset_split: str, limit: int, hf_token: str) -> Dict:
    env = _common_env(model_name, dataset_split, limit, hf_token)

    if combo == "skill_few_shot_base":
        env += [
            {"name": "GCS_PREFIX", "value": "combo/base"},
            {"name": "PROMPT_STRATEGY", "value": "skill_few_shot"},
            {"name": "FEW_SHOT_K", "value": str(k)},
            {"name": "SKILL_GCS_PATH", "value": skill_gcs_path},
        ]
        worker = {"machine_spec": machine_spec_for(model_name), "replica_count": 1}
        entrypoint = "run_job_base_eval.py"

    elif combo in ("sft_few_shot", "triple"):
        sft_strategy = "few_shot" if combo == "sft_few_shot" else "skill_few_shot"
        env += [
            {"name": "GCS_PREFIX", "value": "combo/sft"},
            {"name": "EVAL_ONLY", "value": "true"},
            {"name": "GCS_ADAPTER_PREFIX", "value": f"sft/{model_name}/adapter/final_adapter"},
            {"name": "SFT_STRATEGY", "value": sft_strategy},
            {"name": "FEW_SHOT_K", "value": str(k)},
            {"name": "MAX_SEQ_LENGTH", "value": COMBO_MAX_SEQ},
        ]
        if combo == "triple":
            env.append({"name": "SKILL_GCS_PATH", "value": skill_gcs_path})
        worker = {"machine_spec": machine_spec_for(model_name, sft=True), "replica_count": 1}
        entrypoint = "run_job_sft.py"

    else:
        raise ValueError(f"Unknown combo: {combo}")

    return {
        "display_name": f"combo-{combo}-k{k}-{model_name}",
        "entrypoint": entrypoint,
        "worker": worker,
        "env": env,
    }


def submit_jobs(models: List[str], combos: List[str], skill_gcs_path: str,
                dataset_split: str, limit: int, hf_token: str, dry_run: bool,
                few_shot_ks: List[int] = FEW_SHOT_KS) -> None:
    if not dry_run:
        aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=f"gs://{GCS_BUCKET}")

    n = 0
    for combo in combos:
        for model_name in models:
            for k in few_shot_ks:
                spec = build_job_spec(combo, model_name, k, skill_gcs_path,
                                      dataset_split, limit, hf_token)
                if dry_run:
                    env_str = " ".join(f"{e['name']}={e['value']}" for e in spec["env"]
                                        if e["name"] != "HF_TOKEN")
                    print(f"[dry-run] {spec['display_name']}  ({spec['entrypoint']}, "
                          f"{spec['worker']['machine_spec']['machine_type']})\n    {env_str}")
                else:
                    job = aiplatform.CustomJob(
                        display_name=spec["display_name"],
                        worker_pool_specs=[{
                            **spec["worker"],
                            "container_spec": {
                                "image_uri": image_for(model_name),
                                "command": ["python3", spec["entrypoint"]],
                                "env": spec["env"],
                            },
                        }],
                    )
                    job.submit()
                    print(f"Submitted: {spec['display_name']:45s} → {job.resource_name}")
                n += 1

    print(f"\n{'(dry-run) ' if dry_run else ''}{n} job(s) {'planned' if dry_run else 'submitted'}.")
    if not dry_run:
        print("\nMonitor at:")
        print(f"  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
        print("\nRetrieve results after completion:")
        print(f"  gsutil -m cp -r gs://{GCS_BUCKET}/combo/ results/combo/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit Vertex AI combined-method jobs")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, metavar="MODEL")
    parser.add_argument("--combos", nargs="+", default=ALL_COMBOS, choices=ALL_COMBOS)
    parser.add_argument("--few-shot-ks", nargs="+", type=int, default=FEW_SHOT_KS,
                        help="Few-shot k values to run (default: 1 3)")
    parser.add_argument("--skill", default="skills/SKILL_vuln_detection_v9.md",
                        help="Local path to the skill file used by skill combos")
    parser.add_argument("--dataset-split", default="mivia_test",
                        help="Dataset split to evaluate on (default: mivia_test)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap samples for a smoke test (0 = full split)")
    hf_token = os.getenv("HF_TOKEN")
    parser.add_argument("--hf-token", default=hf_token,
                        help="HuggingFace token. Defaults to HF_TOKEN env var.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the job specs without uploading the skill or submitting")
    args = parser.parse_args()

    if not PROJECT_ID or not GCS_BUCKET:
        sys.exit(
            "Set GCP_PROJECT and GCS_RESULTS_BUCKET environment variables before submitting "
            "(see README). GCP_REGION is optional and defaults to europe-west4."
        )

    invalid = [m for m in args.models if m not in ALL_MODELS]
    if invalid:
        print(f"ERROR: Unknown model(s): {invalid}")
        sys.exit(1)

    needs_skill = any(c in ("skill_few_shot_base", "triple") for c in args.combos)
    skill_gcs_path = ""
    if needs_skill:
        skill_path = Path(args.skill)
        if not skill_path.exists():
            print(f"ERROR: Skill file not found: {skill_path}")
            sys.exit(1)
        skill_gcs_path = (f"skills/{skill_path.name}" if args.dry_run
                          else upload_skill_to_gcs(skill_path))

    submit_jobs(args.models, args.combos, skill_gcs_path,
                args.dataset_split, args.limit, args.hf_token, args.dry_run,
                few_shot_ks=args.few_shot_ks)


if __name__ == "__main__":
    main()
