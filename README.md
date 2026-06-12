# Master Thesis — LLM-Based Vulnerability Detection

Experiment code for a master thesis comparing three approaches to automated source-code vulnerability detection with Large Language Models. All three are evaluated on the **same project-disjoint test set**, with the **same four models**, identical prompt structure, and identical inference configuration:

1. **Base LLM Evaluation** — zero-shot, few-shot (k=1, 3), and Chain-of-Thought prompting
2. **Supervised Fine-Tuning (SFT)** — QLoRA parameter-efficient fine-tuning
3. **Agent Skill Optimization** — a skill prompt iteratively refined on the validation set; at inference it is a static prompt prefix (no tool calls)

A combined-method matrix (SFT + skill + few-shot) is also evaluated. Experiments run as Vertex AI Custom Jobs on GCP, one job per model × strategy, in parallel.

## Models

| Short name | HuggingFace ID |
|---|---|
| `llama3-8b` | `meta-llama/Meta-Llama-3-8B-Instruct` |
| `qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` |
| `deepseek-coder-6.7b` | `deepseek-ai/deepseek-coder-6.7b-instruct` |
| `codellama-7b` | `codellama/CodeLlama-7b-Instruct-hf` |

Optional larger models (enterprise scale-up, not part of the core comparison) are defined in `src/config.py` under `LARGE_MODELS` and auto-route to A100 machines.

## Datasets

The data is **not committed** to this repository. The pipeline downloads the three public sources from HuggingFace on first run, then cleans, normalizes, SHA1-deduplicates, and merges them before building the splits:

| Dataset | HuggingFace ID |
|---|---|
| BigVul | `DynaOuchebara/BigVul` |
| MegaVul | `hitoshura25/megavul` |
| CVEfixes | `hitoshura25/cvefixes` |

The merge is partitioned into a **project-disjoint 80/10/10 split** (no project appears in more than one split). Each vuln-fix pair expands into two labelled rows. Resulting sizes: Train 45,964 / Val 11,372 / Test 4,911 rows.

The independent evaluation benchmark is the **MVFSC** test set (MIVIA Vulnerable Function Source Code) introduced by Carletti et al. (ARES 2025, DOI 10.1007/978-3-032-00644-8_8): 972 functions in the source paper, 872 after SHA-1 deduplication (439 vulnerable / 433 non-vulnerable). The exact files used here, with full provenance and citation, are included at [`data/mvfsc/`](data/mvfsc/) under GPL-3.0. The pipeline loads it by default from the HuggingFace mirror `ferran-solanes/MIVIA`, which requires an `HF_TOKEN` (see Setup).

Splits are deterministic (`seed=42`), so anyone with dataset access can rebuild them exactly with `python -m src.data_utils`.


## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your HuggingFace token (needed for the private MVFSC dataset and gated
models):

```bash
echo "HF_TOKEN=hf_your_token_here" > .env
```

Build the dataset splits (downloads and caches the public datasets on first run):

```bash
python -m src.data_utils
```

### GCP prerequisites

```bash
gcloud auth application-default login
gcloud config set project <project-id>

# Provision infrastructure (GCS bucket + Artifact Registry)
cd infrastructure && terraform init && terraform apply && cd ..

# Build and push the Docker image
gcloud auth configure-docker <region>-docker.pkg.dev
docker build -t <region>-docker.pkg.dev/<project-id>/docker-image/base-eval:latest -f docker/Dockerfile .
docker push <region>-docker.pkg.dev/<project-id>/docker-image/base-eval:latest
```

## Experiment Workflow

All phases run as Vertex AI Custom Jobs. Each submit script builds and submits GPU jobs that
execute the corresponding Docker entrypoint, one job per model × strategy, in parallel. Each
phase evaluates on the MVFSC test set directly. Results are written to a GCS bucket for
retrieval, avoiding recomputation.

### Phase 1 — Base LLM Evaluation

Run all prompt strategies (zero-shot, few-shot k=1/3, CoT) on the four models.

```bash
python scripts/submit_vertexai_phase1.py \
    --models llama3-8b qwen2.5-7b deepseek-coder-6.7b codellama-7b --hf-token $HF_TOKEN
```

### Phase 2 — Supervised Fine-Tuning

Train QLoRA adapters, then evaluate on the held-out, project-disjoint test split.

```bash
python scripts/submit_vertexai_phase2.py \
    --models llama3-8b qwen2.5-7b deepseek-coder-6.7b codellama-7b --hf-token $HF_TOKEN
```

### Phase 3 — Skill Optimization

Evaluate a skill version. The skill is iteratively refined on the validation set; at inference it
is a static prompt prefix. Use `--test-set` for final runs.

```bash
python scripts/submit_vertexai_phase3.py \
    --skill skills/SKILL_vuln_detection_v9.md \
    --models llama3-8b qwen2.5-7b deepseek-coder-6.7b codellama-7b --hf-token $HF_TOKEN
```

### Combined-method matrix

Fuse SFT, skill, and few-shot (24 jobs: 4 models × k=1,3 × 3 combos).

```bash
python scripts/submit_vertexai_combo.py --combos skill_few_shot sft_few_shot triple --few-shot-ks 1 3
```

### Retrieve results & aggregate

```bash
gsutil -m cp -r "gs://$GCS_RESULTS_BUCKET/" results/runs/
python scripts/aggregate_results.py --phase base-eval --results-dir results/   # or: sft | skill | combo
python -m src.pipeline.export --results-dir results/runs
```

Monitor jobs at the
[Vertex AI Console](https://console.cloud.google.com/vertex-ai/training/custom-jobs).

## Key Hyperparameters

**Inference** (all approaches):
- Temperature 0 (greedy), top-p 1.0, seed 42
- Max sequence length 4096 (zero-shot/CoT) or 8192 (few-shot/skill), max new tokens 512
- 4-bit quantization (BitsAndBytes NF4), bfloat16

**QLoRA fine-tuning** (authoritative source: `src/config.py` `SFTConfig`):
- LoRA rank 16, alpha 32, dropout 0.05
- Learning rate 2e-5, paged AdamW 8-bit, 2 epochs
- Effective batch size 32, early-stop patience 3 on validation loss
- Target modules: q/k/v/o + gate/up/down_proj, NF4 + bfloat16, completion-only loss
- Max sequence length 4096 (2048 for Qwen models — OOM)

## Documentation

- [Configuration System](docs/configuration.md)
- [Phase Workflow](docs/phase_workflow.md)
- [Output Schema](docs/output_schema.md)
- [Phase 1 Guide](docs/phases/phase1-guide.md)
- [Phase 2 Guide](docs/phases/phase2-guide.md)
- [Phase 3 Guide](docs/phases/phase3-guide.md)
