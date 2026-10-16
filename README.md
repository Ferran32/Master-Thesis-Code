# Master Thesis — LLM-Based Vulnerability Detection

Experiment code for a master thesis comparing three approaches to automated source-code vulnerability detection with Large Language Models. All three are evaluated on the **same test set**, with the **same four models**, identical prompt structure, and identical inference configuration. The test set is deduplicated against training/validation at the exact-snippet (SHA-1) level, not held out by project: 18 of its 31 distinct projects also appear by name in the training/validation corpus (see the thesis's project-overlap audit, §4.3.2).

1. **Base LLM Evaluation** — zero-shot, few-shot (k=1, 3), and Chain-of-Thought prompting
2. **Label-only QLoRA fine-tuning (SFT)** — QLoRA parameter-efficient fine-tuning
3. **Agent Skill Optimization** — a skill prompt iteratively refined on the validation set; at inference it is a static prompt prefix (no tool calls)

A **combined-method matrix** (SFT + skill + few-shot) is also implemented in this repository (`scripts/submit_vertexai_combo.py`) and jobs were run, but it is **not evaluated or reported in the thesis** — treat any combo results under `results/combo/` as exploratory, not thesis findings. Experiments run as Vertex AI Custom Jobs on GCP, one job per model × strategy, in parallel.

## Models

| Short name | HuggingFace ID |
|---|---|
| `llama3-8b` | `meta-llama/Meta-Llama-3-8B-Instruct` |
| `qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` |
| `deepseek-coder-6.7b` | `deepseek-ai/deepseek-coder-6.7b-instruct` |
| `codellama-7b` | `codellama/CodeLlama-7b-Instruct-hf` |

Optional larger models (enterprise scale-up, not part of the core comparison) are defined in `src/config.py` under `LARGE_MODELS` and auto-route to A100 machines.

## Datasets

The BigVul/MegaVul/CVEfixes training corpus is **not committed** to this repository — this is the only dataset the "not committed" claim applies to. The pipeline downloads these three public sources from HuggingFace on first run, then cleans, normalizes, SHA1-deduplicates, and merges them before building the splits:

| Dataset | HuggingFace ID |
|---|---|
| BigVul | `DynaOuchebara/BigVul` |
| MegaVul | `hitoshura25/megavul` |
| CVEfixes | `hitoshura25/cvefixes` |

The merge is split in two stages (`src/data_utils.py`): first, 10% of **projects** are held out entirely to form an in-distribution test split (`_split_by_project`); the remaining projects' rows are then shuffled row-wise into train/val (`_train_val_split`), so train and val share projects with each other (but not with the held-out test split). Each vuln-fix pair expands into two labelled rows. Resulting sizes: Train 45,964 / Val 11,299 / Test 4,911 rows. **The 4,911-row in-distribution test split is not used for any result reported in the thesis** — all reported results use the independent MVFSC benchmark below.

The independent evaluation benchmark — the one the thesis actually reports results on — is the **MVFSC** test set (MIVIA Vulnerable Function Source Code) introduced by Carletti et al. (ARES 2025, DOI 10.1007/978-3-032-00644-8_8): 972 functions in the source paper, 872 after SHA-1 deduplication (439 vulnerable / 433 non-vulnerable). It is deduplicated against train/val at the exact-snippet (SHA-1) level only, not by project: 18 of its 31 distinct projects (by name) also appear in the training/validation corpus. The verbatim upstream file and the 872-row processed evaluation set used for every reported result **are committed in this repository at [`data/mvfsc/`](data/mvfsc/README.md)** (`mvfsc_test_set.xlsx`, SHA-256 `b12cde118202b31a65b25cd835d4cc2f205238c66aa65c0a4030088220362593`; see that directory's README for the processed file's hash and full provenance). The experiment pipeline itself loads the same data from the private HuggingFace mirror `ferran-solanes/MIVIA` (requires `HF_TOKEN`, see Setup) and caches the processed result at `data/splits/mivia_test/`; to use the committed CSV directly without an HF token, read `data/mvfsc/mvfsc_test_set_processed.csv` instead of calling `load_mivia_test()`.

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

### Phase 2 — Label-only QLoRA fine-tuning (SFT)

Train QLoRA adapters, then evaluate on the MVFSC test set (snippet-deduplicated against train/val, not project-disjoint from it — see Datasets above). Every vulnerable training completion is the same fixed explanation string plus an optional `cwe_id`; non-vulnerable completions are the bare `{"vulnerable": false}`. This is the labeling used for the SFT results reported in the thesis (`src/legacy/run_sft_train.py::_build_label_json`); a snippet-diff-based label scheme has since been implemented in `src/data_utils.py` but the corresponding Phase 2 rerun has not yet been submitted.

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
gsutil -m cp -r gs://ferrans-thesis-results/ results/runs/
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
- Max sequence length 2048 (all four core models — the thesis-reported adapters were all trained at this length, confirmed per-model from each `training_summary.json` under `results/sft/`)

`SFTConfig.max_seq_length` defaults to 2048, but `scripts/submit_vertexai_phase2.py` overrides it per model via a `MODEL_MAX_SEQ` dict whose fallback for any unlisted model is 4096; only `qwen2.5-7b` (plus the optional larger models) is currently listed, so resubmitting Phase 2 today would request 4096 for `llama3-8b`, `deepseek-coder-6.7b`, and `codellama-7b` — different from what was actually used for the reported adapters. Add those three to `MODEL_MAX_SEQ` (value 2048) before resubmitting, to keep the rerun comparable to the reported run.
