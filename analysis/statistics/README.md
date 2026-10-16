# Statistical analyses: near-duplicate audit, paired tests, project-disjoint re-scoring, truncation and per-CWE audits

CPU-only analyses on data the thesis already produced. No model inference,
training, or GPU job is run here, and no existing result file under
`results/`, `data/`, or elsewhere in the repo is modified.

## Setup

```bash
# from the repository root
.venv/bin/python3 -m pip install -r requirements.txt   # already satisfied in this repo's venv
.venv/bin/python3 -m pip install scipy datasketch matplotlib python-dotenv
```

## Run (one command each)

```bash
.venv/bin/python3 analysis/statistics/validate_gate.py        # Step 0, ~1 sec
.venv/bin/python3 analysis/statistics/near_duplicates.py       # ~3 min
.venv/bin/python3 analysis/statistics/paired_tests.py          # ~5 sec
.venv/bin/python3 analysis/statistics/confusion_matrices.py    # confusion matrices + full metrics, 28 configs
.venv/bin/python3 analysis/statistics/project_disjoint.py      # project-disjoint vs. shared re-scoring
.venv/bin/python3 analysis/statistics/truncation_audit.py      # prompt-length / truncation audit
.venv/bin/python3 analysis/statistics/per_cwe.py                # per-CWE recall/FN breakdown
```

All scripts read from `data/splits/{train,val,mivia_test}` and the 28 existing
`predictions.jsonl` files under `results/`, and write only into
`analysis/statistics/results/`.

## Layout

- `common.py` — path resolution for the 28 configs, `normalize()` reproduced
  verbatim from `src/data_utils.py::_normalize`, prediction-file loading.
- `lexer.py` — the simple C/C++ tokenizer used for shingling.
- `metrics.py` — shared P/R/F1/F2/accuracy/PPR + exact binomial test.
- `validate_gate.py` — recomputes the confusion matrix of all 28
  configurations from the prediction files and checks it against the
  thesis's published tables; result cached to `results/validation_gate.json`.
- `near_duplicates.py` — token-shingle MinHash/LSH near-duplicate audit of
  the 872 MVFSC functions against train+val, plus the effect of removing
  flagged rows on every configuration's metrics.
- `paired_tests.py` — within-model paired McNemar tests and bootstrap CIs
  for the F2/precision differences between configurations.
- `confusion_matrices.py` — confusion matrices and full metrics (P/R/F1/F2/
  accuracy/PPR) for all 28 configurations, cross-checked against the
  thesis's published F2 values.
- `project_disjoint.py` — splits the 872 MVFSC functions into a disjoint
  subset (projects absent from train+val) and a shared subset (projects
  present in train+val), and recomputes all 28 configs' metrics on each
  subset with bootstrap CIs and per-model ROC convex hull area.
- `truncation_audit.py` — how many of the 872 MVFSC prompts exceed each
  model's max sequence length, per prompt strategy, and how many
  Chain-of-Thought outputs hit `max_new_tokens=512` before a parseable
  verdict.
- `per_cwe.py` — CWE-class distribution and per-CWE recall/false-negative
  breakdown for all 28 configurations.
- `fetch_mvfsc_pairs.py` — one-off provenance script (already run) that
  reconstructs the vulnerable/patched pair_id for each of the 872 MVFSC rows
  by re-running `src.data_utils.load_mivia_test()`'s expansion loop against
  the raw `ferran-solanes/MIVIA` HF dataset, keeping the (pair_id, half) the
  original loop discards. Needs `HF_TOKEN` + network; its output is cached at
  `data/mvfsc_pairs.json` so the main scripts never need to re-run it.
- `data/mvfsc_pairs.json` — cached pair_id map, row `i` aligned 1:1 with
  `data/splits/mivia_test` row `i` and with `idx` in every predictions file
  (verified: 0/872 code_sha1 or label mismatches against the cached MVFSC
  dataset).
- `results/` — all CSV/PNG/JSON outputs, plus `SUMMARY.md`.

## Validation gate

`near_duplicates.py` and `paired_tests.py` both assume the 28 prediction
files reproduce the thesis's published confusion matrices (within ±1 cell,
rounding). That check is done once, separately, by `validate_gate.py`, and
its result is recorded verbatim in `SUMMARY.md` §1 — it is not re-run on
every invocation of the other scripts (they'd otherwise re-validate 28
files on every run for no benefit; the check is deterministic given static
input files).
