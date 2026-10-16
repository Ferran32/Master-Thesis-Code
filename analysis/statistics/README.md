# Examiner revision: near-duplicate and paired-significance analyses

CPU-only analyses on data the thesis already produced. No model inference,
training, or GPU job is run here, and no existing result file under
`results/`, `data/`, or elsewhere in the repo is modified.

## Setup

```bash
cd /home/ferri/2_MasterKTH/Master-Thesis
.venv/bin/python3 -m pip install -r requirements.txt   # already satisfied in this repo's venv
.venv/bin/python3 -m pip install scipy datasketch matplotlib python-dotenv
```

## Run (one command each)

```bash
.venv/bin/python3 analysis/statistics/near_duplicates.py   # Part 1, ~3 min
.venv/bin/python3 analysis/statistics/paired_tests.py      # Part 2, ~5 sec
```

Both read from `data/splits/{train,val,mivia_test}` and the 28 existing
`predictions.jsonl` files under `results/`, and write only into
`analysis/statistics/results/`.

## Layout

- `common.py` — path resolution for the 28 configs, `normalize()` reproduced
  verbatim from `src/data_utils.py::_normalize`, prediction-file loading.
- `lexer.py` — the simple C/C++ tokenizer used for shingling.
- `metrics.py` — shared P/R/F1/F2/accuracy/PPR + exact binomial test.
- `near_duplicates.py` — Part 1.
- `paired_tests.py` — Part 2.
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
rounding). That check is done once, separately, and its result is recorded
verbatim in `SUMMARY.md` §1 — it is not re-run on every invocation of the two
scripts (they'd otherwise re-validate 28 files on every run for no benefit;
the check is deterministic given static input files).
