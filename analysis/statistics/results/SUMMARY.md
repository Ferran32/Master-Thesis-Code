# Statistical analyses: near-duplicate audit, paired tests, and related checks

All analyses are CPU-only, read-only with respect to the rest of the repo,
and use seed 42 wherever randomness is involved. No model inference,
training, or GPU job was run. No existing result file was modified.

Scripts: `analysis/statistics/{validate_gate,near_duplicates,paired_tests}.py`.
Raw outputs: `analysis/statistics/results/*.csv`, `*.json`, `*.png`.

---

## 1. Inputs and validation gate

| Input | Path | Rows | Key columns |
|---|---|---|---|
| Training split | `data/splits/train` (HF `Dataset`, Arrow) | 45,964 | `text`, `label`, `code_sha1`, `cve_id`, `cwe_id`, `project`, `vulnerable_code`, `patch` |
| Validation split | `data/splits/val` (HF `Dataset`, Arrow) | 11,299 | same as train |
| MVFSC test set | `data/splits/mivia_test` (HF `Dataset`, Arrow; this *is* `mvfsc_test_set_processed.csv` — cached post-processing of the private `ferran-solanes/MIVIA` HF dataset, 872 rows, 439 vulnerable / 433 non-vulnerable, 33 project strings → 31 distinct projects) | 872 | `text`, `label`, `code_sha1`, `cve_id` (empty for all rows), `cwe_id`, `project` |
| MVFSC pair IDs | `analysis/statistics/data/mvfsc_pairs.json` (new, derived) | 872 | `idx`, `pair_id`, `half` (`vuln`/`fixed`), `code_sha1` |
| Per-sample predictions, 28 configs | `results/base-eval/...` (16), `results/sft/*/predictions.jsonl` (4), `results/skill/{v9,v6}_test/*/predictions.jsonl` (8) | 872 each | `idx`, `label`, `predicted_label`, `parse_success` |
| Normalisation function | `src/data_utils.py::_normalize` | — | reproduced verbatim in `common.normalize()` |

**MVFSC pair ID note.** `mivia_test` has no `pair_id`/CVE column as stored. It is
built by `src.data_utils.load_mivia_test()`, which iterates 451 raw
vuln/fixed pairs from the private HF dataset and expands each into up to two
rows (skipping empty halves and exact-SHA1 duplicates already seen),
producing exactly 872 rows. We re-ran that same loop against the raw
`ferran-solanes/MIVIA` dataset (network + `HF_TOKEN`, no inference) while
keeping the `(pair_id, half)` the original loop discards
(`fetch_mvfsc_pairs.py`), and verified byte-for-byte that the result aligns
1:1 with the cached `mivia_test` dataset: **0/872 `code_sha1` or `label`
mismatches**. 430 of 451 raw pairs have both halves present in the 872 rows;
9 are vuln-only, 12 are fixed-only.

**Validation gate result: PASS.** All 28 configurations' recomputed confusion
matrices matched the thesis's published tables within the ±1 rounding
tolerance (`results/validation_gate.json`), and ground-truth labels are
identical across all 28 files after aligning on `idx`
(`ground_truth_identical_across_all_28: true`). Parse failures were treated
as negative predictions throughout, per the thesis convention.

| Model | Config | TP | FP | FN | TN | Parse fail |
|---|---|--:|--:|--:|--:|--:|
| Llama 3 8B | zero-shot | 125 | 132 | 314 | 301 | 2 |
| Llama 3 8B | few-shot k=1 | 86 | 78 | 353 | 355 | 0 |
| Llama 3 8B | few-shot k=3 | 175 | 167 | 264 | 266 | 0 |
| Llama 3 8B | CoT | 151 | 144 | 288 | 289 | 2 |
| Qwen2.5 7B | zero-shot | 1 | 1 | 438 | 432 | 2 |
| Qwen2.5 7B | few-shot k=1 | 16 | 17 | 423 | 416 | 0 |
| Qwen2.5 7B | few-shot k=3 | 46 | 50 | 393 | 383 | 0 |
| Qwen2.5 7B | CoT | 5 | 3 | 434 | 430 | 2 |
| DeepSeek-Coder 6.7B | zero-shot | 57 | 48 | 382 | 385 | 11 |
| DeepSeek-Coder 6.7B | few-shot k=1 | 294 | 277 | 145 | 156 | 1 |
| DeepSeek-Coder 6.7B | few-shot k=3 | 246 | 239 | 193 | 194 | 0 |
| DeepSeek-Coder 6.7B | CoT | 76 | 83 | 363 | 350 | 9 |
| Code Llama 7B | zero-shot | 224 | 206 | 215 | 227 | 28 |
| Code Llama 7B | few-shot k=1 | 287 | 293 | 152 | 140 | 71 |
| Code Llama 7B | few-shot k=3 | 226 | 221 | 213 | 212 | 383 |
| Code Llama 7B | CoT | 253 | 242 | 186 | 191 | 28 |
| Llama 3 8B | SFT | 130 | 116 | 309 | 317 | 2 |
| Qwen2.5 7B | SFT | 159 | 154 | 280 | 279 | 53 |
| DeepSeek-Coder 6.7B | SFT | 273 | 277 | 166 | 156 | 8 |
| Code Llama 7B | SFT | 238 | 231 | 201 | 202 | 49 |
| Llama 3 8B | skill v9 | 199 | 193 | 240 | 240 | 0 |
| Qwen2.5 7B | skill v9 | 15 | 16 | 424 | 417 | 0 |
| DeepSeek-Coder 6.7B | skill v9 | 182 | 173 | 257 | 260 | 2 |
| Code Llama 7B | skill v9 | 183 | 175 | 256 | 258 | 505 |
| Llama 3 8B | skill v6 | 200 | 198 | 239 | 235 | 0 |
| Qwen2.5 7B | skill v6 | 10 | 11 | 429 | 422 | 0 |
| DeepSeek-Coder 6.7B | skill v6 | 137 | 139 | 302 | 294 | 0 |
| Code Llama 7B | skill v6 | 190 | 169 | 249 | 264 | 512 |

---

## 2. Part 1 — Near-duplicate counts and sanity check

**Method as specified**: `_normalize()` (reused verbatim) → simple C/C++
lexer (`lexer.py`: identifiers/keywords keep their literal text as distinct
tokens; numeric/string/char literals abstracted to `NUM`/`STR`/`CHAR`;
operators/punctuation verbatim; comments/whitespace dropped) → token 3-gram
shingle sets → MinHash (128 permutations, seed 42) + LSH at **bucket
threshold 0.5** (well below the 0.7 floor the task requires recall for) →
exact Jaccard on LSH candidates only.

**LSH recall check.** Brute-force exact Jaccard was computed for 25 randomly
sampled MVFSC functions against the *entire* 55,655-function corpus
(train+val) and compared to the LSH-mediated result. All 7 cases with a true
best-match Jaccard ≥ 0.5 (up to 0.9939) matched the LSH result exactly; the
other 18 (true best-match < 0.23) correctly returned 0 LSH candidates. No
recall failures were found in this spot check.

**Skip count** (< 10 tokens): MVFSC 0/872, train 1,274/45,964, val 334/11,299.

**Sanity check (Jaccard = 1.0 to an exact normalised duplicate): FAILED, 2
cases found** (expected 0, since exact SHA-1 duplicates were already
removed). Both cases are genuine near-duplicates that exact-SHA1 dedup
missed by one character: MVFSC idx 405 (Qt, vulnerable) vs. a train-split
function differing only by a single space before `{` — not caught by
`_normalize()` because `_normalize()` only *collapses* whitespace runs, it
does not insert or remove whitespace where none exists against where one
does. Same for MVFSC idx 702 (TensorFlow, non-vulnerable) vs. a train-split
*vulnerable* function with the same one-space difference. These are
reported, not silently dropped (`results/nd_sanity_check.json`); they also
appear at the top of the "effect on results" filtering below, since both
are flagged at every threshold tested. This is a secondary finding in its
own right: whitespace-adjacent near-duplicates slip past exact-hash
dedup, and a token-based approach catches them where a text-hash approach
cannot.

**Jaccard distribution over all 872 MVFSC functions** (best match against
train+val; `results/nd_jaccard_distribution.csv`, histogram at
`results/nd_jaccard_histogram.png`):

| Cutoff | n at/above | Share of 872 |
|---|--:|--:|
| 0.5 | 217 | 24.9% |
| 0.6 | 204 | 23.4% |
| 0.7 | 185 | 21.2% |
| 0.8 | 167 | 19.2% |
| 0.9 | 129 | 14.8% |

**Counts by threshold, broken down** (`results/nd_counts_by_threshold.csv`):

| Threshold | n flagged | from train | from val | MVFSC vuln | MVFSC non-vuln | same label as match | opposite label | in 18 overlap projects |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.7 | 185 | 145 | 40 | 91 | 94 | 130 | 55 | 154 |
| 0.8 | 167 | 132 | 35 | 83 | 84 | 118 | 49 | 139 |
| 0.9 | 129 | 102 | 27 | 63 | 66 | 95 | 34 | 105 |

The 18-project name-overlap set was derived from the actual intersection of
project name strings between MVFSC and train+val (case-insensitive), not
assumed: `contiki-ng, ffmpeg, gnutls, gpac, harfbuzz, jasper, libarchive,
libmobi, lua, mruby, opencv, openssh, redis, squid, tensorflow, vim,
wireshark, yara` — exactly 18, matching the thesis's stated count. Per-project
breakdowns are in `results/nd_per_project_t{0.7,0.8,0.9}.csv`; at t=0.8 the
heaviest contributors are TensorFlow/Tensorflow/tensorflow (3 casing
variants combined: 41), wireshark (20), libarchive (15), vim (16), OpenSSH
(12), Contiki-NG (10).

**Reading these counts.** "Same label as match" (118–130 of the flagged
functions, depending on threshold) is the dangerous case worth flagging:
a near-identical function appears with the *same* vulnerable/non-vulnerable
label in both MVFSC and train/val, so a model could in principle have
memorised the label rather than learned to detect the pattern. "Opposite
label" (34–55) is consistent with the normal vuln/patched-pair structure
(the model saw the *other* half of the same pair during training/val, which
is a much weaker form of leakage since the two functions actually differ at
the vulnerability site).

---

## 3. Part 1 — Effect on results (near-duplicate-filtered subsets)

Metrics recomputed for all 28 configurations with flagged MVFSC functions
removed, in two variants: (a) remove only the flagged rows; (b) remove the
whole vuln/patched pair whenever either half is flagged (keeps the subset
balanced). Full tables for all three thresholds and both variants:
`results/nd_filtered_metrics_t{0.7,0.8,0.9}.csv` and
`results/nd_filtered_metrics_all.csv`; the unfiltered baseline is
`results/nd_filtered_metrics_full.csv`.

**Threshold 0.8, variant (a), n=705** (full table; `binom_p_precision_vs_base_rate`
and variant (b) are in the CSVs):

| Model | Config | P | R | F1 | F2 | Acc | PPR | ΔF2 vs full |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| Code Llama 7B | zero-shot | 0.529 | 0.534 | 0.531 | 0.533 | 0.525 | 0.509 | +0.021 |
| Code Llama 7B | few-shot k=1 | 0.499 | 0.674 | 0.574 | 0.630 | 0.494 | 0.682 | +0.016 |
| Code Llama 7B | few-shot k=3 | 0.511 | 0.520 | 0.515 | 0.518 | 0.506 | 0.514 | +0.005 |
| Code Llama 7B | CoT | 0.514 | 0.604 | 0.556 | 0.584 | 0.512 | 0.593 | +0.022 |
| Code Llama 7B | SFT | 0.509 | 0.553 | 0.530 | 0.544 | 0.505 | 0.549 | +0.009 |
| Code Llama 7B | skill v9 | 0.516 | 0.421 | 0.464 | 0.437 | 0.508 | 0.413 | +0.004 |
| Code Llama 7B | skill v6 | 0.543 | 0.461 | 0.498 | 0.475 | 0.532 | 0.428 | +0.026 |
| DeepSeek 6.7B | zero-shot | 0.551 | 0.138 | 0.220 | 0.162 | 0.508 | 0.126 | +0.009 |
| DeepSeek 6.7B | few-shot k=1 | 0.507 | 0.671 | 0.578 | 0.631 | 0.505 | 0.668 | −0.001 |
| DeepSeek 6.7B | few-shot k=3 | 0.499 | 0.537 | 0.517 | 0.529 | 0.494 | 0.543 | −0.020 |
| DeepSeek 6.7B | CoT | 0.474 | 0.180 | 0.261 | 0.205 | 0.485 | 0.192 | +0.007 |
| DeepSeek 6.7B | SFT | 0.507 | 0.643 | 0.567 | 0.610 | 0.504 | 0.641 | +0.018 |
| DeepSeek 6.7B | skill v9 | 0.507 | 0.413 | 0.455 | 0.429 | 0.501 | 0.411 | −0.002 |
| DeepSeek 6.7B | skill v6 | 0.495 | 0.301 | 0.374 | 0.326 | 0.492 | 0.306 | −0.011 |
| Llama 3 8B | zero-shot | 0.490 | 0.270 | 0.348 | 0.296 | 0.489 | 0.278 | −0.014 |
| Llama 3 8B | few-shot k=1 | 0.519 | 0.191 | 0.279 | 0.219 | 0.502 | 0.186 | −0.005 |
| Llama 3 8B | few-shot k=3 | 0.517 | 0.382 | 0.439 | 0.403 | 0.508 | 0.373 | −0.014 |
| Llama 3 8B | CoT | 0.507 | 0.315 | 0.388 | 0.340 | 0.499 | 0.313 | −0.028 |
| Llama 3 8B | SFT | 0.528 | 0.317 | 0.397 | 0.345 | 0.512 | 0.303 | +0.020 |
| Llama 3 8B | skill v9 | 0.507 | 0.430 | 0.465 | 0.443 | 0.501 | 0.428 | −0.020 |
| Llama 3 8B | skill v6 | 0.502 | 0.444 | 0.471 | 0.454 | 0.497 | 0.447 | −0.010 |
| Qwen2.5 7B | zero-shot | — | 0.000 | — | — | 0.495 | 0.000 | — |
| Qwen2.5 7B | few-shot k=1 | 0.478 | 0.031 | 0.058 | 0.038 | 0.494 | 0.033 | −0.007 |
| Qwen2.5 7B | few-shot k=3 | 0.493 | 0.101 | 0.168 | 0.120 | 0.494 | 0.104 | −0.004 |
| Qwen2.5 7B | CoT | 0.600 | 0.008 | 0.017 | 0.011 | 0.497 | 0.007 | −0.004 |
| Qwen2.5 7B | SFT | 0.504 | 0.376 | 0.431 | 0.396 | 0.498 | 0.377 | +0.012 |
| Qwen2.5 7B | skill v9 | 0.429 | 0.025 | 0.048 | 0.031 | 0.491 | 0.030 | −0.011 |
| Qwen2.5 7B | skill v6 | 0.385 | 0.014 | 0.027 | 0.017 | 0.491 | 0.018 | −0.011 |

(Qwen zero-shot makes zero positive predictions on this subset too — P is
undefined, as in the full set.)

**Magnitude across both variants and all three thresholds** (all 28×3×2 = 168
filtered cells): max |ΔF2| = 0.030, mean |ΔF2| ≈ 0.010–0.012. Signs are
mixed within every model (some configs go up, some down after filtering) —
there is no systematic direction consistent with "near-duplicates are
propping up the reported numbers."

---

## 4. Part 1 — Example near-duplicate pairs

Full CSV: `results/nd_example_pairs.csv`. Top 5 by Jaccard:

| MVFSC idx | Project | Label | Corpus split | Corpus project | Corpus label | Jaccard |
|---|---|--:|---|---|--:|--:|
| 405 | Qt | vulnerable | train | `1ca02cf2...` (commit-hash project label, MegaVul/CVEfixes source) | vulnerable | 1.000 |
| 702 | tensorflow | non-vulnerable | train | `7e5f9fbf...` (commit-hash project label) | vulnerable | 1.000 |
| 209 | libarchive | vulnerable | train | libarchive | vulnerable | 0.997 |
| 210 | libarchive | non-vulnerable | train | libarchive | vulnerable | 0.997 |
| 712 | tensorflow | non-vulnerable | train | `67784700...` (commit-hash project label, non-vulnerable) | non-vulnerable | 0.997 |

The top two (idx 405, 702; see §2) are whitespace-only near-duplicates that
slipped past exact-SHA1 dedup. Idx 209/210 are the MVFSC vuln/patched pair
both independently matching the *same* train-split vulnerable function at
Jaccard ≈ 1.0 — i.e. the train corpus independently contains a near-copy of
the libarchive vulnerability (same-label leakage for idx 209, opposite-label
for idx 210, since 210 is MVFSC's own patched/non-vulnerable half).

---

## 5. Part 2 — Paired significance tests

60 within-model pairwise comparisons across the 6 headline configs
(zero-shot, few-shot k=1, few-shot k=3, CoT, SFT, skill v9), plus 4
skill-v9-vs-v6 comparisons. McNemar exact test on per-sample correctness
(`b`, `c`, two-sided exact binomial p), Holm–Bonferroni-adjusted within each
15-test model family, and paired bootstrap 95% CIs (B=2000, seed 42,
percentile method, joint resampling of the 872 indices) for the F2
difference and the precision difference. Full tables:
`results/pt_pairwise_60.csv`, `results/pt_v9_vs_v6.csv`.

**Headline comparisons the thesis currently cites** (`results/pt_highlight.json`):

| Comparison | Unpaired F2 Δ (thesis) | Unpaired 95% CI (thesis) | Paired F2 Δ | Paired 95% CI | McNemar b, c | McNemar p | Holm p |
|---|--:|---|--:|---|---|--:|--:|
| Llama 3 8B, skill v9 − few-shot k=3 | +0.046 | [−0.013, 0.110] | +0.046 | **[0.006, 0.088]** | 98, 96 | 0.943 | 1.0 |
| Llama 3 8B, skill v9 − SFT | +0.139 | [0.079, 0.202] | +0.139 | [0.075, 0.206] | 210, 202 | 0.730 | 1.0 |

The point estimates match the thesis's unpaired numbers almost exactly
(pairing doesn't change the point estimate — it changes the CI). For
**skill v9 vs. few-shot k=3**, pairing *narrows* the interval enough that it
now **excludes zero** — a comparison the thesis currently reports as not
distinguishable from zero (unpaired CI crossed zero) is, under the paired
bootstrap, a statistically resolved advantage for skill v9. The v9-vs-SFT
comparison was already significant unpaired and remains so paired, with a
very similar interval.

**Across all 60 comparisons**: 50/60 paired F2-difference 95% CIs exclude
zero, but **0/60 are significant by Holm-adjusted McNemar** (even before
Holm adjustment, the smallest raw McNemar p was ≈0.26). This is not a
contradiction: F2 is recall-weighted and aggregates a specific mix of
TP/FP/FN, so two configs can differ substantially in F2 while each is
"correct" about the same *number* of samples overall (McNemar's `b` and `c`
are close — e.g. 124/124, 197/203 — because one config trades recall for
precision rather than being uniformly more accurate). Treat the paired F2
CIs as the primary evidence for "which strategy is better on this metric,"
and McNemar as answering the narrower question "is one strategy more often
simply right," which these results show is a weaker, less distinguishing
test at this sample size for configs with similar accuracy but different
precision/recall balance.

**v9 vs. v6 selection tie** (`results/pt_v9_vs_v6.csv`):

| Model | F2 v9 | F2 v6 | Paired Δ | 95% CI | McNemar b, c | p |
|---|--:|--:|--:|---|---|--:|
| Llama 3 8B | 0.463 | 0.464 | −0.001 | [−0.027, 0.025] | 45, 41 | 0.746 |
| Qwen2.5 7B | 0.042 | 0.028 | +0.014 | [−0.000, 0.029] | 6, 6 | 1.000 |
| DeepSeek-Coder 6.7B | 0.431 | 0.337 | **+0.094** | **[0.060, 0.128]** | 73, 62 | 0.390 |
| Code Llama 7B | 0.433 | 0.449 | −0.016 | [−0.057, 0.027] | 90, 103 | 0.388 |

Three of four models show no resolvable difference between v9 and v6 — the
thesis's "tied on validation" framing holds up under pairing for Llama 3,
Qwen2.5, and Code Llama. **DeepSeek-Coder is the exception**: its paired F2
CI for v9 − v6 is entirely positive and excludes zero, i.e. v9 is a real,
paired-significant improvement over v6 for this model specifically, even
though the two versions were selected as tied on the *validation* average
across models.

---

## 6. Findings (plain statement)

1. **Near-duplicates exist beyond exact SHA-1 matches.** 21.2% of MVFSC's
   872 functions have a token-3-gram Jaccard ≥ 0.7 match in train/val (19.2%
   at ≥0.8, 14.8% at ≥0.9). Two of these are near-*exact* duplicates
   (Jaccard effectively 1.0) that differ from a train-split function by a
   single whitespace character and were missed by the thesis's exact-hash
   dedup — a genuine, if small, leakage gap worth noting in the thesis text.
2. **Removing them does not materially change any reported metric.** Across
   all 28 configurations, 3 thresholds, and both removal variants (168
   filtered cells total), the largest F2 shift was 0.030 and the average was
   about 0.011, with no systematic direction — some configs' F2 rises after
   filtering, some fall, roughly in balance. The thesis's headline numbers
   are not propped up by near-duplicate leakage.
3. **Paired testing changes the significance picture for one of the two
   comparisons the thesis currently cites.** Llama 3 8B's skill v9 vs.
   few-shot k=3 advantage (+0.046 F2), reported as not statistically
   resolved under the unpaired bootstrap (CI crossed zero), **is** resolved
   under the paired bootstrap (CI [0.006, 0.088]) — because pairing removes
   the between-sample noise that two independent bootstraps both pay for
   separately. The skill v9 vs. SFT comparison (+0.139 F2) was already
   significant and remains so.
4. **McNemar's test, by contrast, finds no significant pairwise differences
   anywhere** (0/60, even pre-Holm). This is a metric-sensitivity artifact,
   not a contradiction: F2 is recall-weighted and two strategies can have
   very different recall/precision trade-offs while being "correct" on
   essentially the same *count* of samples. Report paired bootstrap F2 CIs
   as the primary significance evidence; report McNemar as a secondary,
   stricter check that this dataset size cannot resolve for most
   comparisons.
5. **The v9/v6 skill-selection tie holds for 3 of 4 models but not
   DeepSeek-Coder**, where v9 is a paired-significant improvement over v6
   (+0.094 F2, CI [0.060, 0.128]). Worth a sentence in §4.6 since the
   current text frames the tie as holding across the board.

---

## 7. Runtime and package versions

- `near_duplicates.py`: ~3m 5s wall clock (6-core machine, Linux WSL2).
- `paired_tests.py`: ~4s.
- `validate_gate.py`: ~1s.
- Python 3.13.12, Linux-6.6.87.2-microsoft-standard-WSL2-x86_64.
- pandas 3.0.2, numpy 2.4.4, scipy 1.17.1, matplotlib 3.10.8,
  datasets 4.8.4, datasketch 2.0.0.
