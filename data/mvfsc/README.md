# MVFSC test set

This directory contains the **MIVIA Vulnerable Function Source Code (MVFSC)** test
set, the independent evaluation benchmark used for all evaluation in this thesis.
MVFSC was built by its own authors to be project-disjoint from *their* training
data; relative to this thesis's training corpus, 18 of its 31 distinct projects
(by name) are also present in the training/validation split, so here it is a
snippet-deduplicated (exact SHA-1) benchmark rather than a project-disjoint one.

## Source

The benchmark was introduced by:

> Vincenzo Carletti, Pasquale Foggia, Carlo Mazzocca, Giuseppe Parrella and
> Mario Vento. *Evaluating Large Language Models for Vulnerability Detection
> Under Realistic Conditions.* In *Availability, Reliability and Security
> (ARES 2025)*, Lecture Notes in Computer Science, vol. 15999, pp. 135–152.
> Springer Nature Switzerland, 2025. DOI: 10.1007/978-3-032-00644-8_8.

The dataset is composed exclusively of functions extracted from real-world,
production-level open-source C/C++ projects. Each function is labelled vulnerable
or neutral and is mapped to a high-level CWE class. The authors built it by
merging and normalising four public corpora (Big-Vul, DiverseVul, Devign,
ReVeal) and removing duplicates, so that the resulting functions come from
projects disjoint from the usual training distributions.

## Files

| File | Description |
|------|-------------|
| `mvfsc_test_set.xlsx` | The upstream benchmark, verbatim as distributed by the authors. 972 functions (487 vulnerable / 485 neutral) across 34 project labels. Columns: `index`, `hash`, `vul`, `cwe`, `project`, `reference`, `code`. SHA-256 `b12cde118202b31a65b25cd835d4cc2f205238c66aa65c0a4030088220362593`. |
| `mvfsc_test_set_processed.csv` | The exact evaluation set consumed by this project: 872 functions (439 vulnerable / 433 neutral) after SHA-1 deduplication of identical snippets. Columns: `code_sha1`, `vulnerable` (1/0), `cwe_id`, `project`, `code`. |

## Provenance and how the processed set was derived

The upstream file `mvfsc_test_set.xlsx` was obtained on 9 April 2026 from the authors'
companion repository, distributed during double-blind review through the
[Anonymous GitHub](https://anonymous.4open.science) proxy
(`anonymous.4open.science/r/code_analysis-cftb`). That proxy is ephemeral; the
verbatim copy committed here, together with the SHA-256 above, is the stable
record of exactly what was used.

The 872-row evaluation set is produced from the upstream functions by SHA-1
deduplication of code snippets, implemented in
[`src/data_utils.py`](../../src/data_utils.py) (`load_mivia_test`). Because the
intermediate re-pairing step ran outside this repository, the processed CSV is
provided directly so the evaluation set can be reproduced without that step:
every result in the thesis is computed on `mvfsc_test_set_processed.csv`.

## Licence

The dataset is redistributed under the **GNU General Public License v3.0**, the
licence under which the authors released it. The full text is in
[`LICENSE`](LICENSE). This licence applies to the dataset files in this
directory only, not to the source code in the rest of the repository. Please
cite the ARES 2025 paper above if you use the data.
