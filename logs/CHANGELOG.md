## Combined-method experiments — 2026-06-02

### New experiment: stacking few-shot + fine-tuning + skill (supervisor request)

Nikolaos asked to measure how much models improve when methods are combined. Three combos were
implemented and run on the MVFSC test set, each at k=1 and k=3:
1. **skill + few-shot (base models)** — clean prompting stack, no SFT confound (the headline).
2. **SFT + few-shot** — few-shot prompt over the existing fine-tuned adapters.
3. **triple = SFT + skill + few-shot** — the supervisor's literal ask (exploratory; inherits the
   SFT label-degeneracy + train/inference format-mismatch confounds).

Skill v9 is used for all skill combos. SFT combos reuse the existing adapters (no retraining).

**Implementation**: new `build_skill_few_shot_prompt()` and `build_sft_messages(..., user_prompt=)`
in `src/prompt_template.py`; `skill_few_shot` dispatch + `VALID_STRATEGIES` entry;
`docker/run_job_base_eval.py` gains optional `SKILL_GCS_PATH`; `docker/run_job_sft.py` gains
`SFT_STRATEGY`/`FEW_SHOT_K`/`SKILL_GCS_PATH` (eval-only, `MAX_SEQ_LENGTH=8192`);
`scripts/submit_vertexai_combo.py` submits all 24 jobs (`--combos`, `--few-shot-ks`, `--dry-run`,
`--limit`); `aggregate_results.py` dedup key now includes `approach`, with a new `--phase combo`.
Both entrypoints share one image (`docker-image/base-eval:latest`, rebuilt+pushed 2026-06-02).

**First result** — llama3-8b skill+few-shot **k=1**, test set:
| Metric | Value |
|--------|-------|
| F2 | 0.5468 |
| Precision | 0.5137 |
| Recall | 0.5558 |
| F1 | 0.5339 |
| Parse failures | 0/872 |

This beats Llama3's prior best across all phases (Phase 1 / Phase 3 ≈ 0.464). Precision stays in the
0.50–0.53 band — the precision ceiling is not broken. Remaining base jobs (qwen/deepseek/codellama
k=1,k=3 + llama3 k=3 resubmit after an L4-allocation failure) were PENDING at session close.
SFT+few-shot and triple combos not yet submitted.

### Full matrix retrieved — 2026-06-03 (21/24 cells)

All three combos submitted for all 4 models × k=1,k=3 (24 cells). 21 retrieved; 3 SFT-adapter k=1
cells repeatedly fail to land (no partial artifacts → dying early, likely L4 allocation): sft_few_shot
deepseek k1, triple llama3 k1, triple deepseek k1.

Test-set F2 by cell (— = in flight):

| Combo | llama3 k1 | llama3 k3 | qwen k1 | qwen k3 | deepseek k1 | deepseek k3 | codellama k1 | codellama k3 |
|-------|-----------|-----------|---------|---------|-------------|-------------|--------------|--------------|
| skill+few-shot (base) | 0.547 | 0.550 | 0.080 | 0.311 | 0.433 | 0.323 | 0.114 (PF=774) | 0.050 (PF=835) |
| SFT+few-shot | 0.000 | 0.003 | 0.050 | 0.371 | — | 0.635 | 0.643 | 0.322 |
| triple (SFT+skill+few-shot) | — | 0.167 | 0.117 | 0.348 | — | 0.668 | **0.754** | 0.155 |

Headline: triple codellama k1 = **0.754** (R=0.863), matrix peak but confounded (SFT label degeneracy +
train/inference format mismatch). Cleanest strong result: llama3 base skill+few-shot k3 = 0.550. All
llama3 SFT-based cells are near-zero (label-degeneracy confound, NOT job failures). Codellama base
skill+few-shot is wrecked by parse failures. Precision sits 0.48–0.53 everywhere a model actually
predicts — no stack breaks the precision ceiling. A session-only polling loop (CronCreate, every 3h)
syncs GCS, aggregates, and resubmits cells still missing after a full cycle. Aggregate at
`results/combo/aggregate_summary.json`.

---

## Thesis update — 2026-05-26

### Phase 3 llama3-8b v6 test-set result added

- llama3-8b v6 retrieved: F2=0.464 (P=0.503, R=0.456, F1=0.478, PPR=45.6%, PF=0/872).
- v6 marginally outperforms v9 for llama3 (0.464 vs 0.463); thesis updated to report v6 as the best version for that model.
- All v6 test-set results are now complete. Still missing: deepseek-coder-6.7b v9 and codellama-7b v9.
- Thesis locations updated: Ch.4 table row (llama3), Ch.4 intro paragraph and caption, Ch.5 lines 61/63/73/118/124/128.

---

## v11 → v12 — 2026-05-17

### Changes
- Stripped the format header back to v6's exact 3-line minimal form. Removed: the "The first character you write must be `{`." constraint added at v9, and the tail reminder ("Remember: your response must be a single JSON object...") added at v9. Neither of these additions recovered codellama's parse failures from v6's baseline of 34: v9 produced PF=39, v10 PF=47, v11 PF=49. The trajectory is monotonically worse with each format-header addition. Returning to the exact 3-line v6 header (which produced the best-ever PF=34) is the only remaining intervention that has demonstrated success at the format level. The v11 prohibitions ("Do not begin with 'The'", "Do not wrap in backticks") are also removed — both hurt their target models.
- Added a partial-JSON completion prompt as a 4th line in the format block: `Start your response now: {"vulnerable":`. This is the one format intervention not yet tried across v1–v11. It works via a different mechanism than all prior attempts: rather than giving the model a rule or constraint to follow, it presents the literal first tokens of a valid response in-context, letting the model continue the sequence. This is a positive token-level template — the model reads it as text it is continuing, not as an instruction it may ignore. The partial JSON is placed after the schema/safe-verdict examples so the model has seen the full output format before encountering the completion prompt. This is structurally distinct from: (a) declarative rules ("Output one JSON object"), (b) imperative constraints ("must start with `{`"), (c) character-level anchors ("The first character you write must be `{`"), (d) first-person identity statements ("Your entire response is one JSON object:"), (e) negative prohibitions ("Do not begin with 'The'"). None of those mechanisms have solved the generation-momentum problem. A literal partial completion has not been attempted.
- Retained all v9 reasoning improvements: the no-space memcpy ban covering both `memcpy(buf, src, len)` and `memcpy(buf,src,len)`, CWE-20 in the Logic/authorization Step 3 bullet with explicit "Is input accepted without validation of type, range, or format?" check, the strengthened safe-verdict gate requiring "classify as vulnerable" when operations or guards cannot be named, and the full Bounds three-bullet block (off-by-one, multi-step offset, size from input) and inline CWE references in body text.
- Explicitly excluded: the Bounds re-check failsafe sub-bullet from v10 (hurt deepseek F2 0.502→0.382), the first-person completion anchor from v10 (hurt codellama PF 39→47), the negative prose-start prohibition from v11 (hurt codellama PF 39→49), the code-fence/tag wrapping prohibition from v11 (hurt deepseek F2 0.502→0.392).

### Rationale
The codellama parse-failure trajectory across all format interventions since v6:
- v6: PF=34 — 3-line minimal header, JSON-first placement
- v7: PF=43 — added CWE IDs in bold label headers (regression)
- v8: PF=44 — moved CWE IDs to body text (no recovery)
- v9: PF=39 — added first-character constraint + tail reminder (partial recovery but still worse than v6)
- v10: PF=47 — first-person anchor (worse)
- v11: PF=49 — negative prohibition (worst since v7)

The evidence is unambiguous: every addition to the format header since v6 has either not helped or made things worse. The only way to test a genuinely new mechanism without adding length to the format-instruction block is to replace the v9/v11 additions with something structurally different. The partial-JSON completion prompt replaces all removed lines (net length change is neutral) and is the one token-level positive-template mechanism not yet tried.

For deepseek, v9's strong performance (F2=0.502, PF=3) came from two changes: the JSON-first format (already in v6) and the no-space memcpy ban. Neither of those is removed in v12. The regressions at v10 and v11 were caused by additions that suppressed Step 2 naming (Bounds re-check failsafe) and disrupted generation (fence/tag prohibition). v12 removes both of those and keeps the precision rule exactly as it was at v9.

For llama3, improvements have been gradual (v6: 0.592 → v9: 0.606 → v11: 0.644) and appear robust to format changes. No reasoning step changes are made that could disturb llama3's positive trajectory.

### Per-Model Metrics (v12 results — partial, 2026-05-18)
| Model | Precision | Recall | F1 | F2 | Parse Failures |
|-------|-----------|--------|----|-----|----------------|
| llama3-8b | — | — | — | — | pending |
| qwen2.5-7b | 0.6667 | 0.0962 | 0.1667 | 0.1163 | 1 |
| deepseek-coder-6.7b | — | — | — | — | pending |
| codellama-7b | — | — | — | — | pending |

Partial avg F2 (1 model): 0.116. Remaining 3 models submitted 2026-05-18.

### Per-Model Metrics (v9 baseline — best confirmed)
| Model | Precision | Recall | F1 | F2 | Parse Failures |
|-------|-----------|--------|----|-----|----------------|
| llama3-8b | 0.5714 | 0.6154 | 0.5926 | 0.6061 | 1 |
| qwen2.5-7b | 0.6250 | 0.0962 | 0.1667 | 0.1157 | 2 |
| deepseek-coder-6.7b | 0.5098 | 0.5000 | 0.5049 | 0.5019 | 3 |
| codellama-7b | 0.4746 | 0.5385 | 0.5045 | 0.5243 | 39 |

---

## v10 → v11 — 2026-05-12

### Changes
- Added a negative prose-start prohibition to the format block: "Do not begin your response with the word 'The' or any word other than '{'." Codellama's parse failures (39 at v9, 47 at v10) follow a fixed template that always starts with "The provided code is vulnerable" or "The provided code is not vulnerable." Every prior format intervention has been positive ("must start with `{`") or generic ("no prose before the JSON"). A negative instruction that names the exact failure token is structurally distinct from all prior attempts: it targets the specific word that initiates the generation-momentum cascade rather than describing the desired output state. This is the one type of prose-start prohibition not yet tried across v1–v10.
- Added a code-fence and tag wrapping prohibition to the format block: "Do not wrap your response in backticks, code fences (```), or any XML/HTML tags." Deepseek wraps its output in ` ```json `, ` ``` `, `<code>`, or `</skill>` tags — a pattern observed at v6–v9. The parser can extract JSON from some forms of wrapping, but deepseek's parse failures rose from 2 (v8) to 3 (v9), indicating growing instability. The `</skill>` tag wrapping indicates skill-content echo; the backtick wrapping causes extraction failures when the JSON spans multiple fence blocks. A direct prohibition against these specific wrappers has never been included in any skill version.
- Retained all v9 changes: "The first character you write must be `{`" first-character constraint on line 1, tail reminder at the last line, and the extended memcpy ban covering both `memcpy(buf, src, len)` (spaced) and `memcpy(buf,src,len)` (no-space) forms.
- Based on v9 (best avg F2=0.437). No structural changes: Step 1–4 format, three-bullet Bounds block, logic/authorization bullet with CWE-20, safe-verdict gate with "classify as vulnerable" consequence, and precision rule are all retained from v9 unchanged.
- Explicitly excluded from v11: the Bounds re-check failsafe sub-bullet from v10 (hurt deepseek, F2 0.502→0.382) and the first-person completion anchor from v10 (hurt codellama, PF 39→47). Neither change is included.

### Rationale
v10 confirmed two failures: the first-person completion anchor made codellama worse (PF 39→47), and the Bounds re-check failsafe caused deepseek to produce 33 FNs instead of 26. Both changes are discarded.

The codellama prose-start problem has been approached from five angles across v6–v10: (1) moving the format instruction to line 1 (v6 — worked partially, PF fell to 34), (2) adding character-level constraint "first character must be `{`" (v9 — reduced PF to 39), (3) adding tail reminder (v9 dual-placement), (4) moving CWE IDs out of headers (v8 — no improvement), (5) first-person anchor (v10 — regression). The one angle not yet tried is a negative instruction naming the exact failure token. The failure always starts with "The" — banning that word as the first token of output is qualitatively different from all prior instructions, which have described what the model should do rather than naming what it must not start with.

For deepseek, the code-fence prohibition targets a persistent pattern that has never been directly addressed. The `</skill>` tag output indicates the model is echoing skill-wrapper context; banning XML/HTML tags closes this specific echo path. Backtick prohibition eliminates the fence-extraction failures. Both prohibitions fit within the existing format block and do not affect the reasoning steps.

### Per-Model Metrics (v9 baseline — best confirmed)
| Model | Precision | Recall | F1 | F2 | Parse Failures |
|-------|-----------|--------|----|-----|----------------|
| llama3-8b | 0.5714 | 0.6154 | 0.5926 | 0.6061 | 1 |
| qwen2.5-7b | 0.6250 | 0.0962 | 0.1667 | 0.1157 | 2 |
| deepseek-coder-6.7b | 0.5098 | 0.5000 | 0.5049 | 0.5019 | 3 |
| codellama-7b | 0.4746 | 0.5385 | 0.5045 | 0.5243 | 39 |

### Per-Model Metrics (v11 results — 2026-05-13)
| Model | Precision | Recall | F1 | F2 | Parse Failures |
|-------|-----------|--------|----|-----|----------------|
| llama3-8b | 0.6071 | 0.6538 | 0.6296 | 0.6439 | 1 |
| qwen2.5-7b | 0.6667 | 0.1154 | 0.1967 | 0.1382 | 2 |
| deepseek-coder-6.7b | 0.4255 | 0.3846 | 0.4040 | 0.3922 | 3 |
| codellama-7b | 0.4792 | 0.4423 | 0.4600 | 0.4492 | 49 |

**Avg F2 (all 4 models): 0.4059** — regressed from v9 (0.437). v11 is not a new best.

Observations: codellama PF rose to 49 (from 39 at v9) — the "Do not begin with 'The'" negative prohibition did not fix the generation-momentum problem. Deepseek regressed (F2 0.502→0.392) — the code-fence/tag-wrapping prohibition disrupted generation. Llama3 improved to F2=0.644 (best across all versions). Qwen slightly below v9 (0.138 vs 0.116). Best avg remains jointly v6 and v9 at 0.437.

---

## v9 → v10 — 2026-05-10

### Changes
- Changed the JSON format opening line from a declarative instruction ("Output one JSON object and nothing else.") to a first-person completion anchor ("Your entire response is one JSON object:"). The hypothesis is that "Your entire response is" functions as a partial statement the model completes by generating JSON, rather than a rule it may choose to follow or ignore. Every prior format intervention has used declarative or imperative forms ("Output...", "The response must start with...", "The first character you write must be..."). None have broken codellama's generation-momentum problem at v7/v8. A first-person framing is structurally different: the model reads it as an identity statement about its own output, not as an external constraint. This change is at line 1, preserving the JSON-first structural requirement confirmed at v6. The tail reminder from v9 ("Your entire response is one JSON object starting with `{`...") is retained at the last line to maintain dual-placement anchoring.
- Added a Bounds re-check failsafe sub-bullet: "If you name any memcpy, memset, strcpy, strncpy, or array index expression in Step 2 and then conclude 'safe' for Bounds, re-examine: state the exact constant or checked variable that limits the size or index. If you cannot name it, classify as vulnerable." This targets deepseek's dominant FN pattern (CWE-125: 5 FNs at v8; CWE-119: 2 FNs) where the model names the dangerous operation in Step 2 but then returns "safe" without naming the bound check. The re-check instruction creates a forced verification loop: if the model has already performed Step 2 correctly, this sub-bullet forces it to either name the guard or flip to vulnerable.
- Retained all v9 changes: "The first character you write must be `{`" constraint on line 1, the tail reminder, and the extended memcpy ban covering both `memcpy(buf, src, len)` (spaced) and `memcpy(buf,src,len)` (no-space) forms. v9 is not yet evaluated, so its changes are carried forward unchanged.
- Based on v6 (best avg F2=0.437). No structural changes: Step 1–4 format, inline schema, safe-verdict gate, three-bullet Bounds block, logic/authorization bullet, and precision rule are all retained from v6/v9.

### Rationale
v7, v8, and v9 all failed to recover codellama's parse failures from v6's 34 toward the target range. The confirmed root cause (generation-momentum: model starts with prose before JSON) has now been approached three ways: moving CWE IDs out of headers (v8), adding explicit character-level constraint (v9 line 1), and adding a tail reminder (v9 last line). None worked. The first-person completion anchor is the last structurally distinct intervention available within the format-instruction space. It is qualitatively different from all prior attempts because it frames the model's own identity as a JSON producer rather than giving it an output rule.

For deepseek, the FP-side improvement from v9's no-space memcpy ban addresses hallucinated `vulnerable_code` on FP samples. The FN-side problem (31 FNs, CWE-125 and CWE-20 dominating) has not been directly addressed since v7. The Bounds re-check sub-bullet directly targets the specific pattern where deepseek names `memcpy` or array access in Step 2 but then produces a safe verdict without stating what the bound check is. If the model cannot name the guard, the instruction forces it to classify as vulnerable — this is the same mechanism as the safe-verdict gate but applied specifically to the Bounds check.

v9 metrics are pending. v10 is written against the v8 baseline (the best available confirmed data point), carrying forward all v9 changes.

### Per-Model Metrics (v9 baseline — confirmed 2026-05-09)
| Model | Precision | Recall | F1 | F2 | Parse Failures |
|-------|-----------|--------|----|-----|----------------|
| llama3-8b | 0.5714 | 0.6154 | 0.5926 | 0.6061 | 1 |
| qwen2.5-7b | 0.7143 | 0.0962 | 0.1695 | 0.1163 | 1 |
| deepseek-coder-6.7b | 0.5000 | 0.5096 | 0.5048 | 0.5024 | 2 |
| codellama-7b | 0.5000 | 0.5000 | 0.5000 | 0.5238 | 39 |

**Avg F2 (all 4 models): 0.437** — tied with v6 as best across all versions.

---

## v8 → v9 — 2026-05-09

### Changes
- Added explicit first-character constraint: the JSON-first line now reads "The first character you write must be `{`." This is a direct addition to the existing "must start with `{` and end with `}`" instruction. At v8, codellama produced 44 parse failures — essentially unchanged from v7's 43 — despite the JSON-first format being at line 1. Inspecting v8 codellama predictions shows the model ignores the format instruction and generates prose analysis ("The provided code is vulnerable to a buffer overflow attack...") before any JSON. Adding the explicit first-character constraint gives the model a more concrete anchoring instruction at the start of its generation.
- Added a reminder line at the END of the skill: "Remember: your response must be a single JSON object starting with `{` and ending with `}` — no prose before or after." This places the JSON format requirement at the transition point where the model stops reading instructions and begins generating output. The existing JSON-first line at line 1 establishes the frame, and this tail reminder reinforces it immediately before generation begins. This dual-placement strategy (top + bottom of skill) has not been tried in any prior version.
- Extended the `memcpy` template ban: the precision rule previously banned `` `memcpy(buf, src, len)` `` (with spaces). Deepseek v8 predictions show the model consistently outputs the compact variant `` `memcpy(buf,src,len)` `` (no spaces) on false-positive samples — technically different characters, so the existing ban did not apply. The ban now explicitly covers both forms. This targets deepseek's hallucinated FP template that generates CWE-787 verdicts on code that contains no such call.
- Based on v6 (avg F2=0.4365, best across all versions). v8 confirmed: llama3 F2=0.606, qwen F2=0.116, deepseek F2=0.417, codellama F2=0.492 (avg 0.408). v8 regressed from v6 — codellama parse failures did not recover as expected (44 at v8 vs 34 at v6; CWE IDs moved to body text did not fix the issue). All other v6 content (three-bullet Bounds block, logic/authorization bullet, CWE-20 in body, tighter safe-verdict gate with "classify as vulnerable" consequence) is retained from v8.
- No structural changes. The Step 1–4 format, JSON schema line, safe-verdict gate, and all checklist bullets are identical to v6/v8. Only the JSON format reinforcement lines and the memcpy ban extension are new.

### Rationale
v8 was expected to recover codellama's parse failures from 43 (v7) toward 34 (v6) by removing the header-like CWE parenthetical from bold bullet labels. Instead, parse failures remained at 44 — suggesting the root cause of codellama's prose output is not the bullet label format but the model's intrinsic tendency to generate analysis prose before JSON. The evidence is clear: codellama outputs "The provided code is vulnerable to a buffer overflow attack. The `memcpy` function is used..." on every sample that fails to parse. This is a generation-momentum problem: once the model starts with prose, it cannot self-correct. The dual-placement JSON reinforcement (first line + last line) is the next logical intervention — the last line is encountered just before the model generates output, making it the most temporally proximate instruction.

For deepseek, the v8 predictions show the model parsing correctly (2 failures) but hallucinating the template `memcpy(buf,src,len)` in the `vulnerable_code` field on FP samples. The precision rule banning the spaced form did not cover this variant. Extending the ban closes this loophole and should reduce deepseek's FP count (23 at v8, fp_rate=0.479).

qwen remains at a structural capability ceiling (F2=0.116 unchanged from v4). No additional instruction targeted at qwen — focus is on codellama and deepseek improvements.

### Per-Model Metrics (v8 baseline)
| Model | Precision | Recall | F1 | F2 | Parse Failures |
|-------|-----------|--------|----|-----|----------------|
| llama3-8b | 0.5714 | 0.6154 | 0.5926 | 0.6061 | 1 |
| qwen2.5-7b | 0.7143 | 0.0962 | 0.1695 | 0.1163 | 1 |
| deepseek-coder-6.7b | 0.4773 | 0.4038 | 0.4375 | 0.4167 | 2 |
| codellama-7b | 0.4643 | 0.5000 | 0.4815 | 0.4924 | 44 |

---

## v7 → v8 — 2026-05-08

### Changes
- Moved CWE identifiers out of the bold bullet label and into the body text of each Step 3 check. In v7, each bullet header read `**Bounds** (CWE-119, CWE-125, CWE-787):` — the parenthetical CWE IDs immediately after the bold label made the line read like a section heading to codellama, which echoed it as prose and drove parse failures from 34 (v6) to 43 (v7). In v8, the header remains `**Bounds**:` and the CWE coverage note appears as the second sentence of the check body: "Covers CWE-119, CWE-125, CWE-787." Same information, structurally less header-like.
- Based on v6 (avg F2=0.4365), not v7 (avg F2=0.4315). v7 regressed on average despite minor gains for llama3 (0.592→0.667) and deepseek (0.379→0.394), because codellama dropped from F2=0.659 to F2=0.549 and parse failures rose from 34 to 43.
- Preserved the CWE-20 addition to the Logic/authorization bullet from v7 (deepseek had 5 CWE-20 FNs at v6; this bullet addition helped marginally at v7). In v8, CWE-20 remains in the logic/authorization body text.
- Preserved the safe-verdict gate language from v7: "Generic statements ... are not acceptable — if you cannot name the specific operations and guards, re-examine the code and classify as vulnerable." v6 ended with "re-examine the code" without the explicit "classify as vulnerable" consequence; v7 added it and llama3 improved from 21 FNs to 16 FNs.
- No other changes. The JSON-first format block (lines 1–3), numbered Step 1–4 structure, three-bullet memory hiding spots block, detection-bias statement, and precision rule are all retained from v6 verbatim.

### Rationale
v7's only structural change from v6 was adding CWE IDs to the bold bullet label headers. This caused codellama parse failures to rise from 34 to 43, a regression of 9 additional failures. At roughly 50% base rate, each parse failure is an expected 0.5 lost TP — 9 extra failures translates to roughly 4–5 lost TPs. Codellama's F2 dropped from 0.659 to 0.549, more than erasing the gains llama3 and deepseek made. The root cause is confirmed by the v4→v5 pattern: structured headers with named labels trigger codellama to echo those headers as prose and abandon JSON output.

The fix is minimal: keep the CWE information but change its position from the label suffix to an inline sentence in the check body. "Covers CWE-119, CWE-125, CWE-787." reads as body text, not as a heading, and is unlikely to be echoed. The useful content (CWE labeling, CWE-20 addition, tightened safe-verdict gate) is preserved; only the header-style placement is removed.

### Per-Model Metrics (v7 baseline)
| Model | Precision | Recall | F1 | F2 | Parse Failures |
|-------|-----------|--------|----|-----|----------------|
| llama3-8b | 0.581 | 0.692 | 0.632 | 0.667 | 1 |
| qwen2.5-7b | 0.714 | 0.096 | 0.170 | 0.116 | 1 |
| deepseek-coder-6.7b | 0.435 | 0.385 | 0.408 | 0.394 | 2 |
| codellama-7b | 0.518 | 0.558 | 0.537 | 0.549 | 43 |

---

## v6 → v7 — 2026-05-01

### Changes
- Consolidated CWE identifiers directly into Step 3 bullet labels: each check now carries the relevant CWE IDs inline (e.g., "Bounds (CWE-119, CWE-125, CWE-787)", "Logic/authorization (CWE-264, CWE-284, CWE-617, CWE-400, CWE-20)"). This directly targets the two remaining high-FN clusters: llama3's logic/authorization misses (CWE-264, CWE-617, CWE-284, CWE-367 all showed 0% recall at v6) and qwen/deepseek's input-validation misses (CWE-20 had 5 FNs for deepseek and 4 for qwen at v6). Adding CWE-20 explicitly to the Logic/authorization bullet addresses the "missing input validation" class that was not named there in v5/v6.
- Strengthened the safe-verdict fallback gate: added "classify as vulnerable" as the explicit consequence of being unable to name specific operations and guards, closing the loophole where models conclude safe because they cannot articulate a guard. At v6, llama3 produced 9 FNs whose explanations contained only "No untrusted data flows to unchecked operations" — this clause directly addresses that.
- Retained JSON-first format block verbatim from v6: the v6 change (moving output schema to line 1) was confirmed highly effective — codellama parse failures fell from 53 (v5) to 34 (v6), and deepseek parse failures fell from 18 (v5) to 2 (v6). No changes to the format block.
- Retained three-bullet memory bug hiding spots block inside Step 3 Bounds: confirmed working since v4. CWE-125 FNs for llama3 at v6 are 4 (down from 5 at v5), showing the block works but more guidance is needed; the explicit CWE-125 label in the Bounds header should reinforce this.
- Retained logic/authorization bullet from v5 — added CWE-20 explicitly to cover input validation misses seen at v6 for all four models.
- No structural changes: the v6 numbered Step 1–4 format is retained. The JSON-first + numbered steps combination is the most stable format found. No new section types, no phase headers, no sub-headings.

### Rationale
v6 is the best-performing version by average F2 across all four models (0.436 vs. 0.322 for v5). The v6 JSON-first format change was confirmed highly effective: codellama improved from F2=0.529 to F2=0.659, and deepseek improved from F2=0.069 to F2=0.379. llama3 improved slightly (0.571 → 0.592) and qwen remained flat (0.116). v7 is built on v6 as the base.

The dominant remaining failure mode across all models is the logic/authorization and input-validation FN cluster. At v6, llama3 had 21 FNs including CWE-617 (1), CWE-264 (1), CWE-284 (1), CWE-367 (1), CWE-125 (4), CWE-476 (2). Deepseek had 33 FNs with CWE-20 (5), CWE-125 (5), CWE-119 (3) dominating. Qwen had 47 FNs spread across all CWEs — a structural ceiling. Codellama had 16 FNs (strong) but 29 FPs — the over-prediction problem. CWE identifiers inline in the checklist bullets provide the most direct signal to guide model attention to each check without requiring the model to recall them from general training knowledge. The safe-verdict gate fix targets the llama3 FN pattern where the model sees no obvious attacker path and concludes safe without naming what it checked.

### Per-Model Metrics (v6 baseline)
| Model | Precision | Recall | F1 | F2 | Parse Failures |
|-------|-----------|--------|----|-----|----------------|
| llama3-8b | 0.574 | 0.596 | 0.585 | 0.592 | 1 |
| qwen2.5-7b | 0.714 | 0.096 | 0.170 | 0.116 | 1 |
| deepseek-coder-6.7b | 0.442 | 0.365 | 0.400 | 0.379 | 2 |
| codellama-7b | 0.554 | 0.692 | 0.615 | 0.659 | 34 |

---

## v5 → v6 — 2026-05-01

### Changes
- Output format moved to the top: the JSON output schema and the "start with `{`, end with `}`" requirement are now the first three lines of the skill, before the role statement and before any analysis steps. Codellama's parse failures at v5 (53 remaining, down from 87 at v4) persist because the output constraint appears after several paragraphs of analysis instructions — the model generates prose analysis before it encounters the format rule. Placing the format requirement first forces the model to frame its response as JSON before reading any analysis content.
- Restored dedicated memory bug hiding spots block: v4 had a three-bullet "Common hiding spots" sub-section inside Phase 2 (off-by-one, multi-step accumulation, size from input). v5 compressed this into a single merged bullet, and CWE-125 FNs for llama3 jumped from 1 (v4) to 5 (v5). The three specific checks are restored as distinct sub-bullets inside the Step 3 Bounds check, preserving their v4 emphasis level without re-introducing the Phase header echo triggers.
- No structural changes to Step 1–4 content or the safe-verdict gate: the v5 numbered-step format with short imperative labels proved effective for codellama (parse failures 87→53), so the Step 1–4 structure is retained. The logic/authorization bullet from v5 is kept. No changes that would regress codellama back to prose echo.
- Separator line removed: the `---` horizontal rule between the output format block and the role statement is retained as a visual break but the output instructions are kept as the first content the model reads.

### Rationale
v5 is the best-performing version by average F2 across all four models (0.322 vs. 0.258 for v1, the previous best). The single largest remaining improvement opportunity is codellama's 53 parse failures: each failure is treated as label -1, and at 50% base rate ~26 of these are missed TPs. Moving the JSON output constraint to line 1 is the most direct intervention available — this structural placement is not tried in any prior version. The secondary change (restoring the memory bug hiding spots block) targets the llama3 CWE-125 regression specifically. Qwen and deepseek are not addressed with new strategies because their bottlenecks are structural (qwen: safe-default prior that resists skill instructions; deepseek: prompt-wrapper echo that cannot be fixed from the skill side) and no untried instruction-level fix is available.

### Per-Model Metrics (v5 baseline)
| Model | Precision | Recall | F1 | F2 | Parse Failures |
|-------|-----------|--------|----|-----|----------------|
| llama3-8b | 0.630 | 0.558 | 0.592 | 0.571 | 1 |
| qwen2.5-7b | 0.714 | 0.096 | 0.170 | 0.116 | 1 |
| deepseek-coder-6.7b | 0.375 | 0.058 | 0.100 | 0.069 | 18 |
| codellama-7b | 0.575 | 0.519 | 0.546 | 0.529 | 53 |

---

## v4 → v5 — 2026-04-30

### Changes
- Structure: replaced the 3-phase named sections (Phase 1 / Phase 2 / Phase 3) with 4 numbered steps (Step 1–4) using plain imperative labels ("List untrusted inputs", "List dangerous operations", etc.). Codellama reproduced the "Phase 1 — Collect Evidence" header verbatim in its output on almost every sample, driving parse failures from 73 (v3) to 87 (v4). Numbered steps with plain English titles are less likely to be echoed as structural headings.
- Logic/authorization check added to Step 3: v4 had no coverage of CWE-264, CWE-284, CWE-617 (reachable assertion), CWE-367 (TOCTOU), CWE-400. Llama3's FNs at v4 included CWE-617, CWE-367, CWE-264, CWE-400, CWE-388, CWE-284 — all non-memory bugs that the v4 Phase 2 checklist did not address by name. Added "Logic/authorization" bullet to Step 3 covering missing permission checks, bypassable checks, assertion vulnerabilities.
- Safe-verdict gate: the v4 ban on "no untrusted input flows to unchecked operations" phrasing was present but qwen and deepseek still emitted it. Reworded the constraint to state the positive requirement (name two operations and their guards) more prominently, keeping the prohibition as a secondary note.
- Output format instruction: added "Do not output any prose, analysis text, phase headings, code blocks, or markdown before or after the JSON." to directly address codellama's free-text output pattern. The v4 skill said "Output one JSON object and nothing else" but the additional specificity should reduce prose fallback.
- Removed the "Common hiding spots" sub-bullet section from memory bug guidance: in v4 this was inside Phase 2 as a nested block. Moving the key checks into the Step 3 bullet list integrates them as checklist items rather than advisory notes, which is more likely to trigger action on models that follow lists.

### Rationale
v4 delivered significant improvements on llama3 (+0.116 F2) and modest gains on qwen (+0.045) and deepseek (+0.067), but codellama regressed (-0.042) due to a jump in parse failures from 73 to 87. The proximate cause is that the "Phase 1 — Collect Evidence" and "Phase 2 — Evaluate Each Sink" headers in v4 are being reproduced literally by codellama as free-text section headings, and the model then outputs its narrative under those headings without producing JSON. Renaming and restructuring to numbered steps (Step 1–4) with simpler labels removes those echo triggers. The logic/authorization addition addresses llama3's remaining FN cluster (CWE-617, CWE-367, CWE-264, CWE-400, CWE-284) which v4's memory-focused checklist did not cover. Qwen and deepseek are structurally bottlenecked by their strong safe-default prior; incremental skill changes have limited effect on them, but the strengthened safe-verdict gate phrasing gives another opportunity to break the "all inputs are validated" short-circuit.

### Per-Model Metrics (v4 baseline)
| Model | Precision | Recall | F1 | F2 |
|-------|-----------|--------|----|-----|
| llama3-8b | 0.521 | 0.712 | 0.602 | 0.663 |
| qwen2.5-7b | 0.714 | 0.096 | 0.170 | 0.116 |
| deepseek-coder-6.7b | 0.364 | 0.077 | 0.127 | 0.091 |
| codellama-7b | 0.625 | 0.096 | 0.167 | 0.116 |

---

## v3 → v4 — 2026-04-29

### Changes
- Structure: replaced the 10-step numbered procedure with a 3-phase structure (Collect Evidence, Evaluate Each Sink, Verdict) that is harder to short-circuit with a surface-scan "all clear" response (qwen/deepseek/codellama were ticking off all 10 steps with generic "no untrusted input" verdicts)
- Recall bias: strengthened the opening directive from "when suspicion arises, flag as vulnerable" to "default assumption is vulnerable until proven otherwise" (qwen FN rate 94.2% and deepseek FN rate 98.1% show the v3 bias statement was not strong enough to shift their prior)
- Safe-verdict gate: added explicit requirement to name at least two specific sinks and their guards, with the rule that generic phrases like "no untrusted input flows to unchecked operations" are not acceptable and must trigger re-examination (all three low-recall models used that exact phrasing on the majority of their FN samples)
- Output format: removed all fenced code block examples and the document-style title/header structure that caused deepseek (18 parse failures) and codellama (73 parse failures) to echo the skill text back or emit `</skill>` tags in output; the response format is now stated only in prose and inline schema
- `vulnerable_code` rule: added explicit named ban on `memcpy(buf, src, len)` template — llama3 emitted that exact string on 10+ FP samples where those characters did not appear in the input code
- Phase 1 structure: made evidence collection concrete by requiring named variables and actual function calls rather than category-level checks; this targets qwen's pattern of applying Step 2 language without tracing specific code paths
- Precision guard: retained false-positive constraint but reframed it to require a specific expression in the actual code plus a traceable attacker-controlled path, not just a category check

### Rationale
Three models remain near-zero recall after v3 (qwen F2=0.071, deepseek F2=0.024, codellama F2=0.158). The v3 prediction logs show two distinct failure modes that require different fixes. First, deepseek and codellama are producing parse failures by echoing the skill document or emitting `<skill>` tags — this is a formatting/framing problem where the document-style Markdown header causes the model to treat the skill as conversation context rather than instructions. Removing the `# Vulnerability Detection Skill — v3` header and the fenced JSON examples eliminates the most common echo triggers. Second, qwen is parsing correctly but returning `{"vulnerable": false}` with rationalization language drawn from v3's own Step 2 terminology ("no untrusted data flows directly into unchecked operations"). The 10-step procedure enables this by giving the model a sequential checklist it can exit early. Replacing it with a 3-phase structure that requires concrete named evidence at Phase 1 before any verdict is possible forces the model past the surface scan. For llama3, the dominant remaining problem is the hallucinated `memcpy(buf, src, len)` template (10 FP samples), which the v3 "reducing FPs" section did not prevent — an explicit named ban is added.

### Per-Model Metrics (v3 baseline)
| Model | Precision | Recall | F1 | F2 |
|-------|-----------|--------|----|-----|
| llama3-8b | 0.583 | 0.538 | 0.560 | 0.547 |
| qwen2.5-7b | 0.750 | 0.058 | 0.107 | 0.071 |
| deepseek-coder-6.7b | 0.200 | 0.019 | 0.035 | 0.024 |
| codellama-7b | 0.500 | 0.135 | 0.212 | 0.158 |

---

## v2 → v3 — 2026-04-29

### Changes
- Added detection-bias directive after Objective (qwen 94.2% FN rate, deepseek 92.3% FN rate, codellama 86.5% FN rate — models defaulting to safe verdicts without engaging the code)
- Step 3: Added targeted CWE-119/CWE-125 guidance on off-by-one, multi-step index accumulation, and untrusted size values (all 4 models missed CWE-119: 18+ total FN, CWE-125: 17 total FN)
- Step 10: Replaced vague "confirm which steps found nothing suspicious" with concrete checklist requiring trace of actual data flows, real buffer sizes, and real input validation before negative verdict (prior instruction not breaking the false-verdict short-circuit)
- Output format: Tightened to "No prose, no code fences" to address codellama's 69/100 parse failures persisting from v2

### Rationale
Three models remained near zero recall at v2 (qwen F2=0.071, deepseek F2=0.092, codellama F2=0.156). The dominant pattern was an unchanged false-verdict short-circuit: models outputting `{"vulnerable": false, "explanation": "No vulnerability found."}` without working through any analysis step. The v2 Step 10 articulation requirement was present but not strong enough to break this habit. Adding a detection-bias statement at the top anchors the models toward suspicion before they reach the verdict step. The Step 10 rewrite ties the safe path explicitly to Steps 2–4 and the actual code, making it harder to return a negative verdict generically. CWE-119/125 remained the highest-miss CWEs across all four models, so an extra sentence explaining their typical hiding places was added within the existing Step 3 budget.

### Per-Model Metrics (v2 baseline)
| Model | Precision | Recall | F1 | F2 |
|-------|-----------|--------|----|----|
| llama3-8b | 0.548 | 0.654 | 0.597 | 0.630 |
| qwen2.5-7b | 0.750 | 0.058 | 0.107 | 0.071 |
| deepseek-coder-6.7b | 0.400 | 0.077 | 0.129 | 0.092 |
| codellama-7b | 0.438 | 0.135 | 0.206 | 0.156 |

---

## v1 → v2 — 2026-04-28

### Changes
- Step 3: Expanded "out-of-bounds reads and writes" into two distinct bullets naming CWE-125/CWE-119 and CWE-787 with concrete heuristics (all 4 models missed CWE-125: 17 total FN, CWE-119: 16 total FN)
- Step 4: Expanded NULL-pointer check bullet to name CWE-476 and list concrete fallible functions (all 4 models missed: 10 total FN)
- Step 10: Added explicit requirement to articulate why code is safe before returning `vulnerable: false` (qwen 92% FN rate, deepseek 92% FN rate — both producing bare false verdicts without engaging with analysis steps)
- Output format: Added explicit instruction that output must begin with `{` and end with `}` (codellama 72/100 parse failures, deepseek 5/100 — models outputting free-text)
- Reducing FP: Added concrete rule against flagging `memcpy`/`malloc` surface features without a traceable exploit path (llama3 18 FP, deepseek/codellama template-cloned CWE-787 FPs)

### Rationale
Three of the four models (qwen, deepseek, codellama) showed FN rates above 77%, driven by two failure modes: models defaulting to a bare `false` verdict without engaging the analysis steps, and models failing to parse CWE-125/CWE-119 (out-of-bounds reads) as distinct from CWE-787 (writes). The parse failure crisis in codellama (72%) required strengthening the JSON-only output constraint. Adding explicit CWE identifiers and function-level heuristics in Steps 3 and 4 gives all models concrete signals to recognise these vulnerability classes rather than dismissing them as safe.

### Per-Model Metrics (v1 baseline)
| Model | Precision | Recall | F1 | F2 |
|-------|-----------|--------|----|----|
| llama3-8b | 0.625 | 0.577 | 0.600 | 0.586 |
| qwen2.5-7b | 0.800 | 0.077 | 0.140 | 0.094 |
| deepseek-coder-6.7b | 0.400 | 0.077 | 0.129 | 0.092 |
| codellama-7b | 0.545 | 0.231 | 0.324 | 0.261 |

---
