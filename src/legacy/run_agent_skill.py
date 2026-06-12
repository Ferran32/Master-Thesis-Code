#!/usr/bin/env python3
"""Experiment 3 — Agent Skill Optimization."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MAX_SEQ_LEN,
    GENERATION_KWARGS,
    MODELS,
    get_model_id,
    is_instruct_model,
    local_model_path,
    model_short_name,
    resolve_model_source,
    results_path,
)
from src.data_utils import load_mivia_test, load_split
from src.evaluation import compute_metrics, error_analysis, print_metrics, save_result_row
from src.output_parser import parse_cwe, parse_prediction
from src.prompt_template import build_skill_prompt


def _load_model(model_id: str, max_seq_len: int):
    resolved = resolve_model_source(model_id)
    is_local = resolved != model_id
    print(f"[agent_skill] Loading model: {model_id}"
          f"{' (local)' if is_local else ' (HuggingFace)'}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        resolved, trust_remote_code=True, model_max_length=max_seq_len,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    if not is_local:
        local_dir = local_model_path(model_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(local_dir))
        tokenizer.save_pretrained(str(local_dir))
        print(f"[agent_skill] Saved model locally → {local_dir}")

    model.eval()
    return model, tokenizer


def _generate(model, tokenizer, prompt, model_id, max_new_tokens):
    short = model_short_name(model_id)
    if is_instruct_model(short) and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        input_text = prompt

    inputs = tokenizer(
        input_text, return_tensors="pt", truncation=True,
        max_length=tokenizer.model_max_length,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=GENERATION_KWARGS["temperature"],
            top_p=GENERATION_KWARGS["top_p"],
            do_sample=GENERATION_KWARGS["do_sample"],
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def load_skill(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Skill file not found: {p}")
    return p.read_text(encoding="utf-8")


def skill_version_from_path(path: str | Path) -> str:
    return Path(path).stem.replace("SKILL_", "").replace("vuln_detection_", "")


def run_agent_skill(
    model_id: str,
    skill_path: str,
    split: str = "test",
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    limit: Optional[int] = None,
    dataset_source: str = "local",
) -> Dict[str, Any]:
    full_model_id = get_model_id(model_id)
    short = model_short_name(full_model_id)
    skill_text = load_skill(skill_path)
    skill_ver = skill_version_from_path(skill_path)

    out_path = results_path("agent_skill", full_model_id, f"skill_{skill_ver}")
    print(f"[agent_skill] Model   : {full_model_id}")
    print(f"[agent_skill] Skill   : {skill_path} (version: {skill_ver})")
    print(f"[agent_skill] Results → {out_path}")

    ds = load_mivia_test() if dataset_source == "mivia" else load_split(split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"[agent_skill] Samples : {len(ds)}")

    model, tokenizer = _load_model(full_model_id, max_seq_len)

    if out_path.exists():
        out_path.unlink()

    y_true, y_pred, cwe_ids = [], [], []
    start_time = time.time()

    for idx, sample in enumerate(ds):
        prompt = build_skill_prompt(sample["text"], skill_text)
        raw_output = _generate(model, tokenizer, prompt, full_model_id, max_new_tokens)
        pred_label, parsed_json = parse_prediction(raw_output)
        pred_cwe = parse_cwe(parsed_json)

        y_true.append(sample["label"])
        y_pred.append(pred_label)
        cwe_ids.append(sample.get("cwe_id", ""))

        save_result_row(out_path, {
            "idx": idx, "label": sample["label"], "predicted_label": pred_label,
            "cwe_id": sample.get("cwe_id", ""), "predicted_cwe": pred_cwe,
            "raw_output": raw_output, "parsed_json": parsed_json,
        })

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  [{idx + 1}/{len(ds)}] elapsed={elapsed:.0f}s avg={elapsed / (idx + 1):.1f}s/sample")

    elapsed = time.time() - start_time
    metrics = compute_metrics(y_true, y_pred)
    errors = error_analysis(y_true, y_pred, cwe_ids=cwe_ids)

    print_metrics(metrics, title=f"Agent Skill / {short} / {skill_ver}")
    print(f"  FP rate   : {errors['fp_rate']:.4f}")
    print(f"  FN rate   : {errors['fn_rate']:.4f}")
    print(f"  Time      : {elapsed:.1f}s ({elapsed / len(ds):.1f}s/sample)\n")

    summary = {
        "model": full_model_id, "approach": "agent_skill", "skill_version": skill_ver,
        "split": split, "max_seq_len": max_seq_len, "max_new_tokens": max_new_tokens,
        "metrics": metrics, "error_analysis": errors, "elapsed_seconds": round(elapsed, 1),
    }
    with open(out_path.with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def generate_diagnostic_report(results_file: Path, output_path: Optional[Path] = None) -> str:
    from src.evaluation import load_results

    results = load_results(results_file)
    y_true = [r["label"] for r in results]
    y_pred = [r["predicted_label"] for r in results]
    cwe_ids = [r.get("cwe_id", "") for r in results]

    metrics = compute_metrics(y_true, y_pred)
    errors = error_analysis(y_true, y_pred, cwe_ids=cwe_ids)

    lines = [
        "=" * 70, "  SKILL REFINEMENT DIAGNOSTIC REPORT", "=" * 70, "",
        "## Aggregate Metrics",
        f"  Precision : {metrics['precision']:.4f}",
        f"  Recall    : {metrics['recall']:.4f}",
        f"  F1-score  : {metrics['f1']:.4f}",
        f"  F2-score  : {metrics['f2']:.4f}",
        f"  Total     : {metrics['total']}",
        f"  Parse fail: {metrics['parse_failures']}", "",
        "## Confusion Matrix",
        f"  TP={errors['confusion']['tp']}  FP={errors['confusion']['fp']}",
        f"  FN={errors['confusion']['fn']}  TN={errors['confusion']['tn']}",
        f"  FP rate: {errors['fp_rate']:.4f}   FN rate: {errors['fn_rate']:.4f}", "",
    ]

    fp_examples = [r for r in results if r["label"] == 0 and r["predicted_label"] == 1]
    lines.append(f"## False Positives ({len(fp_examples)} total, showing up to 10)")
    for i, r in enumerate(fp_examples[:10]):
        lines.append(f"\n  FP #{i+1} (idx={r['idx']}):")
        lines.append(f"    Output preview: {(r.get('raw_output') or '')[:300]}...")
    lines.append("")

    fn_examples = [r for r in results if r["label"] == 1 and r["predicted_label"] == 0]
    lines.append(f"## False Negatives ({len(fn_examples)} total, showing up to 10)")
    for i, r in enumerate(fn_examples[:10]):
        lines.append(f"\n  FN #{i+1} (idx={r['idx']}, CWE={r.get('cwe_id', '?')}):")
        lines.append(f"    Output preview: {(r.get('raw_output') or '')[:300]}...")
    lines.append("")

    if "per_cwe" in errors:
        lines.append("## Per-CWE Performance")
        lines.append(f"  {'CWE':<12} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>6} {'R':>6} {'F1':>6}")
        lines.append("  " + "-" * 50)
        for cwe_key, m in sorted(errors["per_cwe"].items(), key=lambda x: x[1].get("fn", 0), reverse=True):
            lines.append(
                f"  {cwe_key:<12} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} "
                f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}"
            )
        lines.append("")

    lines.append("## Suggested Skill Improvements")
    if errors["fn_rate"] > 0.3:
        lines.append("  - HIGH FN rate: strengthen detection steps")
        if "per_cwe" in errors:
            for cwe, m in sorted(errors["per_cwe"].items(), key=lambda x: x[1].get("fn", 0), reverse=True)[:5]:
                if m["fn"] > 0:
                    lines.append(f"    → {cwe} (missed {m['fn']} times)")
    if errors["fp_rate"] > 0.3:
        lines.append("  - HIGH FP rate: add stricter criteria for vulnerability confirmation")
    if metrics["parse_failures"] > metrics["total"] * 0.1:
        lines.append("  - HIGH parse failures: reinforce JSON-only output format in skill")
    lines.append("")

    report = "\n".join(lines)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"[agent_skill] Diagnostic report saved to: {output_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 3: Agent Skill Optimization")
    parser.add_argument("--model", required=True, help=f"Model short name or HF ID. Available: {list(MODELS.keys())}")
    parser.add_argument("--skill", required=True, help="Path to the SKILL.md definition file.")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--dataset-source", default="local", choices=["local", "mivia"])
    args = parser.parse_args()

    summary = run_agent_skill(
        model_id=args.model, skill_path=args.skill, split=args.split,
        max_seq_len=args.max_seq_len, max_new_tokens=args.max_new_tokens,
        limit=args.limit, dataset_source=args.dataset_source,
    )

    if args.diagnose:
        full_model_id = get_model_id(args.model)
        skill_ver = skill_version_from_path(args.skill)
        res_path = results_path("agent_skill", full_model_id, f"skill_{skill_ver}")
        report = generate_diagnostic_report(res_path, output_path=res_path.with_suffix(".diagnostic.txt"))
        print(report)


if __name__ == "__main__":
    main()
