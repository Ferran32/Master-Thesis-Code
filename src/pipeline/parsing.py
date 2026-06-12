from __future__ import annotations

from typing import Any, Dict

from src.output_parser import extract_json, parse_prediction, parse_cwe


def parse_sample(raw_output: str) -> Dict[str, Any]:
    pred_label, parsed_json = parse_prediction(raw_output)
    pred_cwe = parse_cwe(parsed_json)

    if pred_label == -1:
        if not raw_output or not raw_output.strip():
            error_type = "empty_output"
        elif extract_json(raw_output) is None:
            error_type = "json_extraction_failed"
        else:
            error_type = "vulnerable_field_missing"
    else:
        error_type = None

    return {
        "predicted_label": pred_label,
        "parsed_json": parsed_json,
        "predicted_cwe": pred_cwe,
        "parse_success": pred_label != -1,
        "error_type": error_type,
    }
