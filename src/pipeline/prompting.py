from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.prompt_template import (
    build_zero_shot_prompt,
    build_few_shot_prompt,
    build_cot_prompt,
    build_skill_prompt,
    build_skill_few_shot_prompt,
)


def build_prompt_from_config(
    code: str,
    strategy: str,
    few_shot_pool: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
    few_shot_k: int = 3,
    skill_text: Optional[str] = None,
    seed: int = 42,
) -> str:
    if strategy == "zero_shot":
        return build_zero_shot_prompt(code)

    if strategy == "few_shot":
        if few_shot_pool is None:
            raise ValueError("few_shot strategy requires few_shot_pool")
        return build_few_shot_prompt(code, few_shot_pool, n=few_shot_k)

    if strategy == "cot":
        return build_cot_prompt(code)

    if strategy == "skill":
        if skill_text is None:
            raise ValueError("skill strategy requires skill_text")
        return build_skill_prompt(code, skill_text)

    if strategy == "skill_few_shot":
        if skill_text is None or few_shot_pool is None:
            raise ValueError("skill_few_shot strategy requires both skill_text and few_shot_pool")
        return build_skill_few_shot_prompt(code, skill_text, few_shot_pool, n=few_shot_k)

    raise ValueError(f"Unknown strategy: {strategy}")


