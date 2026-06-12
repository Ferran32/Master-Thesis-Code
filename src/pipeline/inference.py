from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import (
    model_short_name,
    is_instruct_model,
    resolve_model_source,
    local_model_path,
)

_LOADED_MODEL_KEY: Optional[str] = None
_LOADED_MODEL: Optional[AutoModelForCausalLM] = None
_LOADED_TOKENIZER: Optional[AutoTokenizer] = None

_BPE_ARTIFACTS = str.maketrans({
    "\u0120": " ",   # Ġ -> space  (byte-level BPE space prefix)
    "\u010a": "\n",  # Ċ -> newline
    "\u0109": "\t",  # ĉ -> tab
})


def _clean_bpe_artifacts(text: str) -> str:
    # For deepseek-coder, its tokenizer leaves these in the decoded output.
    return text.translate(_BPE_ARTIFACTS)


def load_model(
    model_id: str,
    max_seq_len: int = 1024,
    quantization: str = "4bit",
    device_map: str = "auto",
    adapter_path: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    global _LOADED_MODEL_KEY, _LOADED_MODEL, _LOADED_TOKENIZER

    cache_key = f"{model_id}::{adapter_path or ''}::{quantization}::{max_seq_len}"
    if _LOADED_MODEL_KEY == cache_key and _LOADED_MODEL is not None:
        return _LOADED_MODEL, _LOADED_TOKENIZER

    resolved = resolve_model_source(model_id)
    is_local = resolved != model_id
    print(f"[inference] Loading model: {model_id}"
          f"{' (local)' if is_local else ' (HuggingFace)'}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=(quantization == "4bit"),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer_source = resolved
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, trust_remote_code=True, model_max_length=max_seq_len,
        truncation_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    if not is_local and not adapter_path:
        local_dir = local_model_path(model_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(local_dir))
        tokenizer.save_pretrained(str(local_dir))
        print(f"[inference] Saved model locally → {local_dir}")

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    _LOADED_MODEL_KEY = cache_key
    _LOADED_MODEL = model
    _LOADED_TOKENIZER = tokenizer
    return model, tokenizer


def unload_model() -> None:
    global _LOADED_MODEL_KEY, _LOADED_MODEL, _LOADED_TOKENIZER
    if _LOADED_MODEL is not None:
        del _LOADED_MODEL
        torch.cuda.empty_cache()
    _LOADED_MODEL_KEY = None
    _LOADED_MODEL = None
    _LOADED_TOKENIZER = None


def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    model_id: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> str:
    short = model_short_name(model_id)

    if messages is not None and hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    elif is_instruct_model(short) and hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    else:
        input_text = prompt

    raw_len = len(tokenizer(input_text, truncation=False)["input_ids"])
    if raw_len > tokenizer.model_max_length:
        import logging
        logging.warning(
            "[truncation] prompt %d tokens > max_seq %d — left-truncating; "
            "format instructions may be lost",
            raw_len, tokenizer.model_max_length,
        )

    inputs = tokenizer(
        input_text, return_tensors="pt", truncation=True,
        max_length=tokenizer.model_max_length,
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(
        gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True,
    )
    text = _clean_bpe_artifacts(text)
    return text.strip()
