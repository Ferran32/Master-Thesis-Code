from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATASETS_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
LOCAL_MODELS_DIR = MODELS_DIR / "hub"
SKILLS_DIR = PROJECT_ROOT / "skills"

MEGAVUL_PROCESSED = DATA_DIR / "megavul" / "pair_only_megavul_dataset"
BIGVUL_PROCESSED = DATA_DIR / "bigvul" / "pair_only_bigvul_dataset"
CVEFIXES_PROCESSED = DATA_DIR / "cvefixes" / "pair_only_cvefixes_dataset"

MEGAVUL_HF = "hitoshura25/megavul"
BIGVUL_HF = "DynaOuchebara/BigVul"
CVEFIXES_HF = "hitoshura25/cvefixes"
MIVIA_HF = "ferran-solanes/MIVIA"

MODELS: Dict[str, str] = {
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-coder-6.7b": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "codellama-7b": "codellama/CodeLlama-7b-Instruct-hf",
    "qwen3.6-27b": "Qwen/Qwen3.6-27B",
    "gemma4-31b": "google/gemma-4-31B-it",
}

INSTRUCT_MODELS = {
    "llama3-8b", "qwen2.5-7b", "deepseek-coder-6.7b", "codellama-7b",
    "qwen3.6-27b", "gemma4-31b",
}

LARGE_MODELS = {"qwen3.6-27b", "gemma4-31b"}

L4_INFERENCE_MACHINE = "g2-standard-8"
L4_SFT_MACHINE = "g2-standard-12"
A100_MACHINE = "a2-highgpu-1g"

SEED = 42
PROMPT_STRATEGIES = ["zero_shot", "few_shot", "cot"]
FEW_SHOT_COUNTS = [0, 1, 3]

DEFAULT_MAX_SEQ_LEN = 8192
DEFAULT_MAX_NEW_TOKENS = 512

GENERATION_KWARGS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "do_sample": False,
    "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
}


@dataclass
class SFTConfig:
    
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]
    )
    learning_rate: float = 2e-5
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 32 
    max_seq_length: int = 2048
    max_train_chars: int = 10000
    optimizer: str = "paged_adamw_8bit"
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "linear"
    seed: int = SEED

    quantization_bits: int = 4
    compute_dtype: str = "bfloat16"

    early_stopping_patience: int = 3
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 290
    save_strategy: str = "steps"
    save_steps: int = 290
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    report_to: str = "none"


SFT_DEFAULTS = SFTConfig()

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1


def get_model_id(short_name: str) -> str:
    if short_name in MODELS:
        return MODELS[short_name]
    if "/" in short_name:
        return short_name
    raise ValueError(
        f"Unknown model '{short_name}'. Available: {list(MODELS.keys())}"
    )


def model_short_name(model_id: str) -> str:
    for short, full in MODELS.items():
        if full == model_id:
            return short
    return model_id.split("/")[-1]


def is_instruct_model(short_name: str) -> bool:
    return short_name in INSTRUCT_MODELS


def is_large_model(short_name: str) -> bool:
    return short_name in LARGE_MODELS


def machine_spec_for(short_name: str, *, sft: bool = False) -> Dict[str, object]:
    if short_name in LARGE_MODELS:
        return {
            "machine_type": A100_MACHINE,
            "accelerator_type": "NVIDIA_TESLA_A100",
            "accelerator_count": 1,
        }
    return {
        "machine_type": L4_SFT_MACHINE if sft else L4_INFERENCE_MACHINE,
        "accelerator_type": "NVIDIA_L4",
        "accelerator_count": 1,
    }


def results_path(experiment: str, model: str, suffix: str = "") -> Path:
    exp_dir = RESULTS_DIR / experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    fname = model_short_name(model)
    if suffix:
        fname += f"_{suffix}"
    fname += ".jsonl"
    return exp_dir / fname


def adapter_path(model: str) -> Path:
    p = MODELS_DIR / "sft" / model_short_name(model)
    p.mkdir(parents=True, exist_ok=True)
    return p


def local_model_path(model_id: str) -> Path:
    return LOCAL_MODELS_DIR / model_id


def resolve_model_source(model_id: str) -> str:
    local = local_model_path(model_id)
    if local.exists() and any(local.iterdir()):
        return str(local)
    return model_id
