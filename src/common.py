from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_workspace_path(path_like: str | Path) -> Path:
    path = Path(str(path_like))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def to_report_to_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, ListConfig)):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    if not text or text.lower() == "none":
        return []
    return [text]


def resolve_torch_dtype(name: str | None):
    if name is None:
        return None
    key = str(name).strip().lower()
    if key in {"", "auto"}:
        return None
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'")
    return mapping[key]


def setup_wandb_env(
    logging_cfg: DictConfig,
    experiment_name: str | None = None,
    tags_override: list[str] | None = None,
) -> None:
    report_to = to_report_to_list(logging_cfg.get("report_to"))
    if "wandb" not in report_to:
        return

    wandb_cfg = logging_cfg.get("wandb") or {}

    project = wandb_cfg.get("project")
    if project:
        os.environ.setdefault("WANDB_PROJECT", str(project))

    entity = wandb_cfg.get("entity")
    if entity:
        os.environ.setdefault("WANDB_ENTITY", str(entity))

    tags = tags_override if tags_override is not None else wandb_cfg.get("tags")
    if tags:
        tag_values = [str(tag) for tag in tags if str(tag).strip()]
        if tag_values:
            os.environ["WANDB_TAGS"] = ",".join(tag_values)

    notes = wandb_cfg.get("notes")
    if notes:
        os.environ.setdefault("WANDB_NOTES", str(notes))

    if experiment_name:
        os.environ.setdefault("WANDB_NAME", str(experiment_name))


def suppress_noisy_library_logs() -> None:
    """
    Reduce noisy INFO logs from HF/http clients during train/preprocess/eval.
    """
    for logger_name in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "transformers",
        "datasets",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
