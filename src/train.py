"""
Hydra-based CPT training entrypoint with Unsloth backend.

Supports:
- full-weight training
- LoRA / QLoRA
- W&B logging
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import unsloth  # must be imported before transformers for kernel patches
import hydra
import torch
from datasets import Dataset, load_from_disk
from omegaconf import DictConfig, OmegaConf
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from .common import resolve_workspace_path, setup_wandb_env, to_report_to_list


class CausalPackedCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_batch = []
        label_batch = []

        for feature in features:
            input_ids = list(feature["input_ids"])
            labels = list(feature.get("labels", feature["input_ids"]))
            pad_len = max_len - len(input_ids)

            input_batch.append(input_ids + [self.pad_token_id] * pad_len)
            label_batch.append(labels + [-100] * pad_len)

        input_tensor = torch.tensor(input_batch, dtype=torch.long)
        label_tensor = torch.tensor(label_batch, dtype=torch.long)
        attention_mask = (input_tensor != self.pad_token_id).long()

        return {
            "input_ids": input_tensor,
            "labels": label_tensor,
            "attention_mask": attention_mask,
        }


def is_vision_model(model_name: str) -> bool:
    vision_keywords = ["qwen3.5", "qwen3_5", "gemma-3", "gemma3"]
    return any(keyword in model_name.lower() for keyword in vision_keywords)


def count_params(model) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def apply_finetuning_mode(model, model_class, cfg: DictConfig):
    method = str(cfg.finetune.method).lower()

    if method == "full":
        if bool(cfg.model.quantization.load_in_4bit):
            raise ValueError("Full-weight mode cannot be combined with load_in_4bit=true.")
        return model

    if method != "lora":
        raise ValueError(f"Unsupported finetune method: {cfg.finetune.method}")

    lora_cfg = cfg.finetune.lora

    common_kwargs = {
        "r": int(lora_cfg.r),
        "lora_alpha": int(lora_cfg.alpha),
        "lora_dropout": float(lora_cfg.dropout),
        "target_modules": [str(module_name) for module_name in lora_cfg.target_modules],
        "bias": str(lora_cfg.bias),
        "use_rslora": bool(lora_cfg.use_rslora),
        "use_dora": bool(lora_cfg.use_dora),
        "use_gradient_checkpointing": "unsloth" if bool(cfg.training.gradient_checkpointing) else True,
        "random_state": int(cfg.training.seed),
    }

    if is_vision_model(cfg.model.pretrained_model_name_or_path):
        common_kwargs["finetune_vision_layers"] = bool(lora_cfg.finetune_vision_layers)
        common_kwargs["finetune_language_layers"] = bool(lora_cfg.finetune_language_layers)
        common_kwargs["finetune_attention_modules"] = bool(lora_cfg.finetune_attention_modules)
        common_kwargs["finetune_mlp_modules"] = bool(lora_cfg.finetune_mlp_modules)

    model = model_class.get_peft_model(model, **common_kwargs)
    return model


def resolve_precision_flags(training_cfg: DictConfig) -> tuple[bool, bool]:
    wants_bf16 = bool(training_cfg.bf16)
    wants_fp16 = bool(training_cfg.fp16)

    can_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_bf16 = wants_bf16 and can_bf16

    if wants_bf16 and not can_bf16 and torch.cuda.is_available():
        use_fp16 = True
    else:
        use_fp16 = wants_fp16 and torch.cuda.is_available()

    return use_bf16, use_fp16


def resolve_resume_checkpoint(resume_value: Any, output_dir: Path) -> str | None:
    if resume_value is None:
        return None

    resume_text = str(resume_value).strip()
    if resume_text.lower() in {"", "none", "null"}:
        return None
    if resume_text.lower() == "auto":
        return get_last_checkpoint(str(output_dir))

    return str(resolve_workspace_path(resume_text))


def freeze_input_embeddings(model) -> int:
    if not hasattr(model, "get_input_embeddings"):
        return 0
    embedding_layer = model.get_input_embeddings()
    if embedding_layer is None:
        return 0

    frozen_params = 0
    for parameter in embedding_layer.parameters():
        if parameter.requires_grad:
            frozen_params += parameter.numel()
        parameter.requires_grad = False
    return frozen_params


def _set_if_supported(
    kwargs: dict[str, Any],
    fields: set[str],
    key: str,
    value: Any,
) -> None:
    if key in fields and value is not None:
        kwargs[key] = value


def contains_cross_sample_packing(dataset: Dataset, probe_rows: int = 64) -> bool:
    if "seq_lengths" not in dataset.column_names:
        return False

    limit = min(len(dataset), probe_rows)
    for idx in range(limit):
        row = dataset[idx]
        seq_lengths = row.get("seq_lengths")
        if isinstance(seq_lengths, list) and len(seq_lengths) > 1:
            return True
    return False


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=" * 80)
    print("CPT Train Config")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg, resolve=True))

    run_name = str(cfg.logging.get("run_name") or f"{cfg.experiment.name}-{cfg.finetune.method}")
    setup_wandb_env(cfg.logging, experiment_name=run_name)

    model_name = str(cfg.model.pretrained_model_name_or_path)
    finetune_method = str(cfg.finetune.method).lower()
    wants_full_finetuning = finetune_method == "full"
    vision = is_vision_model(model_name)
    if vision:
        from unsloth import FastVisionModel as ModelClass
        print(f"loading model (vision backend): {model_name}")
    else:
        from unsloth import FastLanguageModel as ModelClass
        print(f"loading model (language backend): {model_name}")

    from_pretrained_kwargs: dict[str, Any] = {
        "model_name": model_name,
        "max_seq_length": int(cfg.model.max_seq_length),
        "dtype": None,
        "load_in_4bit": bool(cfg.model.quantization.load_in_4bit),
    }
    # Unsloth needs full_finetuning=True to avoid silently routing to LoRA mode.
    if "full_finetuning" in inspect.signature(ModelClass.from_pretrained).parameters:
        from_pretrained_kwargs["full_finetuning"] = wants_full_finetuning

    model, tokenizer = ModelClass.from_pretrained(**from_pretrained_kwargs)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if bool(cfg.training.get("freeze_embeddings", False)):
        frozen_count = freeze_input_embeddings(model)
        print(f"[Freeze] input embeddings frozen params: {frozen_count:,}")

    model = apply_finetuning_mode(model, model_class=ModelClass, cfg=cfg)
    model.config.use_cache = bool(cfg.model.use_cache)
    total_params, trainable_params = count_params(model)
    print(f"total params: {total_params:,}")
    print(f"trainable params: {trainable_params:,}")
    print(f"trainable ratio: {100.0 * trainable_params / total_params:.4f}%")
    if wants_full_finetuning and trainable_params == 0:
        raise RuntimeError(
            "Full-weight mode selected, but trainable params = 0. "
            "Check Unsloth full_finetuning path and model loading config."
        )
    if str(cfg.finetune.method).lower() == "lora":
        model.print_trainable_parameters()

    dataset_root = resolve_workspace_path(cfg.training.dataset_dir)
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train dataset not found: {train_dir}")

    train_dataset = load_from_disk(str(train_dir))
    eval_dataset = load_from_disk(str(val_dir)) if val_dir.exists() else None

    runtime_packing_cfg = cfg.training.get("runtime_packing")
    use_runtime_packing = bool(
        runtime_packing_cfg and bool(runtime_packing_cfg.get("enabled", False))
    )

    if use_runtime_packing and "text" not in train_dataset.column_names:
        raise RuntimeError(
            "Runtime packing requires a `text` column in preprocessed dataset. "
            "Re-run preprocess with preprocessing.packing.enabled=false."
        )

    if bool(cfg.training.prevent_cross_sample_attention) and not use_runtime_packing:
        if contains_cross_sample_packing(train_dataset):
            raise RuntimeError(
                "Detected packed rows with multiple samples (`seq_lengths` length > 1). "
                "Set preprocessing.packing.enabled=false for contamination-safe CPT."
            )
        if eval_dataset is not None and contains_cross_sample_packing(eval_dataset):
            raise RuntimeError(
                "Detected packed eval rows with multiple samples (`seq_lengths` length > 1). "
                "Set preprocessing.packing.enabled=false for contamination-safe CPT."
            )

    output_dir = resolve_workspace_path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_to = to_report_to_list(cfg.logging.report_to)
    use_bf16, use_fp16 = resolve_precision_flags(cfg.training)
    has_eval = eval_dataset is not None and str(cfg.training.evaluation_strategy).lower() != "no"
    eval_strategy = str(cfg.training.evaluation_strategy) if has_eval else "no"
    sft_fields = set(SFTConfig.__dataclass_fields__.keys())
    sft_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": float(cfg.training.learning_rate),
        "optim": str(cfg.training.optim),
        "lr_scheduler_type": str(cfg.training.lr_scheduler_type),
        "warmup_ratio": float(cfg.training.warmup_ratio),
        "num_train_epochs": float(cfg.training.num_train_epochs),
        "max_steps": int(cfg.training.max_steps),
        "per_device_train_batch_size": int(cfg.training.per_device_train_batch_size),
        "per_device_eval_batch_size": int(cfg.training.per_device_eval_batch_size),
        "gradient_accumulation_steps": int(cfg.training.gradient_accumulation_steps),
        "weight_decay": float(cfg.training.weight_decay),
        "max_grad_norm": float(cfg.training.max_grad_norm),
        "logging_steps": int(cfg.training.logging_steps),
        "save_steps": int(cfg.training.save_steps),
        "save_total_limit": int(cfg.training.save_total_limit),
        "eval_steps": int(cfg.training.eval_steps),
        "eval_strategy": eval_strategy,
        "eval_on_start": bool(cfg.training.eval_on_start),
        "dataloader_num_workers": int(cfg.training.dataloader_num_workers),
        "remove_unused_columns": bool(cfg.training.remove_unused_columns),
        "bf16": use_bf16,
        "fp16": use_fp16,
        "gradient_checkpointing": bool(cfg.training.gradient_checkpointing),
        "report_to": report_to if report_to else "none",
        "run_name": run_name,
        "seed": int(cfg.training.seed),
        "save_strategy": str(cfg.training.save_strategy),
    }
    _set_if_supported(
        sft_kwargs,
        sft_fields,
        "overwrite_output_dir",
        bool(cfg.training.overwrite_output_dir),
    )

    if use_runtime_packing:
        _set_if_supported(sft_kwargs, sft_fields, "dataset_text_field", "text")
        _set_if_supported(sft_kwargs, sft_fields, "packing", True)
        _set_if_supported(
            sft_kwargs,
            sft_fields,
            "packing_strategy",
            str(runtime_packing_cfg.get("strategy", "bfd_split")),
        )
        _set_if_supported(
            sft_kwargs,
            sft_fields,
            "padding_free",
            bool(runtime_packing_cfg.get("padding_free", True)),
        )
        runtime_max_len = int(runtime_packing_cfg.get("max_length", cfg.model.max_seq_length))
        _set_if_supported(sft_kwargs, sft_fields, "max_length", runtime_max_len)
        _set_if_supported(sft_kwargs, sft_fields, "max_seq_length", runtime_max_len)
    else:
        _set_if_supported(sft_kwargs, sft_fields, "dataset_text_field", "")
        _set_if_supported(sft_kwargs, sft_fields, "packing", False)

    # Keep eval memory usage low by disabling prediction/logit accumulation.
    _set_if_supported(sft_kwargs, sft_fields, "prediction_loss_only", True)

    args = SFTConfig(**sft_kwargs)

    base_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if base_tokenizer.eos_token_id is not None:
        base_tokenizer.eos_token = base_tokenizer.convert_ids_to_tokens(base_tokenizer.eos_token_id)

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if has_eval else None,
        processing_class=base_tokenizer,
        data_collator=(
            None
            if use_runtime_packing
            else CausalPackedCollator(
                pad_token_id=base_tokenizer.pad_token_id or base_tokenizer.eos_token_id
            )
        ),
    )

    resume_checkpoint = resolve_resume_checkpoint(cfg.training.resume_from_checkpoint, output_dir=output_dir)
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    trainer.save_state()

    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)

    if has_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    print("=" * 80)
    print("Train Complete")
    print("=" * 80)
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
