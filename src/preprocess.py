"""
Hydra-based CPT preprocessing:
1) load rows from HF dataset
2) extract text fields
3) boundary-first chunking
4) append EOS only on each document-end chunk
5) optional TRL packing
6) train/val split and save_to_disk
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import hydra
from datasets import Dataset, IterableDataset, load_dataset
from datasets.exceptions import DatasetGenerationCastError
from omegaconf import DictConfig, OmegaConf, open_dict
from transformers import AutoTokenizer
from trl import pack_dataset

from .common import resolve_workspace_path, suppress_noisy_library_logs


def resolve_dataset_cfgs(cfg: DictConfig) -> list[DictConfig]:
    datasets = cfg.data.get("datasets")
    if datasets is not None:
        return [dataset_cfg for dataset_cfg in datasets]

    dataset = cfg.data.get("dataset")
    if dataset is not None:
        return [dataset]

    raise RuntimeError("Config must include either data.dataset or data.datasets.")


def resolve_train_dataset_cfgs(cfg: DictConfig) -> list[DictConfig]:
    train_datasets = cfg.data.get("train_datasets")
    if train_datasets is not None:
        return [dataset_cfg for dataset_cfg in train_datasets]
    return resolve_dataset_cfgs(cfg)


def resolve_validation_dataset_cfgs(cfg: DictConfig) -> list[DictConfig]:
    validation_datasets = cfg.data.get("validation_datasets")
    if validation_datasets is not None:
        return [dataset_cfg for dataset_cfg in validation_datasets]
    return []


def resolve_local_dataset_snapshot(repo_id: str) -> Path | None:
    """
    Return latest local HF dataset snapshot path if available.
    """
    repo_cache = repo_id.replace("/", "--")
    snapshots_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"datasets--{repo_cache}"
        / "snapshots"
    )
    if not snapshots_dir.exists():
        return None

    snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not snapshots:
        return None
    return max(snapshots, key=lambda p: p.stat().st_mtime)


def iter_rows_from_local_snapshot(dataset_cfg: DictConfig):
    """
    Iterate rows from local snapshot JSONL files first (mono-like fast path).
    Returns None if local snapshot is unavailable or unusable.
    """
    repo_id = str(dataset_cfg.path)
    snapshot = resolve_local_dataset_snapshot(repo_id)
    if snapshot is None:
        return None

    jsonl_files = sorted(snapshot.rglob("*.jsonl"))
    if not jsonl_files:
        return None

    max_rows = dataset_cfg.get("max_rows")
    max_rows = int(max_rows) if max_rows is not None else None
    emitted = 0

    print(
        f"[INFO] Loading dataset from local snapshot: {snapshot} "
        f"({len(jsonl_files)} jsonl files)"
    )

    for data_file in jsonl_files:
        ds = load_dataset("json", data_files=str(data_file), split="train")
        for row in ds:
            if max_rows is not None and emitted >= max_rows:
                return
            yield emitted, row
            emitted += 1


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def extract_text(row: dict[str, Any], text_fields: list[str]) -> str | None:
    text = None
    for field in text_fields:
        if field in row and row[field] is not None:
            text = row[field]
            break

    if text is None:
        return None

    if isinstance(text, list):
        text = "\n".join(str(x) for x in text)
    elif isinstance(text, dict):
        text = json.dumps(text, ensure_ascii=False)
    elif not isinstance(text, str):
        text = str(text)

    text = normalize_text(text)
    return text if text else None


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    return paragraphs or [text]


def chunk_boundary_first(
    text: str,
    tokenizer,
    max_body_tokens: int,
    para_sep_ids: list[int],
) -> list[list[int]]:
    paragraphs = split_paragraphs(text)
    chunks: list[list[int]] = []
    current: list[int] = []

    for paragraph in paragraphs:
        para_ids = tokenizer.encode(paragraph, add_special_tokens=False)
        if not para_ids:
            continue

        if len(para_ids) > max_body_tokens:
            if current:
                chunks.append(current)
                current = []
            for start in range(0, len(para_ids), max_body_tokens):
                chunks.append(para_ids[start : start + max_body_tokens])
            continue

        extra_tokens = (len(para_sep_ids) if current else 0) + len(para_ids)
        if len(current) + extra_tokens <= max_body_tokens:
            if current:
                current.extend(para_sep_ids)
            current.extend(para_ids)
        else:
            chunks.append(current)
            current = list(para_ids)

    if current:
        chunks.append(current)

    return chunks


def iter_dataset_rows(dataset_cfg: DictConfig):
    if bool(dataset_cfg.get("prefer_local_snapshot", True)):
        local_iter = iter_rows_from_local_snapshot(dataset_cfg)
        if local_iter is not None:
            yield from local_iter
            return

    load_kwargs = {
        "path": dataset_cfg.path,
        "name": dataset_cfg.name,
        "split": dataset_cfg.split,
        "streaming": bool(dataset_cfg.streaming),
        "trust_remote_code": bool(dataset_cfg.trust_remote_code),
    }

    try:
        dataset = load_dataset(**load_kwargs)
    except DatasetGenerationCastError:
        if bool(dataset_cfg.streaming):
            raise
        # Some JSONL datasets on HF include mixed schemas across files.
        # Fallback to streaming mode, which avoids strict Arrow schema casting.
        print(
            "[WARN] Dataset schema cast failed with non-streaming load. "
            "Retrying with streaming=True."
        )
        dataset = load_dataset(
            path=dataset_cfg.path,
            name=dataset_cfg.name,
            split=dataset_cfg.split,
            streaming=True,
            trust_remote_code=bool(dataset_cfg.trust_remote_code),
        )

    max_rows = dataset_cfg.get("max_rows")
    if max_rows is not None:
        max_rows = int(max_rows)

    if isinstance(dataset, IterableDataset):
        for idx, row in enumerate(dataset):
            if max_rows is not None and idx >= max_rows:
                break
            yield idx, row
        return

    total = len(dataset)
    end = min(total, max_rows) if max_rows is not None else total
    for idx in range(end):
        yield idx, dataset[idx]


def build_unpacked_records(
    cfg: DictConfig,
    tokenizer,
    dataset_cfgs: list[DictConfig] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError("Tokenizer has no eos_token_id")

    max_length = int(cfg.preprocessing.packing.max_length)
    max_body_tokens = max_length - 1
    para_sep_ids = tokenizer.encode(
        cfg.preprocessing.chunking.paragraph_separator,
        add_special_tokens=False,
    )

    min_chars = int(cfg.preprocessing.cleaning.min_chars)
    default_text_fields = [str(x) for x in cfg.data.text_fields]

    records: list[dict[str, Any]] = []
    doc_count = 0
    skipped_empty = 0
    skipped_short = 0
    chunk_count = 0

    if dataset_cfgs is None:
        dataset_cfgs = resolve_train_dataset_cfgs(cfg)
    for dataset_idx, dataset_cfg in enumerate(dataset_cfgs):
        text_fields = dataset_cfg.get("text_fields")
        if text_fields is None:
            text_fields = default_text_fields
        else:
            text_fields = [str(x) for x in text_fields]

        source_column = dataset_cfg.get("source_column", cfg.data.get("source_column"))
        id_column = dataset_cfg.get("id_column", cfg.data.get("id_column"))
        default_source = str(dataset_cfg.path)

        for row_idx, row in iter_dataset_rows(dataset_cfg):
            text = extract_text(row, text_fields=text_fields)
            if text is None:
                skipped_empty += 1
                continue

            if len(text) < min_chars:
                skipped_short += 1
                continue

            doc_count += 1
            source = str(row.get(source_column, default_source)) if source_column else default_source
            doc_id = (
                str(row.get(id_column, f"row:{dataset_idx}:{row_idx}"))
                if id_column
                else f"row:{dataset_idx}:{row_idx}"
            )

            chunks = chunk_boundary_first(
                text=text,
                tokenizer=tokenizer,
                max_body_tokens=max_body_tokens,
                para_sep_ids=para_sep_ids,
            )

            total_chunks = len(chunks)
            for chunk_idx, chunk_ids in enumerate(chunks):
                if not chunk_ids:
                    continue

                is_doc_end = chunk_idx == total_chunks - 1
                input_ids = list(chunk_ids)

                if is_doc_end and input_ids[-1] != eos_id:
                    input_ids.append(eos_id)

                if len(input_ids) > max_length:
                    input_ids = input_ids[:max_length]

                records.append(
                    {
                        "doc_id": doc_id,
                        "source": source,
                        "chunk_id": chunk_idx,
                        "is_doc_end": is_doc_end,
                        "text": tokenizer.decode(input_ids, skip_special_tokens=False),
                        "input_ids": input_ids,
                        "labels": list(input_ids),
                        "seq_lengths": [len(input_ids)],
                    }
                )
                chunk_count += 1

    stats = {
        "docs_kept": doc_count,
        "chunks_total": chunk_count,
        "rows_skipped_empty": skipped_empty,
        "rows_skipped_short": skipped_short,
        "records_total": len(records),
    }
    return records, stats


def build_records_with_snapshot_fallback(
    cfg: DictConfig,
    tokenizer,
    dataset_cfgs: list[DictConfig],
    split_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records, stats = build_unpacked_records(
        cfg=cfg,
        tokenizer=tokenizer,
        dataset_cfgs=dataset_cfgs,
    )
    if records:
        return records, stats

    if not any(bool(ds.get("prefer_local_snapshot", True)) for ds in dataset_cfgs):
        return records, stats

    print(
        f"[WARN] No records built from local snapshot path for {split_name}. "
        "Retrying with HF Hub streaming."
    )
    with open_dict(cfg):
        for dataset_cfg in dataset_cfgs:
            dataset_cfg.prefer_local_snapshot = False
            dataset_cfg.streaming = True

    return build_unpacked_records(
        cfg=cfg,
        tokenizer=tokenizer,
        dataset_cfgs=dataset_cfgs,
    )


def maybe_pack_dataset(unpacked_ds: Dataset, cfg: DictConfig) -> Dataset:
    if not bool(cfg.preprocessing.packing.enabled):
        return unpacked_ds

    strategy = str(cfg.preprocessing.packing.strategy)
    seq_length = int(cfg.preprocessing.packing.max_length)
    return pack_dataset(
        unpacked_ds.select_columns(["input_ids", "labels"]),
        seq_length=seq_length,
        strategy=strategy,
    )


def split_dataset(dataset: Dataset, cfg: DictConfig) -> tuple[Dataset, Dataset | None]:
    if len(dataset) < 2:
        return dataset, None

    val_ratio = float(cfg.preprocessing.split.val_ratio)
    if val_ratio <= 0.0:
        return dataset, None

    val_size = max(1, int(len(dataset) * val_ratio))
    if val_size >= len(dataset):
        val_size = len(dataset) - 1

    split = dataset.train_test_split(
        test_size=val_size,
        seed=int(cfg.preprocessing.split.seed),
        shuffle=True,
    )
    return split["train"], split["test"]


def prepare_save_path(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{path} already exists. Set preprocessing.overwrite_output=true to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    suppress_noisy_library_logs()

    print("=" * 80)
    print("CPT Preprocess Config")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg, resolve=True))

    tokenizer_name = cfg.model.tokenizer_name_or_path or cfg.model.pretrained_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=bool(cfg.model.trust_remote_code),
    )
    if tokenizer.eos_token_id is None:
        raise RuntimeError(f"Tokenizer '{tokenizer_name}' has no eos token id")

    train_dataset_cfgs = resolve_train_dataset_cfgs(cfg)
    val_dataset_cfgs = resolve_validation_dataset_cfgs(cfg)

    train_records, train_stats = build_records_with_snapshot_fallback(
        cfg=cfg,
        tokenizer=tokenizer,
        dataset_cfgs=train_dataset_cfgs,
        split_name="train",
    )
    if not train_records:
        raise RuntimeError(
            "No train records built from dataset. "
            f"docs_kept={train_stats.get('docs_kept', 0)}, "
            f"rows_skipped_empty={train_stats.get('rows_skipped_empty', 0)}, "
            f"rows_skipped_short={train_stats.get('rows_skipped_short', 0)}. "
            "Check data.text_fields and filtering settings."
        )

    unpacked_train_ds = Dataset.from_list(train_records)
    train_ds = maybe_pack_dataset(unpacked_train_ds, cfg)

    unpacked_val_ds: Dataset | None = None
    val_ds: Dataset | None = None
    val_stats: dict[str, Any] | None = None

    if val_dataset_cfgs:
        val_records, val_stats = build_records_with_snapshot_fallback(
            cfg=cfg,
            tokenizer=tokenizer,
            dataset_cfgs=val_dataset_cfgs,
            split_name="validation",
        )
        if not val_records:
            raise RuntimeError(
                "validation_datasets are configured but no validation records were built. "
                f"docs_kept={val_stats.get('docs_kept', 0) if val_stats else 0}, "
                f"rows_skipped_empty={val_stats.get('rows_skipped_empty', 0) if val_stats else 0}, "
                f"rows_skipped_short={val_stats.get('rows_skipped_short', 0) if val_stats else 0}."
            )
        unpacked_val_ds = Dataset.from_list(val_records)
        val_ds = maybe_pack_dataset(unpacked_val_ds, cfg)
    else:
        train_ds, val_ds = split_dataset(train_ds, cfg)

    output_dir = resolve_workspace_path(cfg.preprocessing.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    meta_path = output_dir / "metadata.json"

    overwrite = bool(cfg.preprocessing.overwrite_output)
    prepare_save_path(train_dir, overwrite=overwrite)
    train_ds.save_to_disk(str(train_dir))

    if val_ds is not None:
        prepare_save_path(val_dir, overwrite=overwrite)
        val_ds.save_to_disk(str(val_dir))
    elif overwrite and val_dir.exists():
        shutil.rmtree(val_dir)

    train_source_paths = [str(dataset_cfg.path) for dataset_cfg in train_dataset_cfgs]
    train_source_splits = [str(dataset_cfg.split) for dataset_cfg in train_dataset_cfgs]
    metadata = {
        "source": train_source_paths[0] if len(train_source_paths) == 1 else train_source_paths,
        "split": train_source_splits[0] if len(train_source_splits) == 1 else train_source_splits,
        "validation_mode": "explicit_datasets" if val_dataset_cfgs else "random_split",
        "text_fields": [str(x) for x in cfg.data.text_fields],
        "packing": {
            "enabled": bool(cfg.preprocessing.packing.enabled),
            "strategy": str(cfg.preprocessing.packing.strategy),
            "max_length": int(cfg.preprocessing.packing.max_length),
        },
        "stats": {
            "train_docs_kept": train_stats["docs_kept"],
            "train_chunks_total": train_stats["chunks_total"],
            "train_rows_skipped_empty": train_stats["rows_skipped_empty"],
            "train_rows_skipped_short": train_stats["rows_skipped_short"],
            "train_records_total": train_stats["records_total"],
            "unpacked_train_rows": len(unpacked_train_ds),
            "train_rows": len(train_ds),
            "unpacked_val_rows": len(unpacked_val_ds) if unpacked_val_ds is not None else 0,
            "val_rows": len(val_ds) if val_ds is not None else 0,
        },
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if val_dataset_cfgs:
        val_source_paths = [str(dataset_cfg.path) for dataset_cfg in val_dataset_cfgs]
        val_source_splits = [str(dataset_cfg.split) for dataset_cfg in val_dataset_cfgs]
        metadata["validation_source"] = (
            val_source_paths[0] if len(val_source_paths) == 1 else val_source_paths
        )
        metadata["validation_split"] = (
            val_source_splits[0] if len(val_source_splits) == 1 else val_source_splits
        )
    if val_stats is not None:
        metadata["stats"]["validation_docs_kept"] = val_stats["docs_kept"]
        metadata["stats"]["validation_chunks_total"] = val_stats["chunks_total"]
        metadata["stats"]["validation_rows_skipped_empty"] = val_stats["rows_skipped_empty"]
        metadata["stats"]["validation_rows_skipped_short"] = val_stats["rows_skipped_short"]
        metadata["stats"]["validation_records_total"] = val_stats["records_total"]
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("Preprocess Complete")
    print("=" * 80)
    print(f"unpacked train rows: {len(unpacked_train_ds):,}")
    if unpacked_val_ds is not None:
        print(f"unpacked val rows  : {len(unpacked_val_ds):,}")
    print(f"train rows   : {len(train_ds):,}")
    print(f"val rows     : {len(val_ds):,}" if val_ds is not None else "val rows     : 0")
    print(f"saved to     : {output_dir}")


if __name__ == "__main__":
    main()
