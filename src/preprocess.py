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
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer
from trl import pack_dataset

from .common import resolve_workspace_path


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
    dataset = load_dataset(
        path=dataset_cfg.path,
        name=dataset_cfg.name,
        split=dataset_cfg.split,
        streaming=bool(dataset_cfg.streaming),
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


def build_unpacked_records(cfg: DictConfig, tokenizer) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
    text_fields = [str(x) for x in cfg.data.text_fields]
    source_column = cfg.data.get("source_column")
    id_column = cfg.data.get("id_column")
    default_source = str(cfg.data.dataset.path)

    records: list[dict[str, Any]] = []
    doc_count = 0
    skipped_empty = 0
    skipped_short = 0
    chunk_count = 0

    for row_idx, row in iter_dataset_rows(cfg.data.dataset):
        text = extract_text(row, text_fields=text_fields)
        if text is None:
            skipped_empty += 1
            continue

        if len(text) < min_chars:
            skipped_short += 1
            continue

        doc_count += 1
        source = str(row.get(source_column, default_source)) if source_column else default_source
        doc_id = str(row.get(id_column, f"row:{row_idx}")) if id_column else f"row:{row_idx}"

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

    unpacked_records, stats = build_unpacked_records(cfg, tokenizer=tokenizer)
    if not unpacked_records:
        raise RuntimeError("No records built from dataset. Check data.text_fields and filtering settings.")

    unpacked_ds = Dataset.from_list(unpacked_records)
    final_ds = maybe_pack_dataset(unpacked_ds, cfg)
    train_ds, val_ds = split_dataset(final_ds, cfg)

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

    metadata = {
        "source": str(cfg.data.dataset.path),
        "split": str(cfg.data.dataset.split),
        "text_fields": [str(x) for x in cfg.data.text_fields],
        "packing": {
            "enabled": bool(cfg.preprocessing.packing.enabled),
            "strategy": str(cfg.preprocessing.packing.strategy),
            "max_length": int(cfg.preprocessing.packing.max_length),
        },
        "stats": {
            **stats,
            "unpacked_rows": len(unpacked_ds),
            "final_rows": len(final_ds),
            "train_rows": len(train_ds),
            "val_rows": len(val_ds) if val_ds is not None else 0,
        },
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("Preprocess Complete")
    print("=" * 80)
    print(f"unpacked rows: {len(unpacked_ds):,}")
    print(f"final rows   : {len(final_ds):,}")
    print(f"train rows   : {len(train_ds):,}")
    print(f"val rows     : {len(val_ds):,}" if val_ds is not None else "val rows     : 0")
    print(f"saved to     : {output_dir}")


if __name__ == "__main__":
    main()
