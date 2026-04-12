# scp_stage2_pd

Independent CPT pipeline using Hydra + Unsloth + W&B.

## Structure

- `configs/*`: data/model/finetune/training/logging configs
- `src/preprocess.py`: preprocessing entrypoint
- `src/train.py`: training entrypoint
- `src/evaluate.py`: perplexity + benchmark evaluation entrypoint
- `Makefile`: independent run targets

## Vast.ai / RunPod (Recommended)

```bash
# Docker image: unsloth/unsloth:latest

git clone https://github.com/contiloop/scp_stage2_pd.git
cd scp_stage2_pd
make set                   # includes causal_conv1d kernel check (rebuilds only if needed)
python -c "from huggingface_hub import login; login(token='hf_xxxxxxx')"
wandb login                # optional
make preprocess

# keep training alive after SSH disconnect
tmux new -s train
make train config=full
```

## Quick Start (Local)

```bash
cd scp_stage2_pd
make set
make preprocess
make train config=full
make eval
```

For Gemma-only workflows (no `causal_conv1d` dependency), you can skip that setup:

```bash
make set SKIP_CAUSAL_CONV1D=1
```

## What Each Make Target Does

- `make preprocess`: downloads/loads raw dataset and writes processed train/val dataset
- `make set`: installs dependencies, checks `causal_conv1d` CUDA kernel runtime, and only rebuilds when needed (Blackwell uses source build fallback)
  - skip `causal_conv1d` setup/verification via `make set SKIP_CAUSAL_CONV1D=1`
  - installs `transformers>=5.4.0` (required for `unsloth/gemma-4-E2B`)
- `make train`: runs CPT training from config (`config=...` selects config file)
- `make train-resume`: resumes from latest checkpoint (`training.resume_from_checkpoint=auto`)
- `make eval`: runs validation PPL for base + CPT model, then lm-eval on CPT model
- `make eval-benchmarks`: benchmark-only, CPT model only
- `make eval-benchmarks-base`: benchmark-only, base model only
- `make eval-benchmarks-both`: benchmark-only, base + CPT
- `make push-to-hub`: upload final model or specific checkpoint to Hugging Face Hub

`limit` controls benchmark sample count (faster smoke tests). Default is `400`:

```bash
make eval-benchmarks limit=400
make eval-benchmarks-base limit=400
make eval-benchmarks-both limit=400
```

`make eval` batch size is controlled separately from training:

```yaml
# configs/evaluation/default.yaml
batch_size: 4
```

Train-time eval batch still uses `training.per_device_eval_batch_size`.

## Key Defaults

- Dataset source: `alwaysgood/wmt24pp-kr` + `alwaysgood/flores-kr`
- Parallel dataset preset: `alwaysgood/wmt24pp-kr` + `alwaysgood/flores-kr` (`data=parallel_wmt24pp_flores`)
- If `data.validation_datasets` is set, preprocessing uses those official validation splits directly.
  If not set, preprocessing falls back to random split (`preprocessing.split.val_ratio`).
- Backend: Unsloth (`FastLanguageModel` / `FastVisionModel`)
- Logging: W&B
- Input embeddings are frozen by default (`training.freeze_embeddings=true`)
- Runtime packing (train-time): `training.runtime_packing.enabled=true` (`bfd_split`, `padding_free=true`)
- Preprocessing packing: `preprocessing.packing.enabled=false` (store unpacked records)
- Checkpoint policy: `save_strategy=steps`, `save_steps=200`, `save_total_limit=1`

## Config Usage

```bash
# default (configs/config.yaml)
make train

# full-weight (configs/full.yaml)
make train config=full

# GPU memory presets (full-weight)
make train config=full_48gb
make train config=full_80gb
make train config=full_96gb_qwen3.5_4b

# switch model profile (example: Gemma 4 E2B)
make train config=full_96gb_qwen3.5_4b ovr="model=gemma_4_e2b"

# full flow on a new instance for Gemma 4 E2B
make set
make preprocess ovr="model=gemma_4_e2b"
make train config=full_96gb_qwen3.5_4b ovr="model=gemma_4_e2b"

# parallel data preprocessing/training (official validation splits from each source)
make preprocess config=full_96gb_qwen3.5_4b
make train config=full_96gb_qwen3.5_4b

# resume from last checkpoint
make train-resume config=full

# upload final output dir (default CKPT=final, eval artifacts included when matched)
make push-to-hub config=full_96gb_qwen3.5_4b HF_REPO=your-name/your-model

# upload specific checkpoint (recommended)
make push-to-hub config=full_96gb_qwen3.5_4b HF_REPO=your-name/your-model CKPT=checkpoint-3500

# model-only upload (skip eval artifacts)
python -m src.push_to_hub --config-path configs --config-name full_96gb_qwen3.5_4b --repo your-name/your-model --checkpoint checkpoint-3500 --no-include-eval
```

GPU preset summary (Qwen/Gemma 4B, seq_len=4096 baseline):

- `full_48gb`: train batch `2`, grad accum `16`, train-eval batch `2`, offline eval batch `4`
- `full_80gb`: train batch `4`, grad accum `8`, train-eval batch `4`, offline eval batch `8`
- `full_96gb_qwen3.5_4b`: train batch `8`, grad accum `4`, train-eval batch `8`, offline eval batch `12`

Create a new file under `configs/` when you need a new experiment setup, then run:

```bash
make train config=<file_name_without_yaml>
```
