# scp_stage1_cpt

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

git clone https://github.com/contiloop/scp_stage1_cpt.git
cd scp_stage1_cpt
make setup
python -c "from huggingface_hub import login; login(token='hf_xxxxxxx')"
wandb login                # optional
make preprocess

# keep training alive after SSH disconnect
tmux new -s train
make train config=full
```

## Quick Start (Local)

```bash
cd scp_stage1_cpt
make setup
make preprocess
make train config=full
make eval
```

## What Each Make Target Does

- `make preprocess`: downloads/loads raw dataset and writes processed train/val dataset
- `make train`: runs CPT training from config (`config=...` selects config file)
- `make train-resume`: resumes from latest checkpoint (`training.resume_from_checkpoint=auto`)
- `make eval`: runs validation PPL for base + CPT model, then lm-eval on CPT model
- `make eval-benchmarks`: benchmark-only, CPT model only
- `make eval-benchmarks-base`: benchmark-only, base model only
- `make eval-benchmarks-both`: benchmark-only, base + CPT

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

- Dataset source: `alwaysgood/korean-financial-cpt`
- Backend: Unsloth (`FastLanguageModel` / `FastVisionModel`)
- Logging: W&B
- Input embeddings are frozen by default (`training.freeze_embeddings=true`)
- Runtime packing (train-time): `training.runtime_packing.enabled=true` (`bfd_split`, `padding_free=true`)
- Preprocessing packing: `preprocessing.packing.enabled=false` (store unpacked records)
- Checkpoint policy: `save_strategy=steps`, `save_steps=500`, `save_total_limit=3`

## Config Usage

```bash
# default (configs/config.yaml)
make train

# full-weight (configs/full.yaml)
make train config=full

# GPU memory presets (full-weight)
make train config=full_48gb
make train config=full_80gb
make train config=full_96gb

# GPU-specific training-time eval OOM probe presets
make train config=full_eval_oom_probe_48gb
make train config=full_eval_oom_probe_80gb
make train config=full_eval_oom_probe_96gb

# LoRA (configs/lora.yaml)
make train config=lora

# resume from last checkpoint
make train-resume config=full
```

GPU preset summary (Qwen/Gemma 4B, seq_len=4096 baseline):

- `full_48gb`: train batch `2`, grad accum `16`, train-eval batch `2`, offline eval batch `4`
- `full_80gb`: train batch `4`, grad accum `8`, train-eval batch `4`, offline eval batch `8`
- `full_96gb`: train batch `8`, grad accum `4`, train-eval batch `8`, offline eval batch `12`
- `full_eval_oom_probe_48gb`: `gpu48` + `training.{max_steps=100, eval_steps=1, save_steps=10}`
- `full_eval_oom_probe_80gb`: `gpu80` + `training.{max_steps=100, eval_steps=1, save_steps=10}`
- `full_eval_oom_probe_96gb`: `gpu96` + `training.{max_steps=100, eval_steps=1, save_steps=10}`

Create a new file under `configs/` when you need a new experiment setup, then run:

```bash
make train config=<file_name_without_yaml>
```
