# scp_stage1

Independent CPT pipeline using Hydra + Unsloth + W&B.

## Structure

- `configs/*`: data/model/finetune/training/logging configs
- `src/preprocess.py`: preprocessing entrypoint
- `src/train.py`: training entrypoint
- `Makefile`: independent run targets

## Quick Start

```bash
cd scp_stage1
make setup
make preprocess
make train
make eval
make eval-benchmarks
make eval-benchmarks-base
```

## Key Defaults

- Dataset source: `alwaysgood/korean-financial-cpt`
- Backend: Unsloth (`FastLanguageModel` / `FastVisionModel`)
- Logging: W&B
- Input embeddings are frozen by default (`training.freeze_embeddings=true`)
- Runtime packing (train-time): `training.runtime_packing.enabled=true` (`bfd_split`, `padding_free=true`)
- Preprocessing packing: `preprocessing.packing.enabled=false` (store unpacked records)
- Checkpoint policy: `save_strategy=steps`, `save_steps=500`, `save_total_limit=3`

## Useful Configs

```bash
# default (configs/config.yaml)
make train

# full-weight (configs/full.yaml)
make train config=full

# resume from last checkpoint
make train-resume config=full

# run benchmarks only (lm-eval)
make eval-benchmarks

# run base-only benchmarks
make eval-benchmarks-base

# reduce benchmark size for quick checks
make eval-benchmarks limit=100

# run base + cpt benchmarks together
make eval-benchmarks-both

# evaluate all checkpoints in output_dir
make eval-all
```

Create a new file under `configs/` when you need a new experiment setup, then run `make train config=<file_name_without_yaml>`.
