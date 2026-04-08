from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from .common import resolve_workspace_path, suppress_noisy_library_logs


def compose_cfg(config_path: str, config_name: str):
    cfg_dir = resolve_workspace_path(config_path)
    with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
        cfg = compose(config_name=config_name, overrides=[])
    OmegaConf.resolve(cfg)
    return cfg


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.split("-", 1)[1])
    except Exception:
        return -1


def resolve_upload_folder(output_dir: Path, checkpoint: str) -> Path:
    ckpt = checkpoint.strip()
    if ckpt.lower() in {"", "final", "none", "null"}:
        return output_dir

    if ckpt.lower() == "latest":
        candidates = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
        if not candidates:
            raise FileNotFoundError(f"No checkpoint-* directories found in: {output_dir}")
        return max(candidates, key=_checkpoint_step)

    as_path = Path(ckpt)
    if as_path.is_absolute():
        if not as_path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {as_path}")
        return as_path

    candidate = output_dir / ckpt
    if not candidate.exists():
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")
    return candidate


def main() -> None:
    suppress_noisy_library_logs()

    parser = argparse.ArgumentParser(description="Upload trained model/checkpoint to Hugging Face Hub")
    parser.add_argument("--config-path", default="../configs")
    parser.add_argument("--config-name", default="config")
    parser.add_argument("--repo", required=True, help="HF repo id, e.g. your-name/your-model")
    parser.add_argument(
        "--checkpoint",
        default="final",
        help="final | latest | checkpoint-XXXX | absolute path",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--commit-message", default=None)
    args = parser.parse_args()

    cfg = compose_cfg(args.config_path, args.config_name)
    output_dir = resolve_workspace_path(cfg.training.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Training output dir not found: {output_dir}")

    upload_folder = resolve_upload_folder(output_dir, args.checkpoint)

    api = HfApi()
    api.create_repo(repo_id=args.repo, repo_type="model", private=bool(args.private), exist_ok=True)

    message = args.commit_message
    if not message:
        if upload_folder == output_dir:
            message = f"Upload final model from {cfg.experiment.name}"
        else:
            message = f"Upload checkpoint {upload_folder.name} from {cfg.experiment.name}"

    print("=" * 80)
    print("Push To Hub")
    print("=" * 80)
    print(f"repo: {args.repo}")
    print(f"source: {upload_folder}")
    print(f"private: {bool(args.private)}")
    print(f"commit: {message}")

    api.upload_folder(
        repo_id=args.repo,
        folder_path=str(upload_folder),
        repo_type="model",
        commit_message=message,
    )

    print("=" * 80)
    print("Upload Complete")
    print("=" * 80)
    print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
