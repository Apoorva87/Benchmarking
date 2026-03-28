#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = REPO_ROOT / "models"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Symlink an existing LM Studio model into the repo-local models directory.")
    parser.add_argument("--source", required=True, help="Path to an existing LM Studio model file.")
    parser.add_argument("--provider", required=True, choices=["llamacpp", "mlx", "jax", "lmstudio"])
    parser.add_argument("--output-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--alias", default=None, help="Optional alias filename to use inside the repo models directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Source model file does not exist: {source}")

    target_dir = Path(args.output_dir).expanduser().resolve() / args.provider / source.parent.name
    target_dir.mkdir(parents=True, exist_ok=True)
    target_name = args.alias or source.name
    target = target_dir / target_name

    if target.exists() or target.is_symlink():
        if target.resolve() == source:
            print(f"Symlink already exists: {target} -> {source}")
            return
        raise SystemExit(f"Target already exists and points elsewhere: {target}")

    target.symlink_to(source)
    print(f"Created symlink: {target} -> {source}")


if __name__ == "__main__":
    main()
