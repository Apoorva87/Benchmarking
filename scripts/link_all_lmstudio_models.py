#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_LMSTUDIO_DIR = Path.home() / ".lmstudio" / "models"

GGUF_SUFFIXES = {".gguf"}
MLX_SUFFIXES = {".safetensors"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Symlink all supported LM Studio models into the repo-local models directory."
    )
    parser.add_argument("--lmstudio-dir", default=str(DEFAULT_LMSTUDIO_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_MODELS_DIR))
    return parser


def ensure_symlink(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.resolve() == source.resolve():
            return f"exists {target}"
        raise RuntimeError(f"Target already exists and points elsewhere: {target}")
    target.symlink_to(source)
    return f"linked {target}"


def provider_targets_for_file(source: Path) -> list[str]:
    suffix = source.suffix.lower()
    if suffix in GGUF_SUFFIXES:
        return ["llamacpp", "lmstudio"]
    if suffix in MLX_SUFFIXES:
        return ["mlx", "lmstudio"]
    return []


def main() -> None:
    args = build_parser().parse_args()
    lmstudio_dir = Path(args.lmstudio_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not lmstudio_dir.exists():
        raise SystemExit(f"LM Studio directory does not exist: {lmstudio_dir}")

    linked_count = 0
    scanned_count = 0
    for source in sorted(lmstudio_dir.rglob("*")):
        if not source.is_file():
            continue
        scanned_count += 1
        providers = provider_targets_for_file(source)
        if not providers:
            continue
        rel_parent = source.parent.relative_to(lmstudio_dir)
        for provider in providers:
            target = output_dir / provider / rel_parent / source.name
            message = ensure_symlink(source, target)
            print(message)
            if message.startswith("linked "):
                linked_count += 1

    print(f"Scanned {scanned_count} files and created {linked_count} new symlinks.")


if __name__ == "__main__":
    main()
