#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model snapshot for a local provider workflow.")
    parser.add_argument("--provider", required=True, choices=["mlx", "jax", "llamacpp"])
    parser.add_argument("--model-id", required=True, help="Hugging Face model id, for example Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", default="main", help="Optional Hugging Face revision.")
    parser.add_argument("--output-dir", default="models", help="Directory where model snapshots should be stored.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Install `huggingface_hub` first, for example: pip install huggingface_hub"
        ) from exc

    provider_dir = Path(args.output_dir) / args.provider / args.model_id.replace("/", "--")
    provider_dir.mkdir(parents=True, exist_ok=True)
    local_path = snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=str(provider_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded {args.model_id} for provider={args.provider} into {local_path}")

    if args.provider == "mlx":
        print("Next step: convert or prepare the checkpoint with your MLX workflow if needed.")
    elif args.provider == "jax":
        print("Next step: wire the downloaded checkpoint into your JAX inference codepath.")
    elif args.provider == "llamacpp":
        print("Next step: verify that the downloaded files include the GGUF artifact expected by llama.cpp.")


if __name__ == "__main__":
    main()

