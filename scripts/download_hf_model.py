#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model snapshot for a local provider workflow.")
    parser.add_argument("--provider", required=True, choices=["mlx", "jax", "llamacpp"])
    parser.add_argument("--model-id", required=True, help="Hugging Face model id, for example Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", default="main", help="Optional Hugging Face revision.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where model snapshots should be stored. Defaults to the repo-local models/ directory.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help="Optional Hugging Face allow pattern. Pass multiple times to download only selected files.",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Optional exact filename to download with hf_hub_download. Useful for a single GGUF artifact.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Install `huggingface_hub` first, for example: pip install huggingface_hub"
        ) from exc

    provider_dir = Path(args.output_dir).expanduser().resolve() / args.provider / args.model_id.replace("/", "--")
    provider_dir.mkdir(parents=True, exist_ok=True)
    if args.file:
        local_path = hf_hub_download(
            repo_id=args.model_id,
            filename=args.file,
            revision=args.revision,
            local_dir=str(provider_dir),
        )
    else:
        download_kwargs = {
            "repo_id": args.model_id,
            "revision": args.revision,
            "local_dir": str(provider_dir),
            "local_dir_use_symlinks": False,
        }
        if args.allow_pattern:
            download_kwargs["allow_patterns"] = args.allow_pattern
        local_path = snapshot_download(**download_kwargs)
    print(f"Downloaded {args.model_id} for provider={args.provider} into {local_path}")

    if args.provider == "mlx":
        print("Next step: convert or prepare the checkpoint with your MLX workflow if needed.")
    elif args.provider == "jax":
        print("Next step: wire the downloaded checkpoint into your JAX inference codepath.")
    elif args.provider == "llamacpp":
        print("Next step: verify that the downloaded files include the GGUF artifact expected by llama.cpp.")


if __name__ == "__main__":
    main()
