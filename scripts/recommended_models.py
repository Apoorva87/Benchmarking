#!/usr/bin/env python3
from __future__ import annotations


DEFAULT_JAX_MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"
DEFAULT_LLAMACPP_MODEL = "bartowski/Qwen_Qwen3.5-35B-A3B-GGUF"
DEFAULT_OLLAMA_MODEL = "qwen3.5:35b-a3b"


def main() -> None:
    print("Recommended baseline model family for cross-provider text benchmarking:")
    print("Qwen3.5-35B-A3B")
    print("")
    print("Provider mappings:")
    print(f"jax: {DEFAULT_JAX_MODEL}")
    print(f"llama.cpp: {DEFAULT_LLAMACPP_MODEL}")
    print(f"ollama: {DEFAULT_OLLAMA_MODEL}")
    print("")
    print("Use the same model family across providers and record the exact format or quantization used.")


if __name__ == "__main__":
    main()
