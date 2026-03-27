#!/usr/bin/env python3
from __future__ import annotations


QWEN_MODELS = {
    "jax": "Qwen/Qwen3.5-35B-A3B-FP8",
    "mlx": "mlx-community/Qwen3.5-35B-A3B-8bit",
    "llama.cpp": "bartowski/Qwen_Qwen3.5-35B-A3B-GGUF",
    "ollama": "qwen3.5:35b-a3b",
}

GPT_OSS_MODELS = {
    "jax": "openai/gpt-oss-120b",
    "mlx": "mlx-community/gpt-oss-120b-MXFP4-Q4",
    "llama.cpp": "bartowski/openai_gpt-oss-120b-GGUF",
    "ollama": "gpt-oss:120b",
}


def main() -> None:
    print("Recommended large-model mappings for cross-provider text benchmarking:")
    print("")
    print("Qwen3.5-35B-A3B")
    for provider, model in QWEN_MODELS.items():
        print(f"{provider}: {model}")
    print("")
    print("gpt-oss-120b")
    for provider, model in GPT_OSS_MODELS.items():
        print(f"{provider}: {model}")
    print("")
    print("Use the same model family across providers and record the exact format or quantization used.")


if __name__ == "__main__":
    main()
