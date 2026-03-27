#!/usr/bin/env python3
from __future__ import annotations


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main() -> None:
    print("Recommended baseline model family for cross-provider text benchmarking:")
    print(DEFAULT_MODEL)
    print("Use the same or closest available model across providers where possible.")
    print("For Ollama or LM Studio, host an equivalent local model before running benchmarks.")


if __name__ == "__main__":
    main()
