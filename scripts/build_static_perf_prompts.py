#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = REPO_ROOT / "genAI" / "data" / "perf_prompts"


def build_small() -> str:
    return (
        "You are helping evaluate local inference engines. "
        "Summarize the tradeoffs between latency, throughput, accuracy, reproducibility, "
        "memory pressure, warmup effects, and operational simplicity in a concise but precise way. "
        "Keep the answer factual, organized, and useful for an engineer comparing providers."
    )


def build_medium() -> str:
    paragraph = (
        "Benchmarking local language models requires disciplined control over prompts, token limits, warmup runs, "
        "sampler settings, concurrency, device placement, and cache behavior. The operator should keep one model family "
        "constant where possible, record quantization and runtime versions, and preserve prompt text exactly across runs. "
        "Meaningful comparisons should include time to first token, steady-state token throughput, total completion time, "
        "memory pressure, and notes about quality regressions or instruction-following failures. "
    )
    return ("\n".join(paragraph for _ in range(18))).strip()


def build_large() -> str:
    paragraph = (
        "Create a detailed technical memo for engineers designing a reusable benchmark harness for local generative models. "
        "The memo should cover prompt curation, deterministic replay, batch sizing, device utilization, memory residency, "
        "sampling policy, cache reuse, thermal impact, power draw, tokenizer effects, system observability, failure recovery, "
        "and long-context stress testing. For each topic, explain why the variable matters, how it can confound cross-provider "
        "comparisons, and how a modular benchmark tool should expose it in configuration and reporting. "
        "Also discuss how multimodal models, image generation systems, and future inference stacks can fit into a shared measurement framework. "
    )
    return ("\n".join(paragraph for _ in range(120))).strip()


def main() -> None:
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    (PROMPT_DIR / "small.txt").write_text(build_small())
    (PROMPT_DIR / "medium.txt").write_text(build_medium())
    (PROMPT_DIR / "large.txt").write_text(build_large())
    print(f"Wrote prompts into {PROMPT_DIR}")


if __name__ == "__main__":
    main()
