# Local GenAI Benchmarking

This repository is a modular benchmarking framework for evaluating local inference stacks and local models across multiple modalities.

The initial design supports:

- LLM benchmarking for text generation tasks
- VLM benchmarking for text-plus-image reasoning tasks
- Future image generation benchmarking for prompt-to-image evaluation
- Provider adapters for Ollama, MLX, LM Studio, JAX, llama.cpp, and future runtimes

## Design goals

- Keep providers separate from benchmark definitions
- Standardize scoring so different backends are comparable
- Make it easy to add a new provider or a new benchmark suite independently
- Leave space for multimodal and image generation evaluation without forcing a rewrite later

## Project layout

```text
genAI/
  benchmarks/         Core benchmark abstractions by modality
  data/               Lightweight sample datasets and fixtures
  providers/          Inference client interfaces and implementations
  runners/            Evaluation execution and result aggregation
  scoring/            Standard score objects and utilities
  suites/             Ready-to-run benchmark suites
  cli.py              Simple command line entrypoint
scripts/              Helper scripts for model download and setup
tests/                Unit tests for framework behavior
docs/                 Architecture notes and future roadmap
```

## Quick start

1. Create a virtual environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

2. Run the test suite:

```bash
pytest
```

3. Execute an example benchmark:

```bash
python -m genAI.cli run --provider ollama --model llama3.2 --suite basic-qa
```

4. Inspect provider-specific setup notes:

```bash
python -m genAI.cli provider-info --provider ollama --model llama3.2
```

5. Review the checked-in model mapping for large-model runs:

```bash
cat genAI/data/model_manifest.json
```

## What is implemented now

- Shared benchmark and scoring primitives
- Provider base classes for text, vision, image generation, and performance measurement
- Initial adapters for Ollama, MLX, LM Studio, JAX, and llama.cpp
- Example LLM suite using QA-style prompts
- Example VLM suite using image-aware prompts and metadata fixtures
- Token generation performance benchmarks with small, medium, and large prompts
- Instruction fidelity benchmarks covering exact answers, JSON formatting, and constraint following
- Execution engine that normalizes results across benchmark types
- Setup helper scripts for Hugging Face model download workflows

## Planned next steps

- Add richer rubric scoring with LLM-as-judge support
- Add latency, throughput, token usage, and cost-style local metrics
- Add image generation quality benchmarks using reference images and prompt fidelity checks
- Add dataset packs and report export formats
- Add provider health checks and batch execution

## Benchmark categories

### 1. Token generation performance

This category sends small, medium, and large prompts through each provider and records:

- time to first token
- token generation rate
- total completion time

### 2. Instruction fidelity

This category checks whether a provider follows constraints reliably, such as:

- exact short answers
- required keywords
- JSON-like structured output
- maximum word count or response shape constraints

This is a strong complement to speed because a fast provider is only useful if it also follows the prompt correctly.

## Provider setup notes

- `mlx`, `jax`, and `llama.cpp` can use the helper scripts in `scripts/` to pull Hugging Face assets locally.
- `ollama` and `lmstudio` typically require a hosted local model endpoint first. Use `python -m genAI.cli provider-info --provider <name> --model <model>` to see the expected setup message.

## Current large-model mapping

For the shared 20B to 30B+ benchmarking track, this repo now standardizes on the `Qwen3.5-35B-A3B` family and maps it like this:

- `jax`: `Qwen/Qwen3.5-35B-A3B-FP8`
- `llama.cpp`: `bartowski/Qwen_Qwen3.5-35B-A3B-GGUF`
- `ollama`: `qwen3.5:35b-a3b`

The exact mapping is tracked in [genAI/data/model_manifest.json](/Users/akarnik/experiments/Benchmarking/genAI/data/model_manifest.json).
