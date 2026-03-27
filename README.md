# Local GenAI Benchmarking

This repository is a modular benchmarking framework for evaluating local inference stacks and local models across multiple modalities.

The initial design supports:

- LLM benchmarking for text generation tasks
- VLM benchmarking for text-plus-image reasoning tasks
- Future image generation benchmarking for prompt-to-image evaluation
- Provider adapters for Ollama, MLX, LM Studio, and future runtimes

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
python -m genAI.cli --provider ollama --model llama3.2 --suite basic-qa
```

## What is implemented now

- Shared benchmark and scoring primitives
- Provider base classes for text, vision, and image generation
- Initial adapters for Ollama, MLX, and LM Studio
- Example LLM suite using QA-style prompts
- Example VLM suite using image-aware prompts and metadata fixtures
- Execution engine that normalizes results across benchmark types

## Planned next steps

- Add richer rubric scoring with LLM-as-judge support
- Add latency, throughput, token usage, and cost-style local metrics
- Add image generation quality benchmarks using reference images and prompt fidelity checks
- Add dataset packs and report export formats
- Add provider health checks and batch execution

