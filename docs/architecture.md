# Architecture Notes

## Core idea

The framework separates three concerns:

1. Providers know how to talk to a local runtime.
2. Benchmarks know how to build prompts, invoke a provider capability, and evaluate responses.
3. Scoring normalizes benchmark-specific judgments into a common result shape.

## Extension model

### Adding a new provider

- Implement one or more provider interfaces from `genAI.providers.base`
- Register it in `genAI.providers.registry`
- Keep transport details local to the provider implementation

### Adding a new benchmark

- Subclass the modality-specific benchmark base class
- Supply samples and evaluation logic
- Return standardized `ScoreBreakdown` instances so reports stay comparable

### Adding image generation later

The framework already has:

- `ImageGenerationProvider`
- `ImageGenerationBenchmark`
- modality-aware result objects

This means image generation evaluation can grow into prompt fidelity, reference similarity, style adherence, or human-review workflows without changing the runner contract.

