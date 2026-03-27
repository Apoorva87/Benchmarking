# Architecture Notes

## Core idea

The framework separates three concerns:

1. Providers know how to talk to a local runtime.
2. Benchmarks know how to build prompts, invoke a provider capability, and evaluate responses.
3. Scoring normalizes benchmark-specific judgments into a common result shape.

For text providers, there is now a second split:

1. Functional generation for correctness or fidelity benchmarks
2. Performance measurement for throughput-oriented benchmarks

## Extension model

### Adding a new provider

- Implement one or more provider interfaces from `genAI.providers.base`
- Register it in `genAI.providers.registry`
- Keep transport details local to the provider implementation
- Add a provider setup message so operators know whether the model should be downloaded locally or hosted externally

### Adding a new benchmark

- Subclass the modality-specific benchmark base class
- Supply samples and evaluation logic
- Return standardized `ScoreBreakdown` instances so reports stay comparable

## Current benchmark families

### Token generation performance

Measures:

- time to first token
- output token count
- token generation rate
- total duration

This benchmark is intentionally separate from quality scoring so we can compare runtimes such as Ollama, MLX, JAX, LM Studio, and llama.cpp on the same prompt set.

### Instruction fidelity

Measures:

- exact answer compliance
- keyword compliance
- JSON-like response structure
- general prompt adherence

### Adding image generation later

The framework already has:

- `ImageGenerationProvider`
- `ImageGenerationBenchmark`
- modality-aware result objects

This means image generation evaluation can grow into prompt fidelity, reference similarity, style adherence, or human-review workflows without changing the runner contract.
