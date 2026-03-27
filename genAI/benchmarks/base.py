from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter
from typing import Any, Generic, Iterable, TypeVar

from genAI.scoring.models import ScoreBreakdown


class Modality(str, Enum):
    TEXT = "text"
    VISION = "vision"
    IMAGE_GENERATION = "image_generation"


@dataclass
class BenchmarkSample:
    sample_id: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    benchmark_name: str
    provider_name: str
    model_name: str
    modality: Modality
    sample_id: str
    response: Any
    score: ScoreBreakdown
    latency_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkExecution:
    response: Any
    score: ScoreBreakdown
    metadata: dict[str, Any] = field(default_factory=dict)


SampleT = TypeVar("SampleT", bound=BenchmarkSample)
ProviderT = TypeVar("ProviderT")


class Benchmark(ABC, Generic[SampleT, ProviderT]):
    name: str
    modality: Modality

    def __init__(self, name: str, modality: Modality) -> None:
        self.name = name
        self.modality = modality

    @abstractmethod
    def samples(self) -> Iterable[SampleT]:
        """Return the benchmark samples."""

    @abstractmethod
    def run_sample(self, provider: ProviderT, sample: SampleT) -> BenchmarkExecution:
        """Execute one sample and return a response, standardized score, and metadata."""

    def run(self, provider: ProviderT) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []
        provider_name = getattr(provider, "provider_name", provider.__class__.__name__)
        model_name = getattr(provider, "model_name", "unknown")
        for sample in self.samples():
            started_at = perf_counter()
            execution = self.run_sample(provider, sample)
            latency_seconds = perf_counter() - started_at
            results.append(
                BenchmarkResult(
                    benchmark_name=self.name,
                    provider_name=provider_name,
                    model_name=model_name,
                    modality=self.modality,
                    sample_id=sample.sample_id,
                    response=execution.response,
                    score=execution.score,
                    latency_seconds=latency_seconds,
                    metadata={**sample.metadata, **execution.metadata},
                )
            )
        return results
