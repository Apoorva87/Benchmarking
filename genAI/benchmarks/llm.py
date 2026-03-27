from __future__ import annotations

from dataclasses import dataclass

from genAI.benchmarks.base import Benchmark, BenchmarkSample, Modality
from genAI.providers.base import TextGenerationProvider


@dataclass
class TextSample(BenchmarkSample):
    expected_answer: str = ""


@dataclass
class PerformanceTextSample(BenchmarkSample):
    size_label: str = ""
    max_new_tokens: int = 128


class LLMBenchmark(Benchmark[TextSample, TextGenerationProvider]):
    def __init__(self, name: str) -> None:
        super().__init__(name=name, modality=Modality.TEXT)
