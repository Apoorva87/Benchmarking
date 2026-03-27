from __future__ import annotations

from dataclasses import dataclass, field

from genAI.benchmarks.base import Benchmark, BenchmarkSample, Modality
from genAI.providers.base import VisionLanguageProvider


@dataclass
class VisionSample(BenchmarkSample):
    image_paths: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)


class VLMBenchmark(Benchmark[VisionSample, VisionLanguageProvider]):
    def __init__(self, name: str) -> None:
        super().__init__(name=name, modality=Modality.VISION)
