from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from genAI.benchmarks.base import Benchmark, BenchmarkSample, Modality
from genAI.providers.base import ImageGenerationProvider


@dataclass
class ImageGenerationSample(BenchmarkSample):
    reference_image: Optional[str] = None


class ImageGenerationBenchmark(Benchmark[ImageGenerationSample, ImageGenerationProvider]):
    def __init__(self, name: str) -> None:
        super().__init__(name=name, modality=Modality.IMAGE_GENERATION)
