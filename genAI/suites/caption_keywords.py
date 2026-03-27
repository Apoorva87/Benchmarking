from __future__ import annotations

import json
from pathlib import Path

from genAI.benchmarks.vlm import VLMBenchmark, VisionSample
from genAI.providers.base import VisionLanguageProvider
from genAI.scoring.models import ScoreBreakdown
from genAI.scoring.standard import keyword_coverage_score


class CaptionKeywordBenchmark(VLMBenchmark):
    def __init__(self) -> None:
        super().__init__(name="caption-keywords")
        self._samples = self._load_samples()

    def _load_samples(self) -> list[VisionSample]:
        data_path = Path(__file__).resolve().parents[1] / "data" / "vlm_caption_keywords.json"
        raw_samples = json.loads(data_path.read_text())
        return [VisionSample(**item) for item in raw_samples]

    def samples(self) -> list[VisionSample]:
        return self._samples

    def run_sample(self, provider: VisionLanguageProvider, sample: VisionSample) -> tuple[str, ScoreBreakdown]:
        response = provider.generate_vision_text(sample.prompt, sample.image_paths)
        score = keyword_coverage_score(sample.expected_keywords, response)
        return response, score

