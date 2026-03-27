from __future__ import annotations

import json
from pathlib import Path

from genAI.benchmarks.llm import LLMBenchmark, TextSample
from genAI.providers.base import TextGenerationProvider
from genAI.scoring.models import ScoreBreakdown
from genAI.scoring.standard import normalized_exact_match_score


class BasicQABenchmark(LLMBenchmark):
    def __init__(self) -> None:
        super().__init__(name="basic-qa")
        self._samples = self._load_samples()

    def _load_samples(self) -> list[TextSample]:
        data_path = Path(__file__).resolve().parents[1] / "data" / "llm_basic_qa.json"
        raw_samples = json.loads(data_path.read_text())
        return [TextSample(**item) for item in raw_samples]

    def samples(self) -> list[TextSample]:
        return self._samples

    def run_sample(self, provider: TextGenerationProvider, sample: TextSample) -> tuple[str, ScoreBreakdown]:
        response = provider.generate_text(sample.prompt)
        score = normalized_exact_match_score(sample.expected_answer, response)
        return response, score

