from __future__ import annotations

import json
from pathlib import Path

from genAI.benchmarks.base import BenchmarkExecution
from genAI.benchmarks.llm import LLMBenchmark, PerformanceTextSample
from genAI.providers.base import TextGenerationProvider
from genAI.scoring.models import ScoreBreakdown


class TokenGenerationSpeedBenchmark(LLMBenchmark):
    def __init__(self) -> None:
        super().__init__(name="token-generation-speed")
        self._samples = self._load_samples()

    def _load_samples(self) -> list[PerformanceTextSample]:
        data_path = Path(__file__).resolve().parents[1] / "data" / "token_generation_speed.json"
        raw_samples = json.loads(data_path.read_text())
        return [PerformanceTextSample(**item) for item in raw_samples]

    def samples(self) -> list[PerformanceTextSample]:
        return self._samples

    def run_sample(self, provider: TextGenerationProvider, sample: PerformanceTextSample) -> BenchmarkExecution:
        metrics = provider.measure_text_generation(sample.prompt, max_new_tokens=sample.max_new_tokens)
        score = ScoreBreakdown(
            overall=1.0,
            rubric={"measurement_completed": 1.0},
            notes=["Performance metrics captured for this prompt size."],
        )
        return BenchmarkExecution(
            response=metrics.response_text,
            score=score,
            metadata={
                "prompt_size": sample.size_label,
                "prompt_char_count": metrics.prompt_char_count,
                "output_token_count": metrics.output_token_count,
                "time_to_first_token_seconds": metrics.time_to_first_token_seconds,
                "token_generation_rate": metrics.token_generation_rate,
                "total_duration_seconds": metrics.total_duration_seconds,
                "metric_notes": metrics.metric_notes,
            },
        )

