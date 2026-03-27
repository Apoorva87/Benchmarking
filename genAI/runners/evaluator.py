from __future__ import annotations

from statistics import mean

from genAI.benchmarks.base import BenchmarkResult
from genAI.scoring.models import AggregateScore


class EvaluationRunner:
    def aggregate(self, results: list[BenchmarkResult]) -> AggregateScore:
        if not results:
            raise ValueError("Cannot aggregate an empty result set.")
        first = results[0]
        return AggregateScore(
            benchmark_name=first.benchmark_name,
            provider_name=first.provider_name,
            model_name=first.model_name,
            modality=first.modality.value,
            average_score=mean(result.score.overall for result in results),
            sample_count=len(results),
            average_latency_seconds=mean(result.latency_seconds for result in results),
        )

