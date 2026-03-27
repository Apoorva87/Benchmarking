from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScoreBreakdown:
    overall: float
    rubric: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_percent(self) -> float:
        return round(self.overall * 100, 2)


@dataclass
class AggregateScore:
    benchmark_name: str
    provider_name: str
    model_name: str
    modality: str
    average_score: float
    sample_count: int
    average_latency_seconds: float
