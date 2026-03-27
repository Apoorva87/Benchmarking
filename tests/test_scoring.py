from genAI.runners.evaluator import EvaluationRunner
from genAI.scoring.standard import keyword_coverage_score, normalized_exact_match_score
from genAI.suites.basic_qa import BasicQABenchmark


class PerfectTextProvider:
    provider_name = "test-provider"
    model_name = "test-model"

    def generate_text(self, prompt: str, **kwargs: object) -> str:
        if "France" in prompt:
            return "Paris"
        return "Mars"


def test_exact_match_score_normalizes_whitespace_and_case() -> None:
    score = normalized_exact_match_score("Paris", "  paris ")
    assert score.overall == 1.0
    assert score.as_percent() == 100.0


def test_keyword_coverage_score_counts_matches() -> None:
    score = keyword_coverage_score(["receipt", "text"], "The receipt contains text and totals.")
    assert score.overall == 1.0


def test_runner_aggregates_benchmark_results() -> None:
    benchmark = BasicQABenchmark()
    results = benchmark.run(PerfectTextProvider())
    aggregate = EvaluationRunner().aggregate(results)
    assert aggregate.average_score == 1.0
    assert aggregate.sample_count == 2
