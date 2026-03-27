from genAI.runners.evaluator import EvaluationRunner
from genAI.providers.llamacpp import LlamaCppProvider
from genAI.scoring.standard import instruction_fidelity_score, keyword_coverage_score, normalized_exact_match_score
from genAI.suites.basic_qa import BasicQABenchmark
from genAI.suites.token_generation_speed import TokenGenerationSpeedBenchmark


class PerfectTextProvider:
    provider_name = "test-provider"
    model_name = "test-model"

    def generate_text(self, prompt: str, **kwargs: object) -> str:
        if "France" in prompt:
            return "Paris"
        return "Mars"

    def measure_text_generation(self, prompt: str, **kwargs: object):
        from genAI.providers.base import GenerationMetrics

        return GenerationMetrics(
            prompt_char_count=len(prompt),
            output_token_count=42,
            time_to_first_token_seconds=0.1,
            token_generation_rate=120.0,
            total_duration_seconds=0.35,
            response_text="synthetic response",
            metric_notes=["Synthetic performance metrics for testing."],
        )


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


def test_instruction_fidelity_score_tracks_constraints() -> None:
    score = instruction_fidelity_score(
        '{"status":"ok","provider":"mlx"}',
        expected_keywords=["ok"],
        required_json_keys=["status", "provider"],
    )
    assert score.overall == 1.0


def test_speed_benchmark_records_metrics() -> None:
    benchmark = TokenGenerationSpeedBenchmark()
    results = benchmark.run(PerfectTextProvider())
    assert len(results) == 3
    assert results[0].metadata["time_to_first_token_seconds"] == 0.1
    assert results[0].metadata["token_generation_rate"] == 120.0


def test_llamacpp_command_uses_hf_repo_for_named_model() -> None:
    provider = LlamaCppProvider("bartowski/Qwen_Qwen3.5-35B-A3B-GGUF")
    command = provider._build_command(prompt="hello", max_new_tokens=32)
    assert "--hf-repo" in command
    assert "bartowski/Qwen_Qwen3.5-35B-A3B-GGUF" in command


def test_llamacpp_timing_parser_extracts_metrics() -> None:
    provider = LlamaCppProvider("dummy")
    output = """
load_backend: loaded BLAS backend
Hello world
llama_print_timings: prompt eval time =    2000.00 ms /   20 tokens
llama_print_timings:        eval time =    5000.00 ms /   50 runs
llama_print_timings:       total time =    7100.00 ms
"""
    metrics = provider._parse_generation_output(prompt="Prompt", output=output)
    assert metrics.response_text == "Hello world"
    assert metrics.output_token_count == 50
    assert round(metrics.time_to_first_token_seconds, 2) == 2.10
    assert round(metrics.token_generation_rate, 2) == 10.0
    assert round(metrics.total_duration_seconds, 2) == 7.1
