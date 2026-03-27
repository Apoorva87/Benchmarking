from __future__ import annotations

from collections.abc import Callable

from genAI.benchmarks.base import Benchmark
from genAI.suites.basic_qa import BasicQABenchmark
from genAI.suites.caption_keywords import CaptionKeywordBenchmark
from genAI.suites.instruction_fidelity import InstructionFidelityBenchmark
from genAI.suites.token_generation_speed import TokenGenerationSpeedBenchmark


SuiteFactory = Callable[[], Benchmark]


_SUITE_FACTORIES: dict[str, SuiteFactory] = {
    "basic-qa": BasicQABenchmark,
    "caption-keywords": CaptionKeywordBenchmark,
    "instruction-fidelity": InstructionFidelityBenchmark,
    "token-generation-speed": TokenGenerationSpeedBenchmark,
}


def list_suites() -> list[str]:
    return sorted(_SUITE_FACTORIES)


def build_suite(suite_name: str) -> Benchmark:
    try:
        factory = _SUITE_FACTORIES[suite_name.lower()]
    except KeyError as exc:
        supported = ", ".join(list_suites())
        raise ValueError(f"Unknown suite '{suite_name}'. Supported suites: {supported}") from exc
    return factory()
