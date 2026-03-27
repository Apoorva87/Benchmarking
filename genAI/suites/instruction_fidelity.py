from __future__ import annotations

import json
from pathlib import Path

from genAI.benchmarks.base import BenchmarkExecution
from genAI.benchmarks.llm import LLMBenchmark, TextSample
from genAI.providers.base import TextGenerationProvider
from genAI.scoring.standard import instruction_fidelity_score


class InstructionFidelityBenchmark(LLMBenchmark):
    def __init__(self) -> None:
        super().__init__(name="instruction-fidelity")
        self._samples = self._load_samples()

    def _load_samples(self) -> list[TextSample]:
        data_path = Path(__file__).resolve().parents[1] / "data" / "instruction_fidelity.json"
        raw_samples = json.loads(data_path.read_text())
        return [TextSample(**item) for item in raw_samples]

    def samples(self) -> list[TextSample]:
        return self._samples

    def run_sample(self, provider: TextGenerationProvider, sample: TextSample) -> BenchmarkExecution:
        response = provider.generate_text(sample.prompt)
        score = instruction_fidelity_score(
            response,
            expected_keywords=sample.metadata.get("expected_keywords", []),
            max_words=sample.metadata.get("max_words"),
            required_json_keys=sample.metadata.get("required_json_keys", []),
        )
        return BenchmarkExecution(response=response, score=score)

