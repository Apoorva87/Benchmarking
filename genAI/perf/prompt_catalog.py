from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROMPT_ROOT = Path(__file__).resolve().parents[1] / "data" / "perf_prompts"


@dataclass(frozen=True)
class PromptSpec:
    key: str
    label: str
    path: Path
    approximate_tokens: int

    def load_text(self) -> str:
        return self.path.read_text()


def load_prompt_specs() -> list[PromptSpec]:
    return [
        PromptSpec(key="small", label="Small (<=100 tokens)", path=PROMPT_ROOT / "small.txt", approximate_tokens=80),
        PromptSpec(key="medium", label="Medium (~1K tokens)", path=PROMPT_ROOT / "medium.txt", approximate_tokens=950),
        PromptSpec(key="large", label="Large (10K+ tokens)", path=PROMPT_ROOT / "large.txt", approximate_tokens=12000),
    ]

