from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True)
class ProviderCapabilities:
    text: bool = False
    vision: bool = False
    image_generation: bool = False


@dataclass
class GenerationMetrics:
    prompt_char_count: int
    output_token_count: int
    time_to_first_token_seconds: float
    token_generation_rate: float
    total_duration_seconds: float
    response_text: str
    metric_notes: list[str]


class BaseProvider(ABC):
    provider_name: str
    model_name: str
    capabilities: ProviderCapabilities

    def __init__(self, provider_name: str, model_name: str, capabilities: ProviderCapabilities) -> None:
        self.provider_name = provider_name
        self.model_name = model_name
        self.capabilities = capabilities

    def setup_message(self) -> str:
        return "No provider-specific setup notes are registered yet."


class TextGenerationProvider(BaseProvider, ABC):
    @abstractmethod
    def generate_text(self, prompt: str, **kwargs: object) -> str:
        """Generate a text response for a prompt."""

    def measure_text_generation(self, prompt: str, **kwargs: object) -> GenerationMetrics:
        started_at = perf_counter()
        response_text = self.generate_text(prompt, **kwargs)
        total_duration_seconds = perf_counter() - started_at
        output_token_count = max(len(response_text.split()), 1)
        token_generation_rate = output_token_count / max(total_duration_seconds, 1e-9)
        return GenerationMetrics(
            prompt_char_count=len(prompt),
            output_token_count=output_token_count,
            time_to_first_token_seconds=total_duration_seconds,
            token_generation_rate=token_generation_rate,
            total_duration_seconds=total_duration_seconds,
            response_text=response_text,
            metric_notes=[
                "Fallback metric path used because token streaming is not implemented for this provider yet.",
                "Time to first token is approximated as total duration in fallback mode.",
            ],
        )


class VisionLanguageProvider(BaseProvider, ABC):
    @abstractmethod
    def generate_vision_text(self, prompt: str, image_paths: list[str], **kwargs: object) -> str:
        """Generate a text response using text plus image context."""


class ImageGenerationProvider(BaseProvider, ABC):
    @abstractmethod
    def generate_image(self, prompt: str, **kwargs: object) -> str:
        """Generate an image and return a local artifact reference."""
