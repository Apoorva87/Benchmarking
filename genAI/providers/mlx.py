from __future__ import annotations

from genAI.providers.base import ProviderCapabilities, TextGenerationProvider


class MLXProvider(TextGenerationProvider):
    def __init__(self, model_name: str) -> None:
        super().__init__(
            provider_name="mlx",
            model_name=model_name,
            capabilities=ProviderCapabilities(text=True),
        )

    def generate_text(self, prompt: str, **kwargs: object) -> str:
        return f"[mlx:{self.model_name}] {prompt}"

