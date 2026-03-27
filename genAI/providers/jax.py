from __future__ import annotations

from genAI.providers.base import ProviderCapabilities, TextGenerationProvider


class JAXProvider(TextGenerationProvider):
    def __init__(self, model_name: str) -> None:
        super().__init__(
            provider_name="jax",
            model_name=model_name,
            capabilities=ProviderCapabilities(text=True),
        )

    def generate_text(self, prompt: str, **kwargs: object) -> str:
        return f"[jax:{self.model_name}] {prompt}"

    def setup_message(self) -> str:
        return (
            f"JAX expects local model artifacts for '{self.model_name}' prepared in a JAX-compatible workflow. "
            "Use `python scripts/download_hf_model.py --provider jax --model-id <hf-model-id>` to fetch a base checkpoint."
        )

