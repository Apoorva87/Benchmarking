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

    def setup_message(self) -> str:
        return (
            f"MLX can use a local Hugging Face checkpoint for '{self.model_name}'. "
            "Recommended examples: `mlx-community/Qwen3.5-35B-A3B-8bit` and "
            "`mlx-community/gpt-oss-120b-MXFP4-Q4`. "
            "Use `python scripts/download_hf_model.py --provider mlx --model-id <hf-model-id>` to fetch assets."
        )
