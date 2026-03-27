from __future__ import annotations

from genAI.providers.base import ProviderCapabilities, TextGenerationProvider


class LlamaCppProvider(TextGenerationProvider):
    def __init__(self, model_name: str) -> None:
        super().__init__(
            provider_name="llamacpp",
            model_name=model_name,
            capabilities=ProviderCapabilities(text=True),
        )

    def generate_text(self, prompt: str, **kwargs: object) -> str:
        return f"[llamacpp:{self.model_name}] {prompt}"

    def setup_message(self) -> str:
        return (
            f"llama.cpp expects a local GGUF model for '{self.model_name}'. "
            "Recommended large-model artifact: `bartowski/Qwen_Qwen3.5-35B-A3B-GGUF` "
            "(https://huggingface.co/bartowski/Qwen_Qwen3.5-35B-A3B-GGUF). "
            "Use `python scripts/download_hf_model.py --provider llamacpp --model-id bartowski/Qwen_Qwen3.5-35B-A3B-GGUF` to download local assets."
        )
