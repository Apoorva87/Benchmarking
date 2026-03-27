from __future__ import annotations

from genAI.providers.base import ProviderCapabilities, TextGenerationProvider, VisionLanguageProvider


class LMStudioProvider(TextGenerationProvider, VisionLanguageProvider):
    def __init__(self, model_name: str) -> None:
        super().__init__(
            provider_name="lmstudio",
            model_name=model_name,
            capabilities=ProviderCapabilities(text=True, vision=True),
        )

    def generate_text(self, prompt: str, **kwargs: object) -> str:
        return f"[lmstudio:{self.model_name}] {prompt}"

    def generate_vision_text(self, prompt: str, image_paths: list[str], **kwargs: object) -> str:
        image_count = len(image_paths)
        return f"[lmstudio:{self.model_name}:images={image_count}] {prompt}"

