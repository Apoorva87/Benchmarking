from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderCapabilities:
    text: bool = False
    vision: bool = False
    image_generation: bool = False


class BaseProvider(ABC):
    provider_name: str
    model_name: str
    capabilities: ProviderCapabilities

    def __init__(self, provider_name: str, model_name: str, capabilities: ProviderCapabilities) -> None:
        self.provider_name = provider_name
        self.model_name = model_name
        self.capabilities = capabilities


class TextGenerationProvider(BaseProvider, ABC):
    @abstractmethod
    def generate_text(self, prompt: str, **kwargs: object) -> str:
        """Generate a text response for a prompt."""


class VisionLanguageProvider(BaseProvider, ABC):
    @abstractmethod
    def generate_vision_text(self, prompt: str, image_paths: list[str], **kwargs: object) -> str:
        """Generate a text response using text plus image context."""


class ImageGenerationProvider(BaseProvider, ABC):
    @abstractmethod
    def generate_image(self, prompt: str, **kwargs: object) -> str:
        """Generate an image and return a local artifact reference."""
