from __future__ import annotations

from collections.abc import Callable

from genAI.providers.base import BaseProvider
from genAI.providers.jax import JAXProvider
from genAI.providers.llamacpp import LlamaCppProvider
from genAI.providers.lmstudio import LMStudioProvider
from genAI.providers.mlx import MLXProvider
from genAI.providers.ollama import OllamaProvider


ProviderFactory = Callable[[str], BaseProvider]


_PROVIDER_FACTORIES: dict[str, ProviderFactory] = {
    "jax": JAXProvider,
    "llamacpp": LlamaCppProvider,
    "ollama": OllamaProvider,
    "mlx": MLXProvider,
    "lmstudio": LMStudioProvider,
}


def list_provider_factories() -> list[str]:
    return sorted(_PROVIDER_FACTORIES)


def build_provider(provider_name: str, model_name: str) -> BaseProvider:
    try:
        factory = _PROVIDER_FACTORIES[provider_name.lower()]
    except KeyError as exc:
        supported = ", ".join(list_provider_factories())
        raise ValueError(f"Unknown provider '{provider_name}'. Supported providers: {supported}") from exc
    return factory(model_name)
