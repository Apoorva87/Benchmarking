from genAI.providers.registry import build_provider, list_provider_factories
from genAI.suites.registry import build_suite, list_suites


def test_provider_registry_lists_supported_backends() -> None:
    assert list_provider_factories() == ["jax", "llamacpp", "lmstudio", "mlx", "ollama"]


def test_suite_registry_lists_supported_suites() -> None:
    assert list_suites() == ["basic-qa", "caption-keywords", "instruction-fidelity", "token-generation-speed"]


def test_build_provider_returns_named_backend() -> None:
    provider = build_provider("ollama", "llama3.2")
    assert provider.provider_name == "ollama"
    assert provider.model_name == "llama3.2"


def test_build_suite_returns_named_benchmark() -> None:
    suite = build_suite("basic-qa")
    assert suite.name == "basic-qa"


def test_provider_setup_message_includes_large_model_guidance() -> None:
    provider = build_provider("ollama", "qwen3.5:35b-a3b")
    assert "qwen3.5:35b-a3b" in provider.setup_message()


def test_mlx_setup_message_mentions_supported_large_models() -> None:
    provider = build_provider("mlx", "mlx-community/Qwen3.5-35B-A3B-8bit")
    message = provider.setup_message()
    assert "Qwen3.5-35B-A3B-8bit" in message
    assert "gpt-oss-120b-MXFP4-Q4" in message
