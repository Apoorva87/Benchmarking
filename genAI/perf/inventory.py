from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPO_ROOT / "models"


@dataclass(frozen=True)
class RunnableModel:
    provider: str
    label: str
    model_ref: str
    source: str


def _discover_llamacpp_models() -> list[RunnableModel]:
    root = MODELS_ROOT / "llamacpp"
    if not root.exists():
        return []
    models: list[RunnableModel] = []
    for path in sorted(root.rglob("*.gguf")):
        if path.name.startswith("mmproj-"):
            continue
        label = str(path.relative_to(root))
        models.append(
            RunnableModel(
                provider="llamacpp",
                label=label,
                model_ref=str(path),
                source="repo-local-gguf",
            )
        )
    return models


def _discover_mlx_models() -> list[RunnableModel]:
    root = MODELS_ROOT / "mlx"
    if not root.exists():
        return []
    models: list[RunnableModel] = []
    for path in sorted(root.rglob("config.json")):
        model_dir = path.parent
        if not (model_dir / "tokenizer.json").exists():
            continue
        if not any(model_dir.glob("*.safetensors")):
            continue
        label = str(model_dir.relative_to(root))
        models.append(
            RunnableModel(
                provider="mlx",
                label=label,
                model_ref=str(model_dir),
                source="repo-local-mlx",
            )
        )
    return models


def _discover_ollama_models() -> list[RunnableModel]:
    command = ["curl", "-s", "http://127.0.0.1:11434/api/tags"]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not completed.stdout.strip():
        return []
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return []
    models: list[RunnableModel] = []
    for model in payload.get("models", []):
        name = model.get("name")
        if not name:
            continue
        details = model.get("details", {})
        family = details.get("family") or "ollama"
        label = f"{name} [{family}]"
        models.append(
            RunnableModel(
                provider="ollama",
                label=label,
                model_ref=name,
                source="ollama-runtime",
            )
        )
    return models


def discover_runnable_models() -> dict[str, list[RunnableModel]]:
    inventory: dict[str, list[RunnableModel]] = {
        "llamacpp": _discover_llamacpp_models(),
        "mlx": _discover_mlx_models(),
        "ollama": _discover_ollama_models(),
    }
    return {provider: models for provider, models in inventory.items() if models}


def inventory_as_jsonable() -> dict[str, list[dict[str, str]]]:
    return {provider: [asdict(model) for model in models] for provider, models in discover_runnable_models().items()}

