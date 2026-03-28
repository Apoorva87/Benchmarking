from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

from genAI.providers.base import GenerationMetrics, ProviderCapabilities, TextGenerationProvider, VisionLanguageProvider


class OllamaProvider(TextGenerationProvider, VisionLanguageProvider):
    def __init__(self, model_name: str) -> None:
        super().__init__(
            provider_name="ollama",
            model_name=model_name,
            capabilities=ProviderCapabilities(text=True, vision=True),
        )

    def generate_text(self, prompt: str, **kwargs: object) -> str:
        metrics = self.measure_text_generation(prompt, **kwargs)
        return metrics.response_text

    def measure_text_generation(self, prompt: str, **kwargs: object) -> GenerationMetrics:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0),
            },
        }
        if "max_new_tokens" in kwargs:
            payload["options"]["num_predict"] = int(kwargs["max_new_tokens"])
        response = self._post_json("http://127.0.0.1:11434/api/generate", payload)
        return self._parse_generation_response(prompt=prompt, data=response)

    def generate_vision_text(self, prompt: str, image_paths: list[str], **kwargs: object) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "images": [self._encode_image(path) for path in image_paths],
            "options": {
                "temperature": kwargs.get("temperature", 0),
            },
        }
        response = self._post_json("http://127.0.0.1:11434/api/generate", payload)
        return str(response.get("response", "")).strip()

    def _post_json(self, url: str, payload: dict[str, object]) -> dict[str, object]:
        command = [
            "curl",
            "-s",
            url,
            "-d",
            json.dumps(payload),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Ollama request failed.\n"
                f"command={' '.join(command)}\n"
                f"stderr={completed.stderr.strip()}"
            )
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid Ollama JSON response: {completed.stdout}") from exc

    def _parse_generation_response(self, prompt: str, data: dict[str, object]) -> GenerationMetrics:
        response_text = str(data.get("response", "")).strip()
        prompt_eval_count = int(data.get("prompt_eval_count", 0) or 0)
        prompt_eval_duration = int(data.get("prompt_eval_duration", 0) or 0)
        eval_count = int(data.get("eval_count", 0) or 0)
        eval_duration = int(data.get("eval_duration", 0) or 0)
        total_duration = int(data.get("total_duration", 0) or 0)
        load_duration = int(data.get("load_duration", 0) or 0)

        prompt_duration_seconds = prompt_eval_duration / 1_000_000_000 if prompt_eval_duration else 0.0
        eval_duration_seconds = eval_duration / 1_000_000_000 if eval_duration else 0.0
        total_duration_seconds = total_duration / 1_000_000_000 if total_duration else prompt_duration_seconds + eval_duration_seconds

        ttft_seconds = prompt_duration_seconds
        if eval_count > 0 and eval_duration_seconds > 0:
            ttft_seconds += eval_duration_seconds / eval_count

        token_generation_rate = eval_count / eval_duration_seconds if eval_count > 0 and eval_duration_seconds > 0 else 0.0

        notes = ["Metrics parsed from Ollama generate response."]
        if load_duration:
            notes.append(f"Model load duration reported: {load_duration / 1_000_000_000:.4f}s.")
        if data.get("thinking"):
            notes.append("Ollama returned a separate reasoning/thinking field.")

        return GenerationMetrics(
            prompt_char_count=len(prompt),
            output_token_count=eval_count,
            time_to_first_token_seconds=ttft_seconds,
            token_generation_rate=token_generation_rate,
            total_duration_seconds=total_duration_seconds,
            response_text=response_text,
            metric_notes=notes,
        )

    def _encode_image(self, image_path: str) -> str:
        payload = Path(image_path).expanduser().read_bytes()
        return base64.b64encode(payload).decode("ascii")

    def setup_message(self) -> str:
        return (
            f"Ollama expects the model '{self.model_name}' to be available locally and hosted by Ollama. "
            "Recommended large-model tag: `qwen3.5:35b-a3b` "
            "(https://ollama.com/library/qwen3.5:35b-a3b). "
            f"Typical setup: `ollama pull {self.model_name}` and then run prompts against the local Ollama service."
        )
