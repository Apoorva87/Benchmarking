from __future__ import annotations

import re
import subprocess
from pathlib import Path

from genAI.providers.base import GenerationMetrics, ProviderCapabilities, TextGenerationProvider


class LlamaCppProvider(TextGenerationProvider):
    _PROMPT_EVAL_RE = re.compile(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", re.IGNORECASE)
    _EVAL_RE = re.compile(r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs", re.IGNORECASE)
    _TOTAL_RE = re.compile(r"total time\s*=\s*([\d.]+)\s*ms", re.IGNORECASE)

    def __init__(self, model_name: str) -> None:
        super().__init__(
            provider_name="llamacpp",
            model_name=model_name,
            capabilities=ProviderCapabilities(text=True),
        )

    def generate_text(self, prompt: str, **kwargs: object) -> str:
        metrics = self.measure_text_generation(prompt, **kwargs)
        return metrics.response_text

    def measure_text_generation(self, prompt: str, **kwargs: object) -> GenerationMetrics:
        command = self._build_command(prompt=prompt, max_new_tokens=int(kwargs.get("max_new_tokens", 128)))
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        combined_output = "\n".join(part for part in [completed.stdout, completed.stderr] if part).strip()
        if completed.returncode != 0:
            raise RuntimeError(
                "llama.cpp generation failed.\n"
                f"command={' '.join(command)}\n"
                f"output={combined_output}"
            )
        return self._parse_generation_output(prompt=prompt, output=combined_output)

    def _build_command(self, prompt: str, max_new_tokens: int) -> list[str]:
        command = [
            "llama-cli",
            "--simple-io",
            "--no-display-prompt",
            "--show-timings",
            "--perf",
            "--temp",
            "0",
            "--n-predict",
            str(max_new_tokens),
            "--prompt",
            prompt,
        ]
        model_path = Path(self.model_name).expanduser()
        if model_path.exists():
            command.extend(["--model", str(model_path)])
        else:
            command.extend(["--hf-repo", self.model_name])
        return command

    def _parse_generation_output(self, prompt: str, output: str) -> GenerationMetrics:
        prompt_eval_match = self._PROMPT_EVAL_RE.search(output)
        eval_match = self._EVAL_RE.search(output)
        total_match = self._TOTAL_RE.search(output)

        prompt_eval_ms = float(prompt_eval_match.group(1)) if prompt_eval_match else 0.0
        eval_ms = float(eval_match.group(1)) if eval_match else 0.0
        eval_runs = int(eval_match.group(2)) if eval_match else 0
        total_ms = float(total_match.group(1)) if total_match else prompt_eval_ms + eval_ms

        first_token_ms = prompt_eval_ms
        if eval_runs > 0 and eval_ms > 0:
            first_token_ms += eval_ms / eval_runs

        token_generation_rate = eval_runs / (eval_ms / 1000.0) if eval_ms > 0 and eval_runs > 0 else 0.0

        response_lines: list[str] = []
        for line in output.splitlines():
            stripped = line.strip()
            lowered = stripped.lower()
            if not stripped:
                continue
            if stripped.startswith("load_backend:"):
                continue
            if "prompt eval time" in lowered or "eval time" in lowered or "total time" in lowered:
                continue
            if lowered.startswith("sampler ") or lowered.startswith("generate:"):
                continue
            response_lines.append(line)

        response_text = "\n".join(response_lines).strip()
        response_text = response_text.replace(prompt, "", 1).strip() if response_text.startswith(prompt) else response_text

        notes = [
            "Metrics parsed from llama.cpp timing output.",
            "TTFT is estimated as prompt-eval time plus one token eval step.",
        ]
        if not prompt_eval_match or not eval_match or not total_match:
            notes.append("One or more timing lines were missing; some metrics may be partial.")

        return GenerationMetrics(
            prompt_char_count=len(prompt),
            output_token_count=eval_runs,
            time_to_first_token_seconds=first_token_ms / 1000.0,
            token_generation_rate=token_generation_rate,
            total_duration_seconds=total_ms / 1000.0,
            response_text=response_text,
            metric_notes=notes,
        )

    def setup_message(self) -> str:
        return (
            f"llama.cpp expects a local GGUF model for '{self.model_name}'. "
            "Recommended large-model artifact: `bartowski/Qwen_Qwen3.5-35B-A3B-GGUF` "
            "(https://huggingface.co/bartowski/Qwen_Qwen3.5-35B-A3B-GGUF). "
            "Use `python scripts/download_hf_model.py --provider llamacpp --model-id bartowski/Qwen_Qwen3.5-35B-A3B-GGUF` to download local assets."
        )
