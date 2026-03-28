from __future__ import annotations

import re
import subprocess

from genAI.providers.base import GenerationMetrics, ProviderCapabilities, TextGenerationProvider


class MLXProvider(TextGenerationProvider):
    _PROMPT_RE = re.compile(r"Prompt:\s+(\d+)\s+tokens,\s+([\d.]+)\s+tokens-per-sec", re.IGNORECASE)
    _GENERATION_RE = re.compile(r"Generation:\s+(\d+)\s+tokens,\s+([\d.]+)\s+tokens-per-sec", re.IGNORECASE)
    _PEAK_MEMORY_RE = re.compile(r"Peak memory:\s+([\d.]+)\s+GB", re.IGNORECASE)

    def __init__(self, model_name: str) -> None:
        super().__init__(
            provider_name="mlx",
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
                "MLX generation failed.\n"
                f"command={' '.join(command)}\n"
                f"output={combined_output}"
            )
        return self._parse_generation_output(prompt=prompt, output=combined_output)

    def _build_command(self, prompt: str, max_new_tokens: int) -> list[str]:
        return [
            ".venv311/bin/python",
            "-m",
            "mlx_lm.generate",
            "--model",
            self.model_name,
            "--prompt",
            prompt,
            "--max-tokens",
            str(max_new_tokens),
            "--verbose",
            "true",
        ]

    def _parse_generation_output(self, prompt: str, output: str) -> GenerationMetrics:
        prompt_match = self._PROMPT_RE.search(output)
        generation_match = self._GENERATION_RE.search(output)
        peak_memory_match = self._PEAK_MEMORY_RE.search(output)

        prompt_tokens = int(prompt_match.group(1)) if prompt_match else 0
        prompt_tps = float(prompt_match.group(2)) if prompt_match else 0.0
        output_tokens = int(generation_match.group(1)) if generation_match else 0
        generation_tps = float(generation_match.group(2)) if generation_match else 0.0
        peak_memory_gb = float(peak_memory_match.group(1)) if peak_memory_match else None

        prompt_duration = prompt_tokens / prompt_tps if prompt_tokens and prompt_tps else 0.0
        generation_duration = output_tokens / generation_tps if output_tokens and generation_tps else 0.0

        response_lines: list[str] = []
        capture = False
        for raw_line in output.splitlines():
            stripped = raw_line.strip()
            lowered = stripped.lower()
            if not stripped:
                continue
            if lowered.startswith("calling `python -m mlx_lm.generate"):
                continue
            if lowered.startswith("prompt:") or lowered.startswith("generation:") or lowered.startswith("peak memory:"):
                continue
            if stripped == "==========":
                capture = not capture
                continue
            if capture:
                response_lines.append(raw_line)

        response_text = "\n".join(response_lines).strip()
        if not response_text:
            for raw_line in output.splitlines():
                stripped = raw_line.strip()
                lowered = stripped.lower()
                if not stripped or lowered.startswith("prompt:") or lowered.startswith("generation:") or lowered.startswith("peak memory:"):
                    continue
                if lowered.startswith("calling `python -m mlx_lm.generate") or lowered.startswith("<frozen runpy>"):
                    continue
                response_lines.append(raw_line)
            response_text = "\n".join(response_lines).strip()

        notes = ["Metrics parsed from mlx_lm.generate output."]
        if peak_memory_gb is not None:
            notes.append(f"Peak memory reported by MLX: {peak_memory_gb:.3f} GB.")
        if not prompt_match or not generation_match:
            notes.append("One or more throughput lines were missing; some metrics may be partial.")

        return GenerationMetrics(
            prompt_char_count=len(prompt),
            output_token_count=output_tokens,
            time_to_first_token_seconds=prompt_duration,
            token_generation_rate=generation_tps,
            total_duration_seconds=prompt_duration + generation_duration,
            response_text=response_text,
            metric_notes=notes,
        )

    def setup_message(self) -> str:
        return (
            f"MLX can use a local Hugging Face checkpoint for '{self.model_name}'. "
            "Recommended examples: `mlx-community/Qwen3.5-35B-A3B-8bit` and "
            "`mlx-community/gpt-oss-120b-MXFP4-Q4`. "
            "A verified local model on this machine is `lmstudio-community/GLM-4.7-Flash-MLX-6bit` via the repo-local symlink tree. "
            "Use `python scripts/download_hf_model.py --provider mlx --model-id <hf-model-id>` to fetch assets."
        )
