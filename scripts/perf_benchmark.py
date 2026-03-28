#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genAI.perf.inventory import inventory_as_jsonable
from genAI.perf.prompt_catalog import load_prompt_specs
from genAI.perf.system_metrics import PowermetricsCollector, SystemMetricsSummary, VmStatCollector
from genAI.providers.registry import build_provider


ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "perf_runs"
DEFAULT_MAX_NEW_TOKENS = 5000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive local performance benchmark tool.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("interactive", help="Run the interactive multi-select benchmark flow.")
    subparsers.add_parser("inventory", help="Print discovered runnable providers and models as JSON.")

    run_parser = subparsers.add_parser("run", help="Run a single performance benchmark directly.")
    run_parser.add_argument("--provider", required=True)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--prompt-size", required=True, choices=[spec.key for spec in load_prompt_specs()])
    run_parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    run_parser.add_argument("--system-metrics", action="store_true")
    run_parser.add_argument("--powermetrics-interval-ms", type=int, default=1000)
    run_parser.add_argument("--output-dir", type=Path)

    batch_parser = subparsers.add_parser("batch", help="Run a batch of benchmark combinations.")
    batch_parser.add_argument("--provider", action="append", dest="providers")
    batch_parser.add_argument("--model", action="append", dest="models")
    batch_parser.add_argument("--prompt-size", action="append", dest="prompt_sizes")
    batch_parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    batch_parser.add_argument("--system-metrics", action="store_true")
    batch_parser.add_argument("--powermetrics-interval-ms", type=int, default=1000)
    batch_parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inventory":
        print(json.dumps(inventory_as_jsonable(), indent=2))
        return
    if args.command == "interactive":
        interactive_run()
        return
    if args.command == "batch":
        results, batch_dir = run_batch(
            batch_specs=build_batch_specs_from_args(args.providers, args.models, args.prompt_sizes),
            max_new_tokens=args.max_new_tokens,
            collect_system_metrics=args.system_metrics,
            powermetrics_interval_ms=args.powermetrics_interval_ms,
            output_dir=args.output_dir,
        )
        print(render_batch_summary(results, batch_dir))
        return
    result = run_benchmark(
        provider_name=args.provider,
        model_ref=args.model,
        prompt_size=args.prompt_size,
        max_new_tokens=args.max_new_tokens,
        collect_system_metrics=args.system_metrics,
        powermetrics_interval_ms=args.powermetrics_interval_ms,
        run_dir=args.output_dir,
    )
    print(render_summary(result))


def interactive_run() -> None:
    inventory = inventory_as_jsonable()
    if not inventory:
        raise SystemExit("No runnable providers or models were discovered.")

    print("Local Performance Benchmark")
    print("===========================")
    print("Pick one or more providers, then one or more models per provider, then one or more prompt sizes.")
    print("Use numbers like `1,3` or ranges like `1-4`. Use `all` to select everything in a list.")
    print("")

    selected_providers = choose_many_options("Choose providers", sorted(inventory))
    batch_specs: list[dict[str, str]] = []
    for provider in selected_providers:
        models = inventory[provider]
        model_labels = [item["label"] for item in models]
        selected_model_labels = choose_many_options(f"Choose models for {provider}", model_labels)
        for label in selected_model_labels:
            selected_model = next(item for item in models if item["label"] == label)
            batch_specs.append(
                {
                    "provider": provider,
                    "label": label,
                    "model_ref": selected_model["model_ref"],
                }
            )

    prompt_specs = load_prompt_specs()
    prompt_labels = [spec.label for spec in prompt_specs]
    selected_prompt_labels = choose_many_options("Choose prompt sizes", prompt_labels)
    prompt_keys = [spec.key for spec in prompt_specs if spec.label in selected_prompt_labels]

    max_new_tokens_text = input(f"Max new tokens [{DEFAULT_MAX_NEW_TOKENS}]: ").strip()
    max_new_tokens = int(max_new_tokens_text or str(DEFAULT_MAX_NEW_TOKENS))
    system_metrics = input("Collect system metrics with powermetrics? [Y/n]: ").strip().lower() not in {"n", "no"}
    output_dir_text = input("Output folder [artifacts/perf_runs/<timestamp>]: ").strip()
    output_dir = Path(output_dir_text).expanduser() if output_dir_text else None

    final_batch_specs = [
        {
            "provider": spec["provider"],
            "model_ref": spec["model_ref"],
            "label": spec["label"],
            "prompt_size": prompt_size,
        }
        for spec, prompt_size in product(batch_specs, prompt_keys)
    ]
    print("")
    print("Planned runs")
    print("------------")
    for run_spec in final_batch_specs:
        print(f"- {run_spec['provider']} | {run_spec['label']} | {run_spec['prompt_size']}")
    print("")

    results, batch_dir = run_batch(
        batch_specs=final_batch_specs,
        max_new_tokens=max_new_tokens,
        collect_system_metrics=system_metrics,
        powermetrics_interval_ms=1000,
        output_dir=output_dir,
    )
    print("")
    print(render_batch_summary(results, batch_dir))


def choose_many_options(prompt: str, options: list[str]) -> list[str]:
    if not options:
        raise SystemExit(f"No options available for: {prompt}")
    print(prompt)
    for index, option in enumerate(options, start=1):
        print(f"  {index}. {option}")
    while True:
        selected = input("> ").strip()
        try:
            indexes = parse_selection(selected, len(options))
        except ValueError as exc:
            print(str(exc))
            continue
        return [options[index - 1] for index in indexes]


def parse_selection(selection: str, option_count: int) -> list[int]:
    normalized = selection.strip().lower()
    if not normalized:
        raise ValueError("Enter one or more numbers, ranges like 1-3, or `all`.")
    if normalized == "all":
        return list(range(1, option_count + 1))

    values: set[int] = set()
    for token in [chunk.strip() for chunk in selection.split(",") if chunk.strip()]:
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            if not start_text.isdigit() or not end_text.isdigit():
                raise ValueError("Ranges must look like `2-5`.")
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError("Ranges must be ascending, for example `2-5`.")
            for value in range(start, end + 1):
                values.add(_validate_selection_value(value, option_count))
            continue
        if not token.isdigit():
            raise ValueError("Selections must be numbers, ranges, or `all`.")
        values.add(_validate_selection_value(int(token), option_count))
    return sorted(values)


def _validate_selection_value(value: int, option_count: int) -> int:
    if not 1 <= value <= option_count:
        raise ValueError(f"Selection {value} is out of range 1-{option_count}.")
    return value


def build_batch_specs_from_args(
    providers: list[str] | None,
    models: list[str] | None,
    prompt_sizes: list[str] | None,
) -> list[dict[str, str]]:
    if not providers or not models or not prompt_sizes:
        raise SystemExit("Batch mode requires --provider, --model, and --prompt-size.")
    specs: list[dict[str, str]] = []
    for provider_name, model_ref, prompt_size in product(providers, models, prompt_sizes):
        specs.append(
            {
                "provider": provider_name,
                "model_ref": model_ref,
                "label": model_ref,
                "prompt_size": prompt_size,
            }
        )
    return specs


def run_batch(
    batch_specs: list[dict[str, str]],
    max_new_tokens: int,
    collect_system_metrics: bool,
    powermetrics_interval_ms: int,
    output_dir: Path | None,
) -> tuple[list[dict[str, object]], Path]:
    if not batch_specs:
        raise SystemExit("No benchmark combinations were selected.")

    batch_dir = output_dir or (ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S"))
    batch_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for index, spec in enumerate(batch_specs, start=1):
        prompt_size = spec["prompt_size"]
        run_name = f"{index:02d}_{slugify(spec['provider'])}_{slugify(spec['label'])}_{slugify(prompt_size)}"
        run_dir = batch_dir / run_name
        print(f"[{index}/{len(batch_specs)}] Running {spec['provider']} | {spec['label']} | {prompt_size}")
        result = run_benchmark(
            provider_name=spec["provider"],
            model_ref=spec["model_ref"],
            prompt_size=prompt_size,
            max_new_tokens=max_new_tokens,
            collect_system_metrics=collect_system_metrics,
            powermetrics_interval_ms=powermetrics_interval_ms,
            run_dir=run_dir,
        )
        results.append(result)

    comparison = {
        "run_count": len(results),
        "batch_dir": str(batch_dir),
        "results": results,
    }
    (batch_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))
    (batch_dir / "comparison.md").write_text(render_batch_markdown(results))
    return results, batch_dir


def run_benchmark(
    provider_name: str,
    model_ref: str,
    prompt_size: str,
    max_new_tokens: int,
    collect_system_metrics: bool,
    powermetrics_interval_ms: int,
    run_dir: Path | None,
) -> dict[str, object]:
    prompt_spec = next(spec for spec in load_prompt_specs() if spec.key == prompt_size)
    prompt_text = prompt_spec.load_text()

    resolved_run_dir = run_dir or (ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S"))
    resolved_run_dir.mkdir(parents=True, exist_ok=True)
    powermetrics_output = resolved_run_dir / "powermetrics.plist"

    provider = build_provider(provider_name, model_ref)
    powermetrics_collector: PowermetricsCollector | None = None
    vm_collector = VmStatCollector()

    ram_before = vm_collector.sample()
    if collect_system_metrics:
        sudo_ready = subprocess.run(["sudo", "-v"], check=False).returncode == 0
        if not sudo_ready:
            raise SystemExit("powermetrics requires sudo. Re-run and approve the sudo prompt, or disable system metrics.")
        powermetrics_collector = PowermetricsCollector(powermetrics_output, sample_rate_ms=powermetrics_interval_ms)
        powermetrics_collector.start()

    metrics = provider.measure_text_generation(prompt_text, max_new_tokens=max_new_tokens)

    system_metrics = SystemMetricsSummary(notes=["powermetrics collection disabled."])
    if powermetrics_collector is not None:
        system_metrics = powermetrics_collector.stop()
    ram_after = vm_collector.sample()
    system_metrics.ram_used_gb = ram_after.ram_used_gb
    system_metrics.ram_free_gb = ram_after.ram_free_gb
    system_metrics.raw_metrics.update(ram_after.raw_metrics)

    result = {
        "provider": provider_name,
        "model": model_ref,
        "prompt": {
            "size": prompt_spec.key,
            "label": prompt_spec.label,
            "approximate_tokens": prompt_spec.approximate_tokens,
            "path": str(prompt_spec.path),
        },
        "generation": asdict(metrics),
        "system_metrics": asdict(system_metrics),
    }
    report_path = resolved_run_dir / "report.json"
    report_path.write_text(json.dumps(result, indent=2))
    result["report_path"] = str(report_path)
    result["run_dir"] = str(resolved_run_dir)
    result["powermetrics_path"] = str(powermetrics_output) if collect_system_metrics else None
    result["ram_before_gb"] = ram_before.ram_used_gb
    result["ram_after_gb"] = ram_after.ram_used_gb
    return result


def render_summary(result: dict[str, object]) -> str:
    generation = result["generation"]
    system_metrics = result["system_metrics"]
    lines = [
        f"provider={result['provider']}",
        f"model={result['model']}",
        f"prompt={result['prompt']['label']} approx_tokens={result['prompt']['approximate_tokens']}",
        f"ttft={generation['time_to_first_token_seconds']:.4f}s",
        f"throughput={generation['token_generation_rate']:.2f} tok/s",
        f"total={generation['total_duration_seconds']:.4f}s",
        f"output_tokens={generation['output_token_count']}",
        f"ram_used_gb={system_metrics.get('ram_used_gb')}",
        f"ram_free_gb={system_metrics.get('ram_free_gb')}",
        f"cpu_utilization={system_metrics.get('cpu_utilization')}",
        f"cpu_e_cluster_utilization={system_metrics.get('cpu_e_cluster_utilization')}",
        f"cpu_p0_cluster_utilization={system_metrics.get('cpu_p0_cluster_utilization')}",
        f"cpu_p1_cluster_utilization={system_metrics.get('cpu_p1_cluster_utilization')}",
        f"active_core_count={system_metrics.get('active_core_count')}",
        f"active_e_core_count={system_metrics.get('active_e_core_count')}",
        f"active_p_core_count={system_metrics.get('active_p_core_count')}",
        f"active_core_labels={','.join(system_metrics.get('active_core_labels') or [])}",
        f"per_core_utilization={_render_per_core_utilization(system_metrics.get('per_core_utilization') or {})}",
        f"gpu_utilization={system_metrics.get('gpu_utilization')}",
        f"cpu_power_watts={system_metrics.get('cpu_power_watts')}",
        f"gpu_power_watts={system_metrics.get('gpu_power_watts')}",
        f"ane_power_watts={system_metrics.get('ane_power_watts')}",
        f"total_power_watts={system_metrics.get('total_power_watts')}",
        f"thermal_pressure={system_metrics.get('thermal_pressure')}",
        f"fabric_bandwidth={system_metrics.get('fabric_bandwidth')}",
        f"cache_bandwidth={system_metrics.get('cache_bandwidth')}",
        f"memory_bandwidth={system_metrics.get('memory_bandwidth')}",
        f"powermetrics={result.get('powermetrics_path')}",
        f"report={result['report_path']}",
    ]
    notes = system_metrics.get("notes") or []
    if notes:
        lines.append("notes=" + " | ".join(notes))
    return "\n".join(lines)


def render_batch_summary(results: list[dict[str, object]], batch_dir: Path) -> str:
    lines = [
        f"Completed {len(results)} run(s)",
        f"Batch output: {batch_dir}",
        "",
        _render_results_table(results),
        "",
        f"Comparison JSON: {batch_dir / 'comparison.json'}",
        f"Comparison Markdown: {batch_dir / 'comparison.md'}",
    ]
    return "\n".join(lines)


def render_batch_markdown(results: list[dict[str, object]]) -> str:
    lines = [
        "# Benchmark Comparison",
        "",
        "| Provider | Model | Prompt | TTFT (s) | Tok/s | Total (s) | CPU % | GPU % | Active Cores | CPU W | GPU W | Total W | Thermal | RAM Used (GB) | Report |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for result in results:
        generation = result["generation"]
        system_metrics = result["system_metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(result["provider"]),
                    _shorten_model(str(result["model"])),
                    str(result["prompt"]["size"]),
                    _format_float(generation["time_to_first_token_seconds"], 4),
                    _format_float(generation["token_generation_rate"], 2),
                    _format_float(generation["total_duration_seconds"], 4),
                    _format_float(system_metrics.get("cpu_utilization"), 2),
                    _format_float(system_metrics.get("gpu_utilization"), 2),
                    str(system_metrics.get("active_core_count")),
                    _format_float(system_metrics.get("cpu_power_watts"), 2),
                    _format_float(system_metrics.get("gpu_power_watts"), 2),
                    _format_float(system_metrics.get("total_power_watts"), 2),
                    str(system_metrics.get("thermal_pressure") or "-"),
                    _format_float(system_metrics.get("ram_used_gb"), 2),
                    str(result["report_path"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _render_results_table(results: list[dict[str, object]]) -> str:
    headers = ["Provider", "Model", "Prompt", "TTFT(s)", "Tok/s", "Total(s)", "CPU%", "GPU%", "Cores", "CPU W", "GPU W", "Total W", "Thermal"]
    rows = [headers]
    for result in results:
        generation = result["generation"]
        system_metrics = result["system_metrics"]
        rows.append(
            [
                str(result["provider"]),
                _shorten_model(str(result["model"]), max_length=32),
                str(result["prompt"]["size"]),
                _format_float(generation["time_to_first_token_seconds"], 4),
                _format_float(generation["token_generation_rate"], 2),
                _format_float(generation["total_duration_seconds"], 4),
                _format_float(system_metrics.get("cpu_utilization"), 2),
                _format_float(system_metrics.get("gpu_utilization"), 2),
                str(system_metrics.get("active_core_count")),
                _format_float(system_metrics.get("cpu_power_watts"), 2),
                _format_float(system_metrics.get("gpu_power_watts"), 2),
                _format_float(system_metrics.get("total_power_watts"), 2),
                str(system_metrics.get("thermal_pressure") or "-"),
            ]
        )

    widths = [max(len(str(row[column])) for row in rows) for column in range(len(headers))]
    lines = []
    for row_index, row in enumerate(rows):
        rendered = "  ".join(str(value).ljust(widths[column]) for column, value in enumerate(row))
        lines.append(rendered)
        if row_index == 0:
            lines.append("  ".join("-" * width for width in widths))
    return "\n".join(lines)


def _render_per_core_utilization(per_core_utilization: dict[str, float]) -> str:
    if not per_core_utilization:
        return ""
    parts = [f"{label}:{utilization:.2f}" for label, utilization in sorted(per_core_utilization.items())]
    return ",".join(parts)


def _format_float(value: Any, precision: int) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{precision}f}"


def _shorten_model(model: str, max_length: int = 48) -> str:
    if len(model) <= max_length:
        return model
    return model[: max_length - 3] + "..."


def slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized or "run"


if __name__ == "__main__":
    main()
