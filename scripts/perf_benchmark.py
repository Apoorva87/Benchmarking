#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genAI.perf.inventory import inventory_as_jsonable
from genAI.perf.prompt_catalog import load_prompt_specs
from genAI.perf.system_metrics import PowermetricsCollector, SystemMetricsSummary, VmStatCollector
from genAI.providers.registry import build_provider


ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "perf_runs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive local performance benchmark tool.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("interactive", help="Run the interactive selection flow.")
    subparsers.add_parser("inventory", help="Print discovered runnable providers and models as JSON.")

    run_parser = subparsers.add_parser("run", help="Run a performance benchmark directly.")
    run_parser.add_argument("--provider", required=True)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--prompt-size", required=True, choices=[spec.key for spec in load_prompt_specs()])
    run_parser.add_argument("--max-new-tokens", type=int, default=128)
    run_parser.add_argument("--system-metrics", action="store_true")
    run_parser.add_argument("--powermetrics-interval-ms", type=int, default=1000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inventory":
        print(json.dumps(inventory_as_jsonable(), indent=2))
        return
    if args.command == "interactive":
        interactive_run()
        return
    result = run_benchmark(
        provider_name=args.provider,
        model_ref=args.model,
        prompt_size=args.prompt_size,
        max_new_tokens=args.max_new_tokens,
        collect_system_metrics=args.system_metrics,
        powermetrics_interval_ms=args.powermetrics_interval_ms,
    )
    print(render_summary(result))


def interactive_run() -> None:
    inventory = inventory_as_jsonable()
    provider = choose_option("Choose a provider", sorted(inventory))
    models = inventory[provider]
    model_choice = choose_option("Choose a model", [item["label"] for item in models])
    selected_model = next(item for item in models if item["label"] == model_choice)
    prompt_specs = load_prompt_specs()
    prompt_choice = choose_option("Choose a prompt size", [spec.label for spec in prompt_specs])
    selected_prompt = next(spec for spec in prompt_specs if spec.label == prompt_choice)

    max_new_tokens_text = input("Max new tokens [128]: ").strip()
    max_new_tokens = int(max_new_tokens_text or "128")
    system_metrics = input("Collect system metrics with powermetrics? [Y/n]: ").strip().lower() not in {"n", "no"}

    result = run_benchmark(
        provider_name=provider,
        model_ref=selected_model["model_ref"],
        prompt_size=selected_prompt.key,
        max_new_tokens=max_new_tokens,
        collect_system_metrics=system_metrics,
        powermetrics_interval_ms=1000,
    )
    print("")
    print(render_summary(result))


def choose_option(prompt: str, options: list[str]) -> str:
    if not options:
        raise SystemExit(f"No options available for: {prompt}")
    print(prompt)
    for index, option in enumerate(options, start=1):
        print(f"  {index}. {option}")
    while True:
        selected = input("> ").strip()
        if selected.isdigit():
            idx = int(selected)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        print("Enter one of the numbered options.")


def run_benchmark(
    provider_name: str,
    model_ref: str,
    prompt_size: str,
    max_new_tokens: int,
    collect_system_metrics: bool,
    powermetrics_interval_ms: int,
) -> dict[str, object]:
    prompt_spec = next(spec for spec in load_prompt_specs() if spec.key == prompt_size)
    prompt_text = prompt_spec.load_text()

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    powermetrics_output = run_dir / "powermetrics.plist"

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
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(result, indent=2))
    result["report_path"] = str(report_path)
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
        f"gpu_utilization={system_metrics.get('gpu_utilization')}",
        f"fabric_bandwidth={system_metrics.get('fabric_bandwidth')}",
        f"cache_bandwidth={system_metrics.get('cache_bandwidth')}",
        f"memory_bandwidth={system_metrics.get('memory_bandwidth')}",
        f"report={result['report_path']}",
    ]
    notes = system_metrics.get("notes") or []
    if notes:
        lines.append("notes=" + " | ".join(notes))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
