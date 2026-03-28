from __future__ import annotations

import plistlib
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SystemMetricsSummary:
    cpu_utilization: float | None = None
    gpu_utilization: float | None = None
    fabric_bandwidth: float | None = None
    cache_bandwidth: float | None = None
    memory_bandwidth: float | None = None
    ram_used_gb: float | None = None
    ram_free_gb: float | None = None
    raw_metrics: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    raw_output_path: str | None = None


class PowermetricsCollector:
    def __init__(self, output_path: Path, sample_rate_ms: int = 1000) -> None:
        self.output_path = output_path
        self.sample_rate_ms = sample_rate_ms
        self._process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = self.output_path.open("w")
        self._process = subprocess.Popen(
            [
                "sudo",
                "-n",
                "powermetrics",
                "--samplers",
                "cpu_power,gpu_power,thermal",
                "--format",
                "plist",
                "-i",
                str(self.sample_rate_ms),
            ],
            stdout=output_handle,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def stop(self) -> SystemMetricsSummary:
        if self._process is None:
            return SystemMetricsSummary(notes=["powermetrics collector was not started."])
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)
        return self._parse_output()

    def _parse_output(self) -> SystemMetricsSummary:
        summary = SystemMetricsSummary(raw_output_path=str(self.output_path))
        if not self.output_path.exists():
            summary.notes.append("powermetrics output file was not created.")
            return summary
        raw_bytes = self.output_path.read_bytes()
        payloads = [chunk for chunk in raw_bytes.split(b"\x00") if chunk.strip()]
        flattened: dict[str, float] = {}
        for payload in payloads:
            try:
                plist = plistlib.loads(payload)
            except Exception:
                continue
            for key, value in _flatten_numeric("", plist).items():
                flattened[key] = value
        summary.raw_metrics = flattened
        summary.cpu_utilization = _find_metric(flattened, [["cpu", "util"], ["cpu", "duty"], ["cpu", "active"]])
        summary.gpu_utilization = _find_metric(flattened, [["gpu", "util"], ["gpu", "duty"], ["gpu", "active"]])
        summary.fabric_bandwidth = _find_metric(flattened, [["fabric", "bandwidth"], ["fabric", "bw"]])
        summary.cache_bandwidth = _find_metric(flattened, [["cache", "bandwidth"], ["cache", "bw"]])
        summary.memory_bandwidth = _find_metric(flattened, [["dram", "bandwidth"], ["memory", "bandwidth"], ["dram", "bw"]])
        if not flattened:
            summary.notes.append("No numeric powermetrics fields were parsed from the plist output.")
        return summary


class VmStatCollector:
    def sample(self) -> SystemMetricsSummary:
        summary = SystemMetricsSummary()
        page_size = self._page_size_bytes()
        vm_output = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False)
        mem_output = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False)
        if vm_output.returncode != 0 or mem_output.returncode != 0:
            summary.notes.append("vm_stat or sysctl sampling failed.")
            return summary
        total_bytes = int(mem_output.stdout.strip())
        stats = self._parse_vm_stat(vm_output.stdout)
        free_pages = stats.get("Pages free", 0) + stats.get("Pages speculative", 0)
        active_pages = stats.get("Pages active", 0) + stats.get("Pages wired down", 0) + stats.get("Pages occupied by compressor", 0)
        summary.ram_free_gb = (free_pages * page_size) / (1024**3)
        summary.ram_used_gb = (active_pages * page_size) / (1024**3)
        summary.raw_metrics["ram_total_gb"] = total_bytes / (1024**3)
        return summary

    def _page_size_bytes(self) -> int:
        completed = subprocess.run(["pagesize"], capture_output=True, text=True, check=False)
        return int(completed.stdout.strip() or "4096")

    def _parse_vm_stat(self, output: str) -> dict[str, int]:
        stats: dict[str, int] = {}
        for line in output.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            cleaned = re.sub(r"[^0-9]", "", value)
            if cleaned:
                stats[key.strip()] = int(cleaned)
        return stats


def _flatten_numeric(prefix: str, value: Any) -> dict[str, float]:
    flattened: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_numeric(next_prefix, child))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            next_prefix = f"{prefix}[{index}]"
            flattened.update(_flatten_numeric(next_prefix, child))
    elif isinstance(value, (int, float)):
        flattened[prefix.lower()] = float(value)
    return flattened


def _find_metric(metrics: dict[str, float], keyword_sets: list[list[str]]) -> float | None:
    for keywords in keyword_sets:
        for key, value in metrics.items():
            if all(keyword in key for keyword in keywords):
                return value
    return None

