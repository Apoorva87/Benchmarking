from pathlib import Path

from genAI.perf.system_metrics import PowermetricsCollector


def test_powermetrics_parser_extracts_real_sample_metrics() -> None:
    sample = Path("tests/fixtures/powermetrics_sample.plist")
    collector = PowermetricsCollector(sample)
    summary = collector._parse_output()

    assert summary.raw_output_path == "tests/fixtures/powermetrics_sample.plist"
    assert summary.cpu_power_watts is not None
    assert round(summary.cpu_power_watts, 2) == 1490.87
    assert summary.gpu_power_watts is not None
    assert round(summary.gpu_power_watts, 4) == 3.9337
    assert summary.total_power_watts is not None
    assert round(summary.total_power_watts, 1) == 1494.8
    assert summary.thermal_pressure == "Nominal"
    assert summary.cpu_utilization is not None
    assert round(summary.cpu_e_cluster_utilization or 0, 3) == 43.525
    assert round(summary.cpu_p0_cluster_utilization or 0, 3) == 99.437
    assert round(summary.cpu_p1_cluster_utilization or 0, 3) == 54.269
    assert round(summary.gpu_utilization or 0, 3) == 1.103
    assert summary.fabric_bandwidth is None
    assert any("bandwidth counters" in note for note in summary.notes)
