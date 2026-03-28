from genAI.perf.prompt_catalog import load_prompt_specs
from scripts.perf_benchmark import parse_selection, render_batch_markdown


def test_prompt_catalog_has_three_sizes() -> None:
    specs = load_prompt_specs()
    assert [spec.key for spec in specs] == ["small", "medium", "large"]
    for spec in specs:
        assert spec.path.exists()


def test_parse_selection_supports_multiple_indexes_and_ranges() -> None:
    assert parse_selection("1,3-4", 5) == [1, 3, 4]
    assert parse_selection("all", 3) == [1, 2, 3]


def test_render_batch_markdown_includes_major_comparison_fields() -> None:
    markdown = render_batch_markdown(
        [
            {
                "provider": "mlx",
                "model": "/tmp/model",
                "prompt": {"size": "small"},
                "generation": {
                    "time_to_first_token_seconds": 0.2,
                    "token_generation_rate": 75.0,
                    "total_duration_seconds": 1.2,
                },
                "system_metrics": {
                    "cpu_utilization": 63.5,
                    "gpu_utilization": 21.0,
                    "active_core_count": 10,
                    "cpu_power_watts": 45.2,
                    "gpu_power_watts": 12.3,
                    "total_power_watts": 58.1,
                    "thermal_pressure": "Nominal",
                    "ram_used_gb": 18.4,
                },
                "report_path": "/tmp/report.json",
            }
        ]
    )
    assert "| Provider | Model | Prompt | TTFT (s) | Tok/s |" in markdown
    assert "| mlx | /tmp/model | small | 0.2000 | 75.00 | 1.2000 | 63.50 | 21.00 | 10 | 45.20 | 12.30 | 58.10 | Nominal | 18.40 | /tmp/report.json |" in markdown
