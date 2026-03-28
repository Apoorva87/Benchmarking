from genAI.perf.prompt_catalog import load_prompt_specs


def test_prompt_catalog_has_three_sizes() -> None:
    specs = load_prompt_specs()
    assert [spec.key for spec in specs] == ["small", "medium", "large"]
    for spec in specs:
        assert spec.path.exists()

