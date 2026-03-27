from __future__ import annotations

import re

from genAI.scoring.models import ScoreBreakdown


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def normalized_exact_match_score(expected: str, actual: str) -> ScoreBreakdown:
    matched = _normalize_text(expected) == _normalize_text(actual)
    score = 1.0 if matched else 0.0
    note = "Exact match after normalization." if matched else "Did not match expected answer."
    return ScoreBreakdown(overall=score, rubric={"exact_match": score}, notes=[note])


def keyword_coverage_score(expected_keywords: list[str], actual: str) -> ScoreBreakdown:
    normalized_response = _normalize_text(actual)
    if not expected_keywords:
        return ScoreBreakdown(overall=1.0, rubric={"keyword_coverage": 1.0}, notes=["No required keywords."])
    hits = sum(1 for keyword in expected_keywords if _normalize_text(keyword) in normalized_response)
    score = hits / len(expected_keywords)
    notes = [f"Matched {hits} of {len(expected_keywords)} expected keywords."]
    return ScoreBreakdown(overall=score, rubric={"keyword_coverage": score}, notes=notes)


def instruction_fidelity_score(
    actual: str,
    expected_keywords: list[str] | None = None,
    max_words: int | None = None,
    required_json_keys: list[str] | None = None,
) -> ScoreBreakdown:
    expected_keywords = expected_keywords or []
    required_json_keys = required_json_keys or []
    normalized_response = _normalize_text(actual)
    rubric: dict[str, float] = {}
    notes: list[str] = []

    if expected_keywords:
        hits = sum(1 for keyword in expected_keywords if _normalize_text(keyword) in normalized_response)
        keyword_score = hits / len(expected_keywords)
        rubric["keyword_coverage"] = keyword_score
        notes.append(f"Matched {hits} of {len(expected_keywords)} required keywords.")

    if max_words is not None:
        word_count = len(actual.split())
        within_limit = 1.0 if word_count <= max_words else 0.0
        rubric["length_compliance"] = within_limit
        notes.append(f"Response used {word_count} words with max allowed {max_words}.")

    if required_json_keys:
        lowered = actual.lower()
        hits = sum(1 for key in required_json_keys if f"\"{key.lower()}\"" in lowered or f"'{key.lower()}'" in lowered)
        json_key_score = hits / len(required_json_keys)
        rubric["json_key_presence"] = json_key_score
        notes.append(f"Found {hits} of {len(required_json_keys)} required JSON keys.")

    if not rubric:
        rubric["baseline"] = 1.0
        notes.append("No explicit constraints supplied; fidelity defaults to baseline pass.")

    overall = sum(rubric.values()) / len(rubric)
    return ScoreBreakdown(overall=overall, rubric=rubric, notes=notes)
