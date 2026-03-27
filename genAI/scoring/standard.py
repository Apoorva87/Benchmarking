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

