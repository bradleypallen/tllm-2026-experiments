"""
Validate the schema of every result JSON in `results/`.

Each snapshot has a recognizable shape:
- 2025 classical:               keys include `bilateral_distribution`, `epistemic_policy`, `f1_macro`, `coverage`.
- 2025 unilateral_direct:       keys include `evaluation_type=='unilateral_forced_choice'`, `f1_macro`, `coverage`.
- 2025 unilateral_uncertain:    keys include `evaluation_type=='unilateral_with_uncertainty'`, `f1_macro`, `coverage`.
- 2025 unilateral_confidence:   keys include `threshold_analysis` with sub-keys for each threshold.
- 2026 proper:                  keys include `approaches` dict with bilateral_classical / paracomplete / paraconsistent / forced_unilateral / ternary / confidence_05; `bilateral_distribution`; `detailed_results`.
- 2026 modal:                   keys include `src_bilateral_distribution`, `modal_bilateral_distribution`, `detailed_results`.

These tests catch silent format drift if anyone re-runs the evaluators with a modified bilateral-truth version.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"


def _load_json(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


@pytest.mark.parametrize("p", sorted((RESULTS / "2025_09" / "classical").glob("*.json")) if (RESULTS / "2025_09" / "classical").exists() else [],
                         ids=lambda p: p.name)
def test_2025_classical_schema(p: Path) -> None:
    if p.name == "MANIFEST.json":
        return
    d = _load_json(p)
    assert d.get("epistemic_policy") == "classical"
    assert "bilateral_distribution" in d
    assert "f1_macro" in d and 0.0 <= d["f1_macro"] <= 1.0
    assert "coverage" in d and 0.0 <= d["coverage"] <= 1.0
    assert d.get("total_samples", 0) > 0


@pytest.mark.parametrize("p", sorted((RESULTS / "2025_09" / "unilateral_direct").glob("*.json")) if (RESULTS / "2025_09" / "unilateral_direct").exists() else [],
                         ids=lambda p: p.name)
def test_2025_unilateral_direct_schema(p: Path) -> None:
    if p.name == "MANIFEST.json":
        return
    d = _load_json(p)
    assert d.get("evaluation_type") == "unilateral_forced_choice"
    assert d.get("prompt_style") == "direct"
    assert "f1_macro" in d and 0.0 <= d["f1_macro"] <= 1.0
    assert d.get("coverage") == 1.0  # forced choice


@pytest.mark.parametrize("p", sorted((RESULTS / "2025_09" / "unilateral_uncertain").glob("*.json")) if (RESULTS / "2025_09" / "unilateral_uncertain").exists() else [],
                         ids=lambda p: p.name)
def test_2025_unilateral_uncertain_schema(p: Path) -> None:
    if p.name == "MANIFEST.json":
        return
    d = _load_json(p)
    assert d.get("evaluation_type") == "unilateral_with_uncertainty"
    assert d.get("prompt_style") == "uncertain"
    assert "uncertainty_rate" in d


@pytest.mark.parametrize("p", sorted((RESULTS / "2025_09" / "unilateral_confidence").glob("*.json")) if (RESULTS / "2025_09" / "unilateral_confidence").exists() else [],
                         ids=lambda p: p.name)
def test_2025_unilateral_confidence_schema(p: Path) -> None:
    if p.name == "MANIFEST.json":
        return
    d = _load_json(p)
    ta = d.get("threshold_analysis", {})
    assert set(ta.keys()) == {"0.5", "0.7", "0.9"}, f"unexpected thresholds: {set(ta.keys())}"
    for thr, sub in ta.items():
        assert "f1_macro" in sub and 0.0 <= sub["f1_macro"] <= 1.0
        assert "coverage" in sub and 0.0 <= sub["coverage"] <= 1.0


EXPECTED_2026_PROPER_APPROACHES = {
    "bilateral_classical", "bilateral_paracomplete", "bilateral_paraconsistent",
    "forced_unilateral", "ternary", "confidence_05",
}


@pytest.mark.parametrize("p", sorted((RESULTS / "2026_03" / "proper").glob("*.json")) if (RESULTS / "2026_03" / "proper").exists() else [],
                         ids=lambda p: p.name)
def test_2026_proper_schema(p: Path) -> None:
    if p.name == "MANIFEST.json":
        return
    d = _load_json(p)
    assert d.get("bilateral_samples") == 3
    assert "approaches" in d
    assert set(d["approaches"].keys()) == EXPECTED_2026_PROPER_APPROACHES, (
        f"unexpected approaches: {set(d['approaches'].keys())}"
    )
    for name, metrics in d["approaches"].items():
        for k in ("coverage", "accuracy", "f1_macro"):
            assert k in metrics, f"{name} missing {k}"
            assert 0.0 <= metrics[k] <= 1.0
    assert d.get("total_samples", 0) > 0
    assert "bilateral_distribution" in d
    assert "detailed_results" in d and len(d["detailed_results"]) == d["total_samples"]


@pytest.mark.parametrize("p", sorted((RESULTS / "2026_03" / "modal").glob("*.json")) if (RESULTS / "2026_03" / "modal").exists() else [],
                         ids=lambda p: p.name)
def test_2026_modal_schema(p: Path) -> None:
    if p.name == "MANIFEST.json":
        return
    d = _load_json(p)
    assert "src_bilateral_distribution" in d
    assert "modal_bilateral_distribution" in d
    assert d.get("total_samples", 0) > 0
