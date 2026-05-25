"""
Mock-LLM smoke test: run ProperBenchmarkEvaluator over a tiny dataset slice
using MockLLMEvaluator and verify the output schema matches what's frozen in
results/2026_03/.

This catches drift in the evaluator's output format if anyone re-runs with a
different bilateral-truth version. It does NOT validate numerical results
(LLM responses are mocked).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "evaluators"))


def test_proper_evaluator_runs_with_mock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("bilateral_truth")

    from bilateral_truth.llm_evaluators import MockLLMEvaluator
    from bilateral_truth.model_router import ModelRouter
    import proper_benchmark_evaluator as pbe

    # Force the ModelRouter to hand back a MockLLMEvaluator regardless of
    # what model name is passed. The mock returns deterministic results
    # without making any API calls.
    def _mock_create(model_name: str, **kwargs):
        return MockLLMEvaluator()
    monkeypatch.setattr(ModelRouter, "create_evaluator", _mock_create)

    # Mock also returns a numerical confidence — but it doesn't implement
    # the non-bilateral methods. Stub them to return deterministic values
    # so the evaluator can run end-to-end.
    from bilateral_truth.truth_values import TruthValueComponent
    monkeypatch.setattr(MockLLMEvaluator, "evaluate_forced_unilateral",
                        lambda self, a, context=None: TruthValueComponent.TRUE, raising=False)
    monkeypatch.setattr(MockLLMEvaluator, "evaluate_ternary",
                        lambda self, a, context=None: TruthValueComponent.FALSE, raising=False)
    monkeypatch.setattr(MockLLMEvaluator, "evaluate_confidence",
                        lambda self, a, context=None: 0.7, raising=False)

    evaluator = pbe.ProperBenchmarkEvaluator(
        model_name="mock",
        dataset_path=str(REPO_ROOT / "datasets" / "truthfulqa_complete.json"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        output_dir=str(tmp_path / "results"),
        bilateral_samples=3,
    )
    output = evaluator.run(max_samples=4, seed=42)

    # Schema checks — must match what the snapshot validator expects.
    assert "approaches" in output
    expected_approaches = {
        "bilateral_classical", "bilateral_paracomplete", "bilateral_paraconsistent",
        "forced_unilateral", "ternary", "confidence_05",
    }
    assert set(output["approaches"].keys()) == expected_approaches
    for name, m in output["approaches"].items():
        for k in ("coverage", "accuracy", "f1_macro"):
            assert k in m
            assert 0.0 <= m[k] <= 1.0
    assert "bilateral_distribution" in output
    assert "detailed_results" in output
    assert len(output["detailed_results"]) == 4
    assert output["bilateral_samples"] == 3

    # And one row of detailed_results has the expected keys:
    row = output["detailed_results"][0]
    for k in ("assertion_id", "assertion_text", "ground_truth",
              "bilateral_tv", "bilateral_classical_projected",
              "forced_unilateral_prediction", "ternary_prediction",
              "confidence_score"):
        assert k in row, f"detailed_results row missing key: {k}"
