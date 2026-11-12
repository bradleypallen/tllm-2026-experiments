"""
Offline tests for analysis/sync_generated_blocks.py, which refreshes the GENERATED table blocks in
the manuscript. The manuscript lives in another repository, so these run on a temporary file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "analysis"))
import sync_generated_blocks as S  # noqa: E402

PAPER = """\\begin{table}[t]
\\caption{The author's caption.}
\\label{tab:a}
% BEGIN GENERATED: alpha -- tllm-2026-experiments@0000000, analysis/emit.py; do not edit by hand
OLD ALPHA
% END GENERATED: alpha
\\end{table}
Prose between the tables that must not change.
% BEGIN GENERATED: beta -- tllm-2026-experiments@0000000, analysis/other.py; do not edit by hand
BETA BODY
% END GENERATED: beta
"""


@pytest.fixture()
def workspace(tmp_path):
    frag = tmp_path / "frag"
    frag.mkdir()
    (frag / "alpha_tabular.tex").write_text("NEW ALPHA\nsecond line\n")
    (frag / "beta_tabular.tex").write_text("BETA BODY\n")
    paper = tmp_path / "paper.tex"
    paper.write_text(PAPER)
    return paper, frag


def test_sync_replaces_bodies_and_stamps_commit(workspace) -> None:
    paper, frag = workspace
    assert S.main([str(paper), "--fragments", str(frag), "--commit", "abc1234"]) == 0
    out = paper.read_text()
    assert "% BEGIN GENERATED: alpha -- tllm-2026-experiments@abc1234, analysis/emit.py; do not edit by hand\n" \
           "NEW ALPHA\nsecond line\n% END GENERATED: alpha\n" in out
    assert "% BEGIN GENERATED: beta -- tllm-2026-experiments@abc1234, analysis/other.py; do not edit by hand\n" \
           "BETA BODY\n% END GENERATED: beta\n" in out
    assert "OLD ALPHA" not in out and "0000000" not in out
    # Everything outside the blocks is untouched.
    for kept in ("\\caption{The author's caption.}", "\\label{tab:a}", "Prose between the tables that must not change."):
        assert kept in out
    # Idempotent.
    assert S.main([str(paper), "--fragments", str(frag), "--commit", "abc1234"]) == 0
    assert paper.read_text() == out


def test_check_reports_drift_without_writing(workspace) -> None:
    paper, frag = workspace
    assert S.main([str(paper), "--fragments", str(frag), "--commit", "abc1234", "--check"]) == 1   # alpha differs
    assert paper.read_text() == PAPER
    S.main([str(paper), "--fragments", str(frag), "--commit", "abc1234"])
    assert S.main([str(paper), "--fragments", str(frag), "--commit", "abc1234", "--check"]) == 0


def test_missing_fragment_and_missing_blocks_fail(workspace, tmp_path) -> None:
    paper, frag = workspace
    (frag / "beta_tabular.tex").unlink()
    with pytest.raises(SystemExit):
        S.main([str(paper), "--fragments", str(frag), "--commit", "abc1234"])
    empty = tmp_path / "empty.tex"
    empty.write_text("no blocks here\n")
    assert S.main([str(empty), "--fragments", str(frag), "--commit", "abc1234"]) == 1


def test_every_committed_fragment_has_the_expected_shape() -> None:
    """Each LNCS fragment is a bare tabular: that is what the manuscript's blocks assume."""
    frags = [p for d in S.FRAGMENT_DIRS for p in sorted(d.glob("*_tabular.tex"))]
    assert {p.name for p in frags} == {
        "table1_coverage_tabular.tex", "e1_persistence_tabular.tex", "transition_tabular.tex",
        "e3_compliance_tabular.tex", "f3prime_p2_tabular.tex", "e4_paraphrase_vs_repeat_tabular.tex"}
    for p in frags:
        t = p.read_text()
        assert t.startswith("\\begin{tabular}") and t.rstrip("\n").endswith("\\end{tabular}"), p.name
        assert "\\caption" not in t and "\\label" not in t and "\\begin{table}" not in t, p.name
