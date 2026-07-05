"""
Definition-16 paraphrase-fidelity audit -- Phases 4-5: ingest the human-labeled
CSV, compute statistics, and generate the paper sentence.

Refuses to run unless EVERY row's `human_pass` is exactly `P` or `F` (the audit
is human-judged; there is no machine fallback).

Inputs :  audit/edge_audit_sample.csv   (labeled by the author)
Outputs:  audit/audit_results.json      (all computed numbers, traceable)
          audit/AUDIT_REPORT.md         (human-readable report)
          audit/paper_sentence.tex      (filled Variant-A sentence; the branch the
                                         data selects live, the other commented out)

Statistics:
  1. overall pass rate with 95% Wilson score interval;
  2. per-benchmark pass rates;
  3. failure-mode clusters (normalized case/whitespace; conservative near-duplicate
     merge = token-set equality; grouping shown) with counts and row_ids;
  4. transformation-type distribution over PASS rows (human column);
  5. machine-suggestion agreement: fraction of typed rows where suggested types are
     a subset of human types (diagnostic only; not for the paper).

Run from the repo root:  python3 audit/compute_audit_stats.py
"""
from __future__ import annotations
import csv
import json
import logging
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [def16-stats] %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

AUDIT = Path(__file__).resolve().parent
IN_CSV = AUDIT / "edge_audit_sample.csv"
OUT_JSON = AUDIT / "audit_results.json"
OUT_MD = AUDIT / "AUDIT_REPORT.md"
OUT_TEX = AUDIT / "paper_sentence.tex"

VALID_TYPES = {"lexical", "reorder", "elaboration", "formatting"}


def wilson_95(x: int, n: int):
    """95% Wilson score interval for x successes out of n."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963985
    p = x / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def norm_mode(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).rstrip(".")


def parse_types(cell: str) -> set[str]:
    return {t.strip().lower() for t in cell.split(",") if t.strip()}


def main():
    rows = list(csv.DictReader(IN_CSV.open(encoding="utf-8")))
    log.info("read %d rows from %s", len(rows), IN_CSV)

    # ── hard gate: complete, valid judgments ────────────────────────────────
    bad = [r["row_id"] for r in rows
           if r.get("human_pass", "").strip().upper() not in ("P", "F")]
    if bad:
        log.error("REFUSING TO RUN: %d rows lack a valid human_pass (P/F): row_ids %s",
                  len(bad), ", ".join(bad))
        sys.exit(1)
    unknown_types = {t for r in rows for t in parse_types(r["human_transformation_types"])
                     if t not in VALID_TYPES}
    if unknown_types:
        log.warning("human_transformation_types outside the four categories: %s",
                    sorted(unknown_types))

    n = len(rows)
    passes = [r for r in rows if r["human_pass"].strip().upper() == "P"]
    fails = [r for r in rows if r["human_pass"].strip().upper() == "F"]
    x = len(passes)
    lo, hi = wilson_95(x, n)
    log.info("overall: %d/%d pass = %.1f%%  (95%% Wilson [%.1f%%, %.1f%%])",
             x, n, 100 * x / n, 100 * lo, 100 * hi)

    # ── per-benchmark ───────────────────────────────────────────────────────
    per_bench = {}
    by_bench = defaultdict(list)
    for r in rows:
        by_bench[r["benchmark"]].append(r)
    for b in sorted(by_bench):
        bx = sum(1 for r in by_bench[b] if r["human_pass"].strip().upper() == "P")
        per_bench[b] = {"pass": bx, "n": len(by_bench[b]),
                        "rate": round(bx / len(by_bench[b]), 4)}
        log.info("  %-11s %d/%d", b, bx, len(by_bench[b]))

    # ── failure-mode clustering ─────────────────────────────────────────────
    # exact groups on normalized text, then conservative merge on token-set equality
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in fails:
        groups[norm_mode(r["human_failure_mode"])].append(r)
    merged: list[dict] = []
    used = set()
    keys = sorted(groups, key=lambda k: (-len(groups[k]), k))
    for k in keys:
        if k in used:
            continue
        members = list(groups[k])
        merged_from = [k]
        for k2 in keys:
            if k2 == k or k2 in used:
                continue
            if set(k.split()) == set(k2.split()):        # token-set equality only
                members += groups[k2]
                merged_from.append(k2)
                used.add(k2)
        used.add(k)
        merged.append({
            "label": k if k else "(no failure mode given)",
            "merged_from": merged_from,
            "count": len(members),
            "row_ids": sorted(int(r["row_id"]) for r in members),
        })
    merged.sort(key=lambda c: (-c["count"], c["label"]))
    for c in merged:
        log.info("  failure cluster (%d): %s  rows=%s",
                 c["count"], c["label"], c["row_ids"])

    # ── transformation-type distribution over PASS rows (human column) ─────
    type_counts = Counter()
    for r in passes:
        for t in parse_types(r["human_transformation_types"]):
            type_counts[t] += 1

    # ── machine-suggestion agreement (diagnostic) ───────────────────────────
    typed = [r for r in rows if parse_types(r["human_transformation_types"])]
    agree = sum(1 for r in typed
                if parse_types(r["suggested_transformation_types"])
                <= parse_types(r["human_transformation_types"]))
    agreement = {"typed_rows": len(typed), "suggested_subset_of_human": agree,
                 "fraction": round(agree / len(typed), 4) if typed else None}

    # ── paper sentence (Phase 5) ────────────────────────────────────────────
    rate_pct = 100 * x / n
    fail_pct = 100 * len(fails) / n
    top = merged[0] if merged else None
    clause = ""
    clause_flag = ""
    if top and fails and top["count"] / len(fails) >= 0.60:
        clause = f" (failures were predominantly {top['label']})"
        clause_flag = ("% AUTHOR: approve/adjust the failure-cluster description above "
                       "(proposed from human_failure_mode text).\n")
    primary = (
        f"A hand audit of 100 variants, sampled uniformly across the four benchmarks "
        f"(seed = 42, at most one variant per assertion), found {rate_pct:.0f}\\% to "
        f"instantiate the transformation types of Definition~\\ref{{def:varrel}} without "
        f"altering the proposition under evaluation{clause}; failing variants were left "
        f"in place, so any residual infidelity inflates the measured migration rates --- "
        f"a variant that shifts the proposition can legitimately fail verification --- "
        f"adding a second reason, alongside the sampling noise bounded in "
        f"\\S\\ref{{sec:diagnostic}}, to read the rates of "
        f"Figure~\\ref{{fig:migration-tf-to-ff}} as upper bounds. Per-edge fidelity "
        f"validation at scale is part of the reliability program of "
        f"\\S\\ref{{sec:future-work}}."
    )
    fallback = (
        f"A hand audit of 100 variants, sampled uniformly across the four benchmarks "
        f"(seed = 42, at most one variant per assertion), found {rate_pct:.0f}\\% to "
        f"instantiate the transformation types of Definition~\\ref{{def:varrel}} without "
        f"altering the proposition under evaluation{clause}; the failing {fail_pct:.0f}\\% "
        f"were left in place and inflate the measured migration rates --- a variant that "
        f"shifts the proposition can legitimately fail verification --- so the rates of "
        f"Figure~\\ref{{fig:migration-tf-to-ff}} are upper bounds on this count too, "
        f"alongside the sampling noise bounded in \\S\\ref{{sec:diagnostic}}. Per-edge "
        f"fidelity validation at scale is part of the reliability program of "
        f"\\S\\ref{{sec:future-work}}."
    )
    live, dead, which = (primary, fallback, "primary (rate >= 90%)") if rate_pct >= 90 \
        else (fallback, primary, "fallback (rate < 90%)")
    dead_commented = "\n".join("% " + ln for ln in dead.split("\n"))
    OUT_TEX.write_text(
        f"% Definition-16 fidelity audit sentence -- generated by compute_audit_stats.py\n"
        f"% Selected branch: {which}. Numbers trace to audit_results.json.\n"
        f"% Placement: S7 setup paragraph, immediately after the same-family caveat.\n"
        + clause_flag +
        live + "\n\n% --- other branch (not selected) ---\n" + dead_commented + "\n"
    )
    log.info("wrote %s (%s)", OUT_TEX, which)

    # ── outputs ─────────────────────────────────────────────────────────────
    results = {
        "n": n,
        "pass": x,
        "fail": len(fails),
        "pass_rate": round(x / n, 4),
        "wilson_95": [round(lo, 4), round(hi, 4)],
        "per_benchmark": per_bench,
        "failure_clusters": merged,
        "failure_clause_used": clause or None,
        "transformation_type_distribution_pass_rows": dict(
            sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "machine_suggestion_agreement": agreement,
        "sentence_branch": which,
    }
    OUT_JSON.write_text(json.dumps(results, indent=2) + "\n")
    log.info("wrote %s", OUT_JSON)

    md = ["# Definition-16 fidelity audit — results", "",
          f"**Overall pass rate: {x}/{n} = {rate_pct:.1f}%**  "
          f"(95% Wilson interval [{100*lo:.1f}%, {100*hi:.1f}%])", "",
          "## Per-benchmark", ""]
    md += [f"- {b}: {per_bench[b]['pass']}/{per_bench[b]['n']}" for b in sorted(per_bench)]
    md += ["", "## Failure modes (clustered)", ""]
    if merged:
        md += [f"- ({c['count']}) {c['label']} — rows {c['row_ids']}"
               + (f"  *(merged from: {c['merged_from']})*" if len(c['merged_from']) > 1 else "")
               for c in merged]
    else:
        md += ["- none (no failures)"]
    md += ["", "## Transformation types over PASS rows (human column)", ""]
    md += [f"- {t}: {c}" for t, c in
           sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0]))] or ["- (none typed)"]
    md += ["", "## Machine-suggestion agreement (diagnostic only)",
           "",
           f"- suggested ⊆ human on {agreement['suggested_subset_of_human']}"
           f"/{agreement['typed_rows']} typed rows"
           + (f" ({100*agreement['fraction']:.0f}%)" if agreement["fraction"] is not None else ""),
           "", f"Sentence branch selected: **{which}** → `paper_sentence.tex`.", ""]
    OUT_MD.write_text("\n".join(md))
    log.info("wrote %s", OUT_MD)


if __name__ == "__main__":
    main()
