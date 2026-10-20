#!/usr/bin/env python3
"""Every number for the November 2026 revision of the BBL paper, from the frozen results.

Specification: "BBL Revision Experiments: E1-E3 Specification" (2026-10-18), §7.  Outputs go to
``results/derived/2026_11/`` as JSON (one file per table) plus ``summary.md``.  No number in the
manuscript is typed by hand; each table below names the paper location it feeds.

From the frozen 2026_03 results (no new calls):
    T-TM   full <t,f> -> NINE transition matrix under paraphrase (modal results), per model + pooled   §7.1
    T-RK   per-model <t,f> -> <f,f> migration rates with cluster-bootstrap CIs and pairwise differences  §7.1
    T-COV  coverage beside macro-F1 for bilateral (dec) and ternary, per cell (Table 1)               Table 1

From results/2026_11 (E1-0, E1-a, E1-b, E2, E2-0, E3), when present:
    T-E1a  transition matrices original -> E1 value, per arm, condition, model                        §7.2
    T-E1b  persistence under E1-0 / E1-a / E1-b with CIs; paired differences (P1); control stability   §7.2
    T-E1c  direction of collapse x gold label                                                           §7.2 / app.
    F-3'   UNCERTAIN / CONFLICTING / either, by original bilateral value, E2 vs same-day E2-0           replaces Fig. 3
    T-E2   P(CONFLICTING | value) per model with differences and CIs (P2)                              §7.2
    T-E3   compliance base rate c(M) with Wilson CIs beside <t,t> share                                §7.2 / app.

Parsing (analysis-stage rule, 2026-10-18): E1 joint responses are RE-PARSED here from stored raw
text; the ``parsed`` field the evaluator wrote at run time is not used. The evaluator's parser
(``evaluators/revision_2026_11_prompts.py``, frozen with the prompts; its hash is recorded in every
``.meta.json``) requires the whole response to be the two labelled fields. The analysis-stage rule
(``parse_joint_first_line`` below) additionally accepts a well-formed answer followed by an appended
explanation after a line break, flagging it ``trailing_text``; it is applied uniformly to every model.
Effect on the 2026-10-18 run: of 5,091 E1-a/E1-b elicitations, 7 are recovered this way (all Llama 4
Scout: 3 in E1-a, 4 in E1-b) and 725 remain PARSE_FAIL (reasoning in prose, swapped tokens, or two
alternative answers). The frozen evaluator files are unchanged by this rule.

Run:  python analysis/revision_2026_11.py            (all tables whose inputs exist)
      python analysis/revision_2026_11.py --frozen   (T-TM, T-RK, T-COV only)
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
sys.path.insert(0, str(REPO / "evaluators"))

from decision_rule import commit_bool, ALL_NINE  # noqa: E402  canonical verdict rule (dec)
import revision_2026_11_prompts as P  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [rev-2026-11] %(message)s")
log = logging.getLogger(__name__)

PROPER = REPO / "results/2026_03/proper"
MODAL = REPO / "results/2026_03/modal"
NEW = REPO / "results/2026_11"
OUT = REPO / "results/derived/2026_11"

SEED, B = 42, 10_000

MODELS6 = [  # (stem in filenames, OpenRouter/API id, paper label)
    ("claude-opus-4-1-20250805", "claude-opus-4-1-20250805", "Claude Opus 4.1"),
    ("meta-llama_llama-4-maverick", "meta-llama/llama-4-maverick", "Llama 4 Maverick"),
    ("meta-llama_llama-4-scout", "meta-llama/llama-4-scout", "Llama 4 Scout"),
    ("google_gemini-2.5-flash", "google/gemini-2.5-flash", "Gemini 2.5 Flash"),
    ("deepseek_deepseek-chat", "deepseek/deepseek-chat", "DeepSeek-V3"),
    ("qwen_qwen-2.5-72b-instruct", "qwen/qwen-2.5-72b-instruct", "Qwen 2.5 72B"),
]
MODELS5 = MODELS6[1:]
BENCHES = ["truthfulqa", "simpleqa", "mmlu-pro", "factscore"]
LABEL = {mid: lab for _, mid, lab in MODELS6}
NINE = list(ALL_NINE)


# ----------------------------------------------------------------------------- helpers

def percentile(sorted_vals: list[float], q: float) -> float:
    n = len(sorted_vals)
    pos = (q / 100.0) * (n - 1)
    lo = int(pos); hi = min(lo + 1, n - 1); frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def cluster_bootstrap(cells: list, stat, rng: random.Random, b: int = B) -> tuple[float, float, float]:
    """Point estimate and 95% percentile CI of stat(cells) under resampling of cells with replacement.

    ``stat`` maps a list of cells to a float (or None when undefined for that draw)."""
    point = stat(cells)
    n = len(cells)
    draws = []
    for _ in range(b):
        s = stat([cells[rng.randrange(n)] for _ in range(n)])
        if s is not None:
            draws.append(s)
    draws.sort()
    if not draws:
        return point, float("nan"), float("nan")
    return point, percentile(draws, 2.5), percentile(draws, 97.5)


def rate(num: int, den: int) -> float | None:
    return num / den if den else None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def dump(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=2) + "\n")
    log.info("wrote %s.json", name)


# ----------------------------------------------------------------------------- frozen-data tables

def t_tm() -> dict:
    """T-TM: for assertions with VM(s0,p) = <t,f>, the distribution of [[box p]](s0) over NINE."""
    per_model, pooled = {}, collections.Counter()
    pooled_n = 0
    for stem, mid, lab in MODELS6:
        c = collections.Counter(); n = 0
        for bench in BENCHES:
            p = next(MODAL.glob(f"{bench}_complete_{stem}_n3_modal_results.json"), None)
            if p is None:
                continue
            for r in json.loads(p.read_text())["detailed_results"]:
                if r.get("src_bilateral") == "<t,f>":
                    c[r.get("modal_bilateral")] += 1; n += 1
        per_model[lab] = {"n_tf": n, **{v: {"count": c[v], "share": round(c[v] / n, 4) if n else None} for v in NINE}}
        pooled.update(c); pooled_n += n
    out = {"description": "Full <t,f> -> NINE transition under paraphrase: share of <t,f>-at-s0 assertions whose "
                          "[[box p]](s0) takes each value. Figure 2 reports only the <f,f> column.",
           "per_model": per_model,
           "pooled": {"n_tf": pooled_n, **{v: {"count": pooled[v], "share": round(pooled[v] / pooled_n, 4)} for v in NINE}}}
    dump("T-TM_transition_matrix", out)
    return out


def t_rk() -> dict:
    """T-RK: per-model <t,f> -> <f,f> migration with cluster-bootstrap CIs; pairwise differences and rank stability."""
    rng = random.Random(SEED)
    cells_by_model: dict[str, list[tuple[int, int]]] = {}
    for stem, mid, lab in MODELS6:
        cells = []
        for bench in BENCHES:
            p = next(MODAL.glob(f"{bench}_complete_{stem}_n3_modal_results.json"), None)
            if p is None:
                continue
            n = k = 0
            for r in json.loads(p.read_text())["detailed_results"]:
                if r.get("src_bilateral") == "<t,f>":
                    n += 1; k += r.get("modal_bilateral") == "<f,f>"
            cells.append((n, k))
        cells_by_model[lab] = cells

    def mig(cells):
        n = sum(c[0] for c in cells); k = sum(c[1] for c in cells)
        return 100.0 * k / n if n else None

    rates = {}
    for lab, cells in cells_by_model.items():
        pt, lo, hi = cluster_bootstrap(cells, mig, rng)
        rates[lab] = {"n_tf": sum(c[0] for c in cells), "migration_pct": round(pt, 2), "ci95": [round(lo, 2), round(hi, 2)],
                      "note": "4 benchmark cells per model; CI from resampling those 4 cells"}

    # Pairwise differences: joint resampling of the two models' cells.
    labs = list(cells_by_model)
    pairs = {}
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            a, b_ = cells_by_model[labs[i]], cells_by_model[labs[j]]
            joint = list(zip(a, b_))  # cells are aligned by benchmark

            def diff(cells):
                ra = mig([c[0] for c in cells]); rb = mig([c[1] for c in cells])
                return None if ra is None or rb is None else ra - rb
            pt, lo, hi = cluster_bootstrap(joint, diff, rng)
            pairs[f"{labs[i]} - {labs[j]}"] = {"diff_pp": round(pt, 2), "ci95": [round(lo, 2), round(hi, 2)],
                                               "excludes_zero": bool(lo > 0 or hi < 0)}
    order = sorted(rates, key=lambda l: -rates[l]["migration_pct"])
    out = {"description": "Per-model <t,f> -> <f,f> migration under paraphrase (the Figure 2 column) with 95% "
                          "cluster-bootstrap CIs over benchmark cells, and all pairwise differences with CIs. "
                          "Rank orderings whose pairwise CI excludes zero are the ones the data support.",
           "seed": SEED, "resamples": B, "rates": rates, "ranking_by_point_estimate": order, "pairwise": pairs,
           "supported_orderings": [k for k, v in pairs.items() if v["excludes_zero"]]}
    dump("T-RK_migration_rankings", out)
    return out


def t_cov() -> dict:
    """T-COV: coverage beside macro-F1, bilateral (dec) and ternary, per cell; abstention handling stated."""
    sys.path.insert(0, str(REPO / "analysis"))
    from recompute_decision_rule import compute_metrics, rule_ternary  # noqa: E402
    rows = []
    for bench in BENCHES:
        for stem, mid, lab in MODELS6:
            p = PROPER / f"{bench}_{stem}_n3_proper_results.json"
            recs = [r for r in json.loads(p.read_text())["detailed_results"] if r.get("bilateral_tv")]
            gts = [bool(r["ground_truth"]) for r in recs]
            mb = compute_metrics([commit_bool(r["bilateral_tv"]) for r in recs], gts)
            mt = compute_metrics([rule_ternary(r) for r in recs], gts)
            rows.append({"benchmark": bench, "model": lab, "n": mb["total"],
                         "bilateral_f1": round(mb["f1_macro"], 3), "bilateral_coverage": round(mb["coverage"], 3),
                         "ternary_f1": round(mt["f1_macro"], 3), "ternary_coverage": round(mt["coverage"], 3),
                         "delta_f1": round(mb["f1_macro"] - mt["f1_macro"], 3)})
    out = {"abstention_handling": "macro-F1 is computed on covered items only (abstentions excluded from both "
                                  "numerator and denominator); coverage = covered / N is reported beside it. "
                                  "Bilateral abstains on dec = abstain (<t,t> and the neither-cells); ternary on UNCERTAIN. "
                                  "Replicates compute_metrics in proper_benchmark_evaluator.py.",
           "rows": rows}
    dump("T-COV_table1_coverage", out)
    return out


# ----------------------------------------------------------------------------- analysis-stage E1 parsing

# The evaluator's rule (``P.parse_joint``, frozen with the prompts) requires the WHOLE response to be
# the two labelled fields. The analysis-stage rule below additionally accepts a well-formed answer
# that the model followed with an explanation after a line break; that trailing text is ignored and
# the elicitation flagged. Leading text, a missing or swapped field, an illegal token, or two
# alternative answers remain PARSE_FAIL. Applied uniformly to every model by re-parsing raw text.
_JOINT_HEAD_RE = re.compile(
    r"^VERIFICATION:\s*(VERIFIED|CANNOT VERIFY)\s*[;|,]?\s*REFUTATION:\s*(REFUTED|CANNOT REFUTE)\s*$"
)


def parse_joint_first_line(text: str) -> tuple[str, str] | str:
    """E1 joint response -> (verification_token, refutation_token) under the first-line rule, else PARSE_FAIL."""
    strict = P.parse_joint(text)
    if strict != P.PARSE_FAIL:
        return strict
    lines = [ln.strip() for ln in text.strip().upper().split("\n") if ln.strip()]
    if not lines:
        return P.PARSE_FAIL
    head = lines[0]
    if "REFUTATION:" not in head and len(lines) > 1:  # fields split across two lines
        head = head + " " + lines[1]
    head = re.sub(r"\s+", " ", head)
    if head.endswith("."):
        head = head[:-1]
    m = _JOINT_HEAD_RE.match(head)
    return (m.group(1), m.group(2)) if m else P.PARSE_FAIL


def has_trailing_text(text: str) -> bool:
    """True when the first-line rule recovered an answer the evaluator's whole-response rule rejected."""
    return parse_joint_first_line(text) != P.PARSE_FAIL and P.parse_joint(text) == P.PARSE_FAIL


# ----------------------------------------------------------------------------- new-data tables

def reparse_e1_record(rec: dict) -> tuple[str, int, int, int]:
    """Re-derive the E1 bilateral value from raw text under the final parsing rule.

    Returns (tv, n_parse_fail, n_api_error, n_trailing_text) for the record."""
    els = rec["elicitations"]
    pf = ae = tt = 0
    if isinstance(els, dict):  # E1-0: separate verification / refutation calls
        u, w = [], []
        for e in els["verification"]:
            if e.get("error"):
                ae += 1; u.append("e")
            else:
                tok = P.parse_single(e["raw"] or "", P.VERIFICATION_TOKENS)
                pf += tok == P.PARSE_FAIL; u.append(P.component(tok))
        for e in els["refutation"]:
            if e.get("error"):
                ae += 1; w.append("e")
            else:
                tok = P.parse_single(e["raw"] or "", P.REFUTATION_TOKENS)
                pf += tok == P.PARSE_FAIL; w.append(P.component(tok))
        return f"<{P.majority(u)},{P.majority(w)}>", pf, ae, 0
    u, w = [], []
    for e in els:
        if e.get("error"):
            ae += 1; u.append("e"); w.append("e"); continue
        parsed = parse_joint_first_line(e["raw"] or "")
        if parsed == P.PARSE_FAIL:
            pf += 1; u.append("e"); w.append("e")
        else:
            tt += has_trailing_text(e["raw"] or "")
            u.append(P.component(parsed[0])); w.append(P.component(parsed[1]))
    return f"<{P.majority(u)},{P.majority(w)}>", pf, ae, tt


def load_e1(mode: str) -> dict[tuple[str, str], dict]:
    """(model_id, assertion_id) -> record with re-parsed tv, for one arm."""
    out = {}
    for p in (NEW / mode).glob(f"{mode}_*.jsonl"):
        for r in load_jsonl(p):
            tv, pf, ae, tt = reparse_e1_record(r)
            r["tv_final"] = tv; r["n_parse_fail"] = pf; r["n_api_error"] = ae; r["n_trailing_text"] = tt
            out[(r["model"], r["assertion_id"])] = r
    return out


def t_e1() -> dict | None:
    arms = {m: load_e1(m) for m in ("e1-0", "e1-a", "e1-b")}
    if not arms["e1-a"]:
        log.info("no E1 results yet; skipping T-E1*")
        return None
    rng = random.Random(SEED)
    out: dict = {"parsing": "re-parsed from raw under the final rule; <e,e> from parse failure reported separately "
                            "and excluded from persistence denominators (count stated)"}

    # T-E1a transition matrices and T-E1b rates, per arm x condition x model
    def cell_key(r):
        return (r["model"], r["benchmark"])

    rates = {}
    matrices = {}
    for arm, recs in arms.items():
        if not recs:
            continue
        for cond in ("treatment", "control_tf", "control_ff"):
            sub = [r for r in recs.values() if r["condition"] == cond]
            if not sub:
                continue
            orig = {"treatment": "<t,t>", "control_tf": "<t,f>", "control_ff": "<f,f>"}[cond]
            # matrices
            for scope in ["pooled"] + sorted({r["model"] for r in sub}):
                rows = sub if scope == "pooled" else [r for r in sub if r["model"] == scope]
                c = collections.Counter(r["tv_final"] for r in rows)
                matrices[f"{arm}|{cond}|{LABEL.get(scope, scope)}"] = {
                    "from": orig, "n": len(rows), **{v: c[v] for v in NINE},
                    "n_ee_from_failures": sum(1 for r in rows if r["tv_final"] == "<e,e>" and (r["n_parse_fail"] or r["n_api_error"])),
                    "n_with_trailing_text": sum(1 for r in rows if r["n_trailing_text"])}
            # persistence with CIs: cells = (model, benchmark); stat = share of valid items keeping orig
            cells = collections.defaultdict(lambda: [0, 0])
            for r in sub:
                if r["tv_final"] == "<e,e>" and (r["n_parse_fail"] or r["n_api_error"]):
                    continue  # excluded, counted above
                k = cell_key(r); cells[k][0] += 1; cells[k][1] += r["tv_final"] == orig
            for scope in ["pooled"] + sorted({k[0] for k in cells}):
                cs = [tuple(v) for k, v in cells.items() if scope == "pooled" or k[0] == scope]
                if not cs:
                    continue
                pt, lo, hi = cluster_bootstrap(cs, lambda c: rate(sum(x[1] for x in c), sum(x[0] for x in c)), rng)
                n = sum(x[0] for x in cs)
                rates[f"{arm}|{cond}|{LABEL.get(scope, scope)}"] = {
                    "n_valid": n, "persistence": round(pt, 4) if pt is not None else None,
                    "ci95": [round(lo, 4), round(hi, 4)] if n >= 10 else None}
    out["T-E1a_matrices"] = matrices
    out["T-E1b_persistence"] = rates

    # P1: paired difference E1-0 minus E1-a on treatment items present in both arms (and E1-a minus E1-b).
    def paired(arm_x, arm_y, cond="treatment"):
        orig = {"treatment": "<t,t>", "control_tf": "<t,f>", "control_ff": "<f,f>"}[cond]
        X, Y = arms[arm_x], arms[arm_y]
        cells = collections.defaultdict(lambda: [0, 0, 0])  # n, keep_x, keep_y
        for key, rx in X.items():
            ry = Y.get(key)
            if ry is None or rx["condition"] != cond:
                continue
            bad = lambda r: r["tv_final"] == "<e,e>" and (r["n_parse_fail"] or r["n_api_error"])
            if bad(rx) or bad(ry):
                continue
            k = cell_key(rx); cells[k][0] += 1; cells[k][1] += rx["tv_final"] == orig; cells[k][2] += ry["tv_final"] == orig
        res = {}
        for scope in ["pooled"] + sorted({k[0] for k in cells}):
            cs = [tuple(v) for k, v in cells.items() if scope == "pooled" or k[0] == scope]
            if not cs:
                continue
            pt, lo, hi = cluster_bootstrap(cs, lambda c: (rate(sum(x[1] for x in c), sum(x[0] for x in c)) or 0) -
                                                         (rate(sum(x[2] for x in c), sum(x[0] for x in c)) or 0), rng)
            res[LABEL.get(scope, scope)] = {"n_paired": sum(x[0] for x in cs), "diff": round(pt, 4),
                                            "ci95": [round(lo, 4), round(hi, 4)] if sum(x[0] for x in cs) >= 10 else None}
        return res
    if arms["e1-0"]:
        out["P1_E1-0_minus_E1-a_treatment"] = paired("e1-0", "e1-a")
        out["E1-0_minus_E1-a_control_tf"] = paired("e1-0", "e1-a", "control_tf")
        out["E1-0_minus_E1-a_control_ff"] = paired("e1-0", "e1-a", "control_ff")
    if arms["e1-b"]:
        out["E1-a_minus_E1-b_treatment"] = paired("e1-a", "e1-b")

    # T-E1c: direction of collapse x gold, treatment items in E1-a that did not persist
    direction = collections.Counter()
    for r in arms["e1-a"].values():
        if r["condition"] != "treatment" or r["tv_final"] in ("<t,t>", "<e,e>"):
            continue
        u, w = r["tv_final"][1], r["tv_final"][3]
        dropped = ("refutation" if w != "t" else "") + ("verification" if u != "t" else "")
        if u != "t" and w != "t":
            dropped = "both"
        direction[(LABEL[r["model"]], "gold_TRUE" if r["ground_truth"] else "gold_FALSE", dropped or "other")] += 1
    out["T-E1c_collapse_direction"] = {f"{m}|{g}|{d}": n for (m, g, d), n in sorted(direction.items())}
    out["T-E1c_pooled"] = dict(collections.Counter((g, d) for (m, g, d), n in direction.items() for _ in range(n)).items()) \
        if direction else {}
    out["T-E1c_pooled"] = {f"{g}|{d}": n for (g, d), n in out["T-E1c_pooled"].items()}
    dump("T-E1_neutral_reelicitation", out)
    return out


def t_e2() -> dict | None:
    e2 = {(r["model"], r["assertion_id"]): r for p in (NEW / "e2").glob("e2_*.jsonl") for r in load_jsonl(p)}
    if not e2:
        log.info("no E2 results yet; skipping T-E2 / F-3'")
        return None
    e20 = {(r["model"], r["assertion_id"]): r for p in (NEW / "e2-0").glob("e2-0_*.jsonl") for r in load_jsonl(p)}
    rng = random.Random(SEED)

    def pred(r, legal):
        e = r["elicitations"][0]
        return P.API_ERROR if e.get("error") else P.parse_single(e["raw"] or "", legal)

    # F-3': by original bilateral value, shares of UNCERTAIN / CONFLICTING / either (E2) and UNCERTAIN (E2-0, same items)
    f3 = {}
    for v in NINE:
        rows = [r for r in e2.values() if r["orig_bilateral_tv"] == v]
        preds = [pred(r, P.FOUR_OPTION_TOKENS) for r in rows]
        valid = [q for q in preds if q not in (P.PARSE_FAIL, P.API_ERROR)]
        same = [e20[k] for k in e2 if k in e20 and e2[k]["orig_bilateral_tv"] == v]
        p0 = [pred(r, P.TERNARY_TOKENS) for r in same]
        v0 = [q for q in p0 if q not in (P.PARSE_FAIL, P.API_ERROR)]
        f3[v] = {"n": len(rows), "n_valid": len(valid),
                 "uncertain": round(valid.count("UNCERTAIN") / len(valid), 4) if valid else None,
                 "conflicting": round(valid.count("CONFLICTING") / len(valid), 4) if valid else None,
                 "either": round((valid.count("UNCERTAIN") + valid.count("CONFLICTING")) / len(valid), 4) if valid else None,
                 "e2-0_same_day_n": len(v0),
                 "e2-0_same_day_uncertain": round(v0.count("UNCERTAIN") / len(v0), 4) if v0 else None,
                 "parse_fail": preds.count(P.PARSE_FAIL), "api_error": preds.count(P.API_ERROR)}

    # P2: P(CONFLICTING|<t,t>) - P(CONFLICTING|<t,f> u <f,f>), cells = (model, benchmark)
    cells = collections.defaultdict(lambda: [0, 0, 0, 0])  # n_tt, c_tt, n_ctrl, c_ctrl
    for r in e2.values():
        q = pred(r, P.FOUR_OPTION_TOKENS)
        if q in (P.PARSE_FAIL, P.API_ERROR):
            continue
        k = (r["model"], r["benchmark"])
        if r["orig_bilateral_tv"] == "<t,t>":
            cells[k][0] += 1; cells[k][1] += q == "CONFLICTING"
        elif r["orig_bilateral_tv"] in ("<t,f>", "<f,f>"):
            cells[k][2] += 1; cells[k][3] += q == "CONFLICTING"

    def gap(cs):
        a = rate(sum(c[1] for c in cs), sum(c[0] for c in cs)); b_ = rate(sum(c[3] for c in cs), sum(c[2] for c in cs))
        return None if a is None or b_ is None else a - b_
    p2 = {}
    for scope in ["pooled"] + sorted({k[0] for k in cells}):
        cs = [tuple(v) for k, v in cells.items() if scope == "pooled" or k[0] == scope]
        pt, lo, hi = cluster_bootstrap(cs, gap, rng)
        p2[LABEL.get(scope, scope)] = {
            "n_tt": sum(c[0] for c in cs), "p_conflicting_tt": round(rate(sum(c[1] for c in cs), sum(c[0] for c in cs)) or 0, 4),
            "n_ctrl": sum(c[2] for c in cs), "p_conflicting_ctrl": round(rate(sum(c[3] for c in cs), sum(c[2] for c in cs)) or 0, 4),
            "gap": round(pt, 4) if pt is not None else None,
            "ci95": [round(lo, 4), round(hi, 4)] if sum(c[0] for c in cs) >= 10 else None}
    # migration from committed (E2-0 TRUE/FALSE) to CONFLICTING, by original value
    mig = collections.Counter(); den = collections.Counter()
    for k, r in e2.items():
        r0 = e20.get(k)
        if not r0:
            continue
        q0 = pred(r0, P.TERNARY_TOKENS); q = pred(r, P.FOUR_OPTION_TOKENS)
        if q0 in ("TRUE", "FALSE") and q not in (P.PARSE_FAIL, P.API_ERROR):
            den[r["orig_bilateral_tv"]] += 1; mig[r["orig_bilateral_tv"]] += q == "CONFLICTING"
    out = {"F3prime_by_original_value": f3, "P2_conflicting_gap": p2,
           "committed_to_conflicting_same_day": {v: {"n_committed_e2-0": den[v], "conflicting_in_e2": mig[v],
                                                     "share": round(mig[v] / den[v], 4) if den[v] else None} for v in NINE if den[v]}}
    dump("T-E2_four_option", out)
    return out


def t_e3() -> dict | None:
    recs = [r for p in (NEW / "e3").glob("e3_*.jsonl") for r in load_jsonl(p)]
    if not recs:
        log.info("no E3 results yet; skipping T-E3")
        return None
    tt_share = {}
    for stem, mid, lab in MODELS5:
        n = k = 0
        for bench in BENCHES:
            for r in json.loads((PROPER / f"{bench}_{stem}_n3_proper_results.json").read_text())["detailed_results"]:
                n += 1; k += r["bilateral_tv"] == "<t,t>"
        tt_share[mid] = k / n
    rows = {}
    present = {r["model"] for r in recs}
    for _, mid, _ in MODELS5:  # fixed order: a set here made the key order vary with the hash seed
        if mid not in present:
            continue
        sub = [r for r in recs if r["model"] == mid]
        valid = [r for r in sub if r.get("complied") is not None]
        k = sum(1 for r in valid if r["complied"]); n = len(valid)
        lo, hi = wilson(k, n)
        rows[LABEL[mid]] = {"n_valid": n, "complied": k, "c_M": round(k / n, 4) if n else None, "wilson95": [round(lo, 4), round(hi, 4)],
                            "by_probe": {pr: {"n": sum(1 for r in valid if r["probe"] == pr),
                                              "complied": sum(1 for r in valid if r["probe"] == pr and r["complied"])}
                                         for pr in ("refutation", "verification")},
                            "tt_share_2026_03": round(tt_share[mid], 4)}
    # Ties (DeepSeek-V3 and Qwen 2.5 72B share c_M = 0.020) are broken by label so the order is defined.
    order_c = sorted(rows, key=lambda l: (-(rows[l]["c_M"] or 0), l))
    order_tt = sorted(rows, key=lambda l: (-rows[l]["tt_share_2026_03"], l))
    out = {"rows": rows, "rank_by_compliance": order_c, "rank_by_tt_share": order_tt,
           "note": "five points; report the table and rank agreement, not a correlation coefficient; "
                   "ties in the rank lists are broken alphabetically by label"}
    dump("T-E3_compliance_base_rate", out)
    return out


# ----------------------------------------------------------------------------- summary

def summary_md(tm, rk, cov, e1, e2, e3) -> None:
    L = ["# Revision 2026-11: derived numbers", "", f"Seed {SEED}, {B} bootstrap resamples. Files: `results/derived/2026_11/`.", ""]
    L += ["## T-TM  <t,f> -> NINE under paraphrase (pooled, 6 models)", "",
          "| value | count | share |", "| --- | --- | --- |"]
    for v in NINE:
        L.append(f"| {v} | {tm['pooled'][v]['count']} | {tm['pooled'][v]['share']:.3f} |")
    L += ["", "## T-RK  <t,f> -> <f,f> migration by model (95% CI over benchmark cells)", "",
          "| model | n <t,f> | migration % | CI |", "| --- | --- | --- | --- |"]
    for lab in rk["ranking_by_point_estimate"]:
        r = rk["rates"][lab]; L.append(f"| {lab} | {r['n_tf']} | {r['migration_pct']:.1f} | [{r['ci95'][0]:.1f}, {r['ci95'][1]:.1f}] |")
    L += ["", f"Pairwise orderings whose CI excludes zero: {', '.join(rk['supported_orderings']) or 'none'}", ""]
    L += ["## T-COV  Table 1 with coverage", "", "| benchmark | model | bilateral F1 (cov) | ternary F1 (cov) | dF1 |", "| --- | --- | --- | --- | --- |"]
    for r in cov["rows"]:
        L.append(f"| {r['benchmark']} | {r['model']} | {r['bilateral_f1']:.3f} ({r['bilateral_coverage']:.2f}) | "
                 f"{r['ternary_f1']:.3f} ({r['ternary_coverage']:.2f}) | {r['delta_f1']:+.3f} |")
    L += ["", cov["abstention_handling"], ""]
    if e1:
        L += ["## T-E1b  persistence (re-parsed)", "", "| arm | condition | scope | n valid | persistence | CI |", "| --- | --- | --- | --- | --- | --- |"]
        for k, v in e1["T-E1b_persistence"].items():
            arm, cond, scope = k.split("|")
            ci = f"[{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]" if v["ci95"] else "n<10"
            L.append(f"| {arm} | {cond} | {scope} | {v['n_valid']} | {v['persistence']:.3f} | {ci} |")
        if "P1_E1-0_minus_E1-a_treatment" in e1:
            L += ["", "### P1  E1-0 minus E1-a, treatment (paired)", "", "| scope | n | diff | CI |", "| --- | --- | --- | --- |"]
            for s, v in e1["P1_E1-0_minus_E1-a_treatment"].items():
                ci = f"[{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]" if v["ci95"] else "n<10"
                L.append(f"| {s} | {v['n_paired']} | {v['diff']:+.3f} | {ci} |")
        L += ["", "### T-E1c  collapse direction x gold (pooled, E1-a treatment)", ""] + [f"- {k}: {n}" for k, n in e1["T-E1c_pooled"].items()]
    if e2:
        L += ["", "## F-3'  by original bilateral value", "", "| value | n | UNCERTAIN | CONFLICTING | either | E2-0 UNCERTAIN (same day) |", "| --- | --- | --- | --- | --- | --- |"]
        for v, r in e2["F3prime_by_original_value"].items():
            if r["n"]:
                f = lambda x: "-" if x is None else f"{x:.3f}"
                L.append(f"| {v} | {r['n']} | {f(r['uncertain'])} | {f(r['conflicting'])} | {f(r['either'])} | {f(r['e2-0_same_day_uncertain'])} |")
        L += ["", "### P2  P(CONFLICTING|<t,t>) - P(CONFLICTING|controls)", "", "| scope | n <t,t> | P(C|tt) | P(C|ctrl) | gap | CI |", "| --- | --- | --- | --- | --- | --- |"]
        for s, v in e2["P2_conflicting_gap"].items():
            ci = f"[{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]" if v["ci95"] else "n<10"
            gap = "-" if v["gap"] is None else f"{v['gap']:+.3f}"
            L.append(f"| {s} | {v['n_tt']} | {v['p_conflicting_tt']:.3f} | {v['p_conflicting_ctrl']:.3f} | {gap} | {ci} |")
    if e3:
        L += ["", "## T-E3  compliance base rate", "", "| model | n | c(M) | Wilson 95% | <t,t> share (Mar) |", "| --- | --- | --- | --- | --- |"]
        for lab, r in e3["rows"].items():
            L.append(f"| {lab} | {r['n_valid']} | {r['c_M']:.3f} | [{r['wilson95'][0]:.3f}, {r['wilson95'][1]:.3f}] | {r['tt_share_2026_03']:.3f} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    log.info("wrote summary.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", action="store_true", help="only the tables from the frozen 2026_03 data")
    args = ap.parse_args()
    tm, rk, cov = t_tm(), t_rk(), t_cov()
    e1 = e2 = e3 = None
    if not args.frozen:
        e1, e2, e3 = t_e1(), t_e2(), t_e3()
    summary_md(tm, rk, cov, e1, e2, e3)


if __name__ == "__main__":
    main()
