#!/usr/bin/env python3
"""Analysis of E4 / E4-0: paraphrase versus identical repetition, paired within item.

Specification: docs/revision_2026_11_E4_spec_addition.md, §6. Every value is re-derived from the
stored raw response text with the frozen single-token parser; the convenience fields the evaluator
wrote (``tv``, ``box_rep``, ``box_par``) are not used.

For each item: base = <u,v> at s0; box_rep = <min u, max v> over base + the k identical repeats;
box_par = the same over base + the k cached paraphrases. Items with an API error in any call are
excluded from both arms (counted). The primary analyses condition on the SAME-DAY base being <t,f>.

Tables (results/derived/2026_11_e4/T-E4_paraphrase_vs_repeat.json, plus summary.md):
    T-E4a  distribution of box_rep and box_par over NINE, pooled and per model
    T-E4b  any-change and -> <f,f> rates per arm; paired difference (paraphrase - repeat) with a
           95% cluster-bootstrap CI over (model, benchmark) cells; discordance counts; P3 verdict
    T-E4c  per-model paraphrase-minus-repeat excess with CIs; pairwise model orderings on the excess
    secondary  same-day base distribution of all items (drift since the March run), and the March
           run's own figures for the same items, for reference

Run:  python analysis/revision_e4.py            (reads results/2026_11_e4/)
      python analysis/revision_e4.py --dir results/2026_11_e4/dry_run --out /tmp/x   (offline check)
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
sys.path.insert(0, str(REPO / "evaluators"))

import revision_2026_11_prompts as P  # noqa: E402  frozen parser
from revision_2026_11 import cluster_bootstrap, rate, NINE, SEED, B  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [rev-e4] %(message)s")
log = logging.getLogger(__name__)

MODELS5 = [("meta-llama/llama-4-maverick", "Llama 4 Maverick"), ("meta-llama/llama-4-scout", "Llama 4 Scout"),
           ("google/gemini-2.5-flash", "Gemini 2.5 Flash"), ("deepseek/deepseek-chat", "DeepSeek-V3"),
           ("qwen/qwen-2.5-72b-instruct", "Qwen 2.5 72B")]
LABEL = dict(MODELS5)
BENCHES = ["truthfulqa", "simpleqa", "mmlu-pro", "factscore"]
_ORD = {"f": 0, "e": 1, "t": 2}
TF, FF = "<t,f>", "<f,f>"


def box(tvs: list[str]) -> str:
    return f"<{'fet'[min(_ORD[t[1]] for t in tvs)]},{'fet'[max(_ORD[t[3]] for t in tvs)]}>"


def reparse(rec: dict) -> dict | None:
    """Re-derive base / box_rep / box_par from raw text; None if any call was an API error."""
    tvs = {"base": [], "rep": [], "par": []}
    for s in rec["situations"]:
        comp = []
        for e, legal in ((s["verification"], P.VERIFICATION_TOKENS), (s["refutation"], P.REFUTATION_TOKENS)):
            if e.get("error"):
                return None
            comp.append(P.component(P.parse_single(e["raw"] or "", legal)))
        tvs[s["arm"]].append(f"<{comp[0]},{comp[1]}>")
    base = tvs["base"][0]
    return {"model": rec["model"], "benchmark": rec["benchmark"], "assertion_id": rec["assertion_id"],
            "base": base, "box_rep": box([base] + tvs["rep"]), "box_par": box([base] + tvs["par"]),
            "march_box": rec["orig_modal_box"], "k": rec["k"]}


def ci(lo: float, hi: float, nd: int = 4) -> list[float]:
    return [round(lo, nd), round(hi, nd)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=REPO / "results" / "2026_11_e4")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "derived" / "2026_11_e4")
    args = ap.parse_args()

    raw, metas = [], {}
    for mid, lab in MODELS5:
        stem = mid.replace("/", "_")
        p = args.dir / f"e4_{stem}.jsonl"
        if not p.exists():
            log.info("no E4 file for %s; omitted", lab)
            continue
        raw += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        m = json.loads((args.dir / f"e4_{stem}.meta.json").read_text())
        metas[lab] = {k: m.get(k) for k in ("n_items", "n_records", "n_elicitations", "parse_fail_rate", "api_error_rate",
                                           "provider_pinned", "providers_seen", "models_returned", "started_at",
                                           "finished_at", "harness_commit")}
    models = [(mid, lab) for mid, lab in MODELS5 if lab in metas]
    recs = [reparse(r) for r in raw]
    n_api = sum(r is None for r in recs)
    recs = [r for r in recs if r is not None]
    prim = [r for r in recs if r["base"] == TF]
    log.info("%d records, %d excluded for API errors, %d with same-day base <t,f>", len(raw), n_api, len(prim))
    rng = random.Random(SEED)

    # ---- T-E4a: distributions over NINE --------------------------------------------------------
    def dist(rows: list[dict], key: str) -> dict:
        c = collections.Counter(r[key] for r in rows)
        return {"n": len(rows), **{v: {"count": c[v], "share": round(c[v] / len(rows), 4) if rows else None} for v in NINE}}

    scopes = [("pooled", prim)] + [(lab, [r for r in prim if r["model"] == mid]) for mid, lab in models]
    t_a = {scope: {"repeat": dist(rows, "box_rep"), "paraphrase": dist(rows, "box_par")} for scope, rows in scopes}

    # ---- T-E4b: rates and paired difference -----------------------------------------------------
    def cells_for(rows: list[dict]) -> list[tuple]:
        """One cell per (model, benchmark): (n, rep_changed, par_changed, rep_ff, par_ff)."""
        c = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
        for r in rows:
            k = (r["model"], r["benchmark"])
            c[k][0] += 1
            c[k][1] += r["box_rep"] != TF; c[k][2] += r["box_par"] != TF
            c[k][3] += r["box_rep"] == FF; c[k][4] += r["box_par"] == FF
        order = [(mid, b) for mid, _ in MODELS5 for b in BENCHES]
        return [tuple(c[k]) for k in order if k in c]

    def pooled(cs, i):
        return rate(sum(x[i] for x in cs), sum(x[0] for x in cs))

    t_b = {}
    for scope, rows in scopes:
        cs = cells_for(rows)
        if not cs:
            continue
        n = sum(x[0] for x in cs)
        out = {"n": n, "n_cells": len(cs)}
        for name, ri, pi in (("any_change", 1, 2), ("to_ff", 3, 4)):
            pr, lr, hr = cluster_bootstrap(cs, lambda c, i=ri: pooled(c, i), rng)
            pp, lp, hp = cluster_bootstrap(cs, lambda c, i=pi: pooled(c, i), rng)
            pd, ld, hd = cluster_bootstrap(cs, lambda c, a=pi, b=ri: (pooled(c, a) or 0) - (pooled(c, b) or 0), rng)
            out[name] = {"repeat": round(pr, 4), "repeat_ci95": ci(lr, hr), "paraphrase": round(pp, 4),
                         "paraphrase_ci95": ci(lp, hp), "difference": round(pd, 4), "difference_ci95": ci(ld, hd)}
        out["discordance_any_change"] = {
            "neither": sum(r["box_rep"] == TF and r["box_par"] == TF for r in rows),
            "paraphrase_only": sum(r["box_rep"] == TF and r["box_par"] != TF for r in rows),
            "repeat_only": sum(r["box_rep"] != TF and r["box_par"] == TF for r in rows),
            "both": sum(r["box_rep"] != TF and r["box_par"] != TF for r in rows)}
        t_b[scope] = out
    p3 = t_b.get("pooled", {}).get("any_change", {})
    p3_out = {"statement": "pooled any-change rate under paraphrase exceeds that under identical repetition, "
                           "and the CI of the paired difference excludes zero",
              "difference": p3.get("difference"), "ci95": p3.get("difference_ci95"),
              "holds": bool(p3 and p3["difference_ci95"][0] > 0)}

    # ---- T-E4c: per-model excess and pairwise orderings ---------------------------------------
    by_model = {lab: {b: [0, 0, 0] for b in BENCHES} for _, lab in models}   # n, rep_changed, par_changed
    for r in prim:
        c = by_model[LABEL[r["model"]]][r["benchmark"]]
        c[0] += 1; c[1] += r["box_rep"] != TF; c[2] += r["box_par"] != TF

    def excess(cs):
        n = sum(c[0] for c in cs)
        return None if not n else (sum(c[2] for c in cs) - sum(c[1] for c in cs)) / n

    labs = [lab for _, lab in models]
    t_c = {"excess_any_change": {}, "pairwise": {}}
    for lab in labs:
        cs = [tuple(by_model[lab][b]) for b in BENCHES]
        pt, lo, hi = cluster_bootstrap(cs, excess, rng)
        t_c["excess_any_change"][lab] = {"n": sum(c[0] for c in cs), "excess": round(pt, 4), "ci95": ci(lo, hi),
                                         "note": "4 benchmark cells; CI from resampling those cells"}
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            joint = [(tuple(by_model[labs[i]][b]), tuple(by_model[labs[j]][b])) for b in BENCHES]

            def diff(cs):
                a, b_ = excess([c[0] for c in cs]), excess([c[1] for c in cs])
                return None if a is None or b_ is None else a - b_
            pt, lo, hi = cluster_bootstrap(joint, diff, rng)
            t_c["pairwise"][f"{labs[i]} - {labs[j]}"] = {"diff": round(pt, 4), "ci95": ci(lo, hi),
                                                         "excludes_zero": bool(lo > 0 or hi < 0)}
    t_c["ranking_by_excess"] = sorted(labs, key=lambda l: (-t_c["excess_any_change"][l]["excess"], l))
    t_c["supported_orderings"] = [k for k, v in t_c["pairwise"].items() if v["excludes_zero"]]

    # ---- secondary: drift since March, and March's own figures on the same items ---------------
    base_c = collections.Counter(r["base"] for r in recs)
    sec = {"n_items_no_api_error": len(recs), "n_excluded_api_error": n_api,
           "same_day_base": {v: base_c[v] for v in NINE if base_c[v]},
           "share_base_not_tf": round(1 - base_c[TF] / len(recs), 4) if recs else None,
           "march_run_same_items": {
               "all_items": {"n": len(recs), "any_change": round(rate(sum(r["march_box"] != TF for r in recs), len(recs)) or 0, 4),
                             "to_ff": round(rate(sum(r["march_box"] == FF for r in recs), len(recs)) or 0, 4)},
               "items_with_same_day_base_tf": {"n": len(prim), "any_change": round(rate(sum(r["march_box"] != TF for r in prim), len(prim)) or 0, 4),
                                               "to_ff": round(rate(sum(r["march_box"] == FF for r in prim), len(prim)) or 0, 4)}}}

    result = {"spec": "docs/revision_2026_11_E4_spec_addition.md", "seed": SEED, "resamples": B,
              "models": labs, "runs": metas, "T-E4a_distributions": t_a, "T-E4b_rates": t_b, "P3": p3_out,
              "T-E4c_per_model": t_c, "secondary": sec}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "T-E4_paraphrase_vs_repeat.json").write_text(json.dumps(result, indent=2) + "\n")

    # ---- summary.md ------------------------------------------------------------------------------
    L = ["# E4 / E4-0: paraphrase versus identical repetition", "",
         f"Models: {', '.join(labs)}. Seed {SEED}, {B} bootstrap resamples over (model, benchmark) cells.",
         f"Items with no API error: {len(recs)} (excluded: {n_api}); same-day base ⟨t,f⟩: {len(prim)} "
         f"({sec['share_base_not_tf']:.1%} of items no longer ⟨t,f⟩ at s₀).", "",
         "## T-E4b  change from a same-day ⟨t,f⟩ base", "",
         "| scope | n | repeat: any change | paraphrase: any change | difference [95% CI] | repeat: →⟨f,f⟩ | paraphrase: →⟨f,f⟩ | difference [95% CI] |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for scope, o in t_b.items():
        a, f = o["any_change"], o["to_ff"]
        L.append(f"| {scope} | {o['n']} | {a['repeat']:.3f} | {a['paraphrase']:.3f} | {a['difference']:+.3f} [{a['difference_ci95'][0]:.3f}, {a['difference_ci95'][1]:.3f}] | "
                 f"{f['repeat']:.3f} | {f['paraphrase']:.3f} | {f['difference']:+.3f} [{f['difference_ci95'][0]:.3f}, {f['difference_ci95'][1]:.3f}] |")
    L += ["", f"**P3** ({p3_out['statement']}): {'holds' if p3_out['holds'] else 'does not hold'} "
              f"(difference {p3_out['difference']}, CI {p3_out['ci95']}).", "",
          "## T-E4a  distribution of the aggregate over NINE (pooled)", "", "| value | repeat | paraphrase |", "| --- | --- | --- |"]
    for v in NINE:
        r_, p_ = t_a["pooled"]["repeat"][v], t_a["pooled"]["paraphrase"][v]
        L.append(f"| {v} | {r_['count']} ({r_['share']:.3f}) | {p_['count']} ({p_['share']:.3f}) |")
    L += ["", "## T-E4c  per-model excess (paraphrase − repeat, any change)", "", "| model | n | excess | 95% CI |", "| --- | --- | --- | --- |"]
    for lab in t_c["ranking_by_excess"]:
        e = t_c["excess_any_change"][lab]
        L.append(f"| {lab} | {e['n']} | {e['excess']:+.3f} | [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}] |")
    L += ["", f"Pairwise orderings whose CI excludes zero: {', '.join(t_c['supported_orderings']) or 'none'}", "",
          "## Runs", "", "| model | records | elicitations | parse fail | API error | provider | finished |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for lab, m in metas.items():
        L.append(f"| {lab} | {m['n_records']} | {m['n_elicitations']} | {m['parse_fail_rate']} | {m['api_error_rate']} | {m['providers_seen']} | {m['finished_at']} |")
    (args.out / "summary.md").write_text("\n".join(L) + "\n")

    # ---- LNCS tabular fragment (body only; the caption is the author's) ---------------------------
    TFx, FFx = r"$\langle\mathfrak{t},\mathfrak{f}\rangle$", r"$\langle\mathfrak{f},\mathfrak{f}\rangle$"
    # Column heads are abbreviated (Rep. = identical repeats, Par. = paraphrases, Diff. = Par. - Rep.)
    # because the spelled-out heads made the table 47pt wider than the LNCS text block (122mm).
    T = [r"\begin{tabular}{@{}l@{\quad}r@{\qquad}r@{\quad}r@{\quad}l@{\qquad}r@{\quad}r@{}}", r"\toprule",
         rf" & & \multicolumn{{3}}{{l}}{{Any change from {TFx} (\%)}} & \multicolumn{{2}}{{l@{{}}}}{{To {FFx} (\%)}} \\",
         r"Model & $n$ & Rep. & Par. & Diff. [95\% CI] & Rep. & Par. \\", r"\midrule"]

    def frow(label, o):
        a, f = o["any_change"], o["to_ff"]
        return (f"{label} & {o['n']} & {100 * a['repeat']:.1f} & {100 * a['paraphrase']:.1f} & "
                f"${100 * a['difference']:+.1f}$ $[{100 * a['difference_ci95'][0]:.1f}, {100 * a['difference_ci95'][1]:.1f}]$ & "
                f"{100 * f['repeat']:.1f} & {100 * f['paraphrase']:.1f} \\\\")
    for lab in labs:
        T.append(frow(lab, t_b[lab]))
    T += [r"\midrule", frow("Pooled", t_b["pooled"]), r"\bottomrule", r"\end{tabular}"]
    frag = args.out / "tables" / "lncs"
    frag.mkdir(parents=True, exist_ok=True)
    (frag / "e4_paraphrase_vs_repeat_tabular.tex").write_text("\n".join(T) + "\n")
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
