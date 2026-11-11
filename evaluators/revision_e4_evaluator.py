#!/usr/bin/env python3
"""Evaluator for E4 / E4-0: paraphrase versus identical repetition, paired within item.

Specification: docs/revision_2026_11_E4_spec_addition.md (approved 2026-11-11).
Item list: datasets/revision_2026_11_e4/e4_items.jsonl.

For each item (model, assertion) with source question s0 and cached paraphrases s1..sk:

    base          s0, one verification + one refutation call           -> b = <u0, v0>
    E4-0 (rep)    s0 again, k times, one + one call each
    E4   (par)    s1..sk, one + one call each

issued in the order base, rep 1, par 1, rep 2, par 2, ... so neither arm is systematically later.
One elicitation per situation, as in the 2026-03 paraphrase run. The two aggregates share the base:

    box_rep = <min u, max v> over base + repeats        box_par = <min u, max v> over base + paraphrases

Prompts are the frozen verification / refutation prompts and system prompt of
``revision_2026_11_prompts.py``, unchanged; the context slot carries the situation text. The client,
retry policy, provider pins, temperature and max_tokens are imported from the October evaluator
(``revision_2026_11_evaluator.py``), which this file does not modify.

Output: ``results/2026_11_e4/e4_<model_safe>.jsonl`` (appended per item; resume skips completed
items) plus ``.meta.json`` with provenance. Raw response text is kept for every elicitation.

Examples
    python evaluators/revision_e4_evaluator.py --model qwen/qwen-2.5-72b-instruct --dry-run --limit 5
    python evaluators/revision_e4_evaluator.py --model qwen/qwen-2.5-72b-instruct --limit 5 --out-dir results/2026_11_e4/smoke
    python evaluators/revision_e4_evaluator.py --model qwen/qwen-2.5-72b-instruct
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "evaluators"))

import revision_2026_11_prompts as P  # noqa: E402  frozen prompts and single-token parser
import revision_2026_11_evaluator as R  # noqa: E402  October client, pins, retry policy (unmodified)

log = logging.getLogger("revision_e4")

ITEMS = PROJECT_ROOT / "datasets" / "revision_2026_11_e4" / "e4_items.jsonl"
SPEC = PROJECT_ROOT / "docs" / "revision_2026_11_E4_spec_addition.md"
OUT_DIR = PROJECT_ROOT / "results" / "2026_11_e4"
# Provider pins are the October ones except where an endpoint has since been withdrawn (spec §10,
# amendment of 2026-11-11): DigitalOcean no longer serves Llama 4 Maverick, so E4 pins Parasail (fp8).
E4_PROVIDER = {**R.DEFAULT_PROVIDER, "meta-llama/llama-4-maverick": "parasail"}
_ORD = {"f": 0, "e": 1, "t": 2}
_INV = "fet"


def box(tvs: list[str]) -> str:
    """Definition 7: <min u, max v> under f < e < t, over values written '<u,v>'."""
    us, vs = [t[1] for t in tvs], [t[3] for t in tvs]
    return f"<{_INV[min(_ORD[u] for u in us)]},{_INV[max(_ORD[v] for v in vs)]}>"


def schedule(item: dict) -> list[tuple[str, int, str]]:
    """(arm, index, situation text) in issue order: base, then repeats and paraphrases interleaved."""
    s0 = item["source_question"]
    out = [("base", 0, s0)]
    for i, variant in enumerate(item["variants"], 1):
        out.append(("rep", i, s0))
        out.append(("par", i, variant))
    return out


def _elicit(client, user: str, legal: tuple[str, ...]) -> dict[str, Any]:
    resp = R.call_with_retries(client, P.BILATERAL_SYSTEM_PROMPT, user)
    parsed = P.API_ERROR if resp.get("error") else P.parse_single(resp["text"], legal)
    return {"raw": resp.get("text"), "parsed": parsed, "provider": resp.get("provider"),
            "model_returned": resp.get("model_returned"), "latency_ms": resp.get("latency_ms"),
            "error": resp.get("error")}


def evaluate_item(item: dict, client) -> dict[str, Any]:
    situations = []
    for arm, idx, text in schedule(item):
        fmt = {"assertion": item["assertion_text"], "context": text}
        v = _elicit(client, P.VERIFICATION_PROMPT.format(**fmt), P.VERIFICATION_TOKENS)
        r = _elicit(client, P.REFUTATION_PROMPT.format(**fmt), P.REFUTATION_TOKENS)
        situations.append({"arm": arm, "index": idx, "situation": text, "verification": v, "refutation": r,
                           "tv": f"<{P.component(v['parsed'])},{P.component(r['parsed'])}>"})
    base = situations[0]["tv"]
    els = [e for s in situations for e in (s["verification"], s["refutation"])]
    return {
        "model": item["model"], "benchmark": item["benchmark"], "assertion_id": item["assertion_id"],
        "ground_truth": item["ground_truth"], "orig_modal_src": item["orig_modal_src"],
        "orig_modal_box": item["orig_modal_box"], "k": len(item["variants"]), "mode": "e4",
        "situations": situations, "base_tv": base,
        "box_rep": box([base] + [s["tv"] for s in situations if s["arm"] == "rep"]),
        "box_par": box([base] + [s["tv"] for s in situations if s["arm"] == "par"]),
        "n_parse_fail": sum(e["parsed"] == P.PARSE_FAIL for e in els),
        "n_api_error": sum(e["parsed"] == P.API_ERROR for e in els),
    }


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(R.MODEL_STEMS))
    ap.add_argument("--items", type=Path, default=ITEMS)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--limit", type=int, help="process at most N items (smoke tests)")
    ap.add_argument("--provider", help="OpenRouter provider slug to pin (default: E4_PROVIDER)")
    ap.add_argument("--no-pin", action="store_true", help="do not pin a provider; record whatever serves")
    ap.add_argument("--allow-fallbacks", action="store_true", help="permit fallback on pinned-provider outage (flagged)")
    ap.add_argument("--dry-run", action="store_true", help="mock client, no network, output to <out-dir>/dry_run/")
    args = ap.parse_args()

    provider = None if args.no_pin else (args.provider or E4_PROVIDER[args.model])
    client = (R.MockClient(args.model) if args.dry_run
              else R.OpenRouterClient(args.model, provider, args.allow_fallbacks))

    items = [r for r in R.load_jsonl(args.items) if r["model"] == args.model]
    if args.limit:
        items = items[: args.limit]

    out_dir = args.out_dir / ("dry_run" if args.dry_run else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"e4_{R.MODEL_STEMS[args.model]}"
    out_path, meta_path = out_dir / f"{stem}.jsonl", out_dir / f"{stem}.meta.json"

    done = {r["assertion_id"] for r in R.load_jsonl(out_path)} if out_path.exists() else set()
    todo = [it for it in items if it["assertion_id"] not in done]
    planned = sum(2 * len(schedule(it)) for it in items)
    log.info("e4 × %s: %d items (%d planned calls), %d already complete, %d to run (provider=%s)",
             args.model, len(items), planned, len(done), len(todo), provider or "unpinned")

    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({
        "mode": "e4", "model": args.model, "samples_per_situation": 1,
        "harness_commit": R.git_commit(), "prompt_hashes": R.prompt_hashes(),
        "item_list_sha256": _sha(args.items), "spec_sha256": _sha(SPEC),
        "n_items": len(items), "planned_calls": planned, "request_params": client.request_params(),
        "provider_pinned": provider, "allow_fallbacks": args.allow_fallbacks, "dry_run": args.dry_run,
        "python": platform.python_version(), "platform": platform.platform(),
        "started_at": meta.get("started_at") or dt.datetime.now(dt.timezone.utc).isoformat(),
    })
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    t0 = time.time()
    with out_path.open("a") as fh:
        for i, item in enumerate(todo, 1):
            rec = evaluate_item(item, client)
            rec["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 25 == 0 or i == len(todo):
                log.info("  %s %d/%d done (%.0fs)", args.model, i, len(todo), time.time() - t0)

    recs = R.load_jsonl(out_path)
    els = [e for r in recs for s in r["situations"] for e in (s["verification"], s["refutation"])]
    n = len(els) or 1
    pf = sum(e["parsed"] == P.PARSE_FAIL for e in els) / n
    ae = sum(e["parsed"] == P.API_ERROR for e in els) / n
    providers = sorted({e["provider"] for e in els if e.get("provider")})
    models_ret = sorted({e["model_returned"] for e in els if e.get("model_returned")})
    meta.update({"finished_at": dt.datetime.now(dt.timezone.utc).isoformat(), "n_records": len(recs),
                 "n_elicitations": len(els), "parse_fail_rate": round(pf, 4), "api_error_rate": round(ae, 4),
                 "providers_seen": providers, "models_returned": models_ret})
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    log.info("done %s: %d records, %d elicitations, parse_fail=%.1f%%, api_error=%.1f%%, providers=%s, models=%s",
             args.model, len(recs), len(els), 100 * pf, 100 * ae, providers, models_ret)


if __name__ == "__main__":
    main()
