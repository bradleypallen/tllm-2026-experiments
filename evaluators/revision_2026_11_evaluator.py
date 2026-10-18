#!/usr/bin/env python3
"""Evaluator for the November 2026 revision experiments: E1-0, E1-a, E1-b, E2, E2-0, E3.

Specification: "BBL Revision Experiments: E1-E3 Specification" (2026-10-18).
Prompts: ``evaluators/revision_2026_11_prompts.py`` / ``prompts/2026_11_revision_prompts.md``.
Item lists: ``datasets/revision_2026_11/`` (built by ``build_item_lists.py``).

Modes
    e1-0   frozen verification + refutation prompts, n=3 each, on the E1 baseline subset
    e1-a   joint neutral prompt WITH permission clause, n=3, on all E1 items
    e1-b   joint neutral prompt WITHOUT permission clause, n=3, on E1 treatment items
    e2     four-option prompt, n=1, on all 1,000 frozen (model, assertion) pairs for the model
    e2-0   frozen three-option ternary prompt, n=1, on the E1 baseline subset
    e3     frozen refutation prompt on robust-TRUE items / verification prompt on robust-FALSE items, n=1

Calls are issued directly to OpenRouter's OpenAI-compatible endpoint (``openai`` SDK) rather than
through ``bilateral-truth``, so that the serving provider can be pinned (``provider.order``,
``allow_fallbacks``) and the ``provider`` and ``model`` fields of every response recorded.  Prompts,
temperature and context handling are identical to the 2026-03 run; the full request parameters are
written to the run's ``.meta.json``.

Output: one JSONL per (mode, model) under ``results/2026_11/<mode>/``, appended per item as it
completes (resume skips completed items), plus ``<same stem>.meta.json`` with provenance.

Examples
    # mock smoke test, no network
    python evaluators/revision_2026_11_evaluator.py --mode e1-a --model qwen/qwen-2.5-72b-instruct --dry-run --limit 5
    # live smoke test, 5 items, provider recorded but not pinned
    python evaluators/revision_2026_11_evaluator.py --mode e1-a --model qwen/qwen-2.5-72b-instruct --limit 5 --no-pin
    # full run with the default pinned provider
    python evaluators/revision_2026_11_evaluator.py --mode e1-a --model qwen/qwen-2.5-72b-instruct
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "evaluators"))

import revision_2026_11_prompts as P  # noqa: E402

try:
    from dotenv import load_dotenv
    if (PROJECT_ROOT / ".env").exists():
        load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("revision_2026_11")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODES = ("e1-0", "e1-a", "e1-b", "e2", "e2-0", "e3")

# File-safe stems (as used in results/2026_03 filenames) <-> OpenRouter ids.
MODEL_STEMS = {
    "meta-llama/llama-4-maverick": "meta-llama_llama-4-maverick",
    "meta-llama/llama-4-scout": "meta-llama_llama-4-scout",
    "google/gemini-2.5-flash": "google_gemini-2.5-flash",
    "deepseek/deepseek-chat": "deepseek_deepseek-chat",
    "qwen/qwen-2.5-72b-instruct": "qwen_qwen-2.5-72b-instruct",
}

# Default pinned provider per model (spec §2).  OpenRouter provider slugs; override with --provider.
# DeepSeek pinned from the 2026-10-18 smoke test: DeepInfra served 62/75 unpinned calls with zero
# parse failures (fp4 endpoint); StreamLake served 13 and produced all four DeepSeek parse failures.
DEFAULT_PROVIDER = {
    "meta-llama/llama-4-maverick": "digitalocean",
    "meta-llama/llama-4-scout": "deepinfra",
    "google/gemini-2.5-flash": "google-ai-studio",
    "deepseek/deepseek-chat": "deepinfra",
    "qwen/qwen-2.5-72b-instruct": "deepinfra",
}

TEMPERATURE = 0.0
MAX_TOKENS = 32
RETRIES = 3
BACKOFF_SECONDS = (2, 4, 8)


# ---------------------------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------------------------

class OpenRouterClient:
    def __init__(self, model: str, provider: str | None, allow_fallbacks: bool, reasoning_off: bool = False):
        from openai import OpenAI  # imported lazily so --dry-run needs no SDK
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            sys.exit("OPENROUTER_API_KEY not set (see REPRODUCIBILITY.md)")
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key,
                             default_headers={"HTTP-Referer": "https://github.com/bradleypallen/tllm-2026-experiments",
                                              "X-Title": "tllm-2026-experiments revision_2026_11"})
        self.model = model
        self.extra_body: dict[str, Any] = {}
        if provider:
            self.extra_body["provider"] = {"order": [provider], "allow_fallbacks": allow_fallbacks}
        if reasoning_off:
            # Thinking models (Gemini 2.5 Flash) may spend max_tokens on reasoning and return empty
            # content; this asks OpenRouter to disable reasoning. Recorded in request_params.
            self.extra_body["reasoning"] = {"enabled": False}

    def request_params(self) -> dict[str, Any]:
        return {"endpoint": OPENROUTER_BASE_URL, "model": self.model, "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS, "extra_body": self.extra_body}

    def complete(self, system: str, user: str) -> dict[str, Any]:
        """One call; returns {text, provider, model_returned, latency_ms} or raises."""
        t0 = time.time()
        raw = self.client.chat.completions.with_raw_response.create(
            model=self.model, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            extra_body=self.extra_body or None,
        )
        body = json.loads(raw.text)
        if "error" in body and not body.get("choices"):
            raise RuntimeError(f"OpenRouter error: {body['error']}")
        text = body["choices"][0]["message"]["content"] or ""
        return {"text": text, "provider": body.get("provider"), "model_returned": body.get("model"),
                "latency_ms": int((time.time() - t0) * 1000)}


class MockClient:
    """Deterministic offline stand-in for --dry-run: exercises parsing, voting and the schema."""

    def __init__(self, model: str):
        self.model = model

    def request_params(self) -> dict[str, Any]:
        return {"endpoint": "mock", "model": self.model, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS}

    def complete(self, system: str, user: str) -> dict[str, Any]:
        h = int(hashlib.sha256((system + user).encode()).hexdigest(), 16)
        r = random.Random(h)
        if "VERIFICATION: <token>" in user:
            if r.random() < 0.05:
                text = "I think it is VERIFIED but also"  # exercise PARSE_FAIL
            else:
                text = f"VERIFICATION: {r.choice(P.VERIFICATION_TOKENS)}; REFUTATION: {r.choice(P.REFUTATION_TOKENS)}"
        elif "CONFLICTING" in system:
            text = r.choice(P.FOUR_OPTION_TOKENS) + ("." if r.random() < 0.2 else "")
        elif "UNCERTAIN" in system:
            text = r.choice(P.TERNARY_TOKENS)
        elif "can be refuted" in user:
            text = r.choice(P.REFUTATION_TOKENS)
        else:
            text = r.choice(P.VERIFICATION_TOKENS)
        return {"text": text, "provider": "mock", "model_returned": self.model, "latency_ms": 0}


def call_with_retries(client, system: str, user: str) -> dict[str, Any]:
    last_err = None
    for attempt in range(RETRIES):
        try:
            return client.complete(system, user)
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            log.warning("attempt %d/%d failed: %s", attempt + 1, RETRIES, last_err)
            if attempt < RETRIES - 1:
                time.sleep(BACKOFF_SECONDS[attempt])
    return {"text": None, "provider": None, "model_returned": None, "latency_ms": None, "error": last_err}


# ---------------------------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def items_for(mode: str, model: str, items_path: Path | None) -> tuple[list[dict], str]:
    """Return (items, item_list_sha256) for the mode/model."""
    d = PROJECT_ROOT / "datasets" / "revision_2026_11"
    if mode == "e2":
        stem = MODEL_STEMS[model]
        rows, hasher = [], hashlib.sha256()
        for bench in ("truthfulqa", "simpleqa", "mmlu-pro", "factscore"):
            p = PROJECT_ROOT / "results" / "2026_03" / "proper" / f"{bench}_{stem}_n3_proper_results.json"
            hasher.update(p.read_bytes())
            for r in json.loads(p.read_text())["detailed_results"]:
                rows.append({"model": model, "benchmark": bench, "assertion_id": r["assertion_id"],
                             "assertion_text": r["assertion_text"], "source_question": r.get("source_question", ""),
                             "ground_truth": bool(r["ground_truth"]), "orig_bilateral_tv": r["bilateral_tv"],
                             "orig_ternary_prediction": r["ternary_prediction"], "condition": "all"})
        return rows, hasher.hexdigest()
    path = items_path or (d / ("e3_items.jsonl" if mode == "e3" else "e1_items.jsonl"))
    rows = [r for r in load_jsonl(path) if r["model"] == model]
    if mode in ("e1-0", "e2-0"):
        rows = [r for r in rows if r.get("baseline_subset")]
    elif mode == "e1-b":
        rows = [r for r in rows if r.get("condition") == "treatment"]
    return rows, hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------
# Per-item evaluation
# ---------------------------------------------------------------------------------------------

def _elicit(client, system: str, user: str, legal: tuple[str, ...]) -> dict[str, Any]:
    resp = call_with_retries(client, system, user)
    if resp.get("error"):
        parsed = P.API_ERROR
    else:
        parsed = P.parse_single(resp["text"], legal)
    return {"raw": resp.get("text"), "parsed": parsed, "provider": resp.get("provider"),
            "model_returned": resp.get("model_returned"), "latency_ms": resp.get("latency_ms"),
            "error": resp.get("error")}


def _elicit_joint(client, system: str, user: str) -> dict[str, Any]:
    resp = call_with_retries(client, system, user)
    if resp.get("error"):
        parsed: Any = P.API_ERROR
    else:
        parsed = P.parse_joint(resp["text"])
    return {"raw": resp.get("text"), "parsed": list(parsed) if isinstance(parsed, tuple) else parsed,
            "provider": resp.get("provider"), "model_returned": resp.get("model_returned"),
            "latency_ms": resp.get("latency_ms"), "error": resp.get("error")}


def evaluate_item(mode: str, item: dict, client, samples: int) -> dict[str, Any]:
    fmt = {"assertion": item["assertion_text"], "context": item.get("source_question", "")}
    out: dict[str, Any] = {k: item[k] for k in ("model", "benchmark", "assertion_id", "ground_truth",
                                                 "orig_bilateral_tv", "orig_ternary_prediction")}
    out["condition"] = item.get("condition") or item.get("polarity")
    out["mode"] = mode

    if mode == "e1-0":
        v = [_elicit(client, P.BILATERAL_SYSTEM_PROMPT, P.VERIFICATION_PROMPT.format(**fmt), P.VERIFICATION_TOKENS)
             for _ in range(samples)]
        r = [_elicit(client, P.BILATERAL_SYSTEM_PROMPT, P.REFUTATION_PROMPT.format(**fmt), P.REFUTATION_TOKENS)
             for _ in range(samples)]
        u = P.majority([P.component(e["parsed"]) for e in v])
        w = P.majority([P.component(e["parsed"]) for e in r])
        out.update(elicitations={"verification": v, "refutation": r}, tv=f"<{u},{w}>")

    elif mode in ("e1-a", "e1-b"):
        prompt = P.JOINT_PROMPT_E1A if mode == "e1-a" else P.JOINT_PROMPT_E1B
        el = [_elicit_joint(client, P.BILATERAL_SYSTEM_PROMPT, prompt.format(**fmt)) for _ in range(samples)]
        u = P.majority([P.component(e["parsed"][0]) if isinstance(e["parsed"], list) else "e" for e in el])
        w = P.majority([P.component(e["parsed"][1]) if isinstance(e["parsed"], list) else "e" for e in el])
        out.update(elicitations=el, tv=f"<{u},{w}>")

    elif mode == "e2":
        el = _elicit(client, P.FOUR_OPTION_SYSTEM_PROMPT, P.FOUR_OPTION_PROMPT.format(**fmt), P.FOUR_OPTION_TOKENS)
        out.update(elicitations=[el], prediction=el["parsed"])

    elif mode == "e2-0":
        el = _elicit(client, P.TERNARY_SYSTEM_PROMPT, P.TERNARY_PROMPT.format(**fmt), P.TERNARY_TOKENS)
        out.update(elicitations=[el], prediction=el["parsed"])

    elif mode == "e3":
        if item["probe"] == "refutation":
            el = _elicit(client, P.BILATERAL_SYSTEM_PROMPT, P.REFUTATION_PROMPT.format(**fmt), P.REFUTATION_TOKENS)
            complied = el["parsed"] == "REFUTED"
        else:
            el = _elicit(client, P.BILATERAL_SYSTEM_PROMPT, P.VERIFICATION_PROMPT.format(**fmt), P.VERIFICATION_TOKENS)
            complied = el["parsed"] == "VERIFIED"
        out.update(probe=item["probe"], elicitations=[el], prediction=el["parsed"],
                   complied=None if el["parsed"] in (P.PARSE_FAIL, P.API_ERROR) else complied)
    else:
        raise ValueError(mode)
    return out


# ---------------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------------

def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def prompt_hashes() -> dict[str, str]:
    src = (PROJECT_ROOT / "evaluators" / "revision_2026_11_prompts.py").read_bytes()
    md = (PROJECT_ROOT / "prompts" / "2026_11_revision_prompts.md").read_bytes()
    return {"revision_2026_11_prompts.py": hashlib.sha256(src).hexdigest(),
            "2026_11_revision_prompts.md": hashlib.sha256(md).hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--model", required=True, choices=sorted(MODEL_STEMS))
    ap.add_argument("--items", type=Path, help="override item list (JSONL)")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "2026_11")
    ap.add_argument("--samples", type=int, default=3, help="elicitations per prompt for E1 modes (default 3)")
    ap.add_argument("--limit", type=int, help="process at most N items (smoke tests)")
    ap.add_argument("--provider", help="OpenRouter provider slug to pin (default: spec §2 table)")
    ap.add_argument("--no-pin", action="store_true", help="do not pin a provider; record whatever serves")
    ap.add_argument("--allow-fallbacks", action="store_true", help="permit fallback on pinned-provider outage (flagged)")
    ap.add_argument("--dry-run", action="store_true", help="mock client, no network, output to <out-dir>/dry_run/")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to sleep between items")
    ap.add_argument("--reasoning-off", action="store_true",
                    help="send reasoning={enabled:false}; use if a thinking model returns empty content under max_tokens=32")
    args = ap.parse_args()

    samples = args.samples if args.mode.startswith("e1") else 1
    provider = None if args.no_pin else (args.provider or DEFAULT_PROVIDER[args.model])
    client = (MockClient(args.model) if args.dry_run
              else OpenRouterClient(args.model, provider, args.allow_fallbacks, args.reasoning_off))

    items, items_sha = items_for(args.mode, args.model, args.items)
    if args.limit:
        items = items[: args.limit]

    out_dir = args.out_dir / ("dry_run" if args.dry_run else "") / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.mode}_{MODEL_STEMS[args.model]}"
    out_path, meta_path = out_dir / f"{stem}.jsonl", out_dir / f"{stem}.meta.json"

    done = {r["assertion_id"] for r in load_jsonl(out_path)} if out_path.exists() else set()
    todo = [it for it in items if it["assertion_id"] not in done]
    log.info("%s × %s: %d items, %d already complete, %d to run (samples=%d, provider=%s)",
             args.mode, args.model, len(items), len(done), len(todo), samples, provider or "unpinned")

    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({
        "mode": args.mode, "model": args.model, "samples": samples,
        "harness_commit": git_commit(), "prompt_hashes": prompt_hashes(), "item_list_sha256": items_sha,
        "n_items": len(items), "request_params": client.request_params(),
        "provider_pinned": provider, "allow_fallbacks": args.allow_fallbacks, "dry_run": args.dry_run,
        "python": platform.python_version(), "platform": platform.platform(),
        "started_at": meta.get("started_at") or dt.datetime.now(dt.timezone.utc).isoformat(),
    })
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    t0 = time.time()
    with out_path.open("a") as fh:
        for i, item in enumerate(todo, 1):
            rec = evaluate_item(args.mode, item, client, samples)
            rec["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if "first_response" not in meta:
                first = (rec["elicitations"]["verification"][0] if isinstance(rec["elicitations"], dict)
                         else rec["elicitations"][0])
                meta["first_response"] = {"provider": first.get("provider"), "model_returned": first.get("model_returned")}
                meta_path.write_text(json.dumps(meta, indent=2) + "\n")
            if i % 25 == 0 or i == len(todo):
                log.info("  %d/%d done (%.0fs)", i, len(todo), time.time() - t0)
            if args.sleep:
                time.sleep(args.sleep)

    # Completion summary for the smoke-test gate (spec §9): parse-fail and error rates.
    recs = load_jsonl(out_path)
    els = []
    for r in recs:
        e = r["elicitations"]
        els.extend(e["verification"] + e["refutation"] if isinstance(e, dict) else e)
    n = len(els) or 1
    pf = sum(1 for e in els if e["parsed"] == P.PARSE_FAIL) / n
    ae = sum(1 for e in els if e["parsed"] == P.API_ERROR) / n
    providers = sorted({e.get("provider") for e in els if e.get("provider")})
    models_ret = sorted({e.get("model_returned") for e in els if e.get("model_returned")})
    meta.update({"finished_at": dt.datetime.now(dt.timezone.utc).isoformat(), "n_elicitations": len(els),
                 "parse_fail_rate": round(pf, 4), "api_error_rate": round(ae, 4),
                 "providers_seen": providers, "models_returned": models_ret})
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    log.info("done: %d records, %d elicitations, parse_fail=%.1f%%, api_error=%.1f%%, providers=%s, models=%s",
             len(recs), len(els), 100 * pf, 100 * ae, providers, models_ret)
    if pf > 0.05:
        log.warning("PARSE_FAIL rate %.1f%% is above the 5%% gate (spec §9): inspect raw responses before continuing",
                    100 * pf)


if __name__ == "__main__":
    main()
