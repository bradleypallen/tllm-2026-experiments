#!/usr/bin/env python3
"""Drive the November 2026 revision experiments in the order the specification prescribes (§9).

Order
    1. smoke  : 5 live items per model per mode, provider recorded but not pinned, so the pinned
                provider and the returned model string can be confirmed before spending the rest
    2. Qwen first, all modes (single provider; discover an outage before, not during, the run)
    3. e2 then e2-0 on the remaining models
    4. e1-0, then e1-a, then e1-b on the remaining models (Gemini last within each arm)
    5. e3 last (droppable)

Each (mode, model) cell is one invocation of ``revision_2026_11_evaluator.py``; cells resume, so
re-running this script after an interruption continues where it stopped.  ``--parallel`` runs the
models of a stage concurrently (one subprocess per model); modes within a model stay sequential.

Examples
    python evaluators/run_revision_suite.py --smoke                 # live smoke test, 375 calls (5 items x 15 calls/item x 5 models)
    python evaluators/run_revision_suite.py --dry-run               # whole plan with the mock client
    python evaluators/run_revision_suite.py                         # the real thing, sequential
    python evaluators/run_revision_suite.py --parallel --skip e3    # four models at once, no E3
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVALUATOR = HERE / "revision_2026_11_evaluator.py"

QWEN = "qwen/qwen-2.5-72b-instruct"
OTHERS = ["meta-llama/llama-4-maverick", "meta-llama/llama-4-scout", "deepseek/deepseek-chat",
          "google/gemini-2.5-flash"]  # Gemini last: largest treatment set
MODES_IN_ORDER = ["e2", "e2-0", "e1-0", "e1-a", "e1-b", "e3"]


REASONING_OFF: set[str] = set()


def run_cell(mode: str, model: str, extra: list[str]) -> int:
    cmd = [sys.executable, str(EVALUATOR), "--mode", mode, "--model", model, *extra]
    if model in REASONING_OFF:
        cmd.append("--reasoning-off")
    print(f"\n=== {mode} × {model} ===\n$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="5 live items per cell, unpinned, then stop")
    ap.add_argument("--dry-run", action="store_true", help="mock client, no network")
    ap.add_argument("--parallel", action="store_true", help="run the models of a stage concurrently")
    ap.add_argument("--skip", nargs="*", default=[], choices=MODES_IN_ORDER, help="modes to skip")
    ap.add_argument("--only", nargs="*", default=None, choices=MODES_IN_ORDER, help="modes to run")
    ap.add_argument("--reasoning-off", nargs="*", default=[], metavar="MODEL",
                    help="models to run with reasoning disabled (e.g. google/gemini-2.5-flash); passed through")
    args = ap.parse_args()

    modes = [m for m in MODES_IN_ORDER if m not in args.skip and (args.only is None or m in args.only)]
    REASONING_OFF.update(args.reasoning_off)
    extra: list[str] = []
    if args.dry_run:
        extra.append("--dry-run")
    if args.smoke:
        # Separate output dir: smoke items are served unpinned and must not be resumed into the real run.
        extra += ["--limit", "5", "--no-pin", "--out-dir", str(HERE.parent / "results" / "2026_11" / "smoke")]

    t0 = time.time()
    failures: list[tuple[str, str]] = []

    def model_block(model: str) -> None:
        for mode in modes:
            if run_cell(mode, model, extra) != 0:
                failures.append((mode, model))

    # Stage 1/2: Qwen alone, all modes.
    model_block(QWEN)

    if args.smoke:
        # Smoke covers every model so providers and model strings are confirmed for all of them.
        for model in OTHERS:
            model_block(model)
    else:
        # Stage 3/4/5: remaining models, mode by mode so E2 completes before E1 starts (spec §9).
        for mode in modes:
            if args.parallel:
                with ThreadPoolExecutor(max_workers=len(OTHERS)) as ex:
                    for model in OTHERS:
                        ex.submit(lambda m=model: failures.append((mode, m)) if run_cell(mode, m, extra) else None)
            else:
                for model in OTHERS:
                    if run_cell(mode, model, extra) != 0:
                        failures.append((mode, model))

    print(f"\nfinished in {(time.time() - t0) / 60:.1f} min; failures: {failures or 'none'}")
    if args.smoke:
        print("Smoke test complete. Check each .meta.json under results/2026_11/smoke/ for providers_seen, "
              "models_returned and parse_fail_rate before launching the full run; set DeepSeek's provider "
              "pin from what the smoke test returned (spec §2).")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
