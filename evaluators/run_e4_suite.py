#!/usr/bin/env python3
"""Run E4 / E4-0 for the five models, one subprocess per model, concurrently.

Each model is one invocation of ``revision_e4_evaluator.py``; cells resume, so re-running this
script after an interruption continues where it stopped.

    python evaluators/run_e4_suite.py --dry-run     # mock client, no network
    python evaluators/run_e4_suite.py --smoke       # 5 live items per model, pinned, to results/2026_11_e4/smoke/
    python evaluators/run_e4_suite.py               # the full run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVALUATOR = HERE / "revision_e4_evaluator.py"
MODELS = ["qwen/qwen-2.5-72b-instruct", "meta-llama/llama-4-maverick", "meta-llama/llama-4-scout",
          "deepseek/deepseek-chat", "google/gemini-2.5-flash"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="5 live items per model, pinned, separate output dir")
    ap.add_argument("--dry-run", action="store_true", help="mock client, no network")
    ap.add_argument("--models", nargs="*", default=MODELS, choices=MODELS)
    args = ap.parse_args()

    extra: list[str] = []
    if args.dry_run:
        extra.append("--dry-run")
    if args.smoke:
        # Separate directory: smoke items must not be resumed into the real run.
        extra += ["--limit", "5", "--out-dir", str(HERE.parent / "results" / "2026_11_e4" / "smoke")]

    def run(model: str) -> tuple[str, int]:
        cmd = [sys.executable, str(EVALUATOR), "--model", model, *extra]
        print(f"$ {' '.join(cmd)}", flush=True)
        return model, subprocess.call(cmd)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(args.models)) as ex:
        results = list(ex.map(run, args.models))
    failures = [m for m, rc in results if rc != 0]
    print(f"\nfinished in {(time.time() - t0) / 60:.1f} min; failures: {failures or 'none'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
