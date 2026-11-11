#!/usr/bin/env python3
"""Build the item list for E4 / E4-0 (paraphrase versus identical repetition).

Specification: docs/revision_2026_11_E4_spec_addition.md, §3.

Items are every (model, assertion) pair whose value at s0 in the frozen 2026-03 paraphrase
(modal) run is <t,f> -- the denominators of Figure 2 -- for the five retained models (Claude
Opus 4.1 was retired on 2026-08-05). No sampling, so no seed: the list is a deterministic
function of ``results/2026_03/modal/`` and ``variant_cache/``.

Each record carries the source question (s0) and the cached paraphrases the March run used; the
builder fails if the cache and the March result file disagree on either.

Outputs (in this directory, or ``--out-dir``):
    e4_items.jsonl    one record per (model, assertion)
    MANIFEST.json     SHA-256 and size, in the convention of tests/test_manifest_integrity.py
    SELECTION.json    counts by model and benchmark

Usage:
    python datasets/revision_2026_11_e4/build_e4_items.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODAL = ROOT / "results" / "2026_03" / "modal"
CACHE = ROOT / "variant_cache"
OUT = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [build-e4] %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODELS = [  # (file stem, OpenRouter id)
    ("meta-llama_llama-4-maverick", "meta-llama/llama-4-maverick"),
    ("meta-llama_llama-4-scout", "meta-llama/llama-4-scout"),
    ("google_gemini-2.5-flash", "google/gemini-2.5-flash"),
    ("deepseek_deepseek-chat", "deepseek/deepseek-chat"),
    ("qwen_qwen-2.5-72b-instruct", "qwen/qwen-2.5-72b-instruct"),
]
BENCHES = [  # (benchmark id as in results/2026_03/proper and variant_cache, modal file stem)
    ("truthfulqa", "truthfulqa_complete"),
    ("simpleqa", "simpleqa_complete"),
    ("mmlu-pro", "mmlupro_complete"),
    ("factscore", "factscore_complete"),
]


def build() -> tuple[list[dict], dict]:
    cache = {b: json.loads((CACHE / f"{b}_n3_variants.json").read_text())["variants"] for b, _ in BENCHES}
    rows: list[dict] = []
    counts: dict[str, dict[str, int]] = {}
    for stem, mid in MODELS:
        counts[mid] = {}
        for bench, modal_stem in BENCHES:
            path = MODAL / f"{modal_stem}_{stem}_n3_modal_results.json"
            if not path.exists():
                raise FileNotFoundError(f"missing modal cell: {path}")
            n = 0
            for r in json.loads(path.read_text())["detailed_results"]:
                if r["src_bilateral"] != "<t,f>":
                    continue
                c = cache[bench].get(r["assertion_id"])
                if c is None or c["source_question"] != r["source_question"] or c["variants"] != r["variants"]:
                    raise ValueError(f"variant cache disagrees with the March run for {bench}/{r['assertion_id']}")
                rows.append({
                    "model": mid, "benchmark": bench, "assertion_id": r["assertion_id"],
                    "assertion_text": r["assertion"], "source_question": r["source_question"],
                    "variants": r["variants"], "ground_truth": r["expected_label"] == "correct",
                    "orig_modal_src": r["src_bilateral"], "orig_modal_box": r["modal_bilateral"],
                    "orig_variant_bilaterals": r["variant_bilaterals"],
                })
                n += 1
            counts[mid][bench] = n
    return rows, counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    rows, counts = build()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
    (args.out_dir / "e4_items.jsonl").write_bytes(body)
    manifest = {"e4_items.jsonl": {"sha256": hashlib.sha256(body).hexdigest(), "size_bytes": len(body)}}
    (args.out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    selection = {"source": "results/2026_03/modal (src_bilateral == <t,f>)", "records": len(rows),
                 "by_model": {m: {"total": sum(c.values()), **c} for m, c in counts.items()},
                 "by_benchmark": {b: sum(counts[m][b] for m in counts) for b, _ in BENCHES}}
    (args.out_dir / "SELECTION.json").write_text(json.dumps(selection, indent=2) + "\n")
    calls = sum(2 * (1 + 2 * len(r["variants"])) for r in rows)
    log.info("wrote %d items (%s); planned calls %d", len(rows),
             ", ".join(f"{m.split('/')[-1]} {sum(c.values())}" for m, c in counts.items()), calls)


if __name__ == "__main__":
    main()
