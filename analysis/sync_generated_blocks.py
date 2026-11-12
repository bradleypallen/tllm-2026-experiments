"""
Refresh the generated table blocks in the manuscript from this repository's LNCS fragments.

The manuscript (paper.tex in bradleypallen/bbl-paper-lncs-v2) holds each generated table body
between a pair of marker lines:

    % BEGIN GENERATED: <name> -- tllm-2026-experiments@<commit>, <script>; do not edit by hand
    ...tabular...
    % END GENERATED: <name>

For every such block this script replaces the body with <name>_tabular.tex from the fragment
directories and stamps the commit, so that no number in the manuscript is typed by hand and each
table names the revision it came from. The table environment, caption and label around a block
are the author's and are never touched.

Default: stamp the short hash of this repository's HEAD. The script refuses to write if a fragment
has uncommitted changes, because the stamp would then name a commit that does not contain it.

Run:
    python3 analysis/sync_generated_blocks.py /path/to/paper.tex            # rewrite in place
    python3 analysis/sync_generated_blocks.py /path/to/paper.tex --check    # report only; exit 1 on drift
    python3 analysis/sync_generated_blocks.py /path/to/paper.tex --commit <tag-or-hash>
"""
from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [sync-blocks] %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
FRAGMENT_DIRS = [REPO / "results/derived/2026_11/tables/lncs", REPO / "results/derived/2026_11_e4/tables/lncs"]
REPO_NAME = "tllm-2026-experiments"
BLOCK = re.compile(
    rf"% BEGIN GENERATED: (?P<name>\S+) -- {REPO_NAME}@(?P<commit>\S+), (?P<script>[^;\n]+); do not edit by hand\n"
    r"(?P<body>.*?)% END GENERATED: (?P=name)\n", re.S)


def find_fragment(name: str, dirs: list[Path]) -> Path:
    for d in dirs:
        p = d / f"{name}_tabular.tex"
        if p.exists():
            return p
    raise SystemExit(f"no fragment {name}_tabular.tex in {[str(d) for d in dirs]}")


def sync(text: str, dirs: list[Path], commit: str) -> tuple[str, list[dict]]:
    """Return the text with every block refreshed, and one report row per block."""
    report: list[dict] = []

    def repl(m: re.Match) -> str:
        src = find_fragment(m["name"], dirs)
        body = src.read_text().rstrip("\n") + "\n"
        report.append({"name": m["name"], "fragment": src, "was_commit": m["commit"],
                       "body_changed": body != m["body"], "stamp_changed": commit != m["commit"]})
        return (f"% BEGIN GENERATED: {m['name']} -- {REPO_NAME}@{commit}, {m['script']}; do not edit by hand\n"
                f"{body}% END GENERATED: {m['name']}\n")

    return BLOCK.sub(repl, text), report


def head_commit() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def dirty(paths: list[Path]) -> list[str]:
    """Fragments that differ from HEAD or are untracked."""
    inside = [str(p) for p in paths if REPO in p.resolve().parents]
    if not inside:
        return []
    out = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", *inside],
                         capture_output=True, text=True, check=True).stdout
    return [l[3:] for l in out.splitlines() if l.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("paper", type=Path, help="path to the manuscript's paper.tex")
    ap.add_argument("--commit", help="commit or tag to stamp (default: short hash of this repo's HEAD)")
    ap.add_argument("--fragments", type=Path, nargs="+", default=FRAGMENT_DIRS,
                    help="directories searched, in order, for <name>_tabular.tex")
    ap.add_argument("--check", action="store_true",
                    help="write nothing; exit 1 if any block's body differs from its fragment")
    args = ap.parse_args(argv)

    text = args.paper.read_text()
    commit = args.commit or head_commit()
    new, report = sync(text, args.fragments, commit)
    if not report:
        log.error("no GENERATED blocks found in %s", args.paper)
        return 1
    for r in report:
        log.info("%-26s body %-9s stamp %s -> %s", r["name"], "CHANGED" if r["body_changed"] else "identical",
                 r["was_commit"], commit)
    drift = [r["name"] for r in report if r["body_changed"]]

    if args.check:
        log.info("%d blocks checked, %d differ from their fragments%s", len(report), len(drift),
                 f": {', '.join(drift)}" if drift else "")
        return 1 if drift else 0

    stale = dirty([r["fragment"] for r in report])
    if stale and not args.commit:
        log.error("fragments with uncommitted changes, commit them first or pass --commit: %s", ", ".join(stale))
        return 1
    if new == text:
        log.info("%d blocks already up to date at %s; nothing written", len(report), commit)
        return 0
    args.paper.write_text(new)
    log.info("refreshed %d blocks at %s in %s (%d bodies changed)", len(report), commit, args.paper, len(drift))
    return 0


if __name__ == "__main__":
    sys.exit(main())
