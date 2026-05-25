# Results

Two frozen experimental snapshots. Each snapshot directory is **immutable** — never overwrite.

| Snapshot | When | bilateral-truth SHA | N per cell | Models × benchmarks | Notes |
|----------|------|---------------------|------------|---------------------|-------|
| [`2025_09/`](2025_09/) | 2025-09-12 / 2025-09-13 | `8545986` | 1,000 | 7 × 4 = 28 | Confidence-based ternary, n=1 bilateral, GPT-4.1 / GPT-4.1-mini / Claude Haiku 3.5 in lineup. |
| [`2026_03/`](2026_03/) | 2026-03-18 / 2026-03-19 | `b71915f` | 250 | 6 × 4 = 24 (+modal) | Evidence-based ternary, n=3 bilateral majority voting, DeepSeek + Qwen + Scout in lineup; GPT family + Haiku 3.5 dropped. |

**The "ternary" approach is not the same construct across the two snapshots.** See [`../CHANGELOG.md`](../CHANGELOG.md) and the snapshot READMEs.

`derived/` contains figures and tables computed from these snapshots, organized by run.
