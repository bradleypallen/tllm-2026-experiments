"""
Verify that every file listed in any MANIFEST.json matches its recorded SHA-256
and size. Catches corruption, git-lfs misconfiguration, and accidental edits.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _find_manifests() -> list[Path]:
    """Find every MANIFEST.json in the repo (excluding venvs, etc.)."""
    skip = {"venv", ".venv", "node_modules", ".git", "__pycache__"}
    found = []
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in skip]
        if "MANIFEST.json" in files:
            found.append(Path(root) / "MANIFEST.json")
    return sorted(found)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.mark.parametrize("manifest_path", _find_manifests(),
                         ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_manifest_matches_disk(manifest_path: Path) -> None:
    with open(manifest_path) as f:
        manifest = json.load(f)
    directory = manifest_path.parent
    for filename, expected in manifest.items():
        file_path = directory / filename
        assert file_path.exists(), f"missing: {file_path}"
        actual_size = file_path.stat().st_size
        assert actual_size == expected["size_bytes"], (
            f"size mismatch for {file_path}: "
            f"expected {expected['size_bytes']}, got {actual_size}"
        )
        actual_sha = _sha256(file_path)
        assert actual_sha == expected["sha256"], (
            f"sha256 mismatch for {file_path}: "
            f"expected {expected['sha256']}, got {actual_sha}"
        )


def test_at_least_one_manifest_exists() -> None:
    """Sanity check: this test file would otherwise pass with zero parametrizations."""
    assert _find_manifests(), "no MANIFEST.json files found in the repo"
