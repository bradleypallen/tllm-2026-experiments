#!/usr/bin/env bash
set -euo pipefail

# Create a Python venv and install pinned dependencies.

python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "venv ready. Activate with: source venv/bin/activate"
