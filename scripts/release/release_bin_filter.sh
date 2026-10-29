#!/usr/bin/env bash
# #6502: drop packages Binary Release must never ship (publish=false, *-ci-tools) from a
# `cargo metadata --format-version 1` document before nightly_manifest.py derives bins.
# Usage: release_bin_filter.sh <metadata.json>   (filtered JSON on stdout)
# Used by .github/workflows/binary-release.yml and scripts/tests/release_bin_filter_test.sh.
set -euo pipefail
[ $# -eq 1 ] || { echo "usage: $0 <metadata.json>" >&2; exit 2; }
python3 -I -c '
import json, sys
m = json.load(open(sys.argv[1]))
keep = [p for p in m["packages"] if p.get("publish") != [] and not p["name"].endswith("-ci-tools")]
ids = {p["id"] for p in keep}
m["packages"] = keep
if m.get("workspace_members"):
    m["workspace_members"] = [i for i in m["workspace_members"] if i in ids]
json.dump(m, sys.stdout)
' "$1"
