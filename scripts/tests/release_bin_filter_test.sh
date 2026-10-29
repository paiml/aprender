#!/usr/bin/env bash
# #6502: Binary Release must not ship publish=false / *-ci-tools bins. Exercises the SAME
# filter script the workflow runs, then the real deriver.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
filter="${RELEASE_BIN_FILTER:-scripts/release/release_bin_filter.sh}"
meta=$(mktemp); out=$(mktemp); trap 'rm -f "$meta" "$out"' EXIT
cargo metadata --no-deps --format-version 1 > "$meta"
bash "$filter" "$meta" > "$out"
python3 -I - "$meta" "$out" <<'PY'
import json, sys
raw, flt = (json.load(open(p)) for p in sys.argv[1:3])
bad = {p["name"] for p in raw["packages"] if p.get("publish") == [] or p["name"].endswith("-ci-tools")}
assert bad, "no publish=false package found; test is vacuous"
leaked = sorted(p["name"] for p in flt["packages"] if p["name"] in bad)
if leaked:
    print(f"FAIL: unreleasable packages survive the filter: {leaked}"); sys.exit(1)
assert any(p["name"] == "aprender" or p.get("publish") != [] for p in flt["packages"]), "filter dropped everything"
print(f"PASS: {len(bad)} unreleasable packages filtered, {len(flt['packages'])} kept")
PY
bins=$(python3 -I scripts/nightly_manifest.py bins --metadata "$out")
case ",$bins," in *,aprender-rollup,*) echo "FAIL: aprender-rollup in release bins"; exit 1;; esac
echo "PASS: deriver output has no aprender-rollup"
