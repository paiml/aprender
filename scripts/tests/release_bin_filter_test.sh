#!/usr/bin/env bash
# #6502: Binary Release must not ship publish=false / *-ci-tools bins. Exercises the SAME
# filter script the workflow runs, then the real deriver.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
filter="${RELEASE_BIN_FILTER:-scripts/release/release_bin_filter.sh}"
meta=$(mktemp); out=$(mktemp); trap 'rm -f "$meta" "$out"' EXIT
cargo metadata --no-deps --format-version 1 > "$meta"
bash "$filter" "$meta" > "$out"
bad=$(jq -r '.packages[] | select(.publish == [] or (.name | endswith("-ci-tools"))) | .name' "$meta" | sort)
[ -n "$bad" ] || { echo "FAIL: no publish=false package found; test is vacuous"; exit 1; }
want=$(jq -r '.packages[] | select((.publish == [] or (.name | endswith("-ci-tools"))) | not) | .name' "$meta" | sort)
got=$(jq -r '.packages[].name' "$out" | sort)
leaked=$(comm -12 <(printf '%s\n' "$bad") <(printf '%s\n' "$got") | tr '\n' ' ')
[ -z "$leaked" ] || { echo "FAIL: unreleasable packages survive the filter: $leaked"; exit 1; }
[ "$want" = "$got" ] || { echo "FAIL: filter kept the wrong set (diff want/got):"; diff <(printf '%s\n' "$want") <(printf '%s\n' "$got") | head -20; exit 1; }
echo "PASS: $(printf '%s\n' "$bad" | wc -l) unreleasable packages filtered, $(printf '%s\n' "$got" | wc -l) kept"
bins=$(python3 -I scripts/nightly_manifest.py bins --metadata "$out")
for must in apr pv; do case ",$bins," in *,"$must",*) ;; *) echo "FAIL: released bin $must missing from derived bins"; exit 1;; esac; done
case ",$bins," in *,aprender-rollup,*) echo "FAIL: aprender-rollup in release bins"; exit 1;; esac
echo "PASS: deriver output has no aprender-rollup"
