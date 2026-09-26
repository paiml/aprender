#!/usr/bin/env bash
# check_viz_published_example.sh - examples/viz-facet-coord must build and run
# against the PUBLISHED aprender-viz, not the workspace copy (#3552).
#
# EV-15/EV-16 had no runnable Facet/Coord program to build against. The example
# goes study data -> panels -> Coord::apply -> SVG. This script:
#   1. copies it to a scratch dir outside the workspace and renames Cargo.toml.in,
#   2. proves aprender-viz resolved from crates.io (never a path or git source),
#   3. runs it and checks the SVG: 3 panel rects + background, 12 points,
#   4. mutation control: Facet::wrap -> Facet::none must turn the run RED.
#
# Usage: check_viz_published_example.sh   (CARGO_TARGET_DIR is honoured)
# Exit 0 = pass. 1 = a check failed. 2 = environment (no cargo / no network).

set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
src="$root/examples/viz-facet-coord"
[ -f "$src/Cargo.toml.in" ] || { echo "missing $src/Cargo.toml.in" >&2; exit 2; }
command -v cargo >/dev/null || { echo "cargo not found" >&2; exit 2; }

td=$(mktemp -d "${TMPDIR:-/tmp}/viz-example.XXXXXX") || exit 2
# shellcheck disable=SC2064
trap "rm -rf -- '${td:?}'" EXIT
cp -r "$src/src" "$td/src"
cp "$src/Cargo.toml.in" "$td/Cargo.toml"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$td/target}"

fail=0
cd "$td"
if ! cargo build --quiet; then
    echo "FAIL: example does not build against the published crate" >&2
    exit 1
fi

id=$(cargo metadata --format-version 1 | python3 -c '
import json, sys
m = json.load(sys.stdin)
p = [p for p in m["packages"] if p["name"] == "aprender-viz"]
print(p[0]["version"], p[0]["source"] or "path") if p else print("absent")')
case "$id" in
    "0.69.1 registry+https://github.com/rust-lang/crates.io-index") echo "ok   aprender-viz resolved from crates.io: $id" ;;
    *) echo "FAIL aprender-viz is not the published crate: $id"; fail=1 ;;
esac

bin="$CARGO_TARGET_DIR/debug/viz-facet-coord"
if "$bin" "$td/out.svg" 2> "$td/err"; then
    circles=$(grep -o '<circle' "$td/out.svg" | wc -l)
    rects=$(grep -o '<rect' "$td/out.svg" | wc -l)
    if [ "$circles" -eq 12 ] && [ "$rects" -ge 3 ] && grep -q '</svg>' "$td/out.svg"; then
        echo "ok   SVG: $circles points, $rects rects ($(cat "$td/err"))"
    else
        echo "FAIL SVG shape: circles=$circles rects=$rects"; fail=1
    fi
else
    echo "FAIL example exited non-zero: $(cat "$td/err")"; fail=1
fi

# Mutation control: one panel instead of three must be caught by the example.
sed -i 's/Facet::wrap("lineage", NCOL)/Facet::none()/' src/main.rs
grep -q 'Facet::none()' src/main.rs || { echo "FAIL mutation did not apply"; exit 1; }
# A failed rebuild would leave the passing binary in place and read as a survivor
# (or, worse, a stale mutant as a kill) - stop instead of judging it.
cargo build --quiet || { echo "mutant rebuild failed; not judged" >&2; exit 2; }
if "$bin" "$td/mut.svg" 2> "$td/err"; then
    echo "FAIL mutant (Facet::none) survived"; fail=1
else
    echo "ok   mutant (Facet::none) killed: $(cat "$td/err")"
fi
exit "$fail"
