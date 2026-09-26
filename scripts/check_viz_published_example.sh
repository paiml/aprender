#!/usr/bin/env bash
# check_viz_published_example.sh - examples/viz-facet-coord must build and run
# against the PUBLISHED aprender-viz, not the workspace copy (#3552).
#
# EV-15/EV-16 had no runnable Facet/Coord program to build against. The example
# goes study data -> panels -> Coord::apply -> SVG. A run:
#   0. runs --self-test first (the classifiers below, on known-bad inputs),
#   1. copies the example outside the workspace and renames Cargo.toml.in,
#   2. builds a Facet::wrap -> Facet::none mutant, which must exit non-zero,
#   3. builds the real example; `cargo pkgid` must name the crates.io registry
#      (a path or git source is RED); runs it and checks the SVG shape.
#
# Usage: check_viz_published_example.sh [--self-test]   (CARGO_TARGET_DIR honoured)
# Exit 0 = pass. 1 = a check failed. 2 = environment (no cargo / no network).

set -euo pipefail

WANT_PKGID='registry+https://github.com/rust-lang/crates.io-index#aprender-viz@0.69.1'

source_ok() { [ "$1" = "$WANT_PKGID" ]; }   # source_ok PKGID

svg_ok() { # svg_ok FILE -> 12 points, >=3 panel rects, a closed document
    local circles rects
    circles=$(grep -o '<circle' "$1" | wc -l)
    rects=$(grep -o '<rect' "$1" | wc -l)
    [ "$circles" -eq 12 ] && [ "$rects" -ge 3 ] && grep -q '</svg>' "$1"
}

svg_fixture() { # svg_fixture FILE CIRCLES RECTS CLOSED(0|1)
    local i
    {
        printf '<svg>'
        for i in $(seq "$2"); do printf '<circle/>'; done
        for i in $(seq "$3"); do printf '<rect/>'; done
        [ "$4" -eq 1 ] && printf '</svg>'
        printf '\n'
    } > "$1"
}

self_test() {
    local td fail=0
    td=$(mktemp -d "${TMPDIR:-/tmp}/viz-self.XXXXXX") || exit 2
    # shellcheck disable=SC2064
    trap "rm -rf -- '${td:?}'" EXIT
    row() { # row WANT(0|1) LABEL CMD...
        local want=$1 label=$2 got=0
        shift 2
        "$@" || got=1
        if [ "$got" -eq "$want" ]; then printf 'ok   self-test: %s\n' "$label"
        else printf 'FAIL self-test: %s (want %s, got %s)\n' "$label" "$want" "$got"; fail=1; fi
    }
    row 0 "crates.io 0.69.1 accepted" source_ok "$WANT_PKGID"
    row 1 "workspace path source RED" source_ok "path+file:///src/aprender/crates/aprender-viz#trueno_viz@0.70.0"
    row 1 "git source RED"            source_ok "git+https://github.com/paiml/aprender#aprender-viz@0.69.1"
    row 1 "other version RED"         source_ok "registry+https://github.com/rust-lang/crates.io-index#aprender-viz@0.68.2"
    svg_fixture "$td/good.svg" 12 3 1
    svg_fixture "$td/few.svg" 4 3 1
    svg_fixture "$td/onepanel.svg" 12 1 1
    svg_fixture "$td/open.svg" 12 3 0
    row 0 "well-formed SVG accepted"  svg_ok "$td/good.svg"
    row 1 "missing points RED"        svg_ok "$td/few.svg"
    row 1 "one panel RED"             svg_ok "$td/onepanel.svg"
    row 1 "unclosed document RED"     svg_ok "$td/open.svg"
    return "$fail"
}

case "${1:-}" in
    --self-test) self_test; exit $? ;;
    "") ;;
    *) sed -n '13,14p' "$0" >&2; exit 2 ;;
esac

( self_test ) || { echo "FAIL: self-test" >&2; exit 1; }

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
bin="$CARGO_TARGET_DIR/debug/viz-facet-coord"
fail=0
cd "$td"

# Known-bad first: one panel instead of three must be caught by the example.
cp src/main.rs "$td/main.rs.orig"
sed -i 's/Facet::wrap("lineage", NCOL)/Facet::none()/' src/main.rs
grep -q 'Facet::none()' src/main.rs || { echo "FAIL mutation did not apply"; exit 1; }
# A failed build would leave an older binary in place and be judged instead.
rm -f -- "${bin:?}"
cargo build --quiet || { echo "mutant build failed; not judged" >&2; exit 2; }
if "$bin" "$td/mut.svg" 2> "$td/err"; then
    echo "FAIL mutant (Facet::none) survived"; fail=1
else
    echo "ok   mutant (Facet::none) killed: $(cat "$td/err")"
fi

cp "$td/main.rs.orig" src/main.rs
rm -f -- "${bin:?}"
cargo build --quiet || { echo "FAIL: example does not build against the published crate" >&2; exit 1; }

id=$(cargo pkgid aprender-viz 2>/dev/null) || id="unresolved"
if source_ok "$id"; then echo "ok   aprender-viz resolved from crates.io: $id"
else echo "FAIL aprender-viz is not the published crate: $id"; fail=1; fi

if "$bin" "$td/out.svg" 2> "$td/err"; then
    if svg_ok "$td/out.svg"; then echo "ok   SVG shape ($(cat "$td/err"))"
    else echo "FAIL SVG shape: $(grep -o '<circle' "$td/out.svg" | wc -l) points"; fail=1; fi
else
    echo "FAIL example exited non-zero: $(cat "$td/err")"; fail=1
fi
exit "$fail"
