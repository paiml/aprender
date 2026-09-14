#!/usr/bin/env bash
# check_feature_gated_test_leak.sh -- a sibling crate must not be able to arm a
# test target that is gated on a RUNNER CAPABILITY.
#
# WHAT THIS IS ABOUT (#3242, measured on #3205)
# ---------------------------------------------
# crates/aprender-test-lib/tests/falsify_chromium_driver_is_real.rs opened with
#
#     #![cfg(feature = "browser")]
#
# and its own header explained the containment: "NOT on ci.yml's beat list,
# because the clean-room image is not known to ship a browser; they belong on a
# Chrome-equipped runner the way the GPU falsifiers belong on a CUDA one. Running
# them without a browser FAILS -- deliberately. There is no skip."
#
# A cargo feature could not hold that. crates/aprender-orchestrate's
# [dev-dependencies] line 10 is
#
#     jugar-probar = { path = "../aprender-test-lib",
#                      package = "aprender-test-lib", features = ["browser"] }
#
# and the BSE-17 quick tier runs ONE nextest invocation with `--lib --tests` over
# the whole selection. `--tests` builds dev-dependencies. A PR touching both
# crates unified `browser` on, compiled aprender-test-lib's OWN browser-gated
# target, and ran three browser tests on a browser-less clean-room runner:
#
#     could not launch a real browser: Browser not found. Install Chromium or
#     set CHROMIUM_PATH
#
# 30,281 passed, 3 failed, on the operator's priority PR.
#
# Feature unification is a property of the BUILD GRAPH. "Default features" is a
# property of a PACKAGE. ci_test_tier.sh's header asserts the second over the
# first, and this is the gap.
#
# WHY A RATCHET AND NOT A HARD FAIL
# ---------------------------------
# 24 such edges exist today across 12 crates and 23 (crate, feature) gates --
# 46 of the gated targets are `cuda`. Most are probably harmless: a crate the
# quick tier never selects, or a gate whose capability every runner has. Failing
# the build on all 24 would be an assertion the tree has not earned, which is the
# shape this repo names most often. So: the baseline is the measured number, it
# only ever shrinks, and a NEW edge is what fails.
#
# Usage
#   bash scripts/check_feature_gated_test_leak.sh                 measure + ratchet
#   bash scripts/check_feature_gated_test_leak.sh --self-test     the case table
#   bash scripts/check_feature_gated_test_leak.sh --update        re-record (shrink only)
#   bash scripts/check_feature_gated_test_leak.sh --list          print the edges
set -uo pipefail

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." > /dev/null 2>&1 && pwd )"
BASELINE="$REPO_ROOT/scripts/feature_gated_test_leak_baseline.txt"
CASES="$REPO_ROOT/scripts/lib/feature_leak_cases"

usage() {
    cat <<'USAGE'
check_feature_gated_test_leak.sh -- a sibling must not arm a capability-gated test target.

  (no arguments)   measure the tree and ratchet against the baseline
  --self-test      run the committed case table (fixtures, no cargo)
  --list           print every edge, one per line
  --update         re-record the baseline; refuses a RISE
  --help
USAGE
}

# gates_scan ROOT -> TSV: <crate><TAB><feature>
# Which crates have an integration target gated on a cargo feature. Filesystem
# work, kept separate from the edge computation so the case table can supply its
# own gates without a fixture tree.
gates_scan() {
    local root="$1" d crate f
    while IFS= read -r d; do
        crate="$( basename "$d" )"
        [ -d "$root/$d/tests" ] || continue
        while IFS= read -r f; do
            # head -c: only the file's prologue can carry an inner attribute, and
            # reading whole test files across 80+ crates is the difference
            # between a second and a minute.
            head -c 4000 "$f" 2>/dev/null \
                | grep -oE '^#!\[cfg\(feature[[:space:]]*=[[:space:]]*"[^"]+"\)\]' \
                | sed -E 's/.*"([^"]+)".*/\1/' \
                | while IFS= read -r feat; do printf '%s\t%s\n' "$crate" "$feat"; done
        done < <( find "$root/$d/tests" -name '*.rs' -type f 2>/dev/null )
    done < <( cd "$root" && git ls-files 'crates/*/Cargo.toml' | xargs -r -n1 dirname )
}

# edges METADATA_JSON GATES_TSV -> TSV: <enabler><TAB><kind><TAB><target crate><TAB><feature><TAB><how>
# Pure: JSON and TSV in, edges out. No cargo, no filesystem beyond these two.
edges() {
    local meta="$1" gates="$2"
    [ -r "$meta" ] && [ -r "$gates" ] || { printf 'ENV: unreadable input\n' >&2; return 2; }
    jq -r --rawfile gatesraw "$gates" -f "$REPO_ROOT/scripts/lib/feature_leak_edges.jq" \
        "$meta" 2>/dev/null | sort -u
}

self_test() {
    local fails=0 rows=0 out
    # A literal tab. `grep -E` does not interpret \t, so a pattern written with
    # the escape matches nothing and every row goes green for the wrong reason --
    # which is what the first draft of this table did.
    local T; T="$( printf '\t' )"
    if [ ! -d "$CASES" ]; then printf 'FAIL (vacuity): no fixtures at %s\n' "$CASES"; return 1; fi
    local n; n=$( find "$CASES" -name '*.json' | wc -l | tr -d ' ' )
    if [ "$n" -lt 2 ]; then printf 'FAIL (vacuity): %s fixture(s), expected 2+\n' "$n"; return 1; fi

    _has() { # _has LABEL PATTERN META GATES
        rows=$(( rows + 1 ))
        out="$( edges "$CASES/$3" "$CASES/$4" )"
        if grep -qE "$2" <<<"$out"; then printf 'ok    %s\n' "$1"
        else printf 'FAIL  %s: no edge matching /%s/\n' "$1" "$2"; printf '%s\n' "$out" | sed 's/^/      | /'; fails=1; fi
    }
    _hasnt() {
        rows=$(( rows + 1 ))
        out="$( edges "$CASES/$3" "$CASES/$4" )"
        if grep -qE "$2" <<<"$out"; then printf 'FAIL  %s: edge matching /%s/ should NOT be reported\n' "$1" "$2"; printf '%s\n' "$out" | sed 's/^/      | /'; fails=1
        else printf 'ok    %s\n' "$1"; fi
    }

    # R1 is the measured instance, reduced to two manifests: a dev-dependency
    # naming a feature that gates the DEPENDED-ON crate's own test target.
    _has 'R1 a dev-dependency arms a sibling gated target (the #3205 instance)' \
         "orch${T}dev${T}testlib${T}browser" fx_devdep.json fx_devdep_gates.tsv
    # R2: the same shape reached through a feature table entry `dep/feature`
    # rather than a direct features=[]. Both spellings, or half the surface is
    # invisible -- aprender-test-cli reaches aprender-test-lib/browser this way.
    _has 'R2 a feature table entry dep/feature is the same edge' \
         "cli${T}normal${T}testlib${T}browser" fx_featuremap.json fx_devdep_gates.tsv
    # R3 IS THE DISCRIMINATION ROW. Identical manifests; the only difference is
    # that `browser` gates no test target. Nothing to arm, so nothing to report.
    _hasnt 'R3 a feature that gates NO test target is not an edge' \
           'testlib' fx_devdep.json fx_empty_gates.tsv
    # R4: a crate naming its OWN feature is not a leak -- it owns that target.
    _hasnt 'R4 a crate arming its own gated target is not a leak' \
           "testlib${T}[a-z]+${T}testlib" fx_devdep.json fx_devdep_gates.tsv
    # R5: an optional dependency still counts. `optional = true` means "off until
    # a feature turns it on", not "never on" -- aprender-cgp reaches
    # aprender-gpu/cuda exactly this way.
    _has 'R5 an optional dependency is still an edge' \
         "optdep${T}normal${T}testlib${T}browser" fx_optional.json fx_devdep_gates.tsv

    # An unreadable input is ENV, never an empty (and therefore passing) answer.
    rows=$(( rows + 1 ))
    edges "$CASES/does-not-exist.json" "$CASES/fx_devdep_gates.tsv" > /dev/null 2>&1
    if [ $? -eq 2 ]; then printf 'ok    E1 an unreadable metadata file is ENV (exit 2), not "no edges"\n'
    else printf 'FAIL  E1 an unreadable metadata file did not answer ENV\n'; fails=1; fi

    printf '\n%s row(s), %s\n' "$rows" "$( [ "$fails" -eq 0 ] && echo '0 red / FALSIFIER GREEN' || echo 'RED' )"
    return "$fails"
}

# The work dir is script-scope with an EXIT trap, NOT a `local` with a RETURN
# trap: a RETURN trap fires when measure() returns, so the caller found the
# directory already deleted and awk reported "cannot open file" over an empty
# count -- a guard that prints nothing and exits 0 is a guard that passed
# vacuously.
RUNDIR=""
cleanup() { [ -n "${RUNDIR:-}" ] && rm -rf "${RUNDIR:?}"; }
trap cleanup EXIT

measure() {
    local tmp; tmp="$(mktemp -d)" || return 2
    RUNDIR="$tmp"
    command -v jq > /dev/null 2>&1 || { printf 'ENV: jq is not on PATH\n' >&2; return 2; }
    ( cd "$REPO_ROOT" && cargo metadata --no-deps --format-version 1 ) > "$tmp/meta.json" 2>/dev/null \
        || { printf 'ENV: cargo metadata failed\n' >&2; return 2; }
    gates_scan "$REPO_ROOT" | sort -u > "$tmp/gates.tsv"
    # Vacuity: a tree with no feature-gated test target at all means the SCAN is
    # broken, not that the tree is clean -- and a broken scan reports zero edges,
    # which reads exactly like a pass.
    local ng; ng=$( wc -l < "$tmp/gates.tsv" | tr -d ' ' )
    if [ "${ng:-0}" -lt 5 ]; then
        printf 'FAIL (vacuity): %s feature-gated test target(s) found, expected 5+.\n' "$ng"
        printf 'The scan is broken, not the tree. Fix it rather than this number.\n'
        return 1
    fi
    edges "$tmp/meta.json" "$tmp/gates.tsv" > "$tmp/edges.tsv" || return 2
    printf '%s\n' "$tmp"
}

MODE="${1:-measure}"
case "$MODE" in
    --help|-h) usage; exit 0 ;;
    --self-test|--selftest) self_test; exit $? ;;
esac

TMP="$( measure )" || exit $?
EDGES="$TMP/edges.tsv"
COUNT=$( wc -l < "$EDGES" | tr -d ' ' )
GATES=$( wc -l < "$TMP/gates.tsv" | tr -d ' ' )

if [ "$MODE" = '--list' ]; then
    awk -F'\t' '{ printf "  %-26s [%-6s] -> %s/%-14s  via %s\n", $1, $2, $3, $4, $5 }' "$EDGES"
    printf '%s edge(s) over %s feature-gated test target(s)\n' "$COUNT" "$GATES"
    exit 0
fi

BASE=0
[ -r "$BASELINE" ] && BASE=$( grep -cvE '^[[:space:]]*(#|$)' "$BASELINE" )

if [ "$MODE" = '--update' ]; then
    if [ "$COUNT" -gt "$BASE" ] && [ -r "$BASELINE" ]; then
        printf 'REFUSED: %s edges is a RISE over the baseline %s. This ratchet only shrinks.\n' "$COUNT" "$BASE"
        exit 1
    fi
    { grep -E '^[[:space:]]*#' "$BASELINE" 2>/dev/null || true; cat "$EDGES"; } > "$BASELINE.new"
    mv "$BASELINE.new" "$BASELINE"
    printf 'baseline set to %s edge(s)\n' "$COUNT"
    exit 0
fi

printf '=== cross-crate arming of capability-gated test targets (#3242) ===\n'
printf '  baseline %-4s measured %s   over %s gated target(s)\n' "$BASE" "$COUNT" "$GATES"
if [ "$COUNT" -gt "$BASE" ]; then
    printf '\nFAIL: %s edges, ceiling %s.\n' "$COUNT" "$BASE"
    printf 'A crate can now arm a sibling test target that is gated on a runner capability.\n'
    printf 'Feature unification is a property of the BUILD GRAPH; the quick tier builds the\n'
    printf 'selected crates in ONE graph, so the gate does not hold. Give the target its own\n'
    printf 'feature name that no sibling asks for -- see aprender-test-lib browser-falsify.\n\n'
    printf 'NEW since the baseline:\n'
    comm -13 <( grep -vE '^[[:space:]]*(#|$)' "$BASELINE" 2>/dev/null | sort ) <( sort "$EDGES" ) \
        | awk -F'\t' '{ printf "  %-26s [%-6s] -> %s/%-14s  via %s\n", $1, $2, $3, $4, $5 }'
    exit 1
fi
if [ "$COUNT" -lt "$BASE" ]; then
    printf '  IMPROVED -- run --update to record %s.\n' "$COUNT"
fi
printf '  PASS -- at or under the ceiling.\n'
exit 0
