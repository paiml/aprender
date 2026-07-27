#!/usr/bin/env bash
# Poka-yoke: every beat must be EXECUTED by some workflow.
#
# A "beat" is this project's unit of competitive proof  -  a falsifiable assertion
# against a named external tool that must turn CI red when it stops being true.
# A beat that no workflow runs is worse than no beat: it looks like enforcement,
# it is counted as enforcement, and it proves nothing. That is negative EV.
#
# This has already happened. beat_ollama_decode_throughput_speed  -  the Pillar-4
# marquee "1.371x faster decode than Ollama"  -  sat in ZERO workflows while being
# quoted as an enforced win (fixed in #2319). This script makes that class of
# defect a build failure instead of an audit finding.
#
# TWO POPULATIONS, TWO RULES (getting this wrong is why a hand audit miscounted):
#
#   1. INTEGRATION beats  -  crates/*/tests/beat_*.rs
#      Separate test TARGETS. `cargo test --lib` does NOT run them. They execute
#      only if some workflow names them explicitly: `--test <name>`.
#      => REQUIRE an explicit reference.
#
#   2. LIB beats  -  crates/*/src/**/beat_*.rs
#      `#[cfg(test)] mod` compiled into the lib, so `workspace-test`'s `--lib`
#      already runs them. Requiring `--test <name>` here would be WRONG and would
#      reject a correctly-wired beat.
#      => REQUIRE that they are reachable, i.e. declared by a `mod` somewhere.
#
# Companion to check_runner_labels.sh, same shape: pure text check, no network,
# no build, runs on the clean-room pool.
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT" || exit 2

WORKFLOWS=".github/workflows"
[ -d "$WORKFLOWS" ] || { echo "✗ check_beats_gated: no $WORKFLOWS directory" >&2; exit 2; }

# Every `--test <name>` named anywhere in any workflow. A beat wired into a
# NIGHTLY lane counts as executed  -  this gate asserts "runs somewhere", not
# "blocks a PR". Blocking-ness is a separate policy question per beat.
REFERENCED=$(grep -ohE '\-\-test[[:space:]]+[A-Za-z0-9_]+' "$WORKFLOWS"/*.yml 2>/dev/null \
             | sed -E 's/--test[[:space:]]+//' | sort -u)

fail=0
n_integration=0
n_lib=0

# ---- population 1: integration beats must be explicitly named -----------------
while IFS= read -r f; do
    [ -n "$f" ] || continue
    n_integration=$((n_integration + 1))
    name=$(basename "$f" .rs)
    if ! printf '%s\n' "$REFERENCED" | grep -qx "$name"; then
        echo "✗ UNGATED BEAT: $f"
        echo "    No workflow runs it. It is an integration test TARGET, so"
        echo "    \`cargo test --lib\` does NOT reach it  -  it executes only if a"
        echo "    workflow names it explicitly."
        echo "    Fix: add \`cargo test -p <crate> --test $name\` to the chained"
        echo "    gate at .github/workflows/ci.yml (per-PR blocking), or to a"
        echo "    nightly lane if it needs a GPU/model/daemon."
        fail=1
    fi
done < <(find crates -path '*/tests/beat_*.rs' -not -path '*/src/*' 2>/dev/null | sort)

# ---- population 2: lib beats must be reachable via a `mod` --------------------
while IFS= read -r f; do
    [ -n "$f" ] || continue
    n_lib=$((n_lib + 1))
    name=$(basename "$f" .rs)
    # Reachable if EITHER:
    #   (a) `mod <name>;` / `mod <name> {`   -  the filename is the module name, or
    #   (b) `#[path = ".../<name>.rs"]`      -  declared under a DIFFERENT module
    #       name via a path override.
    #
    # (b) is not hypothetical: beat_fail_closed_config.rs is declared as
    # `#[path = "beat_fail_closed_config.rs"] mod apr_beat_fail_closed_config;`
    # in special_tokens.rs:237. Checking only (a) reports it orphaned when it in
    # fact runs (verified: 3 passed). A gate with false positives trains people
    # to ignore it, which is the same end state as having no gate.
    crate_src=$(dirname "$f")
    while [ "$crate_src" != "." ] && [ "$(basename "$crate_src")" != "src" ]; do
        crate_src=$(dirname "$crate_src")
    done
    if ! grep -rqE "^[[:space:]]*(pub[[:space:]]+)?mod[[:space:]]+${name}[[:space:];{]" \
         "$crate_src" 2>/dev/null \
       && ! grep -rqE "#\[path[[:space:]]*=[[:space:]]*\"[^\"]*${name}\.rs\"\]" \
         "$crate_src" 2>/dev/null; then
        echo "✗ ORPHAN LIB BEAT: $f"
        echo "    No \`mod $name\` declares it, so it is never compiled and never"
        echo "    runs  -  the same silent-no-op class as an unwired integration beat."
        fail=1
    fi
done < <(find crates -path '*/src/*' -name 'beat_*.rs' 2>/dev/null | sort)

total=$((n_integration + n_lib))
if [ "$total" -eq 0 ]; then
    echo "✗ check_beats_gated: found NO beat files at all  -  the discovery globs are"
    echo "   probably wrong after a refactor. Failing rather than passing vacuously." >&2
    exit 2
fi

if [ "$fail" -ne 0 ]; then
    echo ""
    echo "A beat that no workflow executes is NEGATIVE EV: it reads as enforcement," >&2
    echo "gets counted as enforcement, and proves nothing. Wire it or delete it." >&2
    exit 1
fi

echo "✓ check_beats_gated: all $total beats execute ($n_integration integration wired via --test, $n_lib lib via --lib)"
