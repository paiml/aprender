#!/usr/bin/env bash
# check_model_tests_wired.sh — every `model-tests`-gated integration target must
# be RUN by a GitHub workflow, with the feature enabled.
#
# WHY THIS EXISTS (aprender#2522)
# ------------------------------
# `crates/aprender-core/tests/falsification_spec_v10_tests.rs` is the project's
# own falsification spec: 140 gates. It was named in the Makefile and in NO
# workflow. Nobody ran it, so nobody saw that 38 of the 140 were failing — and
# had been since APR-MONO moved every path they were anchored to.
#
# This target class is dark by CONSTRUCTION, which is why the existing wiring
# guards miss it:
#
#   * `#![cfg(feature = "model-tests")]` compiles the file to NOTHING without the
#     feature. A default `cargo test` does not merely skip these gates, it never
#     builds them — so no green run anywhere is evidence about them.
#   * CI's workspace line is `--lib`, which cannot reach an integration target.
#   * The one explicit integration line in ci.yml lists targets by name, and
#     these were not on it.
#
# `check_guards_are_wired.sh` is the sibling meta-guard for `scripts/check_*.sh`.
# This is the same idea for feature-gated test targets: enumerate the universe
# from the SOURCE TREE (what exists), not from the workflows (what someone
# remembered to add) — a guard built from the workflow side can only ever
# confirm what is already there.
#
#   bash scripts/check_model_tests_wired.sh              # check
#   bash scripts/check_model_tests_wired.sh --self-test  # case table
#
# NOTE: option-neutral by policy? No — this script is EXECUTED, never sourced,
# so `set` here affects only its own shell.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FEATURE="model-tests"

# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

# A workflow line WIRES target $1 when, with `#` comments stripped, it both
# names the target after `--test` and enables the feature.
#
# Mention is not execution. The #2522 suite was NAMED in a ci.yml comment before
# this guard existed; a `grep -F "$target"` would have called it wired.
line_wires_target() {
    local line="$1" target="$2" code
    code="${line%%#*}"
    case "$code" in
        *"--test $target"*|*"--test=$target"*) ;;
        *) return 1 ;;
    esac
    case "$code" in
        *"--features $FEATURE"*|*"--features=$FEATURE"*|*"--all-features"*) return 0 ;;
        *) return 1 ;;
    esac
}

# Fold backslash-continued lines into ONE logical command line.
#
# A real CI step wraps: `--features model-tests` on one line, the `--test`
# targets on the next. A per-PHYSICAL-line matcher calls that unwired, which is
# how the first version of this guard failed its own repo. The unit of meaning
# is the logical command.
fold_continuations() {
    awk '{
        if (buf != "") { sub(/^[ \t]+/, "", $0); line = buf " " $0 } else { line = $0 }
        if (line ~ /\\$/) { sub(/\\$/, "", line); buf = line } else { buf = ""; print line }
    }
    END { if (buf != "") print buf }' "$1"
}

# Test targets under crates/*/tests/*.rs whose source is gated on the feature.
gated_targets_in() {
    local root="$1" f base
    for f in "$root"/crates/*/tests/*.rs; do
        [ -f "$f" ] || continue
        grep -qF "#![cfg(feature = \"$FEATURE\")]" "$f" || continue
        base=$(basename "$f" .rs)
        printf '%s\n' "$base"
    done | sort -u
}

unwired_in() {
    local root="$1" target line wired f
    while IFS= read -r target; do
        [ -n "$target" ] || continue
        wired=0
        for f in "$root"/.github/workflows/*.yml "$root"/.github/workflows/*.yaml; do
            [ -f "$f" ] || continue
            while IFS= read -r line; do
                if line_wires_target "$line" "$target"; then
                    wired=1
                    break
                fi
            done < <(fold_continuations "$f")
            [ "$wired" -eq 1 ] && break
        done
        [ "$wired" -eq 0 ] && printf '%s\n' "$target"
    done < <(gated_targets_in "$root")
    return 0
}

# ---------------------------------------------------------------------------
# Case table (rule 7: guard regexes ship a case table)
# ---------------------------------------------------------------------------

self_test() {
    local fails=0 desc line target want got
    # want=0 means "must be treated as WIRED", want=1 means "must NOT be".
    while IFS='|' read -r want target line desc; do
        [ -n "${want:-}" ] || continue
        case "$want" in \#*) continue ;; esac
        if line_wires_target "$line" "$target"; then got=0; else got=1; fi
        if [ "$got" != "$want" ]; then
            printf 'FAIL: %s\n      line=<%s>\n      expected=%s got=%s\n' \
                "$desc" "$line" "$want" "$got"
            fails=$((fails + 1))
        fi
    done <<'TABLE'
0|falsification_spec_v10_tests|        run: cargo test -p aprender-core --features model-tests --test falsification_spec_v10_tests|plain invocation
0|falsification_spec_v10_tests|  run: cargo test --features=model-tests --test=falsification_spec_v10_tests|equals-form flags
0|falsification_spec_v10_tests|  run: cargo test -p aprender-core --all-features --test falsification_spec_v10_tests|--all-features enables it
0|falsification_stress_tests|  run: cargo test --features model-tests --test falsification_spec_v10_tests --test falsification_stress_tests|second target on a multi---test line
1|falsification_spec_v10_tests|  # run: cargo test --features model-tests --test falsification_spec_v10_tests|whole-line comment is a MENTION, not execution
1|falsification_spec_v10_tests|  run: echo skipped # cargo test --features model-tests --test falsification_spec_v10_tests|trailing comment is a MENTION
1|falsification_spec_v10_tests|  run: cargo test -p aprender-core --test falsification_spec_v10_tests|named but feature NOT enabled -- compiles to nothing
1|falsification_spec_v10_tests|  run: cargo test -p aprender-core --features model-tests --lib|feature on, target absent
1|falsification_spec_v10_tests|  # See crates/aprender-core/tests/falsification_spec_v10_tests.rs|bare path mention
1|falsification_stress_tests|  run: cargo test --features model-tests --test falsification_spec_v10_tests|a DIFFERENT target being wired is not this one
TABLE

    # Folding cases: a wrapped command must read as ONE line, and a comment must
    # still not survive folding into executable text.
    local fixture_file folded
    fixture_file=$(mktemp)
    printf '        run: |\n          cargo test -p aprender-core --features model-tests \\\n            --test falsification_spec_v10_tests \\\n            --test falsification_stress_tests\n' > "$fixture_file"
    folded=$(fold_continuations "$fixture_file")
    for target in falsification_spec_v10_tests falsification_stress_tests; do
        if ! line_wires_target "$folded" "$target"; then
            printf 'FAIL: folded multi-line command did not wire %s\n      folded=<%s>\n' \
                "$target" "$folded"
            fails=$((fails + 1))
        fi
    done
    printf '          # cargo test --features model-tests \\\n          #   --test falsification_spec_v10_tests\n' > "$fixture_file"
    folded=$(fold_continuations "$fixture_file")
    while IFS= read -r line; do
        if line_wires_target "$line" falsification_spec_v10_tests; then
            printf 'FAIL: a COMMENTED multi-line command was read as wiring: <%s>\n' "$line"
            fails=$((fails + 1))
        fi
    done <<< "$folded"
    rm -f "$fixture_file"

    # The table above proves the matcher. This proves the ENUMERATOR: a fixture
    # tree with one gated target and no workflow wiring must report exactly that
    # target. Without this, `gated_targets_in` could silently yield nothing and
    # every check would pass vacuously.
    local fixture
    fixture=$(mktemp -d)
    # shellcheck disable=SC2064
    trap "rm -rf '$fixture'" EXIT
    mkdir -p "$fixture/crates/fixture-crate/tests" "$fixture/.github/workflows"
    printf '#![cfg(feature = "%s")]\n' "$FEATURE" > "$fixture/crates/fixture-crate/tests/dark_target.rs"
    printf 'jobs:\n  x:\n    steps:\n      - run: cargo test --lib\n' > "$fixture/.github/workflows/ci.yml"
    got=$(unwired_in "$fixture")
    if [ "$got" != "dark_target" ]; then
        printf 'FAIL: enumerator on fixture tree: expected <dark_target>, got <%s>\n' "$got"
        fails=$((fails + 1))
    fi
    # And the same tree WITH wiring must report nothing.
    printf 'jobs:\n  x:\n    steps:\n      - run: cargo test --features %s --test dark_target\n' \
        "$FEATURE" > "$fixture/.github/workflows/ci.yml"
    got=$(unwired_in "$fixture")
    if [ -n "$got" ]; then
        printf 'FAIL: wired fixture tree still reported <%s>\n' "$got"
        fails=$((fails + 1))
    fi
    # A tree with NO gated targets must not silently pass as "all wired": the
    # caller below refuses an empty universe.
    rm -f "$fixture/crates/fixture-crate/tests/dark_target.rs"
    got=$(gated_targets_in "$fixture")
    if [ -n "$got" ]; then
        printf 'FAIL: empty fixture tree yielded targets <%s>\n' "$got"
        fails=$((fails + 1))
    fi

    if [ "$fails" -gt 0 ]; then
        printf '\n%s case(s) failed\n' "$fails"
        return 1
    fi
    printf 'OK: check_model_tests_wired case table passed (10 matcher + 4 folding + 3 enumerator cases)\n'
    return 0
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if [ "${1:-}" = "--self-test" ]; then
    self_test
    exit $?
fi

targets=$(gated_targets_in "$REPO_ROOT")
if [ -z "$targets" ]; then
    printf 'FAIL: found 0 test targets gated on `%s`.\n' "$FEATURE"
    printf '      A guard over an empty universe passes every assertion put to it.\n'
    printf '      Either the feature was renamed, or crates/*/tests/ moved.\n'
    exit 1
fi
target_count=$(printf '%s\n' "$targets" | wc -l | tr -d ' ')

unwired=$(unwired_in "$REPO_ROOT")
if [ -n "$unwired" ]; then
    printf 'FAIL: %s-gated test target(s) named by no workflow:\n\n' "$FEATURE"
    printf '%s\n' "$unwired" | sed 's/^/  /'
    printf '\nA `#![cfg(feature = "%s")]` target does not merely skip without the\n' "$FEATURE"
    printf 'feature -- it compiles to nothing, so no green CI run says anything about\n'
    printf 'it. Add it to a workflow step that passes --features %s.\n' "$FEATURE"
    exit 1
fi

printf 'OK: all %s `%s`-gated test target(s) are run by a workflow\n' "$target_count" "$FEATURE"
printf '%s\n' "$targets" | sed 's/^/  /'
exit 0
