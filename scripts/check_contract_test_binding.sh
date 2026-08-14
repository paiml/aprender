#!/usr/bin/env bash
#
# check_contract_test_binding.sh — a contract may not cite a test that does not exist.
#
# WHY THIS EXISTS (#2465)
# -----------------------
# Replacing a real test name in a contract's `falsification_tests` with
# `MUTANT_this_test_fn_does_not_exist_anywhere` left `pv validate` reporting the
# contract VALID. Two independent holes:
#
#   1. `pv validate` is schema validation. It does not resolve test references
#      at all, and it is what CI ran. The gate that DOES resolve them
#      (PV-VER-002, `pv lint --strict-test-binding`) ran nowhere.
#   2. That gate read `falsification_tests[].test` only, and skipped every entry
#      without one. 619 of 4206 entries in contracts/ name their test in
#      `test_harness:`/`name:` instead — 94 of those holding a real
#      `cargo test …` invocation. Skipped and bound looked identical in the
#      output.
#
# Hole 2 is fixed in crates/aprender-contracts/src/lint/strict_test_binding.rs.
# This script closes hole 1: it runs the blocking gate and fails the build.
#
# WHAT IT CHECKS
# --------------
# `pv lint contracts/ --strict-test-binding` resolves every cited test filter
# against the source tree, following cargo's own semantics (a `cargo test`
# filter is a SUBSTRING of a test's full `module::path::fn`). Every PV-VER-002
# finding is a contract citing something no `cargo test` invocation can run.
#
# WHY NOT JUST `pv lint --strict`
# -------------------------------
# `--strict` promotes EVERY warning to an error across all nine gates, and the
# tree currently carries 991 unrelated ones (PV-ENF-001 preconditions, …). CI
# would be permanently red for reasons that have nothing to do with test
# binding. This script gates on PV-VER-002 alone.
#
# BASELINE RATCHET
# ----------------
# Pre-existing debt lives in scripts/contract_test_binding_baseline.txt as
# `path<TAB>count`. The guard fails if a contract exceeds its baseline, or if a
# contract NOT in the baseline has any finding. New contracts are at zero from
# day one; the committed sum can only fall. Regenerate with --update-baseline
# (which refuses to raise a count).
#
# VACUITY GUARD
# -------------
# A gate that measures nothing must not pass as clean — the coverage floor
# reported 0/0 for months and read as GREEN. This one refuses to pass unless
# the strict-test-binding gate actually ran (present, not `skipped`) and
# resolved at least MIN_REFS references.
#
# SELF-TEST
# ---------
#   bash scripts/check_contract_test_binding.sh --self-test
# drives the REAL `pv` over a hermetic fixture tree with a five-case
# must-flag / must-not-flag table, then mutates the ratchet's own input to
# prove the comparison turns RED. Verification Discipline #7: re-run the table,
# never re-read the logic.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="${REPO_ROOT}/scripts/contract_test_binding_baseline.txt"
CONTRACT_DIR="${CONTRACT_DIR:-contracts}"
# 363 refs resolve today. A floor well under that catches a broken scan (0 refs)
# without tripping on ordinary contract churn.
MIN_REFS="${MIN_REFS:-250}"

# Scratch dir, cleaned by a single EXIT trap. Deliberately GLOBAL: a `local tmp`
# plus `trap 'rm -rf "$tmp"' EXIT` fires the trap after the function's frame is
# gone, so under `set -u` the cleanup itself aborts with `tmp: unbound variable`.
GUARD_TMP=""
cleanup() {
    if [ -n "$GUARD_TMP" ]; then
        rm -rf "$GUARD_TMP"
    fi
}
trap cleanup EXIT

die() {
    printf '%s\n' "$*" >&2
    exit 1
}

need() {
    command -v "$1" >/dev/null 2>&1 || die "check_contract_test_binding: missing required tool: $1"
}

# ---------------------------------------------------------------------------
# Run `pv lint --strict-test-binding` and emit its JSON on stdout.
#
# `pv` is invoked through `cargo run`, never through PATH and never through a
# path found lying in the target directory. On this box the target dir is
# redirected by a gitignored .cargo/config.toml and is SHARED between the main
# checkout and every worktree, so a `pv` sitting there may have been built from
# a different tree — while writing this guard, a concurrent build from `main`
# replaced the binary mid-session and a measurement silently reverted to the
# pre-fix numbers. Asking cargo to build-and-run is the only resolution that
# cannot pick up someone else's artifact.
# ---------------------------------------------------------------------------
run_pv_lint() {
    local contract_dir="$1" out="$2" err="$3"
    ( cd "$REPO_ROOT" && cargo run -q -p aprender-contracts-cli --bin pv -- \
        lint "$contract_dir" --strict-test-binding --format json --no-cache ) \
        > "$out" 2> "$err"
    # NOTE: pv exits 1 whenever ANY gate reports findings, which it always does
    # on this tree. The exit status is therefore not the verdict — the JSON is.
    # What we must not tolerate is pv failing to produce parseable JSON, which
    # is checked by the caller.
    return 0
}

# Extract `path<TAB>count` for PV-VER-002 findings, sorted. Reads JSON on $1.
findings_by_file() {
    jq -r '[.findings[] | select(.rule_id == "PV-VER-002") | .file]
           | group_by(.)[]
           | "\(.[0])\t\(length)"' "$1" | LC_ALL=C sort
}

# Assert the gate actually ran and measured something. Reads JSON on $1.
assert_gate_measured() {
    local json="$1" gate refs skipped
    gate=$(jq -r '[.gates[] | select(.name == "strict-test-binding")] | length' "$json" 2>/dev/null)
    [ "$gate" = "1" ] || die "VACUOUS: no strict-test-binding gate in pv output - the gate did not run."
    skipped=$(jq -r '.gates[] | select(.name == "strict-test-binding") | .skipped' "$json")
    [ "$skipped" = "false" ] \
        || die "VACUOUS: strict-test-binding gate was SKIPPED (contract validation failed); nothing was measured."
    refs=$(jq -r '.gates[] | select(.name == "strict-test-binding") | .detail.total_refs' "$json")
    case "$refs" in
        ''|*[!0-9]*) die "VACUOUS: strict-test-binding gate reported no total_refs." ;;
    esac
    [ "$refs" -ge "$MIN_REFS" ] \
        || die "VACUOUS: only $refs test references resolved (floor $MIN_REFS). The source scan is broken."
    printf '%s\n' "$refs"
}

# ---------------------------------------------------------------------------
# The ratchet. Compares observed `path<TAB>count` ($1) against a baseline ($2).
# Factored out so --self-test can drive it with fixture data.
# Prints violations; returns 1 if any.
# ---------------------------------------------------------------------------
compare_to_baseline() {
    local observed="$1" baseline="$2" rc=0 path count allowed
    while IFS=$'\t' read -r path count; do
        [ -n "$path" ] || continue
        allowed=$(LC_ALL=C awk -F'\t' -v p="$path" '$1 == p { print $2; found=1 } END { if (!found) print 0 }' "$baseline")
        if [ "$count" -gt "$allowed" ]; then
            printf 'FAIL %s: %s dangling test reference(s), baseline allows %s\n' \
                "$path" "$count" "$allowed"
            rc=1
        fi
    done < "$observed"
    return "$rc"
}

# ---------------------------------------------------------------------------
# Self-test: drive the real pv over a hermetic fixture, then mutate the ratchet.
# ---------------------------------------------------------------------------
write_fixture() {
    local root="$1"
    mkdir -p "$root/contracts" "$root/crates/demo/src"
    cat > "$root/crates/demo/src/lib.rs" <<'RS'
#[cfg(test)]
mod demo_tests {
    #[test]
    fn real_test_exists() {}
}
RS
    cat > "$root/contracts/selftest-v1.yaml" <<'YAML'
metadata:
  version: 1.0.0
  created: '2026-08-14'
  author: check_contract_test_binding self-test
  kind: registry
  description: Hermetic fixture for the contract test-binding guard.
  references:
    - 'scripts/check_contract_test_binding.sh'
kind: KernelContract
name: selftest
version: "1.0.0"
status: ACTIVE
falsification_tests:
  - id: CASE-A-TEST-FIELD
    rule: "a nonexistent fn cited in test: must be flagged"
    prediction: "flagged"
    test: "cargo test -p demo --lib MUTANT_absent_alpha"
    if_fails: "the gate is blind to the test field"
  - id: CASE-B-HARNESS-FIELD
    rule: "a nonexistent fn cited in test_harness: must be flagged"
    prediction: "flagged"
    test_harness: "cargo test -p demo --lib MUTANT_absent_bravo"
    name: "MUTANT_absent_bravo"
    if_fails: "the gate skips entries that bind via test_harness (#2465)"
  - id: CASE-C-NAME-FIELD
    rule: "a nonexistent fn cited in name: alone must be flagged"
    prediction: "flagged"
    name: "MUTANT_absent_charlie"
    if_fails: "the gate skips entries that bind via name (#2465)"
  - id: CASE-D-SHELL-HARNESS
    rule: "a shell harness names a shell command, never a Rust test"
    prediction: "not flagged"
    test_harness: "grep -q 'apr monitor' book/src/cli/monitor.md"
    name: "module_mentioned"
    if_fails: "the gate false-positives on the 525 shell harnesses in contracts/"
  - id: CASE-E-REAL-FN
    rule: "a real fn cited via test_harness resolves"
    prediction: "not flagged"
    test_harness: "cargo test -p demo --lib real_test_exists"
    name: "real_test_exists"
    if_fails: "the gate cannot resolve through test_harness"
YAML
}

self_test() {
    local tmp rc=0 json err msg
    GUARD_TMP=$(mktemp -d) || die "mktemp failed"
    tmp="$GUARD_TMP"

    printf '== self-test: pv case table (must-flag / must-not-flag) ==\n'
    write_fixture "$tmp"
    json="$tmp/out.json"
    err="$tmp/err.txt"
    run_pv_lint "$tmp/contracts" "$json" "$err"
    if ! jq -e . "$json" >/dev/null 2>&1; then
        printf 'FAIL: pv produced no parseable JSON. stderr:\n' >&2
        cat "$err" >&2
        return 1
    fi

    msg=$(jq -r '[.findings[] | select(.rule_id == "PV-VER-002") | .message] | join("\n")' "$json")

    # MUST flag: one per binding field.
    for want in MUTANT_absent_alpha MUTANT_absent_bravo MUTANT_absent_charlie; do
        if printf '%s\n' "$msg" | grep -qF "$want"; then
            printf '  ok    must-flag   %s\n' "$want"
        else
            printf '  FAIL  must-flag   %s (not reported)\n' "$want"
            rc=1
        fi
    done

    # MUST NOT flag: the shell harness and the fn that really exists.
    for unwanted in module_mentioned real_test_exists; do
        if printf '%s\n' "$msg" | grep -qF "$unwanted"; then
            printf '  FAIL  must-not-flag %s (false positive)\n' "$unwanted"
            rc=1
        else
            printf '  ok    must-not-flag %s\n' "$unwanted"
        fi
    done

    # The field name must appear in the finding, so an operator knows the line.
    for field in '].test)' '].test_harness)' '].name)'; do
        if printf '%s\n' "$msg" | grep -qF "$field"; then
            printf '  ok    names field  %s\n' "$field"
        else
            printf '  FAIL  names field  %s (missing from message)\n' "$field"
            rc=1
        fi
    done

    printf '== self-test: ratchet must turn RED when a count exceeds baseline ==\n'
    printf 'contracts/a.yaml\t2\ncontracts/b.yaml\t1\n' > "$tmp/base.txt"

    printf 'contracts/a.yaml\t2\ncontracts/b.yaml\t1\n' > "$tmp/at.txt"
    if compare_to_baseline "$tmp/at.txt" "$tmp/base.txt" >/dev/null; then
        printf '  ok    at-baseline    GREEN\n'
    else
        printf '  FAIL  at-baseline    went RED\n'
        rc=1
    fi

    printf 'contracts/a.yaml\t3\ncontracts/b.yaml\t1\n' > "$tmp/over.txt"
    if compare_to_baseline "$tmp/over.txt" "$tmp/base.txt" >/dev/null; then
        printf '  FAIL  over-baseline  stayed GREEN\n'
        rc=1
    else
        printf '  ok    over-baseline  RED\n'
    fi

    printf 'contracts/unlisted.yaml\t1\n' > "$tmp/new.txt"
    if compare_to_baseline "$tmp/new.txt" "$tmp/base.txt" >/dev/null; then
        printf '  FAIL  new-contract   stayed GREEN\n'
        rc=1
    else
        printf '  ok    new-contract   RED\n'
    fi

    printf 'contracts/a.yaml\t1\n' > "$tmp/under.txt"
    if compare_to_baseline "$tmp/under.txt" "$tmp/base.txt" >/dev/null; then
        printf '  ok    under-baseline GREEN\n'
    else
        printf '  FAIL  under-baseline went RED\n'
        rc=1
    fi

    if [ "$rc" -eq 0 ]; then
        printf 'SELF-TEST PASS\n'
    else
        printf 'SELF-TEST FAIL\n' >&2
    fi
    return "$rc"
}

# ---------------------------------------------------------------------------
main() {
    need jq
    need cargo

    case "${1:-}" in
        --self-test) self_test; exit $? ;;
        --update-baseline) UPDATE=1 ;;
        '') UPDATE=0 ;;
        *) die "usage: $0 [--self-test | --update-baseline]" ;;
    esac

    local tmp json err refs observed
    GUARD_TMP=$(mktemp -d) || die "mktemp failed"
    tmp="$GUARD_TMP"
    json="$tmp/lint.json"
    err="$tmp/lint.err"

    printf 'Running pv lint %s --strict-test-binding ...\n' "$CONTRACT_DIR"
    run_pv_lint "$CONTRACT_DIR" "$json" "$err"
    if ! jq -e . "$json" >/dev/null 2>&1; then
        printf 'pv produced no parseable JSON. The measurement is MISSING, which is a failure.\n' >&2
        printf 'stderr was:\n' >&2
        cat "$err" >&2
        exit 1
    fi

    refs=$(assert_gate_measured "$json") || exit 1

    observed="$tmp/observed.txt"
    findings_by_file "$json" > "$observed"

    local total
    total=$(LC_ALL=C awk -F'\t' '{ s += $2 } END { print s + 0 }' "$observed")

    if [ "${UPDATE:-0}" = "1" ]; then
        # Bootstrap: with no baseline yet there is nothing to ratchet against,
        # so the first write seeds the file. Every later --update-baseline goes
        # through the refuse-to-raise path below. Deleting the (tracked) file to
        # get back here shows up as a deletion in the diff.
        if [ ! -f "$BASELINE" ]; then
            cp "$observed" "$BASELINE"
            printf 'BOOTSTRAP: seeded %s with %s contract(s), %s dangling reference(s).\n' \
                "$BASELINE" "$(wc -l < "$BASELINE" | tr -d ' ')" "$total"
            printf 'This number may only fall from here.\n'
            exit 0
        fi
        local raised=0 path count old
        while IFS=$'\t' read -r path count; do
            [ -n "$path" ] || continue
            old=$(LC_ALL=C awk -F'\t' -v p="$path" '$1 == p { print $2; found=1 } END { if (!found) print 0 }' "$BASELINE")
            if [ "$count" -gt "$old" ]; then
                printf 'REFUSING to raise baseline for %s (%s -> %s). Fix the contract instead.\n' \
                    "$path" "$old" "$count" >&2
                raised=1
            fi
        done < "$observed"
        [ "$raised" -eq 0 ] || exit 1
        cp "$observed" "$BASELINE"
        printf 'Baseline updated: %s entries, %s dangling reference(s) total.\n' \
            "$(wc -l < "$BASELINE" | tr -d ' ')" "$total"
        exit 0
    fi

    [ -f "$BASELINE" ] || die "missing baseline file: $BASELINE, create it with --update-baseline"

    printf 'Resolved %s test references; %s dangling across %s contract(s).\n' \
        "$refs" "$total" "$(wc -l < "$observed" | tr -d ' ')"

    if compare_to_baseline "$observed" "$BASELINE"; then
        printf 'PASS: no contract cites more nonexistent tests than its baseline allows.\n'
        exit 0
    fi

    printf '\n' >&2
    printf 'A contract cites a test that no `cargo test` invocation can run.\n' >&2
    printf 'Fix the citation (or add the test); do NOT raise the baseline.\n' >&2
    printf 'Detail:  cargo run -q -p aprender-contracts-cli --bin pv -- lint %s --strict-test-binding\n' \
        "$CONTRACT_DIR" >&2
    exit 1
}

main "$@"
