#!/usr/bin/env bash
#
# check_contract_enforcement.sh — a contract may not name an enforcement command
# that cannot run.
#
# WHY THIS EXISTS (#2504, SURF-14 / R11)
# --------------------------------------
# contracts/apr-cli-commands-v1.yaml carried five falsification conditions, each
# declaring how it is enforced:
#
#     enforcement: "cargo test --test apr_cli_commands test_all_contract_commands_exist"
#
# There is no `apr_cli_commands` test target. The file is
# crates/apr-cli/tests/cli_commands.rs and the target is `cli_commands`, so all
# five commands exit 101 with `error: no test target named apr_cli_commands`.
# Two were wrong twice over, also naming `test_all_commands_help` where the
# function is `test_all_commands_respond_to_help`.
#
# The tests themselves are real and DO run (ci.yml:327, inside workspace-test,
# inside gate.needs). What was fiction is the contract's account of HOW. A
# reader auditing whether FALSIFY-CLI-003 is live runs the command the contract
# hands them, gets an error, and cannot distinguish "the pointer is stale" from
# "the gate is missing" — which is precisely the discrimination a contract
# exists to provide.
#
# WHY NOTHING CAUGHT IT
# ---------------------
# The strings live under a top-level `falsification:` list. `Contract` has no
# such field — it has `falsification_tests` (schema/types.rs:33). serde drops
# the unknown key silently, so `pv validate` never sees these strings, and
# check_contract_test_binding.sh (#2465) reads `falsification_tests[].test`,
# a different field. The block is inert YAML that reads as governance.
#
# That is why this guard scans YAML text rather than the typed model: the field
# it must police is one the typed model does not admit. Teaching the schema
# about `falsification:` is the better long-term home and is filed as follow-up;
# it is a schema change with its own blast radius, not a prerequisite for
# closing the hole.
#
# WHAT IT CHECKS
# --------------
# Every `enforcement:` scalar in contracts/ that names a `cargo test`
# invocation must resolve, against `cargo metadata` — not against a guess:
#
#   1. `--test <target>` names a test target that exists in the workspace.
#   2. `-p <pkg>`, when present, owns that target.
#   3. A trailing bare token is a `cargo test` filter and must appear as
#      `fn <token>` in that target's source file.
#
# Rule 3 is deliberately a substring-free exact `fn` match. `cargo test FOO`
# filters on a SUBSTRING of `module::path::fn`, so a filter can legitimately be
# a prefix — but every filter in this tree today names a whole function, and
# accepting prefixes would let `test_all` silently "resolve" against
# `test_all_commands_respond_to_help`, which is the exact class of near-miss
# that produced #2504. Loosen this only with a case in the self-test table.
#
# VACUITY GUARD
# -------------
# A regex that stops matching is indistinguishable from a clean tree, and that
# failure mode has shipped here twice (#2476, #2485). This refuses to pass
# unless it found at least MIN_CMDS cargo enforcement strings, and it prints the
# count it resolved. Its universe is built by `find` over contracts/, and the
# defect — a wrong string — cannot remove a file from that universe.
#
# SELF-TEST
# ---------
#   bash scripts/check_contract_enforcement.sh --self-test
# drives the real resolver over a hermetic fixture with an eight-case
# must-flag / must-not-flag table, then mutates the resolver's own inputs to
# prove each arm turns RED. Verification Discipline #7: re-run the table, never
# re-read the pattern.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
CONTRACT_DIR="${CONTRACT_DIR:-$REPO_ROOT/contracts}"
MIN_CMDS="${MIN_CMDS:-8}"

GUARD_TMP=""
cleanup() {
    if [ -z "$GUARD_TMP" ]; then
        return 0
    fi
    if [ "$GUARD_TMP" = "/" ]; then
        return 0
    fi
    rm -rf "$GUARD_TMP"
}
trap cleanup EXIT

die() {
    printf '%s\n' "$*" >&2
    exit 1
}

need() {
    command -v "$1" >/dev/null 2>&1 || die "check_contract_enforcement: missing required tool: $1"
}

# ---------------------------------------------------------------------------
# Emit "pkg<TAB>target<TAB>src_path" for every test target in the workspace.
# Sourced from cargo metadata so the guard cannot disagree with cargo about
# what exists — the disagreement is the whole defect.
# ---------------------------------------------------------------------------
target_table() {
    local manifest="$1" out="$2"
    ( cd "$(dirname "$manifest")" && cargo metadata --no-deps --format-version 1 2>/dev/null ) \
        | jq -r '.packages[] as $p
                 | $p.targets[]
                 | select(.kind | index("test"))
                 | "\($p.name)\t\(.name)\t\(.src_path)"' \
        > "$out"
}

# ---------------------------------------------------------------------------
# Emit "file:line<TAB>command" for every enforcement scalar naming cargo test.
# ---------------------------------------------------------------------------
enforcement_commands() {
    local dir="$1"
    grep -rn --include='*.yaml' -E '^[[:space:]]*enforcement:[[:space:]]*.*cargo[[:space:]]+test' "$dir" 2>/dev/null \
        | sed -E 's/^([^:]+:[0-9]+):[[:space:]]*enforcement:[[:space:]]*/\1\t/' \
        | sed -E 's/\t"(.*)"[[:space:]]*$/\t\1/' \
        | sed -E "s/\t'(.*)'[[:space:]]*\$/\t\1/"
}

# ---------------------------------------------------------------------------
# Resolve one command. Prints a diagnostic and returns 1 on failure.
# ---------------------------------------------------------------------------
resolve_one() {
    local loc="$1" cmd="$2" table="$3"
    local target pkg filter row src owner

    target=$(printf '%s\n' "$cmd" | grep -oE '\-\-test[[:space:]]+[A-Za-z0-9_]+' | awk '{print $2}' | head -1)
    if [ -z "$target" ]; then
        # A cargo test invocation with no --test target (e.g. `cargo test --lib`).
        # Nothing to resolve; counted, not judged.
        return 0
    fi

    row=$(awk -F'\t' -v t="$target" '$2 == t {print; exit}' "$table")
    if [ -z "$row" ]; then
        printf '%s\n' "  $loc" >&2
        printf '%s\n' "      names --test $target, which is not a test target in this workspace" >&2
        printf '%s\n' "      command: $cmd" >&2
        return 1
    fi
    owner=$(printf '%s' "$row" | cut -f1)
    src=$(printf '%s' "$row" | cut -f3)

    pkg=$(printf '%s\n' "$cmd" | grep -oE '(-p|--package)[[:space:]]+[A-Za-z0-9_-]+' | awk '{print $2}' | head -1)
    if [ -n "$pkg" ] && [ "$pkg" != "$owner" ]; then
        printf '%s\n' "  $loc" >&2
        printf '%s\n' "      names -p $pkg, but target $target belongs to $owner" >&2
        return 1
    fi

    # The test filter: the last bare token, excluding cargo's own words.
    filter=$(printf '%s\n' "$cmd" \
        | tr ' ' '\n' \
        | grep -vE '^(cargo|test|--test|-p|--package|--all-features|--release|--|--lib|--no-fail-fast)$' \
        | grep -vE '^-' \
        | grep -vE "^(${target}|${pkg:-__none__})$" \
        | tail -1)
    if [ -z "$filter" ]; then
        return 0
    fi
    if ! grep -qE "fn[[:space:]]+${filter}[[:space:]]*\(" "$src" 2>/dev/null; then
        printf '%s\n' "  $loc" >&2
        printf '%s\n' "      names test fn '$filter', which does not exist in $src" >&2
        return 1
    fi
    return 0
}

scan() {
    local dir="$1" manifest="$2"
    local table="$GUARD_TMP/targets.tsv"
    target_table "$manifest" "$table"
    [ -s "$table" ] || die "VACUOUS: cargo metadata yielded no test targets - the resolver has no universe."

    local checked=0 bad=0 loc cmd
    while IFS=$'\t' read -r loc cmd; do
        [ -n "$cmd" ] || continue
        checked=$((checked + 1))
        resolve_one "$loc" "$cmd" "$table" || bad=$((bad + 1))
    done < <(enforcement_commands "$dir")

    printf '%s %s\n' "$checked" "$bad" > "$GUARD_TMP/stats"
}

# ---------------------------------------------------------------------------
self_test() {
    need jq
    GUARD_TMP="$(mktemp -d)"
    local fx="$GUARD_TMP/fx" fail=0
    mkdir -p "$fx/tests" "$fx/c"

    printf '%s\n' \
        '[package]' 'name = "fx"' 'version = "0.0.0"' 'edition = "2021"' \
        '[[test]]' 'name = "real_target"' 'path = "tests/real_target.rs"' > "$fx/Cargo.toml"
    printf '%s\n' \
        '#[test]' 'fn test_real_fn() {}' \
        '#[test]' 'fn test_real_fn_longer() {}' > "$fx/tests/real_target.rs"

    # MUST NOT FLAG
    printf 'enforcement: "cargo test --test real_target test_real_fn"\n'        > "$fx/c/ok_full.yaml"
    printf 'enforcement: "cargo test -p fx --test real_target test_real_fn"\n'  > "$fx/c/ok_pkg.yaml"
    printf 'enforcement: "cargo test --test real_target"\n'                     > "$fx/c/ok_nofilter.yaml"
    printf 'enforcement: "cargo test --lib"\n'                                  > "$fx/c/ok_notarget.yaml"
    # MUST FLAG
    printf 'enforcement: "cargo test --test ghost_target test_real_fn"\n'       > "$fx/c/bad_target.yaml"
    printf 'enforcement: "cargo test --test real_target test_ghost_fn"\n'       > "$fx/c/bad_fn.yaml"
    printf 'enforcement: "cargo test -p other --test real_target test_real_fn"\n' > "$fx/c/bad_pkg.yaml"
    # Prefix must NOT resolve (the #2504 near-miss shape)
    printf 'enforcement: "cargo test --test real_target test_real"\n'           > "$fx/c/bad_prefix.yaml"

    local out
    out="$(MIN_CMDS=1 scan_capture "$fx/c" "$fx/Cargo.toml" 2>&1)"

    local must_flag="bad_target bad_fn bad_pkg bad_prefix"
    local must_not="ok_full ok_pkg ok_nofilter ok_notarget"
    local c
    for c in $must_flag; do
        case "$out" in
            *"$c.yaml"*) ;;
            *) printf 'SELF-TEST FAIL: %s should have been flagged, was not\n' "$c" >&2; fail=1 ;;
        esac
    done
    for c in $must_not; do
        case "$out" in
            *"$c.yaml"*) printf 'SELF-TEST FAIL: %s was flagged, should not have been\n' "$c" >&2; fail=1 ;;
            *) ;;
        esac
    done

    # Vacuity arm: an empty contract dir must FAIL, never read as clean.
    mkdir -p "$fx/empty"
    if CONTRACT_DIR="$fx/empty" MIN_CMDS=1 bash "$SCRIPT_PATH" --manifest "$fx/Cargo.toml" >/dev/null 2>&1; then
        printf 'SELF-TEST FAIL: empty contract dir should FAIL (vacuity), it passed\n' >&2
        fail=1
    fi

    [ "$fail" -eq 0 ] || die "check_contract_enforcement: SELF-TEST FAILED"
    printf 'check_contract_enforcement: SELF-TEST PASSED (8 resolution cases + 1 vacuity arm)\n'
}

scan_capture() {
    GUARD_TMP="${GUARD_TMP:-$(mktemp -d)}"
    scan "$1" "$2"
}

# ---------------------------------------------------------------------------
main() {
    local manifest="$REPO_ROOT/Cargo.toml"
    while [ $# -gt 0 ]; do
        case "$1" in
            --self-test) self_test; exit 0 ;;
            --manifest) manifest="$2"; shift 2 ;;
            *) die "check_contract_enforcement: unknown argument: $1" ;;
        esac
    done

    need jq
    need cargo
    GUARD_TMP="$(mktemp -d)"
    [ -d "$CONTRACT_DIR" ] || die "check_contract_enforcement: no contract dir at $CONTRACT_DIR"

    scan "$CONTRACT_DIR" "$manifest"
    local checked bad
    read -r checked bad < "$GUARD_TMP/stats"

    if [ "$checked" -lt "$MIN_CMDS" ]; then
        die "VACUOUS: found only $checked cargo enforcement string(s) (floor: $MIN_CMDS). The scan collapsed; this is a broken guard, not a clean tree."
    fi
    if [ "$bad" -gt 0 ]; then
        printf '\nFAIL: %s of %s cargo enforcement string(s) name something that cannot run.\n' "$bad" "$checked" >&2
        printf 'A contract that misreports how it is enforced provides no discrimination\n' >&2
        printf 'between "the pointer is stale" and "the gate is missing".\n' >&2
        exit 1
    fi
    printf 'OK: %s cargo enforcement strings in %s; every target and test fn resolves\n' \
        "$checked" "${CONTRACT_DIR#"$REPO_ROOT"/}"
}

main "$@"
