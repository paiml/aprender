#!/usr/bin/env bash
# check_spec_runtime_integration_proposed.sh - asset-contracts runtime integration is PROPOSED, NOT IMPLEMENTED (aprender#6640).
#
# THE RULE (asset-contracts.md, "Runtime Integration ... (PROPOSED)" + its status table): `execute_with_contract()` and
# `load_model_verified()` are "NOT IMPLEMENTED / Proposed design only". Probe: a non-test Rust `fn <name>` anywhere in the tree
# (excluding target/, tests/, benches/ and the staged docs).
#   table row says NOT IMPLEMENTED + fn exists   -> RED (stale: the table denies code that now exists)
#   table row lacks NOT IMPLEMENTED + fn absent  -> RED (false claim of implementation)
#   table row missing / spec missing             -> exit 2
#
# TIER. LAB (C324): nightly from scripts/lab/, never in a PR, merge-queue or release job.
# Exit 0 consistent, 1 disagree, 2 cannot measure. `--self-test` plants RED and GREEN controls.
# Env: SPEC, SRC (root to search, default .).

set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

spec_default="docs/specifications/aprender-contracts-staging/sub/asset-contracts.md"
symbols=(execute_with_contract load_model_verified)

has_fn() { # root symbol
    local out
    out=$(grep -rIl -E "fn[[:space:]]+$2[[:space:]]*[<(]" --include='*.rs' \
        --exclude-dir=target --exclude-dir=tests --exclude-dir=benches --exclude-dir=.git --exclude-dir=aprender-contracts-staging \
        "$1" 2>/dev/null) || true
    [[ -n "$out" ]]
}

check() {
    local spec=$1 root=$2 bad=0 row exists sym
    [[ -f "$spec" ]] || { echo "cannot measure: $spec missing"; return 2; }
    for sym in "${symbols[@]}"; do
        row=$(grep -E "^\| *\`$sym(\(\))?\`" "$spec" | head -1 || true)
        [[ -n "$row" ]] || { echo "cannot measure: no status-table row for $sym"; return 2; }
        if has_fn "$root" "$sym"; then exists=1; else exists=0; fi
        if grep -q 'NOT IMPLEMENTED' <<<"$row"; then
            [[ $exists -eq 0 ]] || { echo "RED stale: $sym exists in source but the spec says NOT IMPLEMENTED"; bad=$((bad + 1)); }
        else
            [[ $exists -eq 1 ]] || { echo "RED false claim: $sym is not marked NOT IMPLEMENTED but no fn exists"; bad=$((bad + 1)); }
        fi
    done
    echo "${#symbols[@]} symbol(s), $bad disagree"
    [[ $bad -eq 0 ]]
}

self_test() {
    local rc=0
    t=$(mktemp -d); trap 'rm -rf "${t:?}"' EXIT
    mkdir -p "$t/none/src" "$t/have/src" "$t/testonly/tests"
    echo 'pub fn other() {}' > "$t/none/src/lib.rs"
    printf 'pub fn execute_with_contract(&self) {}\npub fn load_model_verified<M>() {}\n' > "$t/have/src/lib.rs"
    echo 'fn execute_with_contract() {}' > "$t/testonly/tests/t.rs"
    printf '| `execute_with_contract()` | **NOT IMPLEMENTED** | x |\n| `load_model_verified()` | **NOT IMPLEMENTED** | x |\n' > "$t/denies.md"
    printf '| `execute_with_contract()` | DONE | x |\n| `load_model_verified()` | DONE | x |\n' > "$t/claims.md"
    printf '| `other` | x |\n' > "$t/norow.md"
    want() { local s=$1 r=$2 code=$3 got=0; check "$t/$s.md" "$t/$r" >/dev/null || got=$?
        if [[ $got -eq $code ]]; then echo "PASS $s/$r -> $code"; else echo "FAIL $s/$r want $code got $got"; rc=1; fi; }
    want denies none 0; want claims have 0; want denies have 1; want claims none 1
    want denies testonly 0; want norow none 2
    return $rc
}

if [[ "${1:-}" == "--self-test" ]]; then self_test; exit $?; fi
check "${SPEC:-$spec_default}" "${SRC:-.}"
