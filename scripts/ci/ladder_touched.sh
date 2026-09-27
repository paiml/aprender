#!/usr/bin/env bash
# ladder_touched.sh — does this change touch the provable-ladder's inputs? (PVL-001 EV-9,
# FLOW-003 Thm 15: a job runs on a PR only when the PR changes the inputs it reads)
#
# The ladder (ci/sections.yml `provable-ladder`) reads the Lean tree, the committed
# discharge summary, the contracts, and the pv binary it builds from the contracts crates.
# A PR that changes none of those cannot change its verdict, so it SKIPs there; `main`
# and the nightly (ladder-nightly.yml) run it in full and keep the caches warm.
#
# OUTPUT (KEY=VALUE on stdout, for `>> "$GITHUB_OUTPUT"`)
#   ladder_touched=1 | 0
#   reason=<one line>
#
# EXIT
#   0  a decision was reached (either polarity)
#   2  undecidable: no diff file, or an EMPTY diff. Never answered as 0 — an empty diff on a
#      shallow clone says nothing about the PR. The caller treats rc 2 as "run the ladder".
#
#   bash scripts/ci/ladder_touched.sh --diff-from FILE   # one changed path per line
#   bash scripts/ci/ladder_touched.sh --self-test        # the case table
set -euo pipefail

# The ladder's input set. ONE declaration; the self-test's mutants each delete one entry
# and require a row to go RED.
LADDER_INPUTS=(
    'crates/aprender-contracts-staging/*'
    'crates/aprender-contracts/*'
    'crates/aprender-contracts-cli/*'
    'crates/aprender-contracts-macros/*'
    'contracts/*'
    'Cargo.lock'
    'Cargo.toml'
    'rust-toolchain.toml'
    'ci/sections.yml'
    'scripts/ci/ladder_touched.sh'
    '.github/workflows/ladder-nightly.yml'
)

# is_input <path> — 0 when the path is in the ladder's input set.
is_input() {
    local p="$1" pat
    for pat in "${LADDER_INPUTS[@]}"; do
        # shellcheck disable=SC2053  # the RHS is a glob on purpose
        [[ $p == $pat ]] && return 0
    done
    return 1
}

decide() {
    local file="$1" n=0 p
    [ -r "$file" ] || { echo "reason=no diff file: $file"; return 2; }
    while IFS= read -r p || [ -n "$p" ]; do
        [ -n "$p" ] || continue
        n=$((n + 1))
        if is_input "$p"; then
            echo "ladder_touched=1"
            echo "reason=input changed: $p"
            return 0
        fi
    done < "$file"
    if [ "$n" -eq 0 ]; then
        echo "reason=empty diff: undecidable"
        return 2
    fi
    echo "ladder_touched=0"
    echo "reason=none of $n changed path(s) is a ladder input"
}

self_test() {
    local td rc fail=0 row want paths got
    td="$(mktemp -d)"
    trap 'rm -rf "${td:?}"' RETURN
    # want<TAB>paths (paths separated by ';'). want: 1, 0, or 2 (undecidable).
    local rows=(
        $'1\tcrates/aprender-contracts-staging/lean/Lean/Foo.lean'
        $'1\tcrates/aprender-contracts-staging/discharge-summary.json'
        $'1\tcrates/aprender-contracts-staging/lean/lake-manifest.json'
        $'1\tcrates/aprender-contracts-staging/lean/lean-toolchain'
        $'1\tcontracts/tensor-layout-v1.yaml'
        $'1\tcrates/aprender-contracts/src/lib.rs'
        $'1\tcrates/aprender-contracts-cli/src/main.rs'
        $'1\tcrates/aprender-contracts-macros/src/lib.rs'
        $'1\tCargo.lock'
        $'1\tCargo.toml'
        $'1\trust-toolchain.toml'
        $'1\tci/sections.yml'
        $'1\tscripts/ci/ladder_touched.sh'
        $'1\t.github/workflows/ladder-nightly.yml'
        $'1\tdocs/a.md;contracts/x.yaml'
        $'0\tdocs/specifications/a.md'
        $'0\tcrates/aprender-core/src/lib.rs'
        $'0\tcrates/aprender-core/Cargo.toml'
        $'0\tcrates/aprender-contractsX/src/lib.rs'
        $'0\tscripts/contracts_gate.sh'
        $'0\tdocs/contracts/x.md'
        $'0\t.github/workflows/ci.yml'
        $'2\t'
    )
    local i=0
    for row in "${rows[@]}"; do
        i=$((i + 1))
        want="${row%%$'\t'*}"
        paths="${row#*$'\t'}"
        if [ -n "$paths" ]; then tr ';' '\n' <<< "$paths" > "$td/d"; else : > "$td/d"; fi
        rc=0; out="$(decide "$td/d")" || rc=$?
        if [ "$rc" -eq 2 ]; then got=2; else got="$(sed -n 's/^ladder_touched=//p' <<< "$out")"; fi
        if [ "$got" != "$want" ]; then
            echo "FAIL row $i: want $want got $got ($paths)"; fail=1
        fi
    done
    [ "$fail" -eq 0 ] || return 1
    echo "self-test: ${#rows[@]}/${#rows[@]} rows"
    # Mutants: drop each input entry in turn; some row must go RED, or that entry is
    # untested. A no-diff file must also stay rc 2.
    local keep=("${LADDER_INPUTS[@]}") j red killed=0
    for j in "${!keep[@]}"; do
        LADDER_INPUTS=("${keep[@]:0:j}" "${keep[@]:j+1}")
        red=0
        for row in "${rows[@]}"; do
            want="${row%%$'\t'*}"; paths="${row#*$'\t'}"
            [ "$want" = 1 ] || continue
            tr ';' '\n' <<< "$paths" > "$td/d"
            out="$(decide "$td/d")" || true
            grep -qx 'ladder_touched=1' <<< "$out" || { red=1; break; }
        done
        if [ "$red" -eq 1 ]; then killed=$((killed + 1)); else echo "SURVIVED mutant: drop ${keep[j]}"; fi
    done
    LADDER_INPUTS=("${keep[@]}")
    rc=0; decide "$td/nonexistent" >/dev/null || rc=$?
    [ "$rc" -eq 2 ] || { echo "FAIL: a missing diff file answered rc $rc, not 2"; return 1; }
    echo "mutants: $killed/${#keep[@]} RED"
    [ "$killed" -eq "${#keep[@]}" ]
}

case "${1:-}" in
    --self-test) self_test ;;
    --diff-from) [ -n "${2:-}" ] || { echo "usage: $0 --diff-from FILE" >&2; exit 64; }; decide "$2" ;;
    *) echo "usage: $0 --diff-from FILE | --self-test" >&2; exit 64 ;;
esac
