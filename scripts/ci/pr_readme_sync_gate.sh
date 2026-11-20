#!/usr/bin/env bash
# pr_readme_sync_gate.sh — a PR that adds or removes a contract file goes red on ITS OWN PR run when the
# README's generated CONTRACT_COUNT blocks were not regenerated (#7343, master #7326).
#
# THE DEFECT: scripts/readme_sync.sh --check (FALSIFY-README-002's generator) only ran in the merge queue and on
# main, so a contract-adding PR passed pr-light and left main red (#7327, #7328).
#
# Usage:  bash scripts/ci/pr_readme_sync_gate.sh              # judge the checked-out tree (pr-light step)
#         bash scripts/ci/pr_readme_sync_gate.sh --self-test  # RED/GREEN case table on a scratch tree
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

d=""
self_test() {
    local rc=0
    d="$(mktemp -d "${TMPDIR:-/tmp}/pr-readme-sync.XXXXXX")"
    trap '[ -z "$d" ] || rm -rf -- "${d:?}"' EXIT
    mkdir -p "$d/scripts" "$d/contracts"
    cp "$REPO_ROOT/scripts/readme_sync.sh" "$d/scripts/readme_sync.sh"
    printf 'a: 1\n' > "$d/contracts/a.yaml"
    printf '{"n_files": 1}\n' > "$d/contracts/census.json"
    printf 'count <!-- CONTRACT_COUNT_START -->1<!-- CONTRACT_COUNT_END -->\n' > "$d/README.md"

    bash "$d/scripts/readme_sync.sh" --check >/dev/null 2>&1 \
        || { echo "FAIL self-test GREEN: a synced tree was refused"; rc=1; }

    printf 'b: 1\n' > "$d/contracts/b.yaml"   # a contract is added, README not regenerated
    if bash "$d/scripts/readme_sync.sh" --check >/dev/null 2>&1; then
        echo "FAIL self-test RED: an added contract with a stale README was accepted"; rc=1
    fi

    bash "$d/scripts/readme_sync.sh" --write >/dev/null 2>&1
    bash "$d/scripts/readme_sync.sh" --check >/dev/null 2>&1 \
        || { echo "FAIL self-test: --write did not restore a green check"; rc=1; }

    rm -f "${d:?}/contracts/b.yaml"           # a contract is removed, README not regenerated
    if bash "$d/scripts/readme_sync.sh" --check >/dev/null 2>&1; then
        echo "FAIL self-test RED: a removed contract with a stale README was accepted"; rc=1
    fi
    [ "$rc" -eq 0 ] && echo "ok    pr_readme_sync_gate self-test: 4 cases"
    return "$rc"
}

case "${1:-}" in
    --self-test) self_test ;;
    "") cd "$REPO_ROOT" && bash scripts/readme_sync.sh --check ;;
    *) echo "usage: $0 [--self-test]" >&2; exit 2 ;;
esac
