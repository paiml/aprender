#!/usr/bin/env bash
# check_parity_receipt_denominator.sh — #4425: run the #3577 denominator in VERIFY mode.
#
# `scripts/parity_receipt_denominator.sh` has three modes, and CI ran only `--self-test`, which reads
# synthetic fixtures. Verify mode — recount the receipts under evidence/parity/** and compare with
# evidence/parity/EXPECTED_RECEIPTS — ran nowhere, so a receipt added or dropped without bumping the
# denominator passed CI: the defect #3577 exists to catch. This file is a `scripts/check_*.sh`, so
# guard_tree.sh's derived universe (`git ls-files 'scripts/check_*.sh'`) runs it on every PR with no
# workflow edit: `--self-test` first (it advertises one), then verify on the real tree.
#
#   bash scripts/check_parity_receipt_denominator.sh              # verify the tree (exit of the denominator)
#   bash scripts/check_parity_receipt_denominator.sh --self-test  # prove this wrapper can go RED
#
# The self-test copies the denominator into a scratch git repo holding ONE v2 receipt and proves the
# wrapper exits 0 when EXPECTED_RECEIPTS says 1, and 1 (never 0, never 2) when it says 2 or 0 — the
# "denominator not bumped" and "receipt dropped" cases. A wrapper that swallowed the exit, or ran the
# denominator's own --self-test in place of verify, turns a row RED.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DENOM="$HERE/parity_receipt_denominator.sh"

verify() {
    [ -f "$DENOM" ] || { printf 'FAIL  %s is missing\n' "$DENOM" >&2; return 2; }
    bash "$DENOM"
}

self_test() {
    local td rc bad=0 want expected
    td="$(mktemp -d)"
    trap 'rm -rf "${td:?}"' RETURN
    mkdir -p "$td/scripts" "$td/evidence/parity"
    cp "$DENOM" "$HERE/check_parity_receipt_denominator.sh" "$td/scripts/"
    printf '{"schema": "apr-parity-receipt/v2"}\n' > "$td/evidence/parity/r1.json"
    git -C "$td" init -q
    for row in "1 0" "2 1" "0 1"; do
        expected=${row% *}; want=${row#* }
        printf '%s\n' "$expected" > "$td/evidence/parity/EXPECTED_RECEIPTS"
        git -C "$td" add -A
        rc=0; bash "$td/scripts/check_parity_receipt_denominator.sh" > /dev/null 2>&1 || rc=$?
        if [ "$rc" = "$want" ]; then
            printf 'PASS  EXPECTED_RECEIPTS=%s over 1 receipt -> exit %s\n' "$expected" "$rc"
        else
            printf 'FAIL  EXPECTED_RECEIPTS=%s over 1 receipt -> exit %s, want %s\n' "$expected" "$rc" "$want"
            bad=1
        fi
    done
    return "$bad"
}

case "${1:-}" in
    --self-test) self_test ;;
    "")          verify ;;
    *)           printf 'usage: %s [--self-test]\n' "$(basename "$0")" >&2; exit 2 ;;
esac
