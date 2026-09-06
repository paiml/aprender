#!/usr/bin/env bash
# release_criteria.sh — PP-066 §4 as EXECUTABLE criteria (SPEC-2.0, driver v5): one exit-coded
# command per credited criterion. C0 is credited FIRST; until `C0` exits 0 every other criterion
# reports [U] and this script exits 1 for it (I9: C0 gates credit, not work).
#
#   bash scripts/release_criteria.sh --list          # the ten criteria and their commands
#   bash scripts/release_criteria.sh C<n>            # run one; exit 0 credited, 1 not, 2 ENV (the box cannot answer)
#   bash scripts/release_criteria.sh --all           # every criterion in order; exit 0 iff all credited
#   bash scripts/release_criteria.sh --self-test     # case table: a criterion never passes vacuously
# C1 C2 C3 C5 C10 C12 moved to 0.67 with their tracks (SPEC-2.0; C5 by the rescope quorum); listed as 0.67, never credited here.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROG=release_criteria
cd "$ROOT"

# criterion <id> -> the command that decides it (a function so --list can print it verbatim)
cmd_of() {
    case "$1" in
        C0)  echo 'bash scripts/release_criteria.sh --c0   # (spec §4 C0 verbatim, through the pin) CB-1700/1701/2100 show no ✗ in "$PMAT" comply check; branch protection strict=true; perf_gate.sh --selftest GREEN (C0-1, C0-2, C0-4)' ;;
        C4)  echo 'bash scripts/check_multiplatform_dogfood.sh --require-resolved-backend cuda   # four host receipts through the R-6 installer, apr devices --json, effective-config backend' ;;
        C6)  echo 'bash scripts/check_guards_observed_red.sh   # each 0.66 guard PR carries mutation RED -> revert in its history (run ids in the body)' ;;
        C7)  echo 'bash scripts/check_no_claim_literals.sh && bash scripts/check_perf_claims_cite_receipts.sh   # the claims ratchet over README, notes, docs/specifications' ;;
        C8)  echo 'bash scripts/run_clean_room.sh   # clean-room p1 via ../infra (hard gate)' ;;
        C9)  echo 'bash scripts/check_receipt_complete.sh --dag docs/specifications/pp-066-dag.yaml   # every 0.66 row credited has a receipt whose marker says complete' ;;
        C11) echo 'bash scripts/check_backend_registry.sh --static   # 15 fixtures (FX-1..15) each observed RED once; zero cfg!(feature) reads in apr-cli backend decisions' ;;
        C13) echo 'bash scripts/check_release_assets.sh v0.66.0   # 5 apr-* tarballs + .sha256 + minisign signature; install.sh ends by printing apr devices' ;;
        C14) echo 'bash scripts/check_model_parity.sh --manifest   # GPU=CPU per manifest model over >= 64 positions, or the GPU refuses it (L0-1a)' ;;
        C1|C2|C3|C5|C10|C12) echo '0.67 (SPEC-2.0: moved with its track; never credited in 0.66)' ;;
        *) return 1 ;;
    esac
}
CREDITED="C0 C4 C6 C7 C8 C9 C11 C13 C14"   # C5 moved to 0.67 by the rescope quorum (Q3 unanimous)

run_one() { # run_one <id> -> 0 credited · 1 not · 2 ENV
    local id=$1 line script
    line=$(cmd_of "$id") || { printf '%s: unknown criterion %s\n' "$PROG" "$id" >&2; return 2; }
    case "$line" in 0.67*) printf '%s: %s is a 0.67 criterion — not credited in 0.66\n' "$PROG" "$id"; return 1 ;; esac
    script=$(printf '%s' "$line" | sed -E 's/^bash ([^ ]+).*/\1/')
    [ -f "$script" ] || { printf '%s: %s ENV — %s does not exist yet (the row that builds it is open); exit 2, never a pass\n' "$PROG" "$id" "$script"; return 2; }
    if [ "$id" != C0 ] && ! bash "$0" C0 >/dev/null 2>&1; then printf '%s: %s [U] — C0 is not credited yet (I9: C0 gates credit)\n' "$PROG" "$id"; return 1; fi
    printf '=== %s: %s\n' "$id" "${line%%#*}"
    local rc=0; bash -c "${line%%#*}" || rc=$?
    if [ "$rc" = 0 ]; then printf 'CREDITED %s\n' "$id"; else printf 'NOT CREDITED %s (rc=%s)\n' "$id" "$rc"; fi
    return "$rc"
}

c0() { # the spec's §4 C0 command, verbatim, through the analyser pin (I6); every leg printed, exit 1 on the first that fails
    . "$ROOT/scripts/pmat_bin.sh" || { printf 'C0: ENV - no analyser at the pin\n'; return 2; }
    local rc=0 out
    out=$("$PMAT" comply check 2>/dev/null | grep -E 'CB-(1700|1701|2100)' || true); printf '%s\n' "$out"
    if [ -z "$out" ] || printf '%s' "$out" | grep -q '✗'; then printf 'C0 leg 1 FAIL: CB-1700/1701/2100 not all ✓ in comply check\n'; rc=1; fi
    if [ "$(gh api repos/paiml/aprender/branches/main/protection --jq .required_status_checks.strict 2>/dev/null)" != true ]; then printf 'C0 leg 2 FAIL: required_status_checks.strict is not true\n'; rc=1; else printf 'C0 leg 2 ok: strict=true\n'; fi
    if bash scripts/perf_gate.sh --selftest >/dev/null 2>&1; then printf 'C0 leg 3 ok: perf_gate.sh --selftest\n'; else printf 'C0 leg 3 FAIL: perf_gate.sh --selftest (#2830 polarity, C0-4)\n'; rc=1; fi
    return "$rc"
}
case "${1:-}" in
    --c0) c0; exit $? ;;
    --list) for c in C0 C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14; do printf '%-4s %s\n' "$c" "$(cmd_of "$c")"; done; exit 0 ;;
    --all) rc=0; for c in $CREDITED; do run_one "$c" || rc=1; done; [ "$rc" = 0 ] && printf 'ALL CREDITED (C0 C4 C6 C7 C8 C9 C11 C13 C14)\n'; exit "$rc" ;;
    --self-test)
        n=0; red=0
        t() { local want=$1 label=$2; shift 2; local rc=0; n=$((n + 1)); "$@" >/dev/null 2>&1 || rc=$?; if [ "$rc" = "$want" ]; then printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"; else printf 'FAIL  row %-2s rc=%s (wanted %s)  %s\n' "$n" "$rc" "$want" "$label"; red=1; fi; }
        t 0 "--list prints the fifteen criteria"                        bash -c "[ \$(bash '$0' --list | grep -c '^C') -eq 15 ]"
        t 0 "--list names exactly nine credited commands and six 0.67"   bash -c "[ \$(bash '$0' --list | grep -c '^C[0-9]* *bash ') -eq 9 ] && [ \$(bash '$0' --list | grep -c '0.67 (SPEC-2.0') -eq 6 ]"
        t 1 "a 0.67 criterion is never credited"                        bash "$0" C1
        t 2 "an unknown criterion is exit 2"                            bash "$0" C99
        t 2 "a criterion whose script does not exist yet is ENV (2), not a pass (C13 before R-5)" bash "$0" C13
        t 1 "before C0 is credited every other criterion is [U] (1)"     bash "$0" C7
        printf '%s/%s rows\n' "$((n - red))" "$n"; [ "$red" = 0 ] || exit 1; exit 0 ;;
    C*) run_one "$1"; exit $? ;;
    *) printf 'usage: %s --list | --all | --self-test | C<n>\n' "$PROG" >&2; exit 2 ;;
esac
