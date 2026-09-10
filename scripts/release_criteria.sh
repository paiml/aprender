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
        C13) echo 'bash scripts/check_release_assets.sh "v$(sed -n '"'"'s/^version = "\(.*\)"/\1/p'"'"' Cargo.toml | head -1)"   # the four apr tarballs (cuda,cpu x x86_64,aarch64) + .sha256 + the eight pv assets; the tag is READ from the root manifest, never typed (no cargo: the release box has none). Seam: RELEASE_ASSETS_FIXTURE' ;;
        C14) echo 'bash scripts/check_model_parity.sh --manifest   # GPU=CPU per manifest model over >= 64 positions, or the GPU refuses it (L0-1a)' ;;
        C1|C2|C3|C5|C10|C12) echo '0.67 (SPEC-2.0: moved with its track; never credited in 0.66)' ;;
        *) return 1 ;;
    esac
}
# DERIVED, not chosen: a criterion is credited in 0.66 iff at least one row the spec's §4
# table names as its owner is still lane 0.66 in docs/specifications/pp-066-dag.yaml.
# Re-derived 2026-09-08 after D-14 rescoped the release to the CUDA parity fix:
#
#   C7  SPEC-2.0 (0.66)              KEEP   the claims ratchet — load-bearing for D-13/D-14's
#                                           vocabulary obligations
#   C8  SPEC-2.0 (0.66)              KEEP   clean-room before publish; `cargo publish` rests on it
#   C9  C0-7     (0.66)              KEEP   every credited row has a complete receipt
#   C14 L0-1a    (0.66)              KEEP   GPU = CPU per manifest model, or the GPU refuses it —
#                                           this IS 0.66's single claim
#   C0  C0-1,C0-2,C0-4  all 0.67     MOVE   cannot be satisfied in 0.66; keeping it made --all
#                                           unpassable AND unrunnable (see the memo below)
#   C4  R-6,R-2         all 0.67     MOVE   four host receipts THROUGH THE R-6 INSTALLER, which
#                                           0.66 does not ship
#   C6  G-10a,G-10b,G-11a            MOVE   two owners do not exist as rows; its script does not
#                                           exist either (it returned ENV 2, never a pass)
#   C11 R-0a,R-0b       0.67/absent  MOVE   the backend registry is 0.67
#   C13 KEY,R-5,R-6     all 0.67     MOVE   no assets and no installer in 0.66 (D-13, D-14)
#
# `--self-test` re-derives this from the DAG and refuses a hand-edited disagreement.
CREDITED="C7 C8 C9 C14"

run_one() { # run_one <id> -> 0 credited · 1 not · 2 ENV
    local id=$1 line script
    line=$(cmd_of "$id") || { printf '%s: unknown criterion %s\n' "$PROG" "$id" >&2; return 2; }
    case "$line" in 0.67*) printf '%s: %s is a 0.67 criterion — not credited in 0.66\n' "$PROG" "$id"; return 1 ;; esac
    script=$(printf '%s' "$line" | sed -E 's/^bash ([^ ]+).*/\1/')
    [ -f "$script" ] || { printf '%s: %s ENV — %s does not exist yet (the row that builds it is open); exit 2, never a pass\n' "$PROG" "$id" "$script"; return 2; }
    # I9's credited-first rule applies only while C0 is itself a credited criterion, and it is
    # evaluated ONCE per process, not once per criterion. Before this, `--all` re-ran C0 for
    # every one of the nine — and C0 shells out to `pmat comply check` and a `gh api` call, so
    # the first step of the release sequence could not return a verdict inside ten minutes.
    # Measured 2026-09-08: C0, C4, C7 and C8 each hit a 120 s timeout; C7 alone runs in ~30 s.
    case " $CREDITED " in
        *" C0 "*)
            if [ "$id" != C0 ]; then
                if [ -z "${C0_VERDICT:-}" ]; then
                    if bash "$0" C0 >/dev/null 2>&1; then C0_VERDICT=ok; else C0_VERDICT=no; fi
                    export C0_VERDICT
                fi
                [ "$C0_VERDICT" = ok ] || { printf '%s: %s [U] — C0 is not credited yet (I9: C0 gates credit)\n' "$PROG" "$id"; return 1; }
            fi ;;
    esac
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
    # The success line prints $CREDITED itself. It used to carry a second, hand-typed copy of
    # the list, so editing the set would have left the banner asserting the old one.
    --all) rc=0; for c in $CREDITED; do run_one "$c" || rc=1; done; [ "$rc" = 0 ] && printf 'ALL CREDITED (%s)\n' "$CREDITED"; exit "$rc" ;;
    --self-test)
        n=0; red=0
        t() { local want=$1 label=$2; shift 2; local rc=0; n=$((n + 1)); "$@" >/dev/null 2>&1 || rc=$?; if [ "$rc" = "$want" ]; then printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"; else printf 'FAIL  row %-2s rc=%s (wanted %s)  %s\n' "$n" "$rc" "$want" "$label"; red=1; fi; }
        t 0 "--list prints the fifteen criteria"                        bash -c "[ \$(bash '$0' --list | grep -c '^C') -eq 15 ]"
        t 0 "--list still names every criterion with a command, credited or not" bash -c "[ \$(bash '$0' --list | grep -c '^C[0-9]* *bash ') -eq 9 ]"
        t 1 "a 0.67 criterion is never credited"                        bash "$0" C1
        t 2 "an unknown criterion is exit 2"                            bash "$0" C99
        t 2 "a criterion whose script does not exist yet is ENV (2), not a pass (C6, check_guards_observed_red.sh)" bash "$0" C6
        # C13 now HAS its script (PMAT-1098, row 67-A1), so "ENV because the file
        # is missing" is no longer the honest row for it. Both polarities are
        # proved instead, through the checker's own offline seam, so this table
        # needs no network and no published release to show that C13 can fail.
        C13_WORK=$(mktemp -d)
        C13_TAG="v$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)"
        bash scripts/check_release_assets.sh --list "$C13_TAG" > "$C13_WORK/complete.txt"
        grep -vx "apr-$C13_TAG-aarch64-unknown-linux-gnu-cpu.tar.gz" "$C13_WORK/complete.txt" > "$C13_WORK/mutant.txt"
        t 0 "C13 is credited when the release carries all sixteen assets" \
            env RELEASE_ASSETS_FIXTURE="$C13_WORK/complete.txt" bash "$0" C13
        t 1 "C13 RED: one apr tarball removed and C13 is NOT credited (it cannot pass vacuously)" \
            env RELEASE_ASSETS_FIXTURE="$C13_WORK/mutant.txt" bash "$0" C13
        t 2 "C13 ENV: an unreadable release is exit 2, never a credit" \
            env RELEASE_ASSETS_FIXTURE="$C13_WORK/absent.txt" bash "$0" C13
        rm -rf "$C13_WORK"
        # D-14 removed C0 from the credited set, so the credited-first rule no longer fires and
        # C7 stands on its own. The row that used to assert "[U] before C0" is replaced by the
        # two that matter now: the set is DERIVED, and the banner is not a second copy of it.
        t 0 "the credited set is exactly what the DAG derives (no hand-edited drift)" bash -c '
            want=$(python3 - <<PYX
import re, yaml
spec = open("docs/specifications/PP-066-release-spec.md", encoding="utf-8").read().split("\n")
lane = {r["id"]: str(r.get("lane")) for r in yaml.safe_load(open("docs/specifications/pp-066-dag.yaml", encoding="utf-8"))["rows"]}
keep = []
for l in spec:
    m = re.match(r"^\|\s*(C\d+)\s*\|(.+)\|([^|]*)\|\s*$", l)
    if not m or ("scripts/" not in m.group(2) and "--c0" not in m.group(2)):
        continue
    owners = [o.strip().split(" ")[0] for o in m.group(3).split(",") if o.strip() and o.strip() != "\u2014"]
    if any(lane.get(o) == "0.66" for o in owners):
        keep.append(m.group(1))
print(" ".join(sorted(set(keep), key=lambda c: int(c[1:]))))
PYX
)
            have=$(grep -E "^CREDITED=" '"$0"' | sed -E "s/^CREDITED=\"(.*)\"/\1/")
            [ "$want" = "$have" ] || { echo "derived [$want] != CREDITED [$have]"; exit 1; }'
        t 0 "the ALL-CREDITED banner prints the set, never a second copy of it" bash -c '
            ! grep -qE "ALL CREDITED \(C[0-9]" '"$0"''
        printf '%s/%s rows\n' "$((n - red))" "$n"; [ "$red" = 0 ] || exit 1; exit 0 ;;
    C*) run_one "$1"; exit $? ;;
    *) printf 'usage: %s --list | --all | --self-test | C<n>\n' "$PROG" >&2; exit 2 ;;
esac
