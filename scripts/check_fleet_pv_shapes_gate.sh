#!/usr/bin/env bash
# check_fleet_pv_shapes_gate.sh -- the SHACL shapes gate, run with the FLEET-PINNED pv on
# the runner this guard is executing on, not the HEAD-built one (PMAT-3567, #3559 row zero).
#
# THE DEFECT. Every "shapes gate green" receipt before 2026-09-20 was produced with
# scripts/pv_bin.sh's HEAD-built pv. The fleet-pinned pv was 0.65.2, which has no
# `extract` subcommand and no `--gate` flag: on any fleet host the gate did not report
# UNMEASURED, it ERRORED, and depending on the caller it reported nothing at all. #3559's
# target is "of a frozen capability ledger, >=80% passes on the installed, fleet-pinned
# pv", and no workflow ran that binary inside a runner. This guard does, and says which
# runner and which binary produced the number.
#
# THREE OUTCOMES, EACH NAMED -- never a pass by silence:
#   PASS        pin declared, fleet pv present, capability probe passes, lint verdict Pass
#   FAIL        pin declared (this box says the tool should be here) but it is absent, cannot
#               do `--gate`, or lint's verdict is not Pass
#   UNMEASURED  no pin: this runner was never converged (infra#708). Exit 0 with a row a
#               consumer reads as not-measured. guard_tree.sh has no Unknown exit contract
#               (measured: no guard emits one), so the row IS the verdict. First-green rule:
#               it may not red a runner class that has never had the tool.
#
# CAPABILITY, NOT VERSION. The probe asks `pv lint --help` for `--gate` and runs
# `pv extract --help`; a version compare would have passed 0.65.2 on any `>= 0.6x` rule.
# VERDICT PARSED, NOT GREPPED. The JSON's `verdict` field, read with python3 -- and pv's own
# planted-violation control (`pc_shape` = fired, `plant_violations` > 0) is required too, so
# a shape checker that fires on nothing cannot report Pass.
#
#   check_fleet_pv_shapes_gate.sh              judge this runner
#   check_fleet_pv_shapes_gate.sh --self-test  case table with stub pvs and a planted violation
#
# Overrides, for the self-test only: FLEET_PV_BIN, FLEET_PV_PIN, FLEET_PV_CONTRACTS.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
# WHERE THE FLEET PUTS pv, from infra's forjar declarations (machines/*/forjar.yaml):
# intel mirrors it into /opt/fleet-bin/bin (root-owned, first on the runner PATH);
# gx10 and lambda install it with `provider: cargo` into ~/.cargo/bin. The guard
# resolves in that order and then PROVES the binary matches the pin -- the 26-day-old
# apr lesson (scripts/apr_bin.sh) applied to pv, which had no such guard (#3567 ask 2).
# Not a bare `pv` on PATH, ever: check_apr_bin_pinned.sh CLASS 3 refuses that form.
FLEET_PV_CANDIDATES="${FLEET_PV_BIN:-/opt/fleet-bin/bin/pv:$HOME/.cargo/bin/pv}"
PV_PIN="${FLEET_PV_PIN:-$HOME/.config/fleet/pv.pin}"
CONTRACTS="${FLEET_PV_CONTRACTS:-$ROOT/contracts}"
RUNNER="${RUNNER_NAME:-unknown}"   # an Actions export; empty outside Actions, and empty is not absent

rmtree() { case "${1:-}" in ''|/) return 0 ;; *) [ -d "$1" ] && rm -rf -- "$1" ;; esac; return 0; }

# capable <pv> -> 0 when the binary can do what the gate needs; prints why not otherwise
capable() {
    local pv=$1 help
    # capture, then match a here-string: `producer | grep -q` under pipefail reads a MATCH
    # as FAIL when grep closes the pipe first (PMAT-3629, the class this tree ratchets).
    help=$("$pv" lint --help 2>/dev/null) || help=""
    grep -q -- '--gate' <<<"$help" || { echo "no --gate on lint"; return 1; }
    "$pv" extract --help >/dev/null 2>&1 || { echo "no extract subcommand"; return 1; }
    return 0
}

# resolve_fleet_pv -> prints the first candidate that exists and is executable; rc 1 if none
resolve_fleet_pv() {
    local c; local IFS=:
    for c in $FLEET_PV_CANDIDATES; do [ -x "$c" ] && { printf '%s\n' "$c"; return 0; }; done
    return 1
}

# judge -> prints ONE verdict row and returns 0 (PASS or UNMEASURED) / 1 (FAIL) / 2 (ENV)
judge() {
    local pin ver why out rc d PV_BIN
    if [ ! -r "$PV_PIN" ]; then
        printf 'UNMEASURED runner=%s reason=no-pin pin=%s -- this runner was never converged (infra#708); the shapes gate is not measured here, and this row is not a pass\n' "$RUNNER" "$PV_PIN"
        return 0
    fi
    # Quorum lane 1 (gemini-3.1-pro-high) on #3633, MEASURED: `tr -d '[:space:]'` without the
    # outer brackets is a literal set on busybox/POSIX tr and turns 0.65.2-rc1 into 0.65.2-r1;
    # GNU tr is merely lenient. No tr: bash strips the class itself, portably.
    pin=$(<"$PV_PIN"); pin=${pin//[[:space:]]/}
    if ! PV_BIN=$(resolve_fleet_pv); then
        printf 'FAIL runner=%s pin=%s candidates=%s -- the pin declares the tool and none of the fleet paths has it; the box disagrees with its own declaration (forjar drift)\n' "$RUNNER" "$pin" "$FLEET_PV_CANDIDATES" >&2
        return 1
    fi
    ver=$("$PV_BIN" --version 2>/dev/null | head -1 | awk '{print $2}')
    if [ "$ver" != "$pin" ]; then
        printf 'FAIL runner=%s pin=%s pv=%s version=%s -- the resolved binary is NOT the pinned one; a number from it would be a number from the wrong tool (the 26-day-old apr class)\n' "$RUNNER" "$pin" "$PV_BIN" "${ver:-?}" >&2
        return 1
    fi
    if ! why=$(capable "$PV_BIN"); then
        printf 'FAIL runner=%s pin=%s pv=%s version=%s -- %s; this is the 0.65.2 condition, the gate cannot run here\n' "$RUNNER" "$pin" "$PV_BIN" "${ver:-?}" "$why" >&2
        return 1
    fi
    d=$(mktemp -d) || return 2
    "$PV_BIN" lint "$CONTRACTS" --gate shapes --format json > "$d/lint.json" 2> "$d/lint.err"; rc=$?
    out=$(python3 - "$d/lint.json" "$rc" <<'PY'
import json, sys
p, rc = sys.argv[1], int(sys.argv[2])
try:
    d = json.load(open(p))
except Exception as e:
    print(f"ENV no JSON verdict (rc={rc}): {e}"); sys.exit(2)
v = d.get("verdict"); pc = d.get("pc_shape"); planted = d.get("plant_violations", 0)
corpus = d.get("by_entity_type", {}); n = d.get("focus_nodes_n", 0); tri = d.get("triples", 0)
armed = ",".join(d.get("armed_shapes", []) or []) or "-"
row = f"verdict={v} focus_nodes={n} triples={tri} armed={armed} corpus={json.dumps(corpus, separators=(',',':'))} planted={planted} pc_shape={pc}"
# pv's own planted-violation control is NOT a ticket requirement (quorum lane 1 on #3633
# called it creep). It is REPORTED in the row whenever pv emits it, and it gates only when
# pv emits it AND says the checker did not fire -- a checker that fires on nothing cannot
# report Pass. A pv that does not emit the field is an unreported control, not a failure.
if "pc_shape" in d and (pc != "fired" or not isinstance(planted, int) or planted < 1):
    print("FAIL " + row + " -- pv reports its planted-violation control did NOT fire; a checker that fires on nothing cannot report Pass"); sys.exit(1)
if "pc_shape" not in d:
    row += " control=unreported"
if v != "Pass":
    print("FAIL " + row); sys.exit(1)
print("PASS " + row); sys.exit(0)
PY
); rc=$?
    rmtree "$d"
    case "$rc" in
        0) printf '%s runner=%s pin=%s pv=%s version=%s\n' "$out" "$RUNNER" "$pin" "$PV_BIN" "$ver" ;;
        1) printf '%s runner=%s pin=%s pv=%s version=%s\n' "$out" "$RUNNER" "$pin" "$PV_BIN" "$ver" >&2 ;;
        *) printf '%s runner=%s pin=%s pv=%s version=%s\n' "$out" "$RUNNER" "$pin" "$PV_BIN" "$ver" >&2 ;;
    esac
    return "$rc"
}

case "${1:-}" in -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac

if [ "${1:-}" = "--self-test" ]; then
    echo "=== fleet pv shapes gate: case table (stub pvs, no network, no real fleet binary) ==="
    d=$(mktemp -d) || exit 2; trap 'rmtree "${d:-}"' EXIT
    bad=0; n=0
    ok()  { n=$((n+1)); printf 'ok    row %-2s %s\n' "$n" "$*"; }
    nok() { n=$((n+1)); printf 'FAIL  row %-2s %s\n' "$n" "$*" >&2; bad=1; }
    mkpv() { # mkpv <path> <mode: capable|old> <verdict for lint json: Pass|Fail> [pc_shape|omit] [version]
        local p=$1 mode=$2 verdict=$3 pc=${4:-fired} v=${5:-0.68.2} pcfield
        if [ "$pc" = omit ]; then pcfield=""; else pcfield="\"pc_shape\":\"$pc\","; fi
        cat > "$p" <<STUB
#!/usr/bin/env bash
case "\$1" in
  --version) echo "pv $v (stub)";;
  lint) case "\$2" in --help) [ "$mode" = capable ] && echo "--gate <G>  gate to run" || echo "(no gate flag)";; *) printf '{"verdict":"$verdict",${pcfield}"plant_violations":3,"focus_nodes_n":7,"triples":40,"armed_shapes":["ont-shapes-v1"],"by_entity_type":{"code":5}}';; esac;;
  extract) [ "$mode" = capable ] && exit 0 || { echo "unrecognized subcommand" >&2; exit 2; };;
esac
STUB
        chmod 755 "$p"
    }
    printf '0.68.2\n' > "$d/pin"; mkdir -p "$d/contracts"

    # 1. everything present and green -> PASS, and the row carries the runner
    mkpv "$d/pv_ok" capable Pass
    out=$(FLEET_PV_BIN="$d/pv_ok" FLEET_PV_PIN="$d/pin" FLEET_PV_CONTRACTS="$d/contracts" RUNNER_NAME=probe-runner bash "$0" 2>&1); rc=$?
    [ "$rc" -eq 0 ] && grep -q '^PASS .*verdict=Pass.*runner=probe-runner' <<<"$out" && ok "capable pv + Pass verdict -> PASS, row names the runner" || nok "expected PASS row, got rc=$rc: $out"

    # 2. PLANTED VIOLATION: lint says Fail -> the guard is RED (verdict parsed, not grepped)
    mkpv "$d/pv_fail" capable Fail
    out=$(FLEET_PV_BIN="$d/pv_fail" FLEET_PV_PIN="$d/pin" FLEET_PV_CONTRACTS="$d/contracts" bash "$0" 2>&1); rc=$?
    [ "$rc" -eq 1 ] && grep -q '^FAIL .*verdict=Fail' <<<"$out" && ok "planted violation (verdict=Fail) -> RED" || nok "expected RED on verdict=Fail, got rc=$rc: $out"

    # 3. the 0.65.2 condition: no --gate, no extract -> RED, named as a capability failure
    mkpv "$d/pv_old" old Pass
    out=$(FLEET_PV_BIN="$d/pv_old" FLEET_PV_PIN="$d/pin" FLEET_PV_CONTRACTS="$d/contracts" bash "$0" 2>&1); rc=$?
    [ "$rc" -eq 1 ] && grep -q 'no --gate on lint' <<<"$out" && ok "0.65.2 condition (no --gate) -> RED by capability, not by version" || nok "expected capability RED, got rc=$rc: $out"

    # 4. pin declared, binary absent -> RED (the box disagrees with its own declaration)
    out=$(FLEET_PV_BIN="$d/does-not-exist" FLEET_PV_PIN="$d/pin" FLEET_PV_CONTRACTS="$d/contracts" bash "$0" 2>&1); rc=$?
    [ "$rc" -eq 1 ] && grep -q 'none of the fleet paths has it' <<<"$out" && ok "pin present, no fleet path has the binary -> RED" || nok "expected RED on absent binary, got rc=$rc: $out"

    # 5. no pin -> UNMEASURED, exit 0, the row says so and names the runner and infra#708
    out=$(FLEET_PV_BIN="$d/pv_ok" FLEET_PV_PIN="$d/no-such-pin" FLEET_PV_CONTRACTS="$d/contracts" RUNNER_NAME=never-converged bash "$0" 2>&1); rc=$?
    [ "$rc" -eq 0 ] && grep -q '^UNMEASURED runner=never-converged.*infra#708' <<<"$out" && ! grep -q '^PASS' <<<"$out" && ok "no pin -> UNMEASURED row (exit 0, never a PASS line)" || nok "expected UNMEASURED row, got rc=$rc: $out"

    # 6. pv's own control silent (pc_shape != fired) -> RED even with verdict=Pass
    mkpv "$d/pv_silent" capable Pass silent
    out=$(FLEET_PV_BIN="$d/pv_silent" FLEET_PV_PIN="$d/pin" FLEET_PV_CONTRACTS="$d/contracts" bash "$0" 2>&1); rc=$?
    [ "$rc" -eq 1 ] && grep -qi 'planted-violation control did not fire' <<<"$out" && ok "verdict=Pass but pc_shape not fired -> RED (a checker firing on nothing cannot pass)" || nok "expected RED on silent control, got rc=$rc: $out"

    # 7. resolution order: first candidate absent, second present -> resolved, PASS names it
    mkpv "$d/pv_second" capable Pass
    out=$(FLEET_PV_BIN="$d/does-not-exist:$d/pv_second" FLEET_PV_PIN="$d/pin" FLEET_PV_CONTRACTS="$d/contracts" bash "$0" 2>&1); rc=$?
    [ "$rc" -eq 0 ] && grep -q "^PASS .*pv=$d/pv_second" <<<"$out" && ok "first fleet path absent, second present -> resolved to the second, row names it" || nok "expected PASS via second candidate, got rc=$rc: $out"

    # 8. version != pin -> RED: the resolved binary is not the pinned one (the 26-day-old apr class)
    mkpv "$d/pv_stale" capable Pass fired 0.65.2
    out=$(FLEET_PV_BIN="$d/pv_stale" FLEET_PV_PIN="$d/pin" FLEET_PV_CONTRACTS="$d/contracts" bash "$0" 2>&1); rc=$?
    [ "$rc" -eq 1 ] && grep -q 'NOT the pinned one' <<<"$out" && ok "binary 0.65.2 under pin 0.68.2 -> RED, named as a pin mismatch (before any lint)" || nok "expected pin-mismatch RED, got rc=$rc: $out"

    # 9. pv omits pc_shape entirely -> PASS with control=unreported in the row (not a false negative)
    mkpv "$d/pv_nopc" capable Pass omit
    out=$(FLEET_PV_BIN="$d/pv_nopc" FLEET_PV_PIN="$d/pin" FLEET_PV_CONTRACTS="$d/contracts" bash "$0" 2>&1); rc=$?
    [ "$rc" -eq 0 ] && grep -q '^PASS .*control=unreported' <<<"$out" && ok "pv without a pc_shape field -> PASS, row says control=unreported" || nok "expected PASS with control=unreported, got rc=$rc: $out"

    # 10. a pin with surrounding whitespace and a suffix survives the strip intact (the tr finding)
    printf '  0.68.2-rc1 \n' > "$d/pin-rc"; mkpv "$d/pv_rc" capable Pass fired 0.68.2-rc1
    out=$(FLEET_PV_BIN="$d/pv_rc" FLEET_PV_PIN="$d/pin-rc" FLEET_PV_CONTRACTS="$d/contracts" bash "$0" 2>&1); rc=$?
    [ "$rc" -eq 0 ] && grep -q 'pin=0.68.2-rc1 ' <<<"$out" && ok "pin '  0.68.2-rc1 \\n' strips to 0.68.2-rc1 -- no tr, no busybox class bug" || nok "expected pin=0.68.2-rc1, got rc=$rc: $out"

    [ "$bad" -eq 0 ] && { printf 'SELF-TEST PASSED: %s rows\n' "$n"; exit 0; }
    printf 'SELF-TEST FAILED\n' >&2; exit 1
fi

echo "=== shapes gate on the FLEET-PINNED pv, on this runner (check_fleet_pv_shapes_gate.sh) ==="
# The row is the verdict. No trailer may begin with PASS unless the row did: a trailer
# reading "PASS-OR-UNMEASURED" is exactly what a consumer's grep would misread (the
# self-test's row 5 refused it).
row=$(judge 2>&1); rc=$?
printf '%s\n' "$row"
case "$rc" in 0) printf '%s\n' "${row%% *}";; 1) echo "FAIL" >&2;; *) echo "ENV (rc=$rc)" >&2;; esac
exit "$rc"
