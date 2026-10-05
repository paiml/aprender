#!/usr/bin/env bash
# check_release_gate.sh -- case table for scripts/release/release_gate.sh (#4805).
#
#   bash scripts/release/check_release_gate.sh [--self-test]
#
# Hermetic: a fixture repository (tagged v1.2.3 at HEAD) holding a stub preflight at its
# own scripts/check_publish_preflight.sh (the gate takes no override, #4833) and a stub
# clean-room lib placed beside a COPY of release_gate.sh, so no row reaches GitHub.
# The real clean-room lib has its own table (scripts/check_cascade_clean_room_gate.sh);
# this one proves how release_gate.sh COMBINES the two verdicts, that it asks the
# UNCHANGED preflight (no mode argument) about the root and the clean-room gate about
# the tag, and that a tag not at HEAD is never judged.
#
# Exit 0 every row held · 1 a row failed. Rows print "ok"/"FAIL".
set -uo pipefail

case "${1:-}" in
    -h|--help) sed -n '2,12p' "$0"; echo "  --self-test   run the case table (the default)"; exit 0 ;;
    ""|--self-test) ;;
    *) echo "usage: $0 [--self-test]" >&2; exit 2 ;;
esac

# the table judges the gate, never the environment it was started from
unset RELEASE_PREFLIGHT
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBJECT="$HERE/release_gate.sh"
WORK="$(mktemp -d)" || exit 1
trap 'rm -rf -- "${WORK:?}"' EXIT
bad=0
ok()  { printf 'ok    %s\n' "$*"; }
nok() { printf 'FAIL  %s\n' "$*"; bad=1; }

FIX="$WORK/repo"
git init -q "$FIX" && git -C "$FIX" -c core.hooksPath=/dev/null -c user.email=t@t -c user.name=t commit -q --allow-empty -m base || exit 1
HEAD_SHA="$(git -C "$FIX" rev-parse HEAD)"
git -C "$FIX" tag v1.2.3 || exit 1

mkdir -p "$WORK/gate"
cp "$SUBJECT" "$WORK/gate/release_gate.sh"
# stub lib: the clean-room verdict comes from $CR_MODE; every question is recorded
cat > "$WORK/gate/lib_clean_room_gate.sh" <<'LIB'
clean_room_gate() {
    printf 'CR %s %s\n' "$1" "$2" >> "$CALLS"
    case "$CR_MODE" in
        green) echo "CLEAN-ROOM PROCEED: run 1 job 2 on tag $2"; return 0 ;;
        other) echo "CLEAN-ROOM REFUSE: newest completed run tested 0123abcd (tag $2)"; return 1 ;;
        none)  echo "CLEAN-ROOM REFUSE: clean-room runs after tag $2 -- found none"; return 1 ;;
    esac
}
LIB
# stub preflight: rc from $PF_RC; records its argument and PUBLISH_PREFLIGHT_ROOT
mkdir -p "$FIX/scripts"
cat > "$FIX/scripts/check_publish_preflight.sh" <<'PF'
printf 'PF [%s] %s\n' "$*" "${PUBLISH_PREFLIGHT_ROOT:-unset}" >> "$CALLS"
echo "stub preflight rc=$PF_RC"
exit "$PF_RC"
PF

# row NAME WANT_RC WANT_LINE PF_RC CR_MODE
row() {
    local name=$1 want=$2 line=$3 out rc=0
    export CALLS="$WORK/calls-$name" PF_RC=$4 CR_MODE=$5
    out="$(bash "$WORK/gate/release_gate.sh" v1.2.3 "$FIX" 2>&1)" || rc=$?
    if [ "$rc" -ne "$want" ]; then nok "$name: rc=$rc, wanted $want"; printf '%s\n' "$out" | sed 's/^/      /'
    elif ! grep -qxF -- "$line" <<< "$out"; then nok "$name: no line '$line'"; printf '%s\n' "$out" | sed 's/^/      /'
    elif ! grep -qxF "PF [] $FIX" "$CALLS" || ! grep -qxF "CR $FIX v1.2.3" "$CALLS"; then
        nok "$name: preflight not asked unchanged about the root, or clean-room not asked about the tag: $(tr '\n' ';' < "$CALLS")"
    else ok "$name -> rc=$rc"; fi
}
V="tag=v1.2.3 sha=$HEAD_SHA"
row both_green_passes            0 "RELEASE-GATE PASS preflight=PASS clean-room=PASS $V"                  0 green
row preflight_red_refuses        1 "RELEASE-GATE REFUSE preflight=REFUSE clean-room=PASS $V -- no release"    1 green
row cleanroom_other_sha_refuses  1 "RELEASE-GATE REFUSE preflight=PASS clean-room=REFUSE $V -- no release"    0 other
row no_cleanroom_run_unmeasured  2 "RELEASE-GATE NOT_MEASURED preflight=PASS clean-room=NOT_MEASURED $V -- no release" 0 none
row preflight_cannot_judge       2 "RELEASE-GATE NOT_MEASURED preflight=NOT_MEASURED clean-room=PASS $V -- no release" 2 green
row refuse_beats_not_measured    1 "RELEASE-GATE REFUSE preflight=REFUSE clean-room=NOT_MEASURED $V -- no release" 1 none

# not a repository -> NOT_MEASURED, never a pass
rc=0; out="$(CALLS="$WORK/c-nr" PF_RC=0 CR_MODE=green bash "$WORK/gate/release_gate.sh" v1.2.3 "$WORK/nope" 2>&1)" || rc=$?
[ "$rc" -eq 2 ] && ok "not a repository -> rc=2" || { nok "not a repository gave rc=$rc"; printf '%s\n' "$out"; }
# the tag names another commit (or nothing) -> rc=2 and NEITHER gate is asked
for t in v9.9.9 v1.2.3-moved; do
    [ "$t" = v1.2.3-moved ] && { git -C "$FIX" -c core.hooksPath=/dev/null -c user.email=t@t -c user.name=t commit -q --allow-empty -m later; git -C "$FIX" tag "$t" HEAD; git -C "$FIX" reset -q --hard HEAD~1; }
    rc=0; out="$(CALLS="$WORK/c-$t" PF_RC=0 CR_MODE=green bash "$WORK/gate/release_gate.sh" "$t" "$FIX" 2>&1)" || rc=$?
    if [ "$rc" -eq 2 ] && [ ! -s "$WORK/c-$t" ]; then ok "tag $t not at HEAD -> rc=2, no gate asked"
    else nok "tag $t not at HEAD gave rc=$rc (calls: $(tr '\n' ';' 2>/dev/null < "$WORK/c-$t"))"; fi
done

# #4833: RELEASE_PREFLIGHT in the environment is an override attempt -> rc=2, and
# neither the override, the root preflight nor the clean-room gate is asked. The root
# preflight refuses (PF_RC=1) and the override would pass, so a gate that honoured it
# would turn a refusal into a release.
cat > "$WORK/hijack.sh" <<'HJ'
echo "HIJACK" >> "$CALLS"
exit 0
HJ
# override_row NAME OVERRIDE -- run with RELEASE_PREFLIGHT=OVERRIDE; prints the rc
override_row() {
    local name=$1 rc=0 out cf
    cf="$WORK/c-$name"
    out="$(CALLS="$cf" PF_RC=1 CR_MODE=green RELEASE_PREFLIGHT="$2" bash "$WORK/gate/release_gate.sh" v1.2.3 "$FIX" 2>&1)" || rc=$?
    if [ "$rc" -eq 2 ] && grep -q '^NOT_MEASURED release_gate.sh: RELEASE_PREFLIGHT is set' <<< "$out" && [ ! -s "$cf" ]; then
        ok "$name -> rc=2, nothing asked"
    else
        nok "$name: rc=$rc (wanted 2), calls: $(tr '\n' ';' 2>/dev/null < "$cf")"; printf '%s\n' "$out" | sed 's/^/      /'
    fi
}
override_row env_override_refused "$WORK/hijack.sh"
override_row env_override_empty_refused ""

# mutant: the clean-room verdict ignored -> the table must go RED (it is not vacuous)
sed 's/^if \[ "\$pf" = PASS \] && \[ "\$cr" = PASS \]; then/if [ "$pf" = PASS ]; then/' "$SUBJECT" > "$WORK/gate/release_gate.sh"
if cmp -s "$SUBJECT" "$WORK/gate/release_gate.sh"; then nok "mutant could not be built -- vacuous"
else
    rc=0; CALLS="$WORK/c-m" PF_RC=0 CR_MODE=other bash "$WORK/gate/release_gate.sh" v1.2.3 "$FIX" > /dev/null 2>&1 || rc=$?
    [ "$rc" -ne 0 ] && nok "mutant (clean-room ignored) still refused, rc=$rc -- the row cannot see it" \
        || ok "mutant: clean-room verdict ignored -> a red clean-room passes, so the row above catches it"
fi

# mutant (#4833): the env seam restored and the refusal removed -> the override passes a
# release the root preflight refused, so env_override_refused must be able to see it
sed -e '/^\[ -z "\${RELEASE_PREFLIGHT+set}" \]/d' \
    -e 's|^preflight="\$ROOT/scripts/check_publish_preflight.sh"$|preflight="${RELEASE_PREFLIGHT:-"$ROOT/scripts/check_publish_preflight.sh"}"|' \
    "$SUBJECT" > "$WORK/gate/release_gate.sh"
if [ "$(diff "$SUBJECT" "$WORK/gate/release_gate.sh" | grep -c '^[<>]')" -ne 3 ]; then nok "seam mutant could not be built -- vacuous"
else
    rc=0; CALLS="$WORK/c-ms" PF_RC=1 CR_MODE=green RELEASE_PREFLIGHT="$WORK/hijack.sh" bash "$WORK/gate/release_gate.sh" v1.2.3 "$FIX" > /dev/null 2>&1 || rc=$?
    [ "$rc" -eq 0 ] && grep -qx HIJACK "$WORK/c-ms" \
        && ok "mutant: env seam restored -> the override passes a refused release, so env_override_refused catches it" \
        || nok "seam mutant gave rc=$rc without the override -- the row cannot see it"
fi

[ "$bad" -eq 0 ] && echo "SELF-TEST PASSED" || echo "SELF-TEST FAILED"
exit "$bad"
