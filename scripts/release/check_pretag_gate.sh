#!/usr/bin/env bash
# check_pretag_gate.sh -- case table for scripts/release/pretag_gate.sh (#4805).
#
#   bash scripts/release/check_pretag_gate.sh [--self-test]
#
# Hermetic: a fixture repository, a stub preflight (PRETAG_PREFLIGHT) and a stub
# clean-room lib placed beside a COPY of pretag_gate.sh, so no row reaches GitHub.
# The real clean-room lib has its own table (scripts/check_cascade_clean_room_gate.sh);
# this one proves how pretag_gate.sh COMBINES the two verdicts, and that it asks both
# gates about exactly HEAD of the root it was given.
#
# Exit 0 every row held · 1 a row failed. Rows print "ok"/"FAIL".
set -uo pipefail

case "${1:-}" in
    -h|--help) sed -n '2,12p' "$0"; echo "  --self-test   run the case table (the default)"; exit 0 ;;
    ""|--self-test) ;;
    *) echo "usage: $0 [--self-test]" >&2; exit 2 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBJECT="$HERE/pretag_gate.sh"
WORK="$(mktemp -d)" || exit 1
trap 'rm -rf -- "${WORK:?}"' EXIT
bad=0
ok()  { printf 'ok    %s\n' "$*"; }
nok() { printf 'FAIL  %s\n' "$*"; bad=1; }

FIX="$WORK/repo"
git init -q "$FIX" && git -C "$FIX" -c core.hooksPath=/dev/null -c user.email=t@t -c user.name=t commit -q --allow-empty -m base || exit 1
HEAD_SHA="$(git -C "$FIX" rev-parse HEAD)"

mkdir -p "$WORK/gate"
cp "$SUBJECT" "$WORK/gate/pretag_gate.sh"
# stub lib: the clean-room verdict comes from $CR_MODE; every question is recorded
cat > "$WORK/gate/lib_clean_room_gate.sh" <<'LIB'
clean_room_gate_sha() {
    printf 'CR %s %s\n' "$1" "$2" >> "$CALLS"
    case "$CR_MODE" in
        green) echo "CLEAN-ROOM PROCEED: run 1 job 2 tested $2 = $2 ($3)"; return 0 ;;
        other) echo "CLEAN-ROOM REFUSE: newest completed run tested 0123abcd, not $2 ($3)"; return 1 ;;
        none)  echo "CLEAN-ROOM REFUSE: clean-room runs after $2 ($3) -- found none"; return 1 ;;
    esac
}
LIB
# stub preflight: rc from $PF_RC; records its argument and PUBLISH_PREFLIGHT_ROOT
cat > "$WORK/preflight.sh" <<'PF'
printf 'PF %s %s\n' "$*" "${PUBLISH_PREFLIGHT_ROOT:-unset}" >> "$CALLS"
echo "stub preflight rc=$PF_RC"
exit "$PF_RC"
PF

# row NAME WANT_RC WANT_LINE PF_RC CR_MODE
row() {
    local name=$1 want=$2 line=$3 out rc=0
    export CALLS="$WORK/calls-$name" PF_RC=$4 CR_MODE=$5 PRETAG_PREFLIGHT="$WORK/preflight.sh"
    out="$(bash "$WORK/gate/pretag_gate.sh" "$FIX" 2>&1)" || rc=$?
    if [ "$rc" -ne "$want" ]; then nok "$name: rc=$rc, wanted $want"; printf '%s\n' "$out" | sed 's/^/      /'
    elif ! grep -qxF -- "$line" <<< "$out"; then nok "$name: no line '$line'"; printf '%s\n' "$out" | sed 's/^/      /'
    elif ! grep -qxF "PF --pre-tag $FIX" "$CALLS" || ! grep -qxF "CR $FIX $HEAD_SHA" "$CALLS"; then
        nok "$name: a gate was not asked about HEAD of the root with --pre-tag: $(tr '\n' ';' < "$CALLS")"
    else ok "$name -> rc=$rc"; fi
}
V="sha=$HEAD_SHA"
row both_green_passes            0 "PRETAG PASS preflight=PASS clean-room=PASS $V"                  0 green
row preflight_red_refuses        1 "PRETAG REFUSE preflight=REFUSE clean-room=PASS $V -- no tag"    1 green
row cleanroom_other_sha_refuses  1 "PRETAG REFUSE preflight=PASS clean-room=REFUSE $V -- no tag"    0 other
row no_cleanroom_run_unmeasured  2 "PRETAG NOT_MEASURED preflight=PASS clean-room=NOT_MEASURED $V -- no tag" 0 none
row preflight_cannot_judge       2 "PRETAG NOT_MEASURED preflight=NOT_MEASURED clean-room=PASS $V -- no tag" 2 green
row refuse_beats_not_measured    1 "PRETAG REFUSE preflight=REFUSE clean-room=NOT_MEASURED $V -- no tag" 1 none

# not a repository -> NOT_MEASURED, never a pass
rc=0; out="$(CALLS="$WORK/c-nr" PF_RC=0 CR_MODE=green bash "$WORK/gate/pretag_gate.sh" "$WORK/nope" 2>&1)" || rc=$?
[ "$rc" -eq 2 ] && ok "not a repository -> rc=2" || { nok "not a repository gave rc=$rc"; printf '%s\n' "$out"; }

# mutant: the clean-room verdict ignored -> the table must go RED (it is not vacuous)
sed 's/^if \[ "\$pf" = PASS \] && \[ "\$cr" = PASS \]; then/if [ "$pf" = PASS ]; then/' "$SUBJECT" > "$WORK/gate/pretag_gate.sh"
if cmp -s "$SUBJECT" "$WORK/gate/pretag_gate.sh"; then nok "mutant could not be built -- vacuous"
else
    rc=0; CALLS="$WORK/c-m" PF_RC=0 CR_MODE=other PRETAG_PREFLIGHT="$WORK/preflight.sh" bash "$WORK/gate/pretag_gate.sh" "$FIX" > /dev/null 2>&1 || rc=$?
    [ "$rc" -ne 0 ] && nok "mutant (clean-room ignored) still refused, rc=$rc -- the row cannot see it" \
        || ok "mutant: clean-room verdict ignored -> a red clean-room passes, so the row above catches it"
fi

[ "$bad" -eq 0 ] && echo "SELF-TEST PASSED" || echo "SELF-TEST FAILED"
exit "$bad"
