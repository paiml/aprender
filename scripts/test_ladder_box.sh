#!/usr/bin/env bash
# test_ladder_box.sh — case table for scripts/lib/ladder_box.sh (#4520 step 1), against a PLANTED PSI file.
# Each case names what must hold; the mutants prove each assertion can fail.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
BOX=scripts/lib/ladder_box.sh
T=$(mktemp -d) || exit 2
trap 'rm -rf "${T:?}"' EXIT
fail=0
ok()  { printf 'PASS  %s\n' "$1"; }
bad() { printf 'FAIL  %s\n' "$1"; fail=1; }
psi() { printf 'some avg10=%s avg60=0.00 avg300=0.00 total=0\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n' "$1" > "$2"; }

# brake <box> <tag>: planted 50% stops the unit, planted 10% resumes it, exit status passes through
brake() {
  local box="$1" tag="$2" ev="$T/$2.jsonl" pf="$T/$2.psi" rc
  psi 50.00 "$pf"
  ( sleep 4; psi 10.00 "$pf" ) &
  # the command ticks every 0.2 s; a real stop leaves a gap of >= 2 s between two ticks
  LADDER_PSI_FILE="$pf" LADDER_PSI_PERIOD=1 timeout 60 bash "$box" --events "$ev" --io-path "$T" -- \
    bash -c 'for i in $(seq 40); do date +%s.%N >> "$0"; sleep 0.2; done; exit 3' "$T/$tag.ticks" >/dev/null 2>&1; rc=$?
  wait
  local gap
  gap=$(awk 'NR>1 && $1-p>m {m=$1-p} {p=$1} END{printf "%d", m}' "$T/$tag.ticks" 2>/dev/null)
  grep -q '"event":"SIGSTOP"' "$ev" 2>/dev/null && grep -q '"event":"SIGCONT"' "$ev" && [ "$rc" = 3 ] \
    && grep -q '"rc":3,"brake_stops":1' "$ev" && [ "${gap:-0}" -ge 2 ]
}
if brake "$BOX" real; then ok "planted PSI 50% → SIGSTOP, 10% → SIGCONT, rc 3 passes through"; else bad "brake on the real box"; cat "$T/real.jsonl" 2>/dev/null; fi

# the boxed command sees LADDER_BOXED=1 and the caller's exported env
if LADDER_PSI_FILE=/dev/null BOXTEST_X=yes timeout 60 bash "$BOX" --events "$T/env.jsonl" --io-path "$T" -- \
     bash -c '[ "$LADDER_BOXED" = 1 ] && [ "$BOXTEST_X" = yes ]' >/dev/null 2>&1; then ok "LADDER_BOXED=1 and caller env reach the unit"
else bad "env passthrough"; fi

# refusals: nothing runs
bash "$BOX" --events "$T/x.jsonl" -- touch "$T/ran1" >/dev/null 2>&1; r1=$?
LADDER_PSI_FILE="$T/nope" bash "$BOX" --events "$T/x.jsonl" --io-path "$T" -- touch "$T/ran2" >/dev/null 2>&1; r2=$?
bash "$BOX" --events "$T/x.jsonl" --io-path /nonexistent-4520 -- touch "$T/ran3" >/dev/null 2>&1; r3=$?
if [ "$r2" = 2 ] && [ "$r3" = 2 ] && [ ! -e "$T/ran2" ] && [ ! -e "$T/ran3" ]; then ok "blind brake / no device → exit 2, nothing ran"
else bad "refusals (r2=$r2 r3=$r3)"; fi
: "$r1"

# the ladder re-execs itself boxed: an unboxed non-dry call goes through ladder_box.sh
if grep -q 'exec bash scripts/lib/ladder_box.sh' scripts/model_ladder.sh && grep -q '"host_box": host_box' scripts/model_ladder.sh
then ok "model_ladder.sh self-boxes and records host_box"; else bad "model_ladder.sh wiring"; fi

# mutants: each must turn the brake case RED
mutant() {
  local name="$1" from="$2" to="$3" m="$T/box-$1.sh"
  python3 - "$BOX" "$m" "$from" "$to" <<'PY' || { bad "mutant $name: anchor not found"; return; }
import sys; t=open(sys.argv[1]).read()
if sys.argv[3] not in t: sys.exit(1)
open(sys.argv[2],"w").write(t.replace(sys.argv[3], sys.argv[4], 1))
PY
  if brake "$m" "mut-$name"; then bad "mutant $name survived"; else ok "mutant $name killed"; fi
}
mutant no-stop   '--signal=SIGSTOP "$UNIT"' '--signal=SIGWINCH "$UNIT"'
mutant no-cont   '[ "$v" -lt $((PSI_CONT * 100)) ]' '[ "$v" -lt 0 ]'
mutant rc-lost   'exit "$rc"' 'exit 0'
mutant threshold '[ "$v" -gt $((PSI_STOP * 100)) ]' '[ "$v" -gt 999999 ]'
exit "$fail"
