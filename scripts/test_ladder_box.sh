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
mi()  { printf 'MemTotal:       %s kB\nMemFree:        1 kB\nMemAvailable:   %s kB\n' "$1" "$2" > "$3"; }
# every case runs against a PLANTED meminfo with room, so a busy host cannot make a case red or green
mi 134217728 67108864 "$T/roomy.meminfo"; export LADDER_MEMINFO_FILE="$T/roomy.meminfo"
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

# memory: a host without 4 GiB of room past the reserve refuses to start; nothing runs
mi 134217728 19922944 "$T/tight.meminfo"
LADDER_MEMINFO_FILE="$T/tight.meminfo" LADDER_PSI_FILE=/dev/null bash "$BOX" --events "$T/m.jsonl" --io-path "$T" -- touch "$T/ranm" >/dev/null 2>&1; rm_rc=$?
[ "$rm_rc" = 2 ] && [ ! -e "$T/ranm" ] && ok "MemAvailable 19G - 16G reserve < 4G → exit 2, nothing ran" || bad "tight-host refusal (rc=$rm_rc)"
# memory floor <box> <tag>: MemAvailable dropping under the floor mid-run stops the unit, exit 75, event mem_floor
memfloor() {
  local box="$1" tag="$2" mf="$T/$2.meminfo" rc n
  mi 134217728 67108864 "$mf"
  ( sleep 3; mi 134217728 1048576 "$mf" ) &
  LADDER_MEMINFO_FILE="$mf" LADDER_PSI_FILE=/dev/null LADDER_PSI_PERIOD=1 timeout 60 bash "$box" --events "$T/$tag.jsonl" --io-path "$T" -- \
    bash -c 'for i in $(seq 60); do echo x >> "$0"; sleep 0.2; done' "$T/$tag.ticks" >/dev/null 2>&1; rc=$?
  wait
  n=$(wc -l < "$T/$tag.ticks" 2>/dev/null || echo 0)
  [ "$rc" = 75 ] && grep -q '"event":"mem_floor"' "$T/$tag.jsonl" && [ "$n" -lt 60 ] \
    && grep -q '"swap_max":0' "$T/$tag.jsonl"
}
if memfloor "$BOX" mf-real; then ok "MemAvailable under the floor mid-run → unit stopped, mem_floor, exit 75"; else bad "memory floor"; cat "$T/mf-real.jsonl" 2>/dev/null; fi
grep -q -- '-p MemorySwapMax=0' "$BOX" && ok "the box sets MemorySwapMax=0" || bad "MemorySwapMax=0 missing"
grep -q -- '-p OOMPolicy=continue' "$BOX" && ok "the box sets OOMPolicy=continue (an OOM kills the cell, not the ladder)" || bad "OOMPolicy=continue missing: a cell OOM loses the whole receipt"
# ...and prove the property does what the comment says: a child over a 64M box is OOM-killed while its
# parent (the ladder's stand-in) lives on and exits with its OWN status. Control: OOMPolicy=stop loses it.
oom_rc() { timeout 60 systemd-run --user --wait --collect -q -p MemoryMax=64M -p MemorySwapMax=0 -p "OOMPolicy=$1" \
  bash -c 'python3 -c "b=bytearray(256<<20); b[::4096]=b\"x\"*len(b[::4096])" 2>/dev/null; exit 7' >/dev/null 2>&1; echo $?; }
if systemctl --user show-environment >/dev/null 2>&1; then
  c=$(oom_rc continue); s=$(oom_rc stop)
  [ "$c" = 7 ] && [ "$s" != 7 ] && ok "OOMPolicy=continue: the over-box child died, the parent exited 7 (stop control: $s)" \
    || bad "OOMPolicy proof: continue rc=$c (want 7), stop rc=$s (want not 7)"
else echo "  note: no user systemd -- OOMPolicy behaviour not proven here (the property grep still ran)"; fi

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
mfmutant() {
  local m="$T/box-mf.sh"
  python3 -c 'import sys; t=open(sys.argv[1]).read(); a="[ \"$ma\" -lt \"$MEM_FLOOR_KB\" ]"; assert a in t; open(sys.argv[2],"w").write(t.replace(a,"false",1))' "$BOX" "$m" \
    || { bad "mutant mem-floor: anchor not found"; return; }
  if memfloor "$m" mut-mf; then bad "mutant mem-floor survived"; else ok "mutant mem-floor killed"; fi
}
mfmutant
exit "$fail"
