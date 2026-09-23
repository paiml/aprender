#!/usr/bin/env bash
# check_ladder_cpu_lane.sh — the ladder's CPU lane runs OUTSIDE the GPU lock (and, opted in, CONCURRENTLY
# with the GPU lane) without being able to touch a GPU or drop out of the receipt (#4034).
#
# WHY. Every apr call in a cell took the fleet GPU lock, the `--no-gpu` lane included. On lambda's
# qwen35-9b-q4km rung, 50 of 83 nvidia-smi samples (60%, about 16.5 of 27.4 min) show no process on
# the GPU while the ladder held the lock (#4033 baseline). model_ladder.sh now runs the CPU lane
# unlocked, with CUDA_VISIBLE_DEVICES set and empty, in the background next to the GPU lane. That
# is only safe if four things hold, and each is a case here:
#
#   LANE RUNNER (behavioural: the SHIPPED script's --lane-probe, with a fake apr that reports what it saw)
#     probe-cpu          the cpu lane holds no lock, sees CUDA_VISIBLE_DEVICES set and EMPTY, oom 1000
#     probe-cuda         every other lane holds the lock at oom 1000, with the devices left alone
#   CALL SITES (static, over the shipped script)
#     sites              apr_cpu_unlocked is called from apr_lane and nowhere else; the lane body calls
#                        apr only as `apr_lane "$b"`, and the serve probe as `apr_lane "$bname"`
#     green-needs-ran    the row builder's green still requires v["ran"], which the RED fragment
#                        below relies on
#   ORCHESTRATION (ladder_run_lanes LIFTED from the script, run against stub lanes)
#     serial-default     with no opt-in the lanes run one at a time: concurrency is OPT-IN until a
#                        lambda 9B A/B shows a gain (cop ruling on #4034); the CPU lane is still unlocked
#     concurrent         MODEL_LADDER_CONCURRENT_LANES=1: cpu and cuda lanes overlap in time, and
#                        be_json keeps the rung's backend order
#     order-cpu-first    the order is the rung's, not "foreground first"
#   (the cases below opt in, because the background lane is where they can go wrong)
#     cpu-silent         a background lane that prints nothing is a RED fragment naming its exit
#                        status, never absent (green is an all() over the backends it is GIVEN)
#     cpu-garbage        a lane with an unparseable fragment is RED the same way
#     cuda-silent        so is a foreground lane
#     cpu-decline        a decline (exit 2) inside the background lane ends the ladder with rc 2
#     cuda-decline-kills-cpu-lane  a decline on the GPU lane ends the ladder AND the background lane:
#                        an orphaned CPU apr would keep a model loaded and write into a deleted $WORK
#
# --self-test plants one regression per rule in a copy of the script, and each must turn a case RED.
#
# Exit: 0 all as expected · 1 a case landed wrong · 2 could not check.
set -uo pipefail
SCRIPT="scripts/model_ladder.sh"; SELF_TEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --script) [ $# -ge 2 ] || { echo "--script needs a value" >&2; exit 2; }; SCRIPT="$2"; shift 2 ;;
    --self-test) SELF_TEST=1; shift ;;
    -h|--help) awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
    *) echo "check_ladder_cpu_lane: unknown argument '$1'" >&2; exit 2 ;;
  esac
done
command -v flock > /dev/null && command -v choom > /dev/null \
  || { echo "  cannot check: flock and choom (util-linux) are required" >&2; exit 2; }
[ -f "$SCRIPT" ] || { echo "  cannot check: $SCRIPT not found" >&2; exit 2; }

T=$(mktemp -d) || { echo "  cannot check: mktemp failed" >&2; exit 2; }
case "$T" in /tmp/?*) ;; *) echo "  cannot check: expected a temp dir under /tmp, got '$T'" >&2; exit 2 ;; esac
cleanup() {
  case "${T:-}" in
    /tmp/?*) if [ -n "$T" ] && [ "$T" != "/" ] && [ -d "$T" ]; then rm -rf -- "$T" || :; fi ;;
  esac
}
trap cleanup EXIT

lift() { # <script> <fn> -> the function's text, from its `name() {` line to the first `}` at column 0
  awk -v F="^$2\\\\(\\\\) \\\\{" '$0 ~ F {f=1} f{print} f && /^\}$/{exit}' "$1"
}

run_cases() { # <script> -> 0 when every case lands
  local src="$1" rc=0 out r bodies
  ok() { printf '  ok    %s\n' "$1"; }
  bad() { printf '  FAIL  %s: %s\n' "$1" "$2"; rc=1; }

  # ── LANE RUNNER ────────────────────────────────────────────────────────────
  : > "$T/lock"
  printf '#!/usr/bin/env bash\nif flock -n "$FAKE_LOCK" true; then l=UNLOCKED; else l=LOCKED; fi\necho "fake-apr $1 lock=$l oom=$(cat /proc/self/oom_score_adj) cvd=${CUDA_VISIBLE_DEVICES-UNSET}."\n' > "$T/apr"
  chmod +x "$T/apr"
  probe() { # <backend> -> the fake apr's line
    env -u CUDA_VISIBLE_DEVICES FAKE_LOCK="$T/lock" MODEL_LADDER_ROOT="$PWD" MODEL_LADDER_GPU_LOCK="$T/lock" \
      DOGFOOD_ALLOW_UNPINNED=1 APR="$T/apr" timeout 60 bash "$src" --lane-probe "$1" run probe 2>&1
  }
  out=$(probe cpu); r=$?
  { [ "$r" = 0 ] && grep -q 'lock=UNLOCKED oom=1000 cvd=\.' <<< "$out"; } && ok probe-cpu \
    || bad probe-cpu "rc=$r, want lock=UNLOCKED oom=1000 cvd=<set, empty>: $out"
  out=$(probe cuda); r=$?
  { [ "$r" = 0 ] && grep -q 'lock=LOCKED oom=1000 cvd=UNSET\.' <<< "$out"; } && ok probe-cuda \
    || bad probe-cuda "rc=$r, want lock=LOCKED oom=1000 cvd=UNSET: $out"

  # ── CALL SITES ─────────────────────────────────────────────────────────────
  out=$(python3 - "$src" <<'PY'
import re, sys
src = open(sys.argv[1]).read().split("\n")
code = [re.sub(r"(^|\s)#.*$", "", l) for l in src]
bad = []
calls = [i for i, l in enumerate(code) if re.search(r"\bapr_cpu_unlocked\b", l) and not re.match(r"\s*apr_cpu_unlocked\(\) \{", l)]
lane = [i for i, l in enumerate(code) if 'if [ "$lane" = cpu ]; then apr_cpu_unlocked "$@"; else apr_locked "$@"; fi' in l]
if len(calls) != 1 or calls != lane:
    bad.append("apr_cpu_unlocked is called at line(s) %s; the only caller must be apr_lane's cpu branch (%s)" % ([c + 1 for c in calls], [c + 1 for c in lane]))
def body(name):
    s = next((i for i, l in enumerate(src) if l.startswith(name + "() {")), None)
    if s is None:
        return None
    e = next(i for i in range(s, len(src)) if src[i] == "}")
    return code[s + 1:e]
cell = body("ladder_backend_cell")
if cell is None:
    bad.append("ladder_backend_cell() is gone")
else:
    direct = [l.strip() for l in cell if re.search(r'\bapr_locked\b|\bapr_cpu_unlocked\b|"\$\{?APR\}?"', l)]
    lanes = [l for l in cell if 'apr_lane "$b" ' in l]
    if direct:
        bad.append("the lane body calls apr around apr_lane: %s" % direct)
    if len(lanes) != 2:
        bad.append("the lane body should call apr_lane \"$b\" for run and chat, found %d" % len(lanes))
probe = body("ladder_serve_probe")
if probe is None or not any('apr_lane "$bname" serve run' in l for l in probe) or any(re.search(r"\bapr_locked\b", l) for l in probe):
    bad.append('ladder_serve_probe must start its server through apr_lane "$bname"')
print("\n".join(bad))
PY
)
  [ -z "$out" ] && ok sites || bad sites "$out"
  grep -q 'and all(v\["ran"\] and not v\["fallback"\]' "$src" && ok green-needs-ran \
    || bad green-needs-ran "the row builder no longer requires v[\"ran\"]; the RED lane fragment is only RED because it does"

  # ── ORCHESTRATION ──────────────────────────────────────────────────────────
  bodies="$(lift "$src" serve_log_tail)"$'\n'"$(lift "$src" ladder_run_lanes)"$'\n'"$(lift "$src" _rm_work)"
  grep -q '^ladder_run_lanes() {' <<< "$bodies" || { bad orchestration "$src defines no ladder_run_lanes()"; return 1; }
  lanes() { # <case> <backends csv> <stub mode> [VAR=value...] -> rc; be json in $T/<case>/be.json
    local name="$1" bes_csv="$2" mode="$3"; shift 3
    mkdir -p "$T/$name"
    env "$@" W="$T/$name" BES="$bes_csv" STUB="$mode" timeout 20 bash -c '
WORK="$W/work"; mkdir -p "$WORK"; rid="r/1"; LADDER_LANE_BG_PID=""
'"$bodies"'
trap _rm_work EXIT
ladder_backend_cell() {
  local s e
  case "$STUB:$1" in
    cpu-silent:cpu)  return 7 ;;
    cuda-silent:cuda) return 3 ;;
    cpu-garbage:cpu) printf "\"cpu\":{broken"; return 0 ;;
    cpu-decline:cpu) echo "decline: ENV planted decline in the cpu lane" >&2; exit 2 ;;
    cuda-decline:cpu) sleep 300 & echo $! > "$W/bg.pid"; wait; return 0 ;;
    cuda-decline:cuda) sleep 0.5; echo "decline: ENV planted decline in the cuda lane" >&2; exit 2 ;;
  esac
  s=$(date +%s.%N); sleep 1.5; e=$(date +%s.%N)
  echo "$1 $s $e" >> "$W/spans"
  printf "\"%s\":{\"ran\":true,\"fallback\":false,\"verbs\":{}}" "$1"
}
measure_like() {
  local be_json="{" first=1 b
  IFS="," read -r -a bes <<< "$BES"
  ladder_run_lanes
  be_json="$be_json}"
  printf "%s" "$be_json" > "$W/be.json"
}
measure_like
' > "$T/$name/out" 2>&1
  }
  keys() { python3 -c 'import json,sys; print(list(json.load(open(sys.argv[1]))))' "$T/$1/be.json" 2> /dev/null; }
  field() { # <case> <backend> <key> -> the value, printed as Python prints it
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]][sys.argv[3]])' "$T/$1/be.json" "$2" "$3" 2> /dev/null
  }
  overlap() { python3 -c '
import sys
s = {}
for l in open(sys.argv[1]):
    b, a, e = l.split(); s[b] = (float(a), float(e))
(a1, e1), (a2, e2) = s["cpu"], s["cuda"]
print("yes" if a1 < e2 and a2 < e1 else "no")' "$T/$1/spans" 2> /dev/null; }

  lanes serial-default cuda,cpu ok; r=$?
  { [ "$r" = 0 ] && [ "$(overlap serial-default)" = no ] && [ "$(keys serial-default)" = "['cuda', 'cpu']" ]; } && ok serial-default \
    || bad serial-default "rc=$r overlap=$(overlap serial-default) keys=$(keys serial-default)"
  lanes concurrent cuda,cpu ok MODEL_LADDER_CONCURRENT_LANES=1; r=$?
  { [ "$r" = 0 ] && [ "$(overlap concurrent)" = yes ] && [ "$(keys concurrent)" = "['cuda', 'cpu']" ]; } && ok concurrent \
    || bad concurrent "rc=$r overlap=$(overlap concurrent) keys=$(keys concurrent) $(head -c 200 "$T/concurrent/out")"
  lanes order-cpu-first cpu,cuda ok MODEL_LADDER_CONCURRENT_LANES=1; r=$?
  { [ "$r" = 0 ] && [ "$(overlap order-cpu-first)" = yes ] && [ "$(keys order-cpu-first)" = "['cpu', 'cuda']" ]; } && ok order-cpu-first \
    || bad order-cpu-first "rc=$r overlap=$(overlap order-cpu-first) keys=$(keys order-cpu-first)"
  lanes cpu-silent cuda,cpu cpu-silent MODEL_LADDER_CONCURRENT_LANES=1; r=$?
  { [ "$r" = 0 ] && [ "$(field cpu-silent cpu ran)" = False ] && [ "$(field cpu-silent cuda ran)" = True ] \
    && field cpu-silent cpu lane_error | grep -q 'lane exit 7'; } && ok cpu-silent \
    || bad cpu-silent "rc=$r be=$(head -c 300 "$T/cpu-silent/be.json" 2> /dev/null)"
  lanes cpu-garbage cuda,cpu cpu-garbage MODEL_LADDER_CONCURRENT_LANES=1; r=$?
  { [ "$r" = 0 ] && [ "$(field cpu-garbage cpu ran)" = False ] && field cpu-garbage cpu lane_error | grep -q 'no parseable'; } \
    && ok cpu-garbage || bad cpu-garbage "rc=$r be=$(head -c 300 "$T/cpu-garbage/be.json" 2> /dev/null)"
  lanes cuda-silent cuda,cpu cuda-silent MODEL_LADDER_CONCURRENT_LANES=1; r=$?
  { [ "$r" = 0 ] && [ "$(field cuda-silent cuda ran)" = False ] && field cuda-silent cuda lane_error | grep -q 'lane exit 3'; } \
    && ok cuda-silent || bad cuda-silent "rc=$r be=$(head -c 300 "$T/cuda-silent/be.json" 2> /dev/null)"
  lanes cpu-decline cuda,cpu cpu-decline MODEL_LADDER_CONCURRENT_LANES=1; r=$?
  { [ "$r" = 2 ] && [ ! -e "$T/cpu-decline/be.json" ] && grep -q '^decline: ENV planted decline in the cpu lane' "$T/cpu-decline/out"; } \
    && ok cpu-decline || bad cpu-decline "rc=$r be_written=$([ -e "$T/cpu-decline/be.json" ] && echo yes || echo no) $(head -c 200 "$T/cpu-decline/out")"
  lanes cuda-decline cuda,cpu cuda-decline MODEL_LADDER_CONCURRENT_LANES=1; r=$?
  sleep 1
  local bgp; bgp=$(cat "$T/cuda-decline/bg.pid" 2> /dev/null)
  { [ "$r" = 2 ] && [ -n "$bgp" ] && ! kill -0 "$bgp" 2> /dev/null; } && ok cuda-decline-kills-cpu-lane \
    || { bad cuda-decline-kills-cpu-lane "rc=$r; the background lane's child (pid ${bgp:-?}) outlived the ladder, or the exit waited on it (rc 124 = timed out)"; [ -n "$bgp" ] && kill "$bgp" 2> /dev/null; }
  return "$rc"
}

if [ "$SELF_TEST" = 1 ]; then
  bad=0
  mutant() { # <label> <case that must go RED> <sed expression>
    local m="$T/m-$1.sh" o
    sed "$3" "$SCRIPT" > "$m"
    if cmp -s "$SCRIPT" "$m"; then echo "  FAIL  mutant $1 did not apply -- case $2 proves nothing"; bad=1; return; fi
    o=$(run_cases "$m" 2>&1)
    if grep -q "FAIL  $2:" <<< "$o"; then printf '  ok    mutant %-20s killed by %s\n' "$1" "$2"
    else echo "  FAIL  mutant $1 SURVIVED: case $2 stayed ok"; bad=1; fi
  }
  mutant unhidden          probe-cpu   's/apr_cpu_unlocked() { CUDA_VISIBLE_DEVICES= nice/apr_cpu_unlocked() { nice/'
  mutant cpu-no-choom      probe-cpu   's/nice -n "\$CPU_LANE_NICE" choom -n 1000 -- "\$APR"/nice -n "$CPU_LANE_NICE" "$APR"/'
  mutant all-unlocked      probe-cuda  's/if \[ "\$lane" = cpu \]; then apr_cpu_unlocked/if true; then apr_cpu_unlocked/'
  mutant cell-bypass       sites       's/apr_lane "\$b" run/apr_cpu_unlocked run/'
  mutant serve-locked      sites       's/apr_lane "\$bname" serve run/apr_locked serve run/'
  mutant default-concurrent serial-default 's/"\${MODEL_LADDER_CONCURRENT_LANES:-0}" = 1/"${MODEL_LADDER_CONCURRENT_LANES:-1}" = 1/'
  mutant never-background  concurrent  's/      \[ "\$b" = cpu \] || continue/      [ "$b" = never ] || continue/'
  mutant no-red-fragment   cpu-silent  's/    if ! python3 -c \(.\)import json,sys; d = json.loads/    if false \&\& ! python3 -c \1import json,sys; d = json.loads/'
  mutant swallow-decline   cpu-decline '/grep -q .\^decline: . "\$lane_err" 2> \/dev\/null; then exit 2; fi/d'
  mutant orphan-cpu-lane   cuda-decline-kills-cpu-lane 's/      pkill -TERM -P "\$p" 2> \/dev\/null; kill -TERM "\$p" 2> \/dev\/null/      :/'
  mutant green-ignores-ran green-needs-ran 's/and all(v\["ran"\] and not v\["fallback"\]/and all(not v["fallback"]/'
  [ "$bad" = 0 ] && { echo "SELF-TEST OK: every planted regression turns a case RED"; exit 0; }
  echo "SELF-TEST FAIL: a planted regression was not caught"; exit 1
fi

echo "ladder CPU lane: unlocked, deviceless, concurrent only when opted in, and never absent from the receipt ($SCRIPT)"
if run_cases "$SCRIPT"; then echo "PASS"; exit 0; fi
echo "FAIL"; exit 1
