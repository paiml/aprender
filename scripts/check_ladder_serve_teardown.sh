#!/usr/bin/env bash
# check_ladder_serve_teardown.sh — `ladder_serve_teardown` must never print `clean`
# while a process it launched is alive (#3943).
#
# WHY. The teardown used to infer liveness from the PORT: kill the wrapper, and if
# /health does not answer, print `clean`. The health-wait timeout calls it for a
# server that never answered /health — a server still LOADING has not bound its port,
# so it "was gone" on the first poll while it kept loading. gx10's qwen35-27b-q4km log
# shows the server reach "listening" after the probe had given up and declared
# `clean`. Killing flock releases the GPU LOCK; it does not stop `apr serve`, which is
# re-parented to init and keeps its VRAM.
#
# WHAT THIS CHECKS, AND WHY IT IS NOT A GREP. It extracts the SHIPPED function from
# model_ladder.sh and runs it against real processes arranged the way the ladder
# arranges them — a backgrounded function whose `$!` is the wrapper, flock under it,
# the server under that — and asserts BOTH the printed verdict AND what is actually
# still running afterwards. A verdict test alone is the defect: the old code printed a
# plausible verdict.
#
#   loading    server sleeps before binding, teardown called while it loads
#              -> must be `escalated`, and the server must be DEAD
#   listening  server bound and answering /health        -> `escalated`, DEAD
#   exits      $! is the server itself, dies with the kill -> `clean`, DEAD
#   foreign    a listener on the port that is NOT in the tree, started from another
#              cwd -> `failed`, and it must be left ALIVE (never signal what is not ours)
#   symlink-cwd  `loading`, with the ladder's cwd reached through a SYMLINK (#4055): gx10 and
#              yoga spell it /mnt/nvme-raid0 -> /home/noah/eph-work/intel-mirror. The cwd check
#              compared the physical /proc cwd with the logical $PWD and refused to kill its
#              own survivor -> must be `escalated`, the server DEAD, and the lock FREE
#   lock-escapee  the server double-forks a child that keeps the inherited GPU-lock fd and
#              escapes the tree (#4055) -> must be `failed`, never `escalated`: the fleet lock
#              is still held by something this teardown launched
#
# THE HEALTH WAIT (#3943 b) is checked the same way: the shipped
# `ladder_serve_wait_health` is extracted and run against a fake loader.
#   slow-log   binds after longer than the stall window, log advancing   -> ready
#   slow-cpu   same, but SILENT: only its CPU time advances              -> ready
#              (the 27B log is three lines; a log-only rule would kill a healthy load)
#   hung       alive, no log, no CPU                                      -> stalled
#   dies       exits mid-load                  -> died, BEFORE the stall window
#   forever    advancing but never binds                                  -> ceiling
#
# --self-test plants the regression (the parentage tree is discarded, which is the old
# port-only behaviour) and requires `loading` to turn this RED. It also plants the two
# #4055 regressions: the logical $PWD instead of `pwd -P` must turn `symlink-cwd` RED, and a
# verdict that ignores the lock must turn `lock-escapee` RED.
#
# Exit: 0 all cases as expected · 1 a case landed wrong · 2 could not check.
#       --self-test: 0 when the planted regression turns this RED, 1 otherwise.
set -euo pipefail

SCRIPT="scripts/model_ladder.sh"
SELF_TEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --script) [ $# -ge 2 ] || { echo "--script needs a value" >&2; exit 2; }; SCRIPT="$2"; shift 2 ;;
    --self-test) SELF_TEST=1; shift ;;
    -h|--help) awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
    *) echo "check_ladder_serve_teardown: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

for tool in flock python3 curl pgrep ss; do
  command -v "$tool" >/dev/null 2>&1 || { echo "  cannot check: '$tool' is not installed" >&2; exit 2; }
done

extract_teardown() {
  local src="$1" body
  [ -f "$src" ] || { echo "  cannot read $src" >&2; return 2; }
  body=$(awk '/^ladder_serve_teardown\(\) \{/{f=1} f{print} f && /^\}$/{exit}' "$src")
  body="$body"$'\n'"$(awk '/^ladder_td_verdict\(\) \{/{f=1} f{print} f && /^\}$/{exit}' "$src")"
  # Anti-vacuity: an extraction that lost the function tests nothing.
  if ! grep -q 'ladder_td_verdict clean' <<< "$body" || ! grep -q 'kill' <<< "$body" \
     || ! grep -q '^ladder_td_verdict() {' <<< "$body"; then
    echo "  the extracted teardown does not print 'clean' or signal anything — this check no longer knows what it is running" >&2
    return 2
  fi
  printf '%s\n' "$body"
}

TMP=$(mktemp -d)
cleanup() { pkill -f "$TMP/fake_server.py" 2>/dev/null || true; rm -rf "$TMP"; }
trap cleanup EXIT

cat > "$TMP/fake_server.py" <<'PY'
import http.server, os, socketserver, sys, time
port, load, pidfile = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
open(pidfile, "w").write(str(os.getpid()))
time.sleep(load)  # "loading the model": the port is not bound yet
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def log_message(self, *a): pass
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("127.0.0.1", port), H) as s:
    s.serve_forever()
PY

cat > "$TMP/fake_loader.py" <<'PY'
import http.server, os, socketserver, sys, time
port, mode, secs = int(sys.argv[1]), sys.argv[2], float(sys.argv[3])
t0 = time.time()
while time.time() - t0 < secs or mode == "forever":
    if mode in ("slow-log", "forever"):
        print("loading layer", flush=True); time.sleep(1)
    elif mode == "slow-cpu":
        sum(i * i for i in range(200000))          # busy: CPU time advances, log does not
    elif mode == "hung":
        time.sleep(1)                              # alive, silent, idle
    elif mode == "dies":
        time.sleep(secs); sys.exit(3)
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def log_message(self, *a): pass
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("127.0.0.1", port), H) as srv:
    srv.serve_forever()
PY

LOCK="$TMP/gpu.lock"
GPU_LOCK="$LOCK"   # the teardown's verdict reads the fleet lock by this name (#4055)
mkdir -p "$TMP/real" && ln -s "$TMP/real" "$TMP/link"
lock_free() { flock -n "$LOCK" true; }
fake_locked() { flock -w 5 "$LOCK" python3 "$TMP/fake_server.py" "$@"; }

wait_health() { local port="$1" i; for i in $(seq 1 50); do curl -fsS --max-time 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && return 0; sleep 0.2; done; return 1; }
wait_pidfile() { local f="$1" i; for i in $(seq 1 50); do [ -s "$f" ] && return 0; sleep 0.1; done; return 1; }
alive() { kill -0 "$1" 2>/dev/null; }

run_cases() { # <teardown-function-body> -> 0 if every case lands, 1 otherwise
  local body="$1" fails=0 port pid spid td fpid
  eval "$body"

  # loading
  port=$(( 20000 + (RANDOM % 20000) ))
  fake_locked "$port" 30 "$TMP/load.pid" & pid=$!
  wait_pidfile "$TMP/load.pid" || { echo "  loading: fake server never started" >&2; return 2; }
  spid=$(cat "$TMP/load.pid")
  td=$(ladder_serve_teardown "$pid" "$port") || true
  if [ "$td" != "escalated" ] || alive "$spid"; then
    echo "  FAIL loading: teardown='$td', server alive=$(alive "$spid" && echo yes || echo no) — want 'escalated' and dead"; fails=1
  else echo "  ok   loading: '$td', server dead"; fi
  kill -9 "$spid" 2>/dev/null || true

  # listening
  port=$(( 20000 + (RANDOM % 20000) ))
  fake_locked "$port" 0 "$TMP/listen.pid" & pid=$!
  wait_pidfile "$TMP/listen.pid" && wait_health "$port" || { echo "  listening: fake server never answered" >&2; return 2; }
  spid=$(cat "$TMP/listen.pid")
  td=$(ladder_serve_teardown "$pid" "$port") || true
  if [ "$td" != "escalated" ] || alive "$spid"; then
    echo "  FAIL listening: teardown='$td', server alive=$(alive "$spid" && echo yes || echo no) — want 'escalated' and dead"; fails=1
  else echo "  ok   listening: '$td', server dead"; fi
  kill -9 "$spid" 2>/dev/null || true

  # exits
  port=$(( 20000 + (RANDOM % 20000) ))
  python3 "$TMP/fake_server.py" "$port" 0 "$TMP/exit.pid" & pid=$!
  wait_health "$port" || { echo "  exits: fake server never answered" >&2; return 2; }
  td=$(ladder_serve_teardown "$pid" "$port") || true
  if [ "$td" != "clean" ] || alive "$pid"; then
    echo "  FAIL exits: teardown='$td', server alive=$(alive "$pid" && echo yes || echo no) — want 'clean' and dead"; fails=1
  else echo "  ok   exits: '$td', server dead"; fi

  # foreign
  port=$(( 20000 + (RANDOM % 20000) ))
  (cd /tmp && exec python3 "$TMP/fake_server.py" "$port" 0 "$TMP/foreign.pid") & fpid=$!
  wait_health "$port" || { echo "  foreign: listener never answered" >&2; return 2; }
  sleep 30 & pid=$!
  td=$(ladder_serve_teardown "$pid" "$port") || true
  if [ "$td" != "failed" ] || ! alive "$fpid"; then
    echo "  FAIL foreign: teardown='$td', foreign alive=$(alive "$fpid" && echo yes || echo no) — want 'failed' and the foreign listener untouched"; fails=1
  else echo "  ok   foreign: '$td', foreign listener untouched"; fi
  kill -9 "$fpid" 2>/dev/null || true

  # symlink-cwd (#4055): the ladder's shell sits in a SYMLINKED dir; the server inherits it
  port=$(( 20000 + (RANDOM % 20000) ))
  rm -f "$TMP/sym.pid" "$TMP/sym.out"
  ( cd "$TMP/link" || exit 2
    fake_locked "$port" 30 "$TMP/sym.pid" & pid=$!
    wait_pidfile "$TMP/sym.pid" || exit 2
    td=$(ladder_serve_teardown "$pid" "$port") || true
    printf '%s' "$td" > "$TMP/sym.out" )
  spid=$(cat "$TMP/sym.pid" 2>/dev/null); td=$(cat "$TMP/sym.out" 2>/dev/null)
  if [ -z "$spid" ]; then echo "  symlink-cwd: fake server never started" >&2; return 2; fi
  if [ "$td" != "escalated" ] || alive "$spid" || ! lock_free; then
    echo "  FAIL symlink-cwd: teardown='$td', server alive=$(alive "$spid" && echo yes || echo no), lock free=$(lock_free && echo yes || echo no) — want 'escalated', dead, free"; fails=1
  else echo "  ok   symlink-cwd: '$td', server dead, lock free"; fi
  kill -9 "$spid" 2>/dev/null || true

  # lock-escapee (#4055): a double-forked child keeps the inherited lock fd, outside the tree
  port=$(( 20000 + (RANDOM % 20000) ))
  rm -f "$TMP/esc.pid" "$TMP/escsrv.pid"
  flock -w 5 "$LOCK" sh -c '(setsid sh -c "echo \$\$ > \"$1\"; exec sleep 300" > /dev/null 2>&1 &); exec python3 "$2" "$3" 0 "$4"' \
    _ "$TMP/esc.pid" "$TMP/fake_server.py" "$port" "$TMP/escsrv.pid" & pid=$!
  wait_pidfile "$TMP/escsrv.pid" && wait_pidfile "$TMP/esc.pid" && wait_health "$port" || { echo "  lock-escapee: fixture never started" >&2; return 2; }
  spid=$(cat "$TMP/escsrv.pid"); fpid=$(cat "$TMP/esc.pid")
  td=$(ladder_serve_teardown "$pid" "$port") || true
  if [ "$td" != "failed" ] || alive "$spid"; then
    echo "  FAIL lock-escapee: teardown='$td', server alive=$(alive "$spid" && echo yes || echo no), lock free=$(lock_free && echo yes || echo no) — want 'failed' (an escapee of ours holds the GPU lock) and the server dead"; fails=1
  else echo "  ok   lock-escapee: '$td' — the escaped child still holds the lock, so the teardown refuses to call it done"; fi
  kill -9 "$spid" "$fpid" 2>/dev/null || true

  return "$fails"
}

extract_wait() {
  local src="$1" body
  body=$(awk '/^ladder_tree_jiffies\(\) \{/{f=1} /^ladder_serve_wait_health\(\) \{/{f=1} f{print} f && /^\}$/{f=0}' "$src")
  if ! grep -q 'printf .ready' <<< "$body" || ! grep -q 'ladder_tree_jiffies()' <<< "$body"; then
    echo "  the extracted health wait is missing its verdicts or its progress probe — this check no longer knows what it is running" >&2
    return 2
  fi
  printf '%s\n' "$body"
}

wait_case() { # <name> <mode> <secs> <stall> <ceiling> <want-verdict> <max-seconds>
  local name="$1" mode="$2" secs="$3" stall="$4" ceil="$5" want="$6" maxs="$7" port pid out verdict took
  port=$(( 20000 + (RANDOM % 20000) ))
  flock -w 5 "$LOCK" python3 "$TMP/fake_loader.py" "$port" "$mode" "$secs" > "$TMP/$name.log" 2>&1 & pid=$!
  out=$(ladder_serve_wait_health "$pid" "$port" "$TMP/$name.log" "$stall" "$ceil") || true
  verdict=${out%% *}; took=${out##* }
  pkill -9 -f "$TMP/fake_loader.py $port " 2>/dev/null || true; kill -9 "$pid" 2>/dev/null || true
  if [ "$verdict" != "$want" ] || [ "$took" -gt "$maxs" ]; then
    echo "  FAIL $name: '$out' — want '$want' within ${maxs}s"; return 1
  fi
  echo "  ok   $name: '$out'"
}

run_wait_cases() { # <wait-function-bodies> -> 0 if every case lands
  local body="$1" fails=0
  eval "$body"
  wait_case slow-log slow-log 6 3 30 ready 10    || fails=1
  wait_case slow-cpu slow-cpu 6 3 30 ready 10    || fails=1
  wait_case hung     hung     60 3 30 stalled 6  || fails=1
  wait_case dies     dies     2 5 30 died 4      || fails=1
  wait_case forever  forever  0 3 6 ceiling 9    || fails=1
  return "$fails"
}

body=$(extract_teardown "$SCRIPT") || exit 2
wbody=$(extract_wait "$SCRIPT") || exit 2

if [ "$SELF_TEST" -eq 1 ]; then
  # Plant the regression: discard the tree right after it is resolved, which is the old
  # port-only behaviour. The `loading` case must catch it.
  planted=$(awk '{print} /^    frontier=\("\$pid"\)$/{p=1} p && /^    done$/ && !d{print "    tree=()  # PLANTED: self-test regression"; d=1}' <<< "$body")
  grep -q 'PLANTED' <<< "$planted" || { echo "  self-test: could not plant the regression — the anchor moved" >&2; exit 2; }
  if run_cases "$planted"; then
    echo "SELF-TEST FAIL: the planted teardown regression passed — this check cannot see the defect"; exit 1
  fi
  # Second plant: progress judged by the LOG ONLY. A silent-but-busy load must then be
  # called hung, which is exactly why the CPU leg exists.
  wplanted=$(sed 's/ || \[ "\$jif" != "\$last_jif" \]; then/; then  # PLANTED: log-only progress/' <<< "$wbody")
  grep -q 'PLANTED' <<< "$wplanted" || { echo "  self-test: could not plant the log-only regression — the anchor moved" >&2; exit 2; }
  # The plant must be VALID code, or a syntax error "turns this RED" for the wrong reason
  # (it did, on the first attempt: every case printed '' because nothing was defined).
  bash -n <(printf '%s\n' "$wplanted") || { echo "  self-test: the planted regression does not parse — a syntax error is not a regression" >&2; exit 2; }
  wout=$(run_wait_cases "$wplanted" 2>&1) || true
  printf '%s\n' "$wout"
  if ! grep -q "FAIL slow-cpu" <<< "$wout" || [ "$(grep -c '^  FAIL' <<< "$wout")" -ne 1 ]; then
    echo "SELF-TEST FAIL: the log-only plant must turn EXACTLY slow-cpu red — anything else means the check is not measuring the CPU leg"; exit 1
  fi
  # The #4055 plants: each must turn ITS case red.
  p3=$(sed 's/^    here=\$(pwd -P)$/    here=$PWD  # PLANTED: the logical cwd/' <<< "$body")
  grep -q 'PLANTED' <<< "$p3" || { echo "  self-test: could not plant the logical-cwd regression — the anchor moved" >&2; exit 2; }
  o3=$(run_cases "$p3" 2>&1) || true
  grep -q '^  FAIL symlink-cwd' <<< "$o3" || { printf '%s\n' "$o3"; echo "SELF-TEST FAIL: the logical \$PWD plant left symlink-cwd green"; exit 1; }
  p4=$(sed 's/^    if \[ -n "\${GPU_LOCK:-}" \] && ino=/    if false \&\& ino=/' <<< "$body")
  grep -q 'if false && ino=' <<< "$p4" || { echo "  self-test: could not plant the lock-blind verdict — the anchor moved" >&2; exit 2; }
  o4=$(run_cases "$p4" 2>&1) || true
  grep -q '^  FAIL lock-escapee' <<< "$o4" || { printf '%s\n' "$o4"; echo "SELF-TEST FAIL: a lock-blind verdict left lock-escapee green"; exit 1; }
  echo "SELF-TEST OK: all four planted regressions turned this RED"; exit 0
fi

rc=0
run_cases "$body" || rc=1
run_wait_cases "$wbody" || rc=1
if [ "$rc" -eq 0 ]; then echo "PASS: teardown never reports clean while a launched process lives; the health wait ends on the server's state, not a clock"; exit 0; fi
echo "FAIL"; exit 1
