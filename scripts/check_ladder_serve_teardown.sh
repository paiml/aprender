#!/usr/bin/env bash
# guard-tree: serial
# (guard_tree.sh runs this guard after the pool, alone: its rows assert a server is ready within 10 s; under the 8-way pool on a clean-room runner they miss the window (#4046))
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
# port-only behaviour) and requires `loading` to turn this RED.
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
  # Anti-vacuity: an extraction that lost the function tests nothing.
  if ! grep -q 'printf .clean.' <<< "$body" || ! grep -q 'kill' <<< "$body"; then
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
fake_locked() { flock -w 5 "$LOCK" python3 "$TMP/fake_server.py" "$@"; }

wait_health() { local port="$1" i; for i in $(seq 1 50); do curl -fsS --max-time 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && return 0; sleep 0.2; done; return 1; }
wait_pidfile() { local f="$1" i; for i in $(seq 1 50); do [ -s "$f" ] && return 0; sleep 0.1; done; return 1; }
alive() { kill -0 "$1" 2>/dev/null; }

run_cases() { # <teardown-function-body> -> 0 if every case lands, 1 otherwise
  local body="$1" fails=0 port pid spid td fpid
  # bashrs SEC001: evals a function body extracted by awk from this repo's own scripts/model_ladder.sh, so the real function is under test; no external input.
  # bashrs disable-next-line=SEC001
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
  # bashrs SEC001: evals function bodies extracted by awk from this repo's own scripts/model_ladder.sh; no external input.
  # bashrs disable-next-line=SEC001
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
  echo "SELF-TEST OK: both planted regressions turned this RED"; exit 0
fi

rc=0
run_cases "$body" || rc=1
run_wait_cases "$wbody" || rc=1
if [ "$rc" -eq 0 ]; then echo "PASS: teardown never reports clean while a launched process lives; the health wait ends on the server's state, not a clock"; exit 0; fi
echo "FAIL"; exit 1
