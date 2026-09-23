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

LOCK="$TMP/gpu.lock"
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

  return "$fails"
}

body=$(extract_teardown "$SCRIPT") || exit 2

if [ "$SELF_TEST" -eq 1 ]; then
  # Plant the regression: discard the tree right after it is resolved, which is the old
  # port-only behaviour. The `loading` case must catch it.
  planted=$(awk '{print} /^    frontier=\("\$pid"\)$/{p=1} p && /^    done$/ && !d{print "    tree=()  # PLANTED: self-test regression"; d=1}' <<< "$body")
  grep -q 'PLANTED' <<< "$planted" || { echo "  self-test: could not plant the regression — the anchor moved" >&2; exit 2; }
  if run_cases "$planted"; then
    echo "SELF-TEST FAIL: the planted regression passed — this check cannot see the defect"; exit 1
  fi
  echo "SELF-TEST OK: the planted regression turned this RED"; exit 0
fi

if run_cases "$body"; then echo "PASS: ladder_serve_teardown never reports clean while a launched process lives"; exit 0; fi
echo "FAIL"; exit 1
