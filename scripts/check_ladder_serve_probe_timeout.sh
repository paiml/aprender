#!/usr/bin/env bash
# check_ladder_serve_probe_timeout.sh — one serve route that never answers must not erase the
# evidence of the others (#4126).
#
# WHY. ladder_serve_probe wrote each route as `"http":$code`, and curl's `%{http_code}` is `000`
# when no response came back. `{"http":000}` is INVALID JSON, so one timed-out route made the
# whole serve object unparseable. The caller then reported "the serve probe produced no parseable
# object ... it exited (status 1) rather than returned", and every route's record was discarded.
# Measured on lambda (qwen35-9b-q4km cpu lane, #4034 A/B arm A), and the same text is on the 27B
# cpu serve of the fc942f6be sweep.
#
# WHAT THIS CHECKS. It lifts the SHIPPED ladder_serve_probe (with the functions it calls) and runs
# it against a fake `apr serve`: /health and every generation route answer 200, except /api/chat,
# which hangs past the route timeout (LADDER_ROUTE_MAX_TIME=2).
#   parses          the probe prints ONE parseable JSON object and returns 1 (a route failed)
#   timeout-named   both /api/chat routes carry http:null and a curl_error naming the timeout
#   cut-kept        /v1/completions sends a 200 then cuts its body: http 200 is KEPT, curl_error names the
#                   failed transfer (curl exit 18), and the route is not ok
#   stall-named     /v1/chat/completions stream sends a 200 then STALLS past the timeout: http 200 kept,
#                   curl_error says "timeout after HTTP 200", never "no response"
#   others-kept     every other route is present with http 200. The evidence survives
# --self-test plants the old bare `"http":$code` and requires `parses` to turn RED.
#
# Exit: 0 all as expected · 1 a case landed wrong · 2 could not check.
set -uo pipefail
SCRIPT="scripts/model_ladder.sh"; SELF_TEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --script) [ $# -ge 2 ] || { echo "--script needs a value" >&2; exit 2; }; SCRIPT="$2"; shift 2 ;;
    --self-test) SELF_TEST=1; shift ;;
    -h|--help) awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
    *) echo "check_ladder_serve_probe_timeout: unknown argument '$1'" >&2; exit 2 ;;
  esac
done
for tool in flock choom python3 curl pgrep; do
  command -v "$tool" > /dev/null || { echo "  cannot check: '$tool' is not installed" >&2; exit 2; }
done
[ -f "$SCRIPT" ] || { echo "  cannot check: $SCRIPT not found" >&2; exit 2; }
[ -f crates/aprender-serve/src/api/router.rs ] || { echo "  cannot check: run from the repo root (the probe derives its routes from router.rs)" >&2; exit 2; }

T=$(mktemp -d -t check_ladder_serve_probe_timeout.XXXXXXXX) || { echo "  cannot check: mktemp failed" >&2; exit 2; }
cleanup() {
  pkill -f "$T/fake_serve.py" 2> /dev/null || :
  case "${T:-}" in /tmp/?*) if [ -n "$T" ] && [ -d "$T" ]; then rm -rf -- "$T" || :; fi ;; esac
}
trap cleanup EXIT

cat > "$T/fake_serve.py" <<'PY'
import http.server, json, socketserver, sys, time
port = int(sys.argv[1])
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); body_in = self.rfile.read(n)
        if self.path == "/api/chat":
            time.sleep(30)                      # never answers within the route timeout
        if self.path == "/v1/chat/completions" and b'"stream":true' in body_in:   # a 200, then the body STALLS
            self.send_response(200); self.send_header("Content-Length", "1000"); self.end_headers()
            self.wfile.write(b'data: {"x"'); self.wfile.flush(); time.sleep(30); return
        if self.path == "/v1/completions":      # a 200 status line, then the body is cut short
            self.send_response(200); self.send_header("Content-Length", "1000"); self.end_headers()
            self.wfile.write(b'{"choices":[{"text":"4"'); self.wfile.flush()
            self.close_connection = True; return
        body = json.dumps({"choices": [{"text": "4", "message": {"role": "assistant", "content": "4"}}],
                           "message": {"role": "assistant", "content": "4"}, "response": "4", "used_gpu": False})
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(body.encode())
class S(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True; daemon_threads = True
with S(("127.0.0.1", port), H) as s:
    s.serve_forever()
PY
# the fake `apr`: `apr serve run <model> --port N [flags]` becomes the fake server on N
cat > "$T/apr" <<EOF
#!/usr/bin/env bash
port=""; while [ \$# -gt 0 ]; do [ "\$1" = --port ] && port=\$2; shift; done
exec python3 "$T/fake_serve.py" "\$port"
EOF
chmod +x "$T/apr"

lift_fns() { # <src> <fn...> -> each function's text; a one-line `f() { ...; }` too
  local src="$1" fn; shift
  for fn in "$@"; do
    awk -v F="^$fn\\\\(\\\\) \\\\{" '$0 ~ F { if ($0 ~ /; }$/) { print; exit } f=1 } f{print} f && /^\}$/{exit}' "$src"
  done
}

run_cases() { # <script> -> 0 when every case lands
  local src="$1" rc=0 body out prc
  ok() { printf '  ok    %s\n' "$1"; }
  bad() { printf '  FAIL  %s: %s\n' "$1" "$2"; rc=1; }
  body=$(lift_fns "$src" apr_locked apr_cpu_unlocked apr_lane lock_timeout serve_log_tail json_str_or_null \
         gibberish_reason serve_reply_text serve_route_bad assistant_reply ladder_tree_jiffies \
         ladder_tree_waits_on_lock ladder_serve_wait_health ladder_td_verdict ladder_serve_teardown ladder_serve_probe)
  grep -q '^ladder_serve_probe() {' <<< "$body" || { bad lift "$src defines no ladder_serve_probe()"; return 1; }
  mkdir -p "$T/work"; : > "$T/lock"
  out=$(GPU_LOCK="$T/lock" LOCK_WAIT=10 LOCK_BUSY=75 APR="$T/apr" WORK="$T/work" CPU_LANE_NICE=10 \
        SERVE_STALL_S=10 SERVE_CEILING_S=30 LADDER_ROUTE_MAX_TIME=2 \
        timeout 120 bash -c "$body"$'\n''ladder_serve_probe /fake.gguf "" r1 cpu' 2> "$T/probe.err"); prc=$?
  printf '%s' "$out" > "$T/probe.json"
  if [ "$prc" = 1 ] && python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$T/probe.json" 2> /dev/null; then ok parses
  else bad parses "rc=$prc (want 1), and the output must be ONE parseable object: $(head -c 240 "$T/probe.json") $(tail -c 200 "$T/probe.err")"; fi
  out=$(python3 - "$T/probe.json" <<'PY' 2>&1
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print("unparseable: %s" % e); sys.exit(0)
r = d.get("routes") or {}
chat = [k for k in r if k.startswith("/api/chat|")]
cut = [k for k in r if k.startswith("/v1/completions|")]
cb = [k for k in cut if not (r[k].get("http") == 200 and "curl exit 18" in str(r[k].get("curl_error") or "") and r[k].get("ok") is False)]
print("CUT:%d:%s" % (len(cut), "; ".join("%s=%s" % (k, r[k]) for k in cb)))
bad = []
if len(chat) != 2:
    bad.append("want both /api/chat routes recorded, got %s" % chat)
for k in chat:
    x = r[k]
    if x.get("http") is not None or "timeout" not in str(x.get("curl_error") or ""):
        bad.append("%s: want http null + a timeout curl_error, got %s" % (k, x))
print("TIMEOUT:" + "; ".join(bad))
st = r.get("/v1/chat/completions|stream=true") or {}
print("STALL:" + ("" if (st.get("http") == 200 and "timeout after HTTP 200" in str(st.get("curl_error") or "") and st.get("ok") is False) else str(st)))
others = [k for k in r if not k.startswith("/api/chat|") and not k.startswith("/v1/completions|") and k != "/v1/chat/completions|stream=true"]
ob = [k for k in others if r[k].get("http") != 200]
print("OTHERS:%d:%s" % (len(others), ",".join(ob)))
PY
)
  if grep -q '^TIMEOUT:$' <<< "$out"; then ok timeout-named; else bad timeout-named "$(grep -m1 -E '^(TIMEOUT|unparseable)' <<< "$out")"; fi
  if grep -q '^STALL:$' <<< "$out"; then ok stall-named; else bad stall-named "$(grep -m1 -E '^(STALL|unparseable)' <<< "$out")"; fi
  if grep -qE '^CUT:[1-9][0-9]*:$' <<< "$out"; then ok cut-kept; else bad cut-kept "$(grep -m1 -E '^(CUT|unparseable)' <<< "$out")"; fi
  if grep -qE '^OTHERS:[1-9][0-9]*:$' <<< "$out"; then ok "others-kept ($(grep -oE '^OTHERS:[0-9]+' <<< "$out" | cut -d: -f2) routes with http 200)"
  else bad others-kept "$(grep -m1 -E '^(OTHERS|unparseable)' <<< "$out")"; fi
  pkill -f "$T/fake_serve.py" 2> /dev/null || :
  return "$rc"
}

if [ "$SELF_TEST" = 1 ]; then
  m="$T/m-bare-code.sh"
  sed 's/{\\"http\\":\$http_json,\\"curl_error\\":\$cerr_json,/{\\"http\\":$code,/' "$SCRIPT" > "$m"
  if cmp -s "$SCRIPT" "$m"; then echo "  FAIL  mutant bare-code did not apply -- the check proves nothing"; exit 1; fi
  o=$(run_cases "$m" 2>&1) || true
  if grep -q 'FAIL  parses' <<< "$o"; then echo "  ok    mutant bare-code killed by parses"; echo "SELF-TEST OK"; exit 0; fi
  printf '%s\n' "$o"; echo "SELF-TEST FAIL: the bare \"http\":\$code plant left parses green"; exit 1
fi
echo "ladder serve probe: a route with no response keeps the record parseable and the other routes' evidence ($SCRIPT)"
if run_cases "$SCRIPT"; then echo "PASS"; exit 0; fi
echo "FAIL"; exit 1
