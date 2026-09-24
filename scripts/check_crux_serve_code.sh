#!/usr/bin/env bash
# check_crux_serve_code.sh: the case table for the CRUX serve and code producers (#3962):
# scripts/lib/crux_serve_routes.py (every route apr serve mounts, both modes) and
# scripts/lib/crux_apr_code.py (`apr code -p`). Hermetic: the server is
# scripts/lib/crux_fake_serve.py and `apr` is a fake script, so it needs no model and
# no GPU.
#
# WHY A TABLE. A stream judged on its status alone passed half the serve routes
# (#3957 F4c). A route derived through a hand-written filter left /api/chat unprobed
# (#3715). Each row below plants one of those failures and requires the RED it names.
# A row that passes on the correct code proves nothing by itself, so each must-RED row
# is also run against a MUTANT of the code that deletes the check it relies on, and
# the row must FAIL there (a mutant must be killed). A case the mutant survives is a
# case that cannot go RED.
#
# Exit: 0 every row behaved and every mutant was killed · 1 a row or mutant broke · 2 ENV.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd) || exit 2
PROG=check_crux_serve_code
command -v python3 > /dev/null 2>&1 || { printf '%s: ENV - python3 is missing\n' "$PROG" >&2; exit 2; }
ROUTES_PY="$ROOT/scripts/lib/crux_serve_routes.py"
CODE_PY="$ROOT/scripts/lib/crux_apr_code.py"
FAKE="$ROOT/scripts/lib/crux_fake_serve.py"
for f in "$ROUTES_PY" "$CODE_PY" "$FAKE"; do
  [ -f "$f" ] || { printf '%s: ENV - %s not found\n' "$PROG" "$f" >&2; exit 2; }
done

TMP=$(mktemp -d) || exit 2
SRV=""
_cleanup() {
  [ -n "$SRV" ] && kill "$SRV" 2> /dev/null
  case "${TMP:-}" in
    /tmp/?*|/var/folders/?*) rm -rf -- "$TMP" || : ;;
    *) : ;;
  esac
}
trap _cleanup EXIT

FAILS=0
ROWS=0
ok() { ROWS=$((ROWS + 1)); printf '  ok   %s\n' "$1"; }
bad() { ROWS=$((ROWS + 1)); FAILS=$((FAILS + 1)); printf '  FAIL %s\n' "$1"; }

free_port() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
start() { # start <index> <fault>: (re)start the fixture server; sets URL
  [ -n "$SRV" ] && { kill "$SRV" 2> /dev/null; wait "$SRV" 2> /dev/null; }
  local port i
  port=$(free_port)
  python3 "$FAKE" --port "$port" --index "$1" --fault "$2" > /dev/null 2>&1 &
  SRV=$!
  URL="http://127.0.0.1:$port"
  for i in $(seq 1 50); do curl -sf "$URL/health" > /dev/null 2>&1 && return 0; sleep 0.1; done
  printf '%s: ENV - fixture server did not come up\n' "$PROG" >&2
  exit 2
}

# The prompts: one single-turn, one two-turn (the state_recall shape).
printf '%s' '{"id":"p1","messages":[{"role":"user","content":"What is 2+2? Answer in <answer></answer>."}]}' > "$TMP/prompt-p1.json"
printf '%s' '{"id":"p2","messages":[{"role":"user","content":"Remember 4."},{"role":"user","content":"What did I ask you to remember?"}]}' > "$TMP/prompt-p2.json"
printf '64' > "$TMP/maxtok-p1.txt"
printf '64' > "$TMP/maxtok-p2.txt"

# field <json file> <python expr on d>: print a value from a driver's output JSON
field() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$1" "$2" 2> /dev/null; }

# drive_case <label> <py> <route> <mode> <prompt> <want rc> <fault prefix|-> [extra drive args...]
drive_case() {
  local label="$1" py="$2" route="$3" mode="$4" pid="$5" want="$6" fp="$7" rc got
  shift 7
  python3 "$py" drive --url "$URL" --route "$route" --mode "$mode" --prompt-file "$TMP/prompt-$pid.json" \
    --max-tokens 64 --out "$TMP/out.json" --timeout 20 "$@" > /dev/null 2>&1
  rc=$?
  got=$(field "$TMP/out.json" 'd.get("protocol_fault") or ""')
  if [ "$rc" != "$want" ]; then CASE_WHY="rc $rc, wanted $want (fault: ${got:-none})"; return 1; fi
  if [ "$fp" != - ]; then
    case "$got" in "$fp"*) ;; *) CASE_WHY="fault '$got', wanted '$fp...'"; return 1 ;; esac
    [ "$(field "$TMP/out.json" 'd.get("text")')" = None ] || { CASE_WHY="a faulted cell carried text"; return 1; }
  fi
  return 0
}
row() { # row <label> drive_case-args...
  local label="$1"
  shift
  CASE_WHY=""
  if drive_case "$label" "$@"; then ok "$label"; else bad "$label: $CASE_WHY"; fi
}

printf -- '--- %s: serve routes (drive) ---\n' "$PROG"
start normal ok
row "S1 chat non-stream answers" "$ROUTES_PY" "POST /v1/chat/completions" nonstream p1 0 -
row "S2 chat stream answers ([DONE] + finish_reason)" "$ROUTES_PY" "POST /v1/chat/completions" stream p1 0 -
row "S3 /generate with no renderer is REFUSED, never sent unrendered text" "$ROUTES_PY" "POST /generate" nonstream p1 4 -
row "S4 /generate rendered answers" "$ROUTES_PY" "POST /generate" nonstream p1 0 - --render-url "$URL"
if [ "$(field "$TMP/out.json" 'len(d["reported"]["rendered_prompts"])')" = 1 ]; then ok "S4b the rendered prompt's sha256 is recorded"
else bad "S4b the rendered prompt's sha256 is recorded"; fi
row "S5 /stream/generate (event: done) answers" "$ROUTES_PY" "POST /stream/generate" stream p1 0 - --render-url "$URL"
row "S6 /api/chat NDJSON stream answers" "$ROUTES_PY" "POST /api/chat" stream p1 0 -
row "S7 two-turn prompt answers turn by turn" "$ROUTES_PY" "POST /api/chat" nonstream p2 0 -
if [ "$(field "$TMP/out.json" 'len(d.get("turns") or [])')" = 2 ]; then ok "S7b both turns recorded"; else bad "S7b both turns recorded"; fi
row "S7c MUST-RED a multi-turn prompt on /api/generate is REFUSED, never asked as its last turn" "$ROUTES_PY" "POST /api/generate" nonstream p2 4 -
row "S7d a one-turn prompt on /api/generate is still asked" "$ROUTES_PY" "POST /api/generate" nonstream p1 0 -
row "S8 a mode the wire lacks is REFUSED" "$ROUTES_PY" "POST /generate" stream p1 4 - --render-url "$URL"
start normal no_done
row "S9  MUST-RED OpenAI SSE with no [DONE]" "$ROUTES_PY" "POST /v1/chat/completions" stream p1 3 stream_truncated
row "S10 MUST-RED realizar SSE with no event: done" "$ROUTES_PY" "POST /stream/generate" stream p1 3 stream_truncated --render-url "$URL"
row "S11 MUST-RED NDJSON with no done:true" "$ROUTES_PY" "POST /api/chat" stream p1 3 stream_truncated
start normal zero_deltas
row "S12 MUST-RED a stream with zero content deltas" "$ROUTES_PY" "POST /v1/chat/completions" stream p1 3 stream_zero_deltas
start normal no_finish
row "S13 MUST-RED OpenAI SSE with no finish_reason" "$ROUTES_PY" "POST /v1/chat/completions" stream p1 3 stream_no_finish
start normal empty_text
row "S14 MUST-RED 200 with empty text" "$ROUTES_PY" "POST /v1/chat/completions" nonstream p1 3 empty_text
start normal http_500
row "S15 MUST-RED HTTP 500" "$ROUTES_PY" "POST /v1/chat/completions" nonstream p1 3 http_500
start normal bad_json
row "S16 MUST-RED an unparseable body" "$ROUTES_PY" "POST /v1/chat/completions" nonstream p1 3 unparseable

printf -- '--- %s: the route universe (plan / sweep / rows) ---\n' "$PROG"
printf '%s\n' '["p1", ["serve run", "serve stream"]]' > "$TMP/list.jsonl"
plan_case() { # <label> <index> <want rc> <python assertion on the plan>
  start "$2" ok
  python3 "$ROUTES_PY" plan --url "$URL" > "$TMP/plan.json" 2> /dev/null
  local rc=$?
  if [ "$rc" = "$3" ] && python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert eval(sys.argv[2])' "$TMP/plan.json" "$4" 2> /dev/null
  then ok "$1"; else bad "$1 (rc $rc: $(head -c 200 "$TMP/plan.json"))"; fi
}
plan_case "P1 the fixture's index classifies fully" normal 0 'not d["unclassified"] and len(d["generation"]) == 7'
plan_case "P2 MUST-RED a route no table knows is unclassified" extra 1 'd["unclassified"] == ["POST /v1/brand-new"]'
plan_case "P3 MUST-RED a router with no index" none 1 'd["no_route_index"]'
# #3962 B4: every generation route has an ORACLE route (the comparator route that asks its question in
# the same representation), and the oracle's wire carries every mode the apr route is driven in.
oracle_assert() { # <routes.py> <python assertion; R is that file loaded as a module>: rc 0 iff it holds
  python3 -c 'import importlib.util,subprocess,sys
spec = importlib.util.spec_from_file_location("R", sys.argv[1]); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
assert eval(sys.argv[2])' "$1" "$2" 2> /dev/null
}
O1='all(R.oracle_route(r) in R.GENERATION and set(g["modes"]) <= set(R.GENERATION[R.oracle_route(r)]["modes"]) for r, g in R.GENERATION.items())'
O2='subprocess.run([sys.executable, sys.argv[1], "oracle-routes"], capture_output=True, text=True).stdout.strip().split(",") == sorted(set(R.ORACLE_ROUTE_BY_KIND.values()))'
O3='R.GENERATION.update({"POST /v1/brand-new": {"kind": "brand_new", "modes": ("nonstream",)}}) or R.oracle_route("POST /v1/brand-new") is None'
if oracle_assert "$ROUTES_PY" "$O1"; then ok "O1 every generation route maps to an oracle route whose wire has its modes"; else bad "O1 oracle map"; fi
if oracle_assert "$ROUTES_PY" "$O2"; then ok "O2 oracle-routes prints exactly the mapped set"; else bad "O2 oracle-routes"; fi
if oracle_assert "$ROUTES_PY" "$O3"; then ok "O3 MUST-RED a route of an unmapped kind has no oracle"; else bad "O3 unmapped kind"; fi

sweep_rows() { # <index> <out dir> [cell-why]: sweep + rows into $TMP/<dir>/manifest.jsonl
  start "$1" ok
  mkdir -p "$TMP/$2"
  : > "$TMP/$2/manifest.jsonl"
  [ "${3:-}" = skip ] || python3 "$ROUTES_PY" sweep --url "$URL" --prompt-list "$TMP/list.jsonl" --prompt-dir "$TMP" \
    --out-dir "$TMP/$2" --render-url "$URL" --timeout 20 > /dev/null 2>&1
  python3 "$ROUTES_PY" rows --out-dir "$TMP/$2" --prompt-list "$TMP/list.jsonl" --manifest "$TMP/$2/manifest.jsonl" \
    --engine apr --sha abc --host h --backend gpu --cell-why "${4:-}" > /dev/null 2>&1
}
manifest_assert() { # <label> <dir> <python assertion over `gen` (gen rows) and `ext` (other rows)>
  if python3 - "$TMP/$2/manifest.jsonl" "$3" <<'PY' 2> /dev/null
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
gen = [r for r in rows if r["kind"] == "gen"]
ext = [r for r in rows if r["kind"] != "gen"]
def out(r):
    return json.load(open(r["stdout"])) if r.get("stdout") else {}
assert eval(sys.argv[2])
PY
  then ok "$1"; else bad "$1"; fi
}
sweep_rows normal ok
manifest_assert "R1 every generation route x mode is a row, all answered" ok \
  'len(gen) == 11 and all(r["rc"] == 0 and out(r)["text"] for r in gen) and {r["verb"] for r in gen} == {"serve run", "serve stream"}'
manifest_assert "R1b the route universe is recorded" ok 'ext and ext[0]["kind"] == "serve_routes" and not ext[0]["unclassified"]'
# aprender-83 (freeze sweep): a multi-turn prompt is never driven on a single-prompt wire, and the
# plan says so -- a skipped pair is recorded, never silently absent.
cp "$TMP/list.jsonl" "$TMP/list.bak"; printf '%s\n' '["p2", ["serve run", "serve stream"]]' > "$TMP/list.jsonl"
sweep_rows normal multiturn
if python3 -c 'import json,sys; p=json.load(open(sys.argv[1]))
na = {(x["route"], x["prompt_id"]) for x in p["not_applicable"]}
assert ("POST /api/generate", "p2") in na and not any(c["route"] == "POST /api/generate" for c in p["cells"])
assert any(c["route"] == "POST /api/chat" and c["prompt_id"] == "p2" for c in p["cells"])' "$TMP/multiturn/plan.json" 2> /dev/null
then ok "W5 a multi-turn prompt skips /api/generate in the plan (recorded) and is still driven on /api/chat"
else bad "W5 multi-turn plan: $(head -c 300 "$TMP/multiturn/plan.json" 2> /dev/null)"; fi
mv "$TMP/list.bak" "$TMP/list.jsonl"
sweep_rows extra extra
manifest_assert "R2 MUST-RED an unclassified route is a RED row per verb, text null" extra \
  'sorted(r["verb"] for r in gen if r["route"] == "POST /v1/brand-new") == ["serve run", "serve stream"] and all(r["rc"] == 3 and out(r)["protocol_fault"].startswith("unclassified_route") and out(r)["text"] is None for r in gen if r["route"] == "POST /v1/brand-new")'
sweep_rows none none
manifest_assert "R3 MUST-RED no route index is a RED row per verb" none \
  'len(gen) == 2 and all(r["route"] == "GET /" and r["rc"] == 3 and out(r)["protocol_fault"].startswith("no_route_index") for r in gen)'
sweep_rows normal nosweep skip "the GPU lock was not had"
manifest_assert "R4 MUST-RED a sweep that never ran is a refused row per verb, never a gap" nosweep \
  'len(gen) == 2 and all(r["refused"] == "the GPU lock was not had" and r["rc"] is None for r in gen)'

printf -- '--- %s: the code verb (apr code -p) ---\n' "$PROG"
FAKE_APR="$TMP/fake-apr"
cat > "$FAKE_APR" <<'SH'
#!/usr/bin/env bash
if [ "${1:-} ${2:-}" = "code --help" ]; then
  [ "${FAKE_APR_FLAGS:-0}" = 1 ] && printf '      --no-gpu\n      --gpu\n      --max-tokens <MAX_TOKENS>\n      --thinking <THINKING>\n'
  exit 0
fi
[ -n "${FAKE_ARGV:-}" ] && printf '%s\n' "$@" > "$FAKE_ARGV"
case "${FAKE_APR_MODE:-ok}" in
  ok)       printf '%s\n' '{"type":"result","subtype":"success","is_error":false,"status":"ok","result":"```python\ndef add(a, b):\n    return a + b\n```","num_turns":1}' ;;
  is_error) printf '%s\n' '{"type":"result","subtype":"error","is_error":true,"status":"error","result":"boom"}' ;;
  not_json) printf 'Launched apr serve\nplain text\n' ;;
  exit1)    printf 'apr: model not found\n' >&2; exit 1 ;;
  empty)    printf '%s\n' '{"type":"result","is_error":false,"status":"ok","result":""}' ;;
esac
SH
chmod +x "$FAKE_APR"
code_case() { # <label> <py> <mode> <prompt> <want rc> <fault prefix|-> [extra wrapper args...]
  local rc got
  FAKE_APR_MODE="$3" python3 "$2" --apr "$FAKE_APR" --model m.gguf --prompt-file "$TMP/prompt-$4.json" \
    --max-tokens 64 --out "$TMP/code.json" --timeout 20 "${@:7}" > /dev/null 2>&1
  rc=$?
  got=$(field "$TMP/code.json" 'd.get("protocol_fault") or ""')
  if [ "$rc" = "$5" ] && { [ "$6" = - ] || case "$got" in "$6"*) true ;; *) false ;; esac; }; then return 0; fi
  CASE_WHY="rc $rc (wanted $5), fault '${got:-none}'"
  return 1
}
crow() { local l="$1"; shift; CASE_WHY=""; if code_case "$l" "$@"; then ok "$l"; else bad "$l: $CASE_WHY"; fi; }
crow "C1 an ok envelope is read raw" "$CODE_PY" ok p1 0 -
if [ "$(field "$TMP/code.json" 'd["text"].startswith("```python")')" = True ]; then ok "C1b the reply is copied raw, fences kept"; else bad "C1b the reply is copied raw"; fi
crow "C2 MUST-RED is_error envelope" "$CODE_PY" is_error p1 3 is_error
crow "C3 MUST-RED stdout that is not the envelope" "$CODE_PY" not_json p1 3 not_json
crow "C4 MUST-RED a non-zero exit" "$CODE_PY" exit1 p1 3 exit_1
crow "C5 MUST-RED an empty result" "$CODE_PY" empty p1 3 empty_text
crow "C6 a multi-turn prompt is REFUSED for the code verb" "$CODE_PY" ok p2 4 -
crow "C7 MUST-RED the cpu lane is REFUSED when apr code has no backend flag" "$CODE_PY" ok p1 4 - --backend cpu
export FAKE_APR_FLAGS=1 FAKE_ARGV="$TMP/argv.txt"
crow "C8 with #3978's flags the cpu lane runs" "$CODE_PY" ok p1 0 - --backend cpu
if tr '\n' ' ' < "$TMP/argv.txt" | grep -q -- '--no-gpu --max-tokens 64 --thinking off --' \
  && [ "$(field "$TMP/code.json" 'd["reported"]["backend_control"]')" = "passed: --no-gpu" ]
then ok "C8b the lane's backend, max_tokens and thinking are PASSED and recorded as controlled"
else bad "C8b controls passed (argv: $(tr '\n' ' ' < "$TMP/argv.txt"))"; fi
unset FAKE_APR_FLAGS FAKE_ARGV

printf -- '--- %s: cell teardown (every server gone, from the process table AND the GPU) ---\n' "$PROG"
TD="$ROOT/scripts/lib/crux_cell_teardown.sh"
FAKE_SMI="$TMP/fake-smi"
printf '#!/usr/bin/env bash\ncat "%s" 2> /dev/null\n' "$TMP/smi-pids" > "$FAKE_SMI"
chmod +x "$FAKE_SMI"
td_case() { # <label> <want rc> <want state prefix> <td script> [pid files...]
  local label="$1" want="$2" pre="$3" td="$4" rc st
  shift 4
  CRUX_NVIDIA_SMI="$FAKE_SMI" CRUX_TEARDOWN_GPU_POLLS=4 bash "$td" "$TMP/td.state" "$@" > /dev/null 2>&1
  rc=$?
  st=$(cat "$TMP/td.state" 2> /dev/null)
  [ "$rc" = "$want" ] && case "$st" in "$pre"*) true ;; *) false ;; esac
}
tdrow() { local l="$1"; shift; if td_case "$l" "$@"; then ok "$l"; else bad "$l (state: $(cat "$TMP/td.state" 2> /dev/null))"; fi; }
stubborn() { # a server that ignores SIGTERM; its pid goes to $1
  bash -c 'trap "" TERM; while :; do sleep 0.2; done' > /dev/null 2>&1 &
  printf '%s\n' "$!" > "$1"
}
: > "$TMP/smi-pids"
stubborn "$TMP/srv1.pid"
tdrow "T1 a server that ignores TERM is KILLed, then proven gone" 0 clean "$TD" "$TMP/srv1.pid" "$TMP/absent.pid"
printf '1\n' > "$TMP/init.pid"
tdrow "T2 MUST-RED a server that survives TERM and KILL fails the cell" 1 "FAILED: server" "$TD" "$TMP/init.pid"
sleep 30 > /dev/null 2>&1 &
printf '%s\n' "$!" > "$TMP/srv3.pid"
printf '%s\n' "$(cat "$TMP/srv3.pid")" > "$TMP/smi-pids"
tdrow "T3 MUST-RED a pid still in nvidia-smi compute-apps fails the cell" 1 "FAILED: pid" "$TD" "$TMP/srv3.pid"
: > "$TMP/smi-pids"
mkdir -p "$TMP/tdrows"
: > "$TMP/tdrows/manifest.jsonl"
python3 "$ROUTES_PY" rows --out-dir "$TMP/ok" --prompt-list "$TMP/list.jsonl" --manifest "$TMP/tdrows/manifest.jsonl" \
  --engine apr --sha abc --host h --backend gpu --cell-fault "cell teardown FAILED: server pid(s) 42 survived" > /dev/null 2>&1
manifest_assert "T4 MUST-RED a failed teardown makes EVERY row of the cell RED" tdrows \
  'len(gen) == 11 and all(r["refused"].startswith("cell teardown FAILED") and r["rc"] is None for r in gen)'

printf -- '--- %s: chat pty reply extraction (#3962 B3) ---\n' "$PROG"
PTY_PY="$ROOT/scripts/lib/crux_pty_chat.py"
pty_case() { # <py> <python assertion over P (the module)>: 0 when the assertion holds
  python3 - "$1" "$2" <<'PY' > /dev/null 2>&1
import importlib.util, sys
spec = importlib.util.spec_from_file_location("p", sys.argv[1]); P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
T = "What is 2+2? Reply with the final answer inside <answer></answer>."
SPIN = "What is 2+2? Reply with the final answer inside <answer></\x08/\x08/answer>.\n<answer>4</answer>"
assert eval(sys.argv[2])
PY
}
prow() { if pty_case "$PTY_PY" "$2"; then ok "$1"; else bad "$1"; fi; }
PX1='P.extract_reply(P.clean(SPIN), T) == ("<answer>4</answer>", None)'
PX2='P.extract_reply("Loading...\n<answer>4</answer>", T)[0] is None and "echo" in P.extract_reply("Loading...\n<answer>4</answer>", T)[1]'
prow "X1 a spinner-corrupted echo (the llama.cpp v2 shape) is stripped; the reply is the answer" "$PX1"
prow "X2 MUST-RED an echo that cannot be found is an ERROR, never the whole transcript as the reply" "$PX2"
prow "X3 --answer-after cuts at its LAST match" 'P.extract_reply(">>> a\n... b\nREPLY", "a\nb", r"^[.][.][.].*$") == ("REPLY", None)'
prow "X4 --answer-after with no match falls back to the echo" 'P.extract_reply("hi there\nREPLY", "hi there", r"^[.][.][.].*$") == ("REPLY", None)'

# ---- mutants: each must-RED row must FAIL against code with its check deleted --------
printf -- '--- %s: mutants (each must be KILLED by its row) ---\n' "$PROG"
mutant() { # <label> <file> <python-literal old> <python-literal new> <case command...>
  local label="$1" src="$2" old="$3" new="$4" m
  shift 4
  m="$TMP/mutant-$(basename "$src")"
  if ! python3 - "$src" "$m" "$old" "$new" <<'PY'
import sys
s = open(sys.argv[1]).read()
if s.count(sys.argv[3]) != 1:
    sys.exit("mutation anchor matched %d times" % s.count(sys.argv[3]))
open(sys.argv[2], "w").write(s.replace(sys.argv[3], sys.argv[4]))
PY
  then bad "$label: the mutation anchor is gone (re-anchor it; a mutant that cannot be planted proves nothing)"; return; fi
  CASE_WHY=""
  if "$@" "$m"; then bad "$label: SURVIVED (its row passed against the mutant)"; else ok "$label: killed"; fi
}
m_plan() { # <assertion> <py>
  python3 "$2" plan --url "$URL" > "$TMP/plan.json" 2> /dev/null
  local rc=$?
  [ "$rc" = 1 ] && python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert eval(sys.argv[2])' "$TMP/plan.json" "$1" 2> /dev/null
}
# Each m_* re-runs one must-RED row with the MUTANT file as its $1 (appended by `mutant`).
m_s9() { start normal no_done; drive_case x "$1" "POST /v1/chat/completions" stream p1 3 stream_truncated; }
m_s12() { start normal zero_deltas; drive_case x "$1" "POST /v1/chat/completions" stream p1 3 stream_zero_deltas; }
m_s13() { start normal no_finish; drive_case x "$1" "POST /v1/chat/completions" stream p1 3 stream_no_finish; }
m_s14() { start normal empty_text; drive_case x "$1" "POST /v1/chat/completions" nonstream p1 3 empty_text; }
m_s3() { start normal ok; drive_case x "$1" "POST /generate" nonstream p1 4 -; }
m_p2() { start extra ok; m_plan 'd["unclassified"] == ["POST /v1/brand-new"]' "$1"; }
m_c2() { code_case x "$1" is_error p1 3 is_error; }
m_c5() { code_case x "$1" empty p1 3 empty_text; }
mutant "M1 S9 vs no terminal-event check" "$ROUTES_PY" 'if meta["terminal"] is None:' 'if False:' m_s9
mutant "M2 S12 vs no zero-delta check" "$ROUTES_PY" '    if not text:
        raise Fault("stream_zero_deltas' '    if False:
        raise Fault("stream_zero_deltas' m_s12
mutant "M3 S13 vs no finish_reason check" "$ROUTES_PY" 'if meta["finish_reason"] is None:' 'if False:' m_s13
mutant "M4 S14 vs no empty-text check" "$ROUTES_PY" '    if not text:
        raise Fault("empty_text' '    if False:
        raise Fault("empty_text' m_s14
mutant "M5 S3 vs raw routes sent unrendered" "$ROUTES_PY" 'if raw and not a.render_url:' 'if False:' m_s3
mutant "M6 P2 vs unknown routes filed as not-generation" "$ROUTES_PY" '        else:
            unclassified.append(r)' '        else:
            other.append({"route": r, "why": "?"})' m_p2
mutant "M7 C2 vs no is_error check" "$CODE_PY" 'if env.get("is_error") or env.get("status") not in (None, "ok"):' 'if False:' m_c2
mutant "M8 C5 vs no empty-result check" "$CODE_PY" 'if not env.get("result"):' 'if False:' m_c5
m_c7() { code_case x "$1" ok p1 4 - --backend cpu; }
mutant "M9 C7 vs the cpu lane run on an apr code that cannot honour it" "$CODE_PY" '    if a.backend == "cpu" and not ctl:' '    if False:' m_c7
m_t1() { stubborn "$TMP/srv1.pid"; td_case x 0 clean "$1" "$TMP/srv1.pid"; }
mutant "M10 T1 vs no KILL escalation" "$TD" '    kill -KILL $left 2> /dev/null' '    :' m_t1
m_t3() { sleep 30 > /dev/null 2>&1 & printf '%s\n' "$!" > "$TMP/srv3.pid"; cp "$TMP/srv3.pid" "$TMP/smi-pids"; td_case x 1 "FAILED: pid" "$1" "$TMP/srv3.pid"; }
mutant "M11 T3 vs no nvidia-smi check" "$TD" 'if [ "${#pids[@]}" -gt 0 ] && command -v "$SMI" > /dev/null 2>&1; then' 'if false; then' m_t3
m_t4() { : > "$TMP/tdrows/manifest.jsonl"; python3 "$1" rows --out-dir "$TMP/ok" --prompt-list "$TMP/list.jsonl" --manifest "$TMP/tdrows/manifest.jsonl" --engine apr --sha abc --host h --backend gpu --cell-fault "cell teardown FAILED: x" > /dev/null 2>&1; python3 -c 'import json,sys; g=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]; g=[r for r in g if r["kind"]=="gen"]; assert g and all(r["refused"] for r in g)' "$TMP/tdrows/manifest.jsonl" 2> /dev/null; }
mutant "M12 T4 vs a teardown fault that does not reach the rows" "$ROUTES_PY" '    if a.cell_fault:' '    if False:' m_t4

m_o3() { oracle_assert "$1" "$O3"; }
mutant "M15 O3 vs an unmapped kind silently judged as chat" "$ROUTES_PY" 'return ORACLE_ROUTE_BY_KIND.get(spec["kind"]) if spec else None' 'return ORACLE_ROUTE_BY_KIND.get(spec["kind"], "POST /v1/chat/completions") if spec else None' m_o3
m_s7c() { start normal ok; drive_case x "$1" "POST /api/generate" nonstream p2 4 -; }
mutant "M16 S7c vs a multi-turn prompt sent last-turn-only" "$ROUTES_PY" '    if spec and spec["kind"] in SINGLE_PROMPT_KINDS and turns > 1:' '    if False:' m_s7c
m_x1() { pty_case "$1" "$PX1"; }
mutant "M13 X1 vs no backspace processing" "$PTY_PY" '        prev, t = t, BACKSPACE.sub("", t)' '        prev = t' m_x1
m_x2() { pty_case "$1" "$PX2"; }
mutant "M14 X2 vs a missing echo read as the reply" "$PTY_PY" '        if k < 0:
            return None, (' '        if False:
            return None, (' m_x2

printf '%s: %d row(s), %d failed\n' "$PROG" "$ROWS" "$FAILS"
[ "$ROWS" -gt 0 ] || { printf '%s: zero rows ran - that is a broken table, not a pass\n' "$PROG" >&2; exit 1; }
[ "$FAILS" -eq 0 ]
