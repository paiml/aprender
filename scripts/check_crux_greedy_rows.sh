#!/usr/bin/env bash
# check_crux_greedy_rows.sh: the case table for the CRUX greedy rows (#3957 F9, aprender-36):
# scripts/lib/crux_cells_greedy.sh + scripts/lib/crux_greedy_llama.py. Hermetic: a stub llama-server (the four
# endpoints the helper calls) and a stub apr stand in for the engines; no GPU, model or real lock.
#
# greedy_cells() is run with the dogfood's OWN cell helpers (cell_add, serve_wait_line, run_cell), extracted
# from scripts/crux_inference_dogfood.sh at test time rather than copied, so they cannot drift.
#
# Rows:
#   1. both engines up        → llama.cpp ON + OFF rows and apr OFF with ids; apr ON REFUSED naming #3723;
#                               the ON text shows <think> (special tokens kept); max_tokens equal across engines
#   2. llama.cpp returns no `tokens` (return_tokens unsupported) → its rows refused BY NAME, never absent
#   3. llama.cpp unavailable (LLAMA_OK=0) → llama rows refused with LLAMA_WHY; apr OFF refused naming why
#   4. apr prints no `tokens`  → apr OFF refused by name
#   5. MUTANT: the apr-ON refusal dropped from the lib → the table sees the missing row
#
# Exit: 0 every row behaved · 1 a row broke · 2 ENV.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd) || exit 2
PROG=check_crux_greedy_rows
command -v python3 >/dev/null 2>&1 || { printf '%s: ENV - python3 is missing\n' "$PROG" >&2; exit 2; }
DOGFOOD="$ROOT/scripts/crux_inference_dogfood.sh"
LIB="$ROOT/scripts/lib/crux_cells_greedy.sh"
for f in "$DOGFOOD" "$LIB" "$ROOT/scripts/lib/crux_greedy_llama.py"; do
  [ -f "$f" ] || { printf '%s: ENV - %s not found\n' "$PROG" "$f" >&2; exit 2; }
done

TMP=$(mktemp -d) || exit 2
_rm_tmp() {
  pkill -f "$TMP/bin/llama-server" 2>/dev/null
  case "${TMP:-}" in
    /tmp/?*|/var/folders/?*) rm -rf -- "$TMP" || : ;;
    *) : ;;
  esac
}
trap _rm_tmp EXIT

PASS=0
FAIL=0
ok()   { printf '  ok    %s\n' "$1"; PASS=$((PASS + 1)); }
broke(){ printf '  BROKE %s\n' "$1"; FAIL=$((FAIL + 1)); }

BIN="$TMP/bin"; mkdir -p "$BIN"
# stub llama-server: `llama-server -m M --port P ...`; STUB_NO_TOKENS=1 drops `tokens` from /completion
cat > "$BIN/llama-server" <<'SH'
#!/usr/bin/env bash
port=""; while [ $# -gt 0 ]; do [ "$1" = --port ] && port="$2"; shift; done
exec python3 - "$port" <<'PY'
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
VOCAB = {1: "<think>", 2: "\n", 3: "</think>", 4: "4", 5: "<|im_end|>"}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, body):
        data = json.dumps(body).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        self.reply({"status": "ok"})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if self.path == "/apply-template":
            on = (req.get("chat_template_kwargs") or {}).get("enable_thinking")
            self.reply({"prompt": "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n" + ("" if on else "<think>\n\n</think>\n\n")})
        elif self.path == "/tokenize":
            self.reply({"tokens": [9, 9, 9, 1 if "<think>" in req["content"] else 8]})
        elif self.path == "/completion":
            on = req["prompt"][-1] == 8  # the OFF prompt ends with the prefilled empty block
            ids = [1, 2, 3, 4, 5] if on else [4, 5]
            body = {"content": "x", "stop_type": "eos"}
            if not os.environ.get("STUB_NO_TOKENS"):
                body["tokens"] = ids[: req["n_predict"]]
            self.reply(body)
        elif self.path == "/detokenize":
            self.reply({"content": "".join(VOCAB.get(t, "?") for t in req["tokens"])})
        else:
            self.reply({})


HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
PY
SH
cat > "$BIN/apr" <<'SH'
#!/usr/bin/env bash
# stub apr run --format json: greedy ids in "tokens"; STUB_APR_NO_TOKENS=1 omits them
if [ -n "${STUB_APR_NO_TOKENS:-}" ]; then printf '{"text": "4"}\n'; else printf '{"text": "4", "tokens": [4, 5], "finish_reason": "stop", "backend": {"ran": "gpu"}}\n'; fi
SH
chmod +x "$BIN/llama-server" "$BIN/apr"

# The dogfood's own cell helpers, extracted by name.
python3 - "$DOGFOOD" "$TMP/helpers.sh" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
out = []
for name in ("cell_add", "serve_wait_line", "free_port", "run_cell"):
    m = re.search(r"^%s\(\) \{.*?^\}\n" % name, s, re.S | re.M) or re.search(r"^%s\(\) \{[^\n]*\}\n" % name, s, re.M)
    if not m:
        sys.exit("helper %s() not found in the dogfood: update this check" % name)
    out.append(m.group(0))
open(sys.argv[2], "w").write("\n".join(out))
PY
[ $? -eq 0 ] || { printf '%s: ENV - could not extract the dogfood cell helpers\n' "$PROG" >&2; exit 2; }

# run_case <name> [lib] [env...]: greedy_cells() for one fake model, rows left in $TMP/<name>/manifest.jsonl
run_case() {
  local name="$1" lib="$2"; shift 2
  local w="$TMP/$name"; mkdir -p "$w/abc123abc123"
  printf '{"messages": [{"role": "user", "content": "What is 2+2?"}]}' > "$w/messages-ctl.json"
  printf 'What is 2+2?' > "$w/prompt-ctl.txt"
  ( cd "$ROOT" && env "$@" bash -c '
      . "$1"; . "$2"
      WORK=$3; MANIFEST=$3/manifest.jsonl; : > "$MANIFEST"
      M=/stub/model.gguf; SHA=$(printf "a%.0s" $(seq 64)); SHA12=abc123abc123; HOST=stub; BACKEND=gpu
      CTX=512; NGL=999; LLAMA_DEV=(); SEED=42; TMO=30; LOCK_WAIT=30; APR_BE="--gpu"
      GPUQ_OK=0; GPU_LOCK=$3/lock; LLAMA_SERVER=$4/llama-server; APR=$4/apr
      LLAMA_OK=${LLAMA_OK:-1}; LLAMA_WHY=${LLAMA_WHY:-}
      GREEDY_PIDS=ctl; GREEDY_MAXTOK=64
      greedy_cells' _ "$TMP/helpers.sh" "$lib" "$w" "$BIN" ) > "$w/run.log" 2>&1
}

rows() { # rows <case> -> one line per row: "row <engine> <thinking> REFUSED <why>|ids=... text=..."
  python3 - "$TMP/$1/manifest.jsonl" <<'PY'
import json, sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r["refused"]:
        print("row %s %s REFUSED %s" % (r["engine"], r["thinking"], r["refused"][:60]))
    else:
        d = json.load(open(r["tokens"]))
        print("row %s %s ids=%s text=%r max=%s special=%s" % (r["engine"], r["thinking"], d["generated_ids"],
              d["generated_text"], d["max_tokens"], d["special"]))
PY
}

printf '%s: greedy rows for #3957 F9\n' "$PROG"

run_case up "$LIB"
got=$(rows up)
want="row llama.cpp on ids=[1, 2, 3, 4, 5] text='<think>\n</think>4<|im_end|>' max=64 special=True
row llama.cpp off ids=[4, 5] text='4<|im_end|>' max=64 special=True
row apr on REFUSED #3723: apr has no thinking toggle — realizar routes every Qw
row apr off ids=[4, 5] text='4<|im_end|>' max=64 special=True"
[ "$got" = "$want" ] && ok "both engines up: llama ON/OFF + apr OFF with ids and special text; apr ON refused #3723" \
  || { broke "both engines up"; printf '%s\n--- want\n%s\n' "$got" "$want" | sed 's/^/        /'; }

run_case notokens "$LIB" STUB_NO_TOKENS=1
got=$(rows notokens | grep '^row llama.cpp')
case "$got" in *"llama.cpp on REFUSED RuntimeError: llama-server /completion returned no"*"llama.cpp off REFUSED RuntimeError"*)
  ok "llama.cpp without return_tokens: both its rows refused by name" ;;
  *) broke "no-tokens llama rows: $got" ;; esac

run_case nollama "$LIB" LLAMA_OK=0 "LLAMA_WHY=llama.cpp unresolved: fixture"
got=$(rows nollama)
case "$got" in *"llama.cpp on REFUSED llama.cpp unresolved: fixture"*"apr off REFUSED apr's ids are decoded through llama-server"*)
  ok "llama.cpp unavailable: its rows carry LLAMA_WHY; apr OFF names why it cannot be decoded" ;;
  *) broke "no-llama rows: $got" ;; esac

run_case noaprtokens "$LIB" STUB_APR_NO_TOKENS=1
got=$(rows noaprtokens | grep '^row apr off')
case "$got" in "row apr off REFUSED RuntimeError: apr --format json carried no "*) ok "apr with no tokens: its OFF row refused by name" ;;
  *) broke "apr no-tokens row: $got" ;; esac

# Row 5: MUTANT — the apr-ON refusal row dropped. The table must notice the missing row.
python3 - "$LIB" "$TMP/mutant-lib.sh" <<'PY'
import sys
s = open(sys.argv[1]).read()
a = '    row("apr", pid, "on", None, NO_ON)\n'
assert s.count(a) == 1, "mutation anchor moved: update this check with the lib"
open(sys.argv[2], "w").write(s.replace(a, ""))
PY
run_case mutant "$TMP/mutant-lib.sh"
got=$(rows mutant)
[ "$got" != "$want" ] && ok "MUTANT (apr ON refusal dropped) is caught by row 1's table" || broke "MUTANT not caught"

printf '%s: %d ok, %d broke\n' "$PROG" "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
exit 0
