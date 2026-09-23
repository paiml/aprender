#!/usr/bin/env bash
# check_crux_greedy_rows.sh: the case table for the CRUX greedy rows (#3957 F9, aprender-36):
# scripts/lib/crux_cells_greedy.sh + scripts/lib/crux_greedy_llama.py. Hermetic: a stub llama-server (the four
# endpoints the helper calls) and a stub apr stand in for the engines; no GPU, model or real lock.
#
# greedy_cells() is run with the dogfood's OWN cell helpers (cell_add, serve_wait_line, run_cell), extracted
# from scripts/crux_inference_dogfood.sh at test time rather than copied, so they cannot drift.
#
# Rows:
#   1. apr WITH `run --thinking` → apr ON and OFF measured; llama.cpp generates from apr's OWN prompt ids
#                               (prompt_source apr, prompt_ids_equal true); the ON text shows <think>
#   1b. apr WITHOUT it          → apr ON REFUSED naming #3723; OFF still measured
#   1c. apr's prompt ≠ llama's template → the parity row runs on apr's ids (prompt_ids_equal FALSE, visible),
#       the OFFICIAL row on the template's own ids (cop ruling on F9, #3990: RED-MODEL needs the official row)
#   2. llama.cpp returns no `tokens` (return_tokens unsupported) → its rows refused BY NAME, never absent
#   3. llama.cpp unavailable (LLAMA_OK=0) → llama rows refused with LLAMA_WHY; apr OFF refused naming why
#   4. apr prints no `tokens`  → apr OFF refused by name
#   5. MUTANT: the apr-ON refusal dropped from the lib → the no-flag row goes missing, and that is caught
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
# stub apr. `run --help` advertises --thinking only when STUB_APR_THINKING=1. `run ... -v` prints the prompt
# apr built on stderr (formatted_prompt + its ids) and the greedy ids on stdout; STUB_APR_NO_TOKENS=1 omits
# them; STUB_APR_PROMPT_DRIFT=1 makes apr's prompt ids differ from llama.cpp's template.
if [ "${2:-}" = --help ]; then echo "  --chat"; [ -n "${STUB_APR_THINKING:-}" ] && echo "  --thinking <on|off>"; exit 0; fi
th=off; while [ $# -gt 0 ]; do [ "$1" = --thinking ] && th="$2"; shift; done
last=1; [ "$th" = on ] && last=8
first=9; [ -n "${STUB_APR_PROMPT_DRIFT:-}" ] && first=7
printf '[DEBUG] formatted_prompt="<|im_start|>user\\nq<|im_end|>\\n"\n[DEBUG] add_bos=false, encoded 4 tokens: [%s, 9, 9, %s]\n' "$first" "$last" >&2
ids="[4, 5]"; [ "$th" = on ] && ids="[1, 2, 3, 4, 5]"
# -v puts `verbose:` lines on stdout BEFORE the JSON, exactly as the real apr does (measured on lambda)
printf 'verbose: apr 0.0.0\nverbose: model = /stub/model.gguf\n'
if [ -n "${STUB_APR_NO_TOKENS:-}" ]; then printf '{"text": "4"}\n'; else printf '{"text": "4", "tokens": %s, "finish_reason": "stop", "backend": {"ran": "gpu"}}\n' "$ids"; fi
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
        print("row %s %s %s REFUSED %s" % (r["engine"], r["thinking"], r["prompt_source"], r["refused"][:60]))
    else:
        d = json.load(open(r["tokens"]))
        print("row %s %s %s ids=%s text=%r max=%s special=%s" % (r["engine"], r["thinking"], r["prompt_source"], d["generated_ids"],
              d["generated_text"], d["max_tokens"], d["special"]))
PY
}

printf '%s: greedy rows for #3957 F9\n' "$PROG"

run_case up "$LIB" STUB_APR_THINKING=1
got=$(rows up)
want_up="row llama.cpp on apr ids=[1, 2, 3, 4, 5] text='<think>\n</think>4<|im_end|>' max=64 special=True
row llama.cpp on official ids=[1, 2, 3, 4, 5] text='<think>\n</think>4<|im_end|>' max=64 special=True
row llama.cpp off apr ids=[4, 5] text='4<|im_end|>' max=64 special=True
row llama.cpp off official ids=[4, 5] text='4<|im_end|>' max=64 special=True
row apr on apr ids=[1, 2, 3, 4, 5] text='<think>\n</think>4<|im_end|>' max=64 special=True
row apr off apr ids=[4, 5] text='4<|im_end|>' max=64 special=True"
[ "$got" = "$want_up" ] && ok "apr with --thinking: apr ON and OFF measured beside llama.cpp, special text kept" \
  || { broke "apr with --thinking"; printf '%s\n--- want\n%s\n' "$got" "$want_up" | sed 's/^/        /'; }
prov=$(python3 -c 'import json,sys
r=[json.loads(l) for l in open(sys.argv[1])]; d=json.load(open([x for x in r if x["engine"]=="llama.cpp" and x["thinking"]=="on" and x["prompt_source"]=="apr"][0]["tokens"]))
print(d["prompt_source"], d["prompt_ids"], d["template_prompt_ids"], d["prompt_ids_equal"])' "$TMP/up/manifest.jsonl")
[ "$prov" = "apr [9, 9, 9, 8] [9, 9, 9, 8] True" ] && ok "the parity row ran on apr's OWN prompt ids, and they equal the template" \
  || broke "llama parity provenance: '$prov'"

run_case noflag "$LIB"
got=$(rows noflag)
got_on=$(printf '%s\n' "$got" | grep '^row apr on')
case "$got_on" in "row apr on apr REFUSED #3723: this apr has no \`run --thinking\` flag"*) ok "apr without --thinking: its ON row refused naming #3723; OFF measured" ;;
  *) broke "apr no-flag ON row: '$got_on'" ;; esac
got_par=$(printf '%s\n' "$got" | grep '^row llama.cpp on ')
case "$got_par" in *"row llama.cpp on apr REFUSED RuntimeError: no apr prompt ids to reuse"*"row llama.cpp on official ids="*)
  ok "...and llama.cpp's ON PARITY row is refused (no apr prompt), while its OFFICIAL row is measured" ;;
  *) broke "no-flag llama ON rows: '$got_par'" ;; esac

run_case drift "$LIB" STUB_APR_THINKING=1 STUB_APR_PROMPT_DRIFT=1
prov=$(python3 -c 'import json,sys
r=[json.loads(l) for l in open(sys.argv[1])]
par=json.load(open([x for x in r if x["engine"]=="llama.cpp" and x["thinking"]=="off" and x["prompt_source"]=="apr"][0]["tokens"]))
off=json.load(open([x for x in r if x["engine"]=="llama.cpp" and x["thinking"]=="off" and x["prompt_source"]=="official"][0]["tokens"]))
print(par["prompt_ids"], par["template_prompt_ids"], par["prompt_ids_equal"], "|", off["prompt_ids"], off["template_prompt_ids"])' "$TMP/drift/manifest.jsonl")
[ "$prov" = "[7, 9, 9, 1] [9, 9, 9, 1] False | [9, 9, 9, 1] [9, 9, 9, 1]" ] \
  && ok "a prompt drift is VISIBLE: parity ran on apr's ids (equal=false); official ran on the template's ids" \
  || broke "prompt drift provenance: '$prov'"

# 1d. the HELPER ITSELF (not only the lib's call convention): handed an apr prompt that DIFFERS, --prompt-source
#     official must still run on the template's ids, and --prompt-source apr on apr's. (A mutant that let the
#     official row reuse apr's ids survived the table until this case: the lib never passes apr's stderr there.)
hp=$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1])')
"$BIN/llama-server" -m /stub --port "$hp" > "$TMP/helper-srv.log" 2>&1 &
hsp=$!
for _ in $(seq 1 50); do python3 -c 'import sys,urllib.request;urllib.request.urlopen("http://127.0.0.1:%s/health"%sys.argv[1],timeout=1)' "$hp" 2>/dev/null && break; sleep 0.1; done
printf '[DEBUG] add_bos=true, encoded 4 tokens: [7, 9, 9, 1]\n' > "$TMP/drift.err"
printf '{"messages": [{"role": "user", "content": "q"}]}' > "$TMP/h-msg.json"
for src in official apr; do
  python3 "$ROOT/scripts/lib/crux_greedy_llama.py" gen --prompt-source "$src" --url "http://127.0.0.1:$hp" \
    --messages "$TMP/h-msg.json" --thinking off --max-tokens 8 --seed 1 --apr-stderr "$TMP/drift.err" --out "$TMP/h-$src.json"
done
kill "$hsp" 2>/dev/null; wait "$hsp" 2>/dev/null
got=$(python3 -c 'import json,sys; print(" | ".join("%s %s" % (s, json.load(open(sys.argv[1] + "/h-%s.json" % s))["prompt_ids"]) for s in ("official", "apr")))' "$TMP")
[ "$got" = "official [9, 9, 9, 1] | apr [7, 9, 9, 1]" ] && ok "helper: official runs on the template's ids even when handed apr's; apr on apr's" \
  || broke "helper prompt selection: '$got'"

run_case notokens "$LIB" STUB_NO_TOKENS=1
got=$(rows notokens | grep '^row llama.cpp')
case "$got" in *"llama.cpp on apr REFUSED RuntimeError"*"llama.cpp on official REFUSED RuntimeError: llama-server /completion returned no"*"llama.cpp off official REFUSED RuntimeError"*)
  ok "llama.cpp without return_tokens: both its rows refused by name" ;;
  *) broke "no-tokens llama rows: $got" ;; esac

run_case nollama "$LIB" LLAMA_OK=0 "LLAMA_WHY=llama.cpp unresolved: fixture"
got=$(rows nollama)
case "$got" in *"llama.cpp on apr REFUSED llama.cpp unresolved: fixture"*"llama.cpp on official REFUSED llama.cpp unresolved: fixture"*"apr off apr REFUSED apr's ids are decoded through llama-server"*)
  ok "llama.cpp unavailable: its rows carry LLAMA_WHY; apr OFF names why it cannot be decoded" ;;
  *) broke "no-llama rows: $got" ;; esac

run_case noaprtokens "$LIB" STUB_APR_THINKING=1 STUB_APR_NO_TOKENS=1
got=$(rows noaprtokens | grep '^row apr off')
case "$got" in "row apr off apr REFUSED RuntimeError: apr --format json carried no "*) ok "apr with no tokens: its OFF row refused by name" ;;
  *) broke "apr no-tokens row: $got" ;; esac

# Row 5: MUTANT — the apr-ON refusal row dropped. The table must notice the missing row.
python3 - "$LIB" "$TMP/mutant-lib.sh" <<'PY'
import sys
s = open(sys.argv[1]).read()
a = '            row("apr", pid, "on", None, NO_ON)\n'
assert s.count(a) == 1, "mutation anchor moved: update this check with the lib"
open(sys.argv[2], "w").write(s.replace(a, ""))
PY
run_case mutant "$TMP/mutant-lib.sh"
got=$(rows mutant | grep -c '^row apr on')
[ "$got" = 0 ] && ok "MUTANT (apr ON refusal dropped) is caught: the no-flag table's ON row is gone" || broke "MUTANT not caught ($got apr-ON rows)"

printf '%s: %d ok, %d broke\n' "$PROG" "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
exit 0
