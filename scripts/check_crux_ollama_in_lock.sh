#!/usr/bin/env bash
# check_crux_ollama_in_lock.sh: every model-loading call the CRUX dogfood makes to ollama runs while the GPU
# lock is HELD (#3964). Hermetic: stub apr and ollama binaries plus a stub HTTP server stand in for the real
# engines, and a private lock file (CRUX_GPU_LOCK) stands in for /tmp/apr-gpu.lock, so it needs no GPU, no
# model and never touches the real lock.
#
# WHY BEHAVIOURAL. A source scan for `"$OLLAMA" run` next to `cell_add` would pass while the cell script ran
# outside the lock; a guard that reads the source it guards has matched itself here before. So the stubs ask
# the only question that matters, at the moment of each call: is the lock held right now? (flock -n on the
# same file fails exactly when another process holds it.)
#
# Load-capable calls: `ollama run` one-shot (run verb) and interactive through crux_pty_chat.py (chat verb),
# and the server's /v1/chat/completions, /api/chat, /api/generate (the serve verb). create/show/rm/ps/stop/--version only read or write the store and load nothing.
#
# Rows:
#   1. the real dogfood, gpu-q path   → every load held, and both kinds of load were seen (not vacuous)
#   2. the real dogfood, flock path   → the same
#   3. MUTANT: run_cell without a lock → at least one load NOT held (the check can fail)
#   4. MUTANT: keep_alive 0 stripped from every ollama call → caught (cop 2026-09-23: gx10 has no host
#      keep_alive=0 default, and an ollama llama-server of up to 8969 MiB was left on its card between legs; the
#      per-REQUEST keep_alive 0 is what empties the card on every host, so every load must carry it)
#
# Exit: 0 every row behaved · 1 a row broke · 2 ENV.
set -uo pipefail
# guard_tree.sh probes `--help` to decide whether to run a self-test. Answer it before any work:
# a probe that fell through to the body ran this whole guard a second time, serially (#4046).
case "${1:-}" in -h|--help) printf 'usage: bash scripts/check_crux_ollama_in_lock.sh (no arguments: runs its rows)\n'; exit 0 ;; esac

ROOT=$(cd "$(dirname "$0")/.." && pwd) || exit 2
PROG=check_crux_ollama_in_lock
for t in python3 flock; do
  command -v "$t" >/dev/null 2>&1 || { printf '%s: ENV - %s is missing\n' "$PROG" "$t" >&2; exit 2; }
done

DOGFOOD="$ROOT/scripts/crux_inference_dogfood.sh"
[ -f "$DOGFOOD" ] || { printf '%s: ENV - %s not found\n' "$PROG" "$DOGFOOD" >&2; exit 2; }

TMP=$(mktemp -d) || exit 2
# every server here is a loopback stub: no proxy may carry those requests (quorum lane 2, 2026-09-23)
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy="127.0.0.1,localhost" NO_PROXY="127.0.0.1,localhost"
# `choom -n 1000` (run_cell's OOM-victim marking) is SHIMMED to a plain exec for this table: the table measures
# whether each load holds the lock, not OOM scoring, and an unprivileged sandbox denies the real one (quorum lane 2, 2026-09-23), which
# turned every row into a false BREAK. The shim runs the command exactly as given after `--`.
mkdir -p "$TMP/shim"
printf '#!/bin/sh\nwhile [ $# -gt 0 ] && [ "$1" != -- ]; do shift; done\n[ $# -gt 0 ] && shift\nexec "$@"\n' > "$TMP/shim/choom"
chmod +x "$TMP/shim/choom"
export PATH="$TMP/shim:$PATH"
SRV_PIDS=()
_cleanup() {
  local p
  for p in "${SRV_PIDS[@]}"; do kill "$p" 2>/dev/null; done
  case "${TMP:-}" in
    /tmp/?*|/var/folders/?*) rm -rf -- "$TMP" || : ;;
    *) : ;;
  esac
}
trap _cleanup EXIT

PASS=0
FAIL=0
ok()   { printf '  ok    %s\n' "$1"; PASS=$((PASS + 1)); }
broke(){ printf '  BROKE %s\n' "$1"; FAIL=$((FAIL + 1)); }

# ---- stubs -----------------------------------------------------------------------
BIN="$TMP/bin"; mkdir -p "$BIN"
cat > "$BIN/stub_server.py" <<'PY'
"""One stub HTTP server: ollama (/api/version, OpenAI chat) or apr serve (/health, OpenAI chat).
Every chat request logs whether the lock was HELD at that moment."""
import fcntl, json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

mode, port, lock, log = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]


def held():
    with open(lock, "a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fh, fcntl.LOCK_UN)
        return False


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, code, body, ctype="application/json"):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/version":
            self.reply(200, json.dumps({"version": "0.0.0-stub"}))
        elif self.path in ("/health", "/v1/models"):
            self.reply(200, json.dumps({"status": "ok"}))
        else:
            self.reply(404, "{}")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        if self.path in ("/v1/chat/completions", "/api/chat", "/api/generate"):
            ka = req.get("keep_alive")
            with open(log, "a") as fh:
                fh.write("LOAD http-%s %s held=%s keepalive=%s\n" % (mode, self.path, "yes" if held() else "no",
                                                                   "0" if ka in (0, "0", "0s") else "missing"))
            if req.get("stream"):
                chunk = json.dumps({"choices": [{"delta": {"content": "4"}, "finish_reason": "stop"}]})
                self.reply(200, "data: %s\n\ndata: [DONE]\n\n" % chunk, "text/event-stream")
            else:
                self.reply(200, json.dumps({"choices": [{"message": {"content": "4"}, "finish_reason": "stop"}],
                                            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}))
        else:
            self.reply(404, "{}")


HTTPServer(("127.0.0.1", port), H).serve_forever()
PY
cat > "$BIN/ollama" <<'SH'
#!/usr/bin/env bash
# stub ollama CLI: `run` is the one load-capable subcommand, and it logs whether the lock is held.
case "${1:-}" in
  --version|-v) echo "ollama version is 0.0.0-stub" ;;
  create|rm|stop|ps) : ;;
  show)
    case " $* " in
      *" --template "*) printf '{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}\n' ;;
      *) printf 'FROM /stub/blobs/sha256-%064d\n' 0 ;;
    esac ;;
  run)
    case " $* " in *" --help "*) echo "Usage: ollama run MODEL [PROMPT] [flags]"; exit 0 ;; esac
    ka=missing
    case " $* " in *" --keepalive 0 "*|*" --keepalive=0 "*|*" --keepalive 0s "*|*" --keepalive=0s "*) ka=0 ;; esac
    if [ -t 0 ]; then
      # interactive, as the chat verb drives it through crux_pty_chat.py: a minimal REPL, one load check per turn
      while :; do
        printf '>>> '
        IFS= read -r line || exit 0
        [ "$line" = /bye ] && exit 0
        if flock -n "$STUB_LOCK" true 2>/dev/null; then h=no; else h=yes; fi
        echo "LOAD cli-chat held=$h keepalive=$ka" >> "$STUB_LOG"
        printf '4\n\n'
      done
    fi
    if flock -n "$STUB_LOCK" true 2>/dev/null; then h=no; else h=yes; fi
    echo "LOAD cli-run held=$h keepalive=$ka" >> "$STUB_LOG"
    echo "4"
    # bashrs SEC001: the word 'eval' here is a literal printf argument (ollama's 'eval count'/'eval rate' output), not an eval call.
    # bashrs disable-next-line=SEC001
    printf 'total duration:       1s\nprompt %s count:    3 token(s)\n%s count:           1 token(s)\n%s rate:            1.00 tokens/s\n' eval eval eval >&2 ;;
  *) echo "stub ollama: unhandled '$*'" >&2; exit 1 ;;
esac
SH
cat > "$BIN/apr" <<'SH'
#!/usr/bin/env bash
# stub apr: the subject, so the dogfood runs; its serve is the stub server in apr mode.
case "${1:-}" in
  --version) echo "apr 0.0.0 (stub)" ;;
  run)
    printf '{"text": "4", "tokens_generated": 1, "tok_per_sec": 1.0, "backend": {"requested": "gpu", "ran": "gpu", "fell_back": false}}\n'
    printf '[DEBUG] formatted_prompt="q"\n[DEBUG] add_bos=false, encoded 1 tokens: [1]\n' >&2 ;;
  chat)
    # apr chat reads one user turn per stdin line and prints a transcript
    while IFS= read -r _turn; do printf 'You: \nAssistant: 4\n'; done
    printf 'You: \nGoodbye!\n' ;;
  serve)
    port=""; while [ $# -gt 0 ]; do [ "$1" = --port ] && port="$2"; shift; done
    exec python3 "$(dirname "$0")/stub_server.py" apr "$port" "$STUB_LOCK" "$STUB_LOG" ;;
  *) echo "stub apr: unhandled '$*'" >&2; exit 1 ;;
esac
SH
chmod +x "$BIN/ollama" "$BIN/apr"
MODEL="$TMP/model.gguf"; printf 'GGUF-stub' > "$MODEL"

free_port() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])'; }

# run_row <name> <tree root> <gpuq bin>: one dogfood run; leaves the stub log at $TMP/<name>.log
run_row() {
  local name="$1" tree="$2" gpuq="$3" port lock log i
  lock="$TMP/$name.lock"; log="$TMP/$name.log"; : > "$lock"; : > "$log"
  port=$(free_port)
  python3 "$BIN/stub_server.py" ollama "$port" "$lock" "$log" & SRV_PIDS+=("$!")
  for i in $(seq 1 50); do curl -sf "http://127.0.0.1:$port/api/version" >/dev/null 2>&1 && break; sleep 0.1; done
  ( cd "$tree" && STUB_LOCK="$lock" STUB_LOG="$log" CRUX_GPU_LOCK="$lock" GPUQ_BIN="$gpuq" \
      GPUQ_DIR="$TMP/$name.q" OLLAMA_BIN="$BIN/ollama" OLLAMA_HOST="127.0.0.1:$port" \
      DOGFOOD_ALLOW_UNPINNED=1 APR="$BIN/apr" \
      timeout 600 bash scripts/crux_inference_dogfood.sh 0.0.0 --model "$MODEL" --engines apr,ollama \
        --verbs run,chat,serve --host stub --out "$TMP/$name.out" --timeout 60 ${RUN_ROW_EXTRA:-} > "$TMP/$name.dogfood.log" 2>&1 )
  echo "$?" > "$TMP/$name.rc"
}

# verdict <name>: "<loads> <not-held> <cli-run loads> <cli-chat loads> <http-ollama loads>"
verdict() {
  local log="$TMP/$1.log"
  printf '%s %s %s %s %s' "$(grep -c '^LOAD ' "$log")" "$(grep -c ' held=no ' "$log")" \
    "$(grep -c '^LOAD cli-run' "$log")" "$(grep -c '^LOAD cli-chat' "$log")" "$(grep -c '^LOAD http-ollama' "$log")"
}

expect_held() { # expect_held <name> <label> — every KIND of load must be seen, or the row is vacuous
  local v loads notheld cli chat http
  v=$(verdict "$1"); read -r loads notheld cli chat http <<< "$v"
  local noka
  noka=$(grep -E '^LOAD (cli-|http-ollama)' "$TMP/$1.log" | grep -vc 'keepalive=0$')
  if [ "$cli" -gt 0 ] && [ "$chat" -gt 0 ] && [ "$http" -gt 0 ] && [ "$notheld" -eq 0 ] && [ "$noka" -eq 0 ]; then
    ok "$2: $loads loads ($cli ollama run, $chat ollama chat turns, $http ollama HTTP), all with the lock held, every ollama load keep_alive 0"
  else
    broke "$2: loads=$loads not-held=$notheld run=$cli chat=$chat http=$http no-keepalive=$noka (dogfood rc $(cat "$TMP/$1.rc"); log $TMP/$1.dogfood.log)"
    tail -5 "$TMP/$1.dogfood.log" | sed 's/^/        /'
  fi
}

printf '%s: every ollama model load runs inside the GPU lock (#3964)\n' "$PROG"

# Row 1: the real dogfood through gpu-q (the production path on lambda and gx10).
GPUQ_REAL="${GPUQ_BIN_UNDER_TEST:-$HOME/.local/bin/gpu-q}"
if [ -x "$GPUQ_REAL" ] && grep -q wait <<< "$("$GPUQ_REAL" --caps 2>/dev/null)"; then
  run_row gpuq "$ROOT" "$GPUQ_REAL"
  expect_held gpuq "gpu-q path"
else
  # Not a gap on THIS host: without a gpu-q that has `wait`, the dogfood itself takes the flock path, which
  # row 2 measures. The gpu-q path is measured wherever gpu-q is installed (lambda, gx10).
  ok "gpu-q path: no gpu-q with \`wait\` at $GPUQ_REAL, so the dogfood here uses the flock path (row 2)"
fi

# Row 2: the real dogfood through the flock fallback.
run_row flock "$ROOT" "/nonexistent/gpu-q"
expect_held flock "flock path"

# Row 2b (#4051): the SAME real run's receipt. Every engine row that ran carries its stamp (the judge ran with
# --require-timing), each under the lock (lock "gpu") with a measured wait. timing_row <name> -> "<required> <stamped> <faults> <gpu-locked stamps>"
timing_row() {
  python3 - "$TMP/$1.out" <<'PY' 2> /dev/null || echo "none 0 -1 0"
import glob, json, sys
fs = [f for f in glob.glob(sys.argv[1] + "/*.json") if not f.endswith("-greedy.json")]
r = json.load(open(fs[0])) if len(fs) == 1 else {}
t = r.get("timing") or {}
gpu = sum(1 for c in r.get("cells", []) for e in (c.get("engines") or {}).values()
          if isinstance(e.get("timing"), dict) and e["timing"].get("lock") == "gpu" and (e["timing"].get("lock_wait_s") or -1) >= 0)
print(t.get("required"), t.get("stamped", 0), len(t.get("faults") or []) if "faults" in t else -1, gpu)
PY
}
read -r treq tst tfa tgpu <<< "$(timing_row flock)"
if [ "$treq" = True ] && [ "$tst" -gt 0 ] && [ "$tfa" = 0 ] && [ "$tgpu" -gt 0 ]; then
  ok "#4051 timing: the real run stamped $tst engine row(s), 0 faults, $tgpu judged cell engine(s) carry a gpu-lock stamp"
else
  broke "#4051 timing: required=$treq stamped=$tst faults=$tfa gpu-stamped=$tgpu -- the real run's receipt does not account for its engine calls"
fi

# Row 3: MUTANT — run_cell runs the cell with no lock. The check must see it.
MT="$TMP/mutant-tree"; mkdir -p "$MT/scripts"
for f in "$ROOT"/scripts/* "$ROOT"/scripts/.[!.]*; do
  [ -e "$f" ] || continue
  [ "$(basename "$f")" = crux_inference_dogfood.sh ] && continue
  ln -s "$f" "$MT/scripts/$(basename "$f")"
done
for d in "$ROOT"/*; do [ "$(basename "$d")" = scripts ] || ln -s "$d" "$MT/$(basename "$d")"; done
python3 - "$DOGFOOD" "$MT/scripts/crux_inference_dogfood.sh" <<'PY'
import sys
s = open(sys.argv[1]).read()
a = '    GPUQ_LOCK="$GPU_LOCK" GPUQ_WAIT="$LOCK_WAIT" "$GPUQ" --prio "$GPU_PRIO" -- bash -c "$acq" "$1" 2> "$1.lock.err"'
b = '    flock -w "$LOCK_WAIT" -E 75 "$GPU_LOCK" choom -n 1000 -- bash -c "$acq" "$1" 2> "$1.lock.err"'
assert s.count(a) == 1 and s.count(b) == 1, "mutation anchors moved: update this check with the dogfood"
s = s.replace(a, '    bash -c "$acq" "$1" 2> "$1.lock.err"').replace(b, '    bash -c "$acq" "$1" 2> "$1.lock.err"')
open(sys.argv[2], "w").write(s)
PY
[ $? -eq 0 ] || { broke "mutant: could not plant (anchors moved)"; }
run_row mutant "$MT" "/nonexistent/gpu-q"
read -r loads notheld cli chat http <<< "$(verdict mutant)"
if [ "$loads" -gt 0 ] && [ "$notheld" -gt 0 ]; then
  ok "MUTANT (run_cell without a lock) is caught: $notheld of $loads loads ran with the lock free"
else
  broke "MUTANT not caught: loads=$loads not-held=$notheld — the check cannot fail"
fi

# Row 4: MUTANT — keep_alive 0 stripped from every ollama call (run, chat pty, serve). The check must see it.
# keep_alive is set in the dogfood (run, chat pty) AND in scripts/lib/crux_cells_serve_code.sh (serve, code; #3962):
# the mutant strips it from BOTH, in a tree where scripts/lib is a real directory so the mutated lib is the one sourced
MK="$TMP/mutant-ka-tree"; mkdir -p "$MK/scripts/lib"
for f in "$ROOT"/scripts/* "$ROOT"/scripts/.[!.]*; do
  [ -e "$f" ] || continue
  case "$(basename "$f")" in crux_inference_dogfood.sh|lib) continue ;; esac
  ln -s "$f" "$MK/scripts/$(basename "$f")"
done
for f in "$ROOT"/scripts/lib/*; do
  [ "$(basename "$f")" = crux_cells_serve_code.sh ] && continue
  ln -s "$f" "$MK/scripts/lib/$(basename "$f")"
done
for d in "$ROOT"/*; do [ "$(basename "$d")" = scripts ] || ln -s "$d" "$MK/$(basename "$d")"; done
python3 - "$DOGFOOD" "$MK/scripts/crux_inference_dogfood.sh" "$ROOT/scripts/lib/crux_cells_serve_code.sh" "$MK/scripts/lib/crux_cells_serve_code.sh" <<'PY'
import re, sys
total = 0
for src, dst in ((sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])):
    try:
        s = open(src).read()
    except OSError:
        continue  # a tree without the serve/code lib: the dogfood carries every call
    pat = re.compile(r"""--keepalive 0 |--extra '\{"keep_alive": 0\}' |, "keep_alive": 0|"keep_alive": 0, ?""")
    total += len(pat.findall(s))
    open(dst, "w").write(pat.sub("", s))
assert total >= 3, "keep_alive anchors moved (%d found): update this check with the dogfood and the serve/code lib" % total
PY
[ $? -eq 0 ] || broke "keep_alive mutant: could not plant (anchors moved)"
run_row mutantka "$MK" "/nonexistent/gpu-q"
kaload=$(grep -cE '^LOAD (cli-|http-ollama)' "$TMP/mutantka.log")
kamiss=$(grep -E '^LOAD (cli-|http-ollama)' "$TMP/mutantka.log" | grep -vc 'keepalive=0$')
if [ "$kaload" -gt 0 ] && [ "$kamiss" -eq "$kaload" ]; then
  ok "MUTANT (keep_alive 0 stripped) is caught: $kamiss of $kaload ollama loads carried no keep_alive 0"
else
  broke "keep_alive MUTANT not caught: ollama loads=$kaload without keep_alive=$kamiss"
fi

# Row 4b (#4051): MUTANT — the engine lines write no stamp. The judge (--require-timing) must then DECLINE by name.
MS="$TMP/mutant-stamp-tree"; mkdir -p "$MS/scripts"
for f in "$ROOT"/scripts/* "$ROOT"/scripts/.[!.]*; do
  [ -e "$f" ] || continue
  [ "$(basename "$f")" = crux_inference_dogfood.sh ] && continue
  ln -s "$f" "$MS/scripts/$(basename "$f")"
done
for d in "$ROOT"/*; do [ "$(basename "$d")" = scripts ] || ln -s "$d" "$MS/$(basename "$d")"; done
python3 - "$DOGFOOD" "$MS/scripts/crux_inference_dogfood.sh" <<'PY'
import sys
s = open(sys.argv[1]).read()
a = 'cell_stamp_close() { # <cell> <prefix>\n'
assert s.count(a) == 1, "stamp anchor moved: update this check with the dogfood"
open(sys.argv[2], "w").write(s.replace(a, a + '  return 0\n'))
PY
[ $? -eq 0 ] || broke "stamp mutant: could not plant (anchor moved)"
run_row mutantst "$MS" "/nonexistent/gpu-q"
read -r mreq mst mfa mgpu <<< "$(timing_row mutantst)"
# The stub engines answer wrong, so the run is RED before timing is read (RED outranks DECLINE); what the mutant
# must change is that the receipt NAMES the unstamped calls, and that nothing passes.
mver=$(python3 -c 'import glob,json,sys; r=json.load(open([f for f in glob.glob(sys.argv[1]+"/*.json") if not f.endswith("-greedy.json")][0])); f=r["timing"]["faults"]; print(r["summary"]["verdict"], "named" if f and all("no timing stamp" in x for x in f) else "unnamed")' "$TMP/mutantst.out" 2> /dev/null)
if [ "$mfa" -gt 0 ] && [ "$tfa" = 0 ] && [ "${mver#PASS}" = "$mver" ] && [ "${mver#* }" = named ]; then
  ok "#4051 MUTANT (no engine-line stamps) is caught: $mfa call(s) named 'no timing stamp' in the receipt (verdict ${mver% *}), vs 0 on the shipped run"
else
  broke "#4051 stamp MUTANT not caught: faults=$mfa (shipped $tfa), verdict/naming '$mver'"
fi

# Row 5: --only-prompts runs only the named prompts FROM the certified file (the file is never rewritten, since its sha
# is what the certification binds). Every cell dir must belong to the named prompt, and there must be some.
PSET="$ROOT/$(sed -n 's/^PROMPTS="\(.*\)"$/\1/p' "$DOGFOOD" | head -1)"
ONLY=$(python3 -c 'import json,sys; print(next(p["id"] for p in json.load(open(sys.argv[1]))["prompts"] if p.get("control")))' "$PSET" 2>/dev/null)
RUN_ROW_EXTRA="--only-prompts $ONLY" run_row only "$ROOT" "/nonexistent/gpu-q"
cells=$(find /tmp -maxdepth 6 -path "*/cell-*.sh" -newer "$TMP/only.lock" 2>/dev/null | grep -c . )
wrong=$(sed -n 's/.*work kept: //p' "$TMP/only.dogfood.log" | head -1 | xargs -r -I{} find {} -name 'cell-*.sh' 2>/dev/null | grep -v -e "cell-$ONLY.sh" -e 'cell-serve.sh' -e 'cell-code.sh' -e 'cell-greedy.sh' | grep -c .)
nloads=$(grep -c '^LOAD ' "$TMP/only.log")
read -r floads _ <<< "$(verdict flock)"
if [ -n "$ONLY" ] && [ "$wrong" = 0 ] && [ "$nloads" -gt 0 ] && [ "$nloads" -lt "$floads" ]; then
  ok "--only-prompts $ONLY: only its cells ran ($nloads loads vs $floads for the whole set)"
else
  broke "--only-prompts: only='$ONLY' foreign cells=$wrong loads=$nloads (whole set $floads) — see $TMP/only.dogfood.log"
fi
RUN_ROW_EXTRA="--only-prompts no-such-prompt" run_row unknown "$ROOT" "/nonexistent/gpu-q"
if grep -q "only-prompts names ids the prompt set does not hold: no-such-prompt" "$TMP/unknown.dogfood.log" && [ "$(cat "$TMP/unknown.rc")" != 0 ]; then
  ok "--only-prompts with an unknown id declines by name (rc $(cat "$TMP/unknown.rc"))"
else
  broke "--only-prompts unknown id: rc $(cat "$TMP/unknown.rc"); $(tail -2 "$TMP/unknown.dogfood.log" | tr '\n' ' ')"
fi

printf '%s: %d ok, %d broke\n' "$PROG" "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
exit 0
