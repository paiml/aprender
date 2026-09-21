#!/usr/bin/env bash
# crux_inference_dogfood.sh: the CRUX inference dogfood (#3739). The SAME GGUF
# and the SAME prompts go through apr, llama.cpp at the pin and ollama, and one
# rule judges them: a cell a comparator answers correctly and apr does not is RED.
#
# WHY. Every apr gate compared apr to apr, and so missed a tokenizer that never
# did real BPE (#3726) and a chat template applied twice (#3672): both sides of
# every comparison shared the defect. A competitor on the same file is the
# external oracle. Operator, verbatim: "same prompts/verbs on llama.cpp and say
# ollama for same model: chat/serve/run/code, etc".
#
# Usage: bash scripts/crux_inference_dogfood.sh <version> --model <gguf> [--model <gguf>]...
#          [--host <id>] [--backend gpu|cpu] [--engines apr,llama.cpp,ollama]
#          [--verbs run] [--out <dir>] [--prompts <file>] [--timeout <s>]
#          [--keep-ollama-models] [--keep-work]
#   <version>  the apr version under test; `apr --version` must report it, or
#              the run declines (a dogfood that measured another version is #3708)
#   --backend  the lane, applied to ALL three engines (default gpu): apr --gpu,
#              llama.cpp -ngl 999, ollama's default; cpu = --no-gpu, -ngl 0, num_gpu 0.
#              An apr that falls back from the lane's backend did not answer the cell.
#   --out      receipt dir (default evidence/crux/<version>); writes <host>-<backend>.{json,md}
#
# Exit: 0 no RED and at least one GREEN cell · 1 any RED · 2 decline (no GREEN
# judged, apr unpinned or the wrong version, a verb this slice cannot drive).
#
# WHAT "SAME" MEANS, and how each engine is resolved:
#   model   one file by path. ollama IMPORTS that file through a Modelfile
#           (`FROM <path>`); a registry pull is a different file. The receipt
#           records the input's sha256 and the blob digest ollama stored, which
#           newer ollama versions do not keep byte-identical to the input.
#   prompts scripts/crux_inference_prompts.json, as chat MESSAGES. Every engine
#           applies the template it reads from the GGUF; that is under test too.
#   apr     scripts/apr_bin.sh (HEAD-built), or DOGFOOD_ALLOW_UNPINNED=1 plus $APR
#           for a published binary, as model_ladder.sh does.
#   llama   scripts/llama_bin.sh ONLY ($LLAMA_BENCH_PATH names the pinned build).
#   ollama  $OLLAMA_BIN, else the forjar-declared $HOME/.local/bin/ollama, else
#           /usr/local/bin/ollama; the SERVER's /api/version is what is recorded.
#   sampling temperature, seed and context come from scripts/llama_pin.toml
#           [protocol], the declaration the parity gates already read.
#
# NO RATE IS COMPUTED HERE. The judge (scripts/lib/crux_inference_judge.py)
# copies the token counts and rates each engine prints about itself into the
# receipt, labelled by engine, and judges none of them. Nothing in this file
# reads a clock. A throughput CLAIM goes through scripts/perf_gate.sh (PERF-009).
#
# GPU sharing (the cop's rule, /mnt/nvme-raid0/agent-wt/gpu-lock-rule.txt):
# every engine invocation holds /tmp/apr-gpu.lock for that command only and
# runs under `choom -n 1000`, so a measurement is the OOM victim, never a CI
# runner. llama.cpp and ollama are held to it as well as apr.
#
# Every status is captured as `cmd; rc=$?`, never through a pipe (#2336, #2360).
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd) || exit 2
cd "$ROOT" || exit 2

decline() { printf 'decline: %s\n' "$*" >&2; exit 2; }

VERSION=""
HOST_ID=""
BACKEND="gpu"
ENGINES="apr,llama.cpp,ollama"
VERBS="run"
OUT_DIR=""
PROMPTS="scripts/crux_inference_prompts.json"
TMO=600
LOCK_WAIT=3600
KEEP_OLLAMA=0
KEEP_WORK=0
MODELS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --model)   [ $# -ge 2 ] || decline "--model needs a value"; MODELS+=("$2"); shift 2 ;;
    --host)    [ $# -ge 2 ] || decline "--host needs a value"; HOST_ID="$2"; shift 2 ;;
    --backend) [ $# -ge 2 ] || decline "--backend needs a value"; BACKEND="$2"; shift 2 ;;
    --engines) [ $# -ge 2 ] || decline "--engines needs a value"; ENGINES="$2"; shift 2 ;;
    --verbs)   [ $# -ge 2 ] || decline "--verbs needs a value"; VERBS="$2"; shift 2 ;;
    --out)     [ $# -ge 2 ] || decline "--out needs a value"; OUT_DIR="$2"; shift 2 ;;
    --prompts) [ $# -ge 2 ] || decline "--prompts needs a value"; PROMPTS="$2"; shift 2 ;;
    --timeout) [ $# -ge 2 ] || decline "--timeout needs a value"; TMO="$2"; shift 2 ;;
    --keep-ollama-models) KEEP_OLLAMA=1; shift ;;
    --keep-work) KEEP_WORK=1; shift ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    -*) decline "unknown argument '$1'" ;;
    *) [ -z "$VERSION" ] || decline "one version, got '$VERSION' and '$1'"; VERSION="$1"; shift ;;
  esac
done
[ -n "$VERSION" ] || decline "usage: $0 <version> --model <gguf> ..."
[ "${#MODELS[@]}" -gt 0 ] || decline "no --model given; this slice takes models by path"
case "$BACKEND" in gpu|cpu) ;; *) decline "--backend is gpu or cpu, got '$BACKEND'" ;; esac
case ",$VERBS," in
  ,run,) ;;
  *) decline "verbs '$VERBS': this slice drives 'run' only; chat/serve/code are the next slices of #3739 and are listed as not covered in every receipt" ;;
esac
want() { case ",$ENGINES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
want apr || decline "apr is the subject; --engines must include it"
[ -f "$PROMPTS" ] || decline "prompt set $PROMPTS not found"

# ---- apr: pinned, and the version asked for ------------------------------------
if [ "${DOGFOOD_ALLOW_UNPINNED:-0}" = "1" ] && [ -n "${APR:-}" ]; then
  : # published-binary mode: the caller pinned $APR deliberately
else
  # shellcheck disable=SC1091
  . scripts/apr_bin.sh || decline "scripts/apr_bin.sh could not pin a HEAD-built apr"
fi
[ -x "${APR:-}" ] || decline "\$APR is not executable: '${APR:-}'"
APR_VERSION_LINE=$("$APR" --version 2>/dev/null | head -1)
APR_GOT=$(printf '%s\n' "$APR_VERSION_LINE" | sed -n 's/^apr \([0-9][0-9A-Za-z.+-]*\).*/\1/p')
[ "$APR_GOT" = "$VERSION" ] || decline "apr under test reports '$APR_VERSION_LINE'; this dogfood was asked for $VERSION"

host_id() {
  if [ -n "$HOST_ID" ]; then printf '%s' "$HOST_ID"; return; fi
  local h; h=$(hostname 2>/dev/null || echo unknown)
  case "$h" in
    gx10*) printf 'gx10' ;;
    *[Ll]ambda*) printf 'lambda' ;;
    *) printf '%s' "$h" ;;
  esac
}
HOST=$(host_id)
[ -n "$OUT_DIR" ] || OUT_DIR="evidence/crux/$VERSION"

# ---- llama.cpp: the pin, through the one resolver ---------------------------------
# shellcheck disable=SC1091
. scripts/llama_bin.sh >/dev/null 2>&1
LLAMA_RC=$?
TEMP=$(llama_pin_get_raw temperature 2>/dev/null)
SEED=$(llama_pin_get_raw seed 2>/dev/null)
CTX=$(llama_pin_get_raw context_length 2>/dev/null)
[ -n "$TEMP" ] && [ -n "$SEED" ] && [ -n "$CTX" ] || decline "scripts/llama_pin.toml [protocol] temperature/seed/context_length unreadable"
LLAMA_OK=0
LLAMA_WHY=""
if want llama.cpp; then
  if [ "$LLAMA_RC" -ne 0 ]; then
    LLAMA_WHY="llama.cpp unresolved: ${LLAMA_PIN_REASON:-unknown} (llama_bin rc $LLAMA_RC)"
  elif [ -z "${LLAMA_CLI:-}" ] || [ -z "${LLAMA_SERVER:-}" ]; then
    LLAMA_WHY="the pinned build has no chat CLI or no server binary beside it"
  else
    LLAMA_OK=1
  fi
fi

# ---- ollama: resolved by declaration, recorded by what the server says ----------
OLLAMA=""
OLLAMA_OK=0
OLLAMA_WHY=""
OLLAMA_CLIENT=""
OLLAMA_SERVER=""
OLLAMA_HOST_URL="http://${OLLAMA_HOST:-127.0.0.1:11434}"
if want ollama; then
  for cand in "${OLLAMA_BIN:-}" "$HOME/.local/bin/ollama" /usr/local/bin/ollama; do
    [ -n "$cand" ] && [ -f "$cand" ] && [ -x "$cand" ] && { OLLAMA="$cand"; break; }
  done
  if [ -z "$OLLAMA" ]; then
    OLLAMA_WHY="no ollama binary at \$OLLAMA_BIN, ~/.local/bin or /usr/local/bin"
  else
    OLLAMA_CLIENT=$("$OLLAMA" --version 2>&1 | sed -n 's/.*client version is \([0-9.]*\).*/\1/p;s/^ollama version is \([0-9.]*\).*/\1/p' | head -1)
    OLLAMA_SERVER=$(curl -sf "$OLLAMA_HOST_URL/api/version" 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("version",""))' 2>/dev/null)
    if [ -z "$OLLAMA_SERVER" ]; then
      OLLAMA_WHY="the ollama server at $OLLAMA_HOST_URL did not answer /api/version"
    else
      OLLAMA_OK=1
    fi
  fi
fi
[ "$LLAMA_OK" = 1 ] || [ "$OLLAMA_OK" = 1 ] || decline "no comparator: llama.cpp (${LLAMA_WHY:-not requested}); ollama (${OLLAMA_WHY:-not requested})"

# ---- work dir -------------------------------------------------------------------
WORK=$(mktemp -d) || decline "mktemp failed"
SRV_PID=""
# The delete is guarded (SEC011): only a path under a temp root is removed.
_cleanup() {
  [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null
  [ "$KEEP_WORK" = 1 ] && { printf 'work kept: %s\n' "$WORK" >&2; return 0; }
  local v="${WORK:-}"
  case "$v" in
    /tmp/?*|/var/folders/?*) if [ -n "$v" ] && [ "$v" != "/" ]; then rm -rf -- "$v" || :; fi ;;
    *) return 0 ;;
  esac
}
trap _cleanup EXIT
MANIFEST="$WORK/manifest.jsonl"; : > "$MANIFEST"
MODELS_JSONL="$WORK/models.jsonl"; : > "$MODELS_JSONL"

# One file per prompt: its id list, and each prompt's last user message.
PIDS=$(python3 - "$PROMPTS" "$WORK" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for p in d["prompts"]:
    open("%s/prompt-%s.txt" % (sys.argv[2], p["id"]), "w").write(p["messages"][-1]["content"])
    json.dump({"messages": p["messages"]}, open("%s/messages-%s.json" % (sys.argv[2], p["id"]), "w"))
    print(p["id"])
PY
) || decline "prompt set $PROMPTS unreadable"
MAXTOK=$(python3 -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["max_tokens"]))' "$PROMPTS") || decline "max_tokens unreadable"

# gpu_run <stdout> <stderr> cmd... — one engine invocation under the cop's rule.
# flock exits 75 (not 1) when the lock is not had in time, so a lock wait can
# never be read as an engine's own failure. The cpu lane touches no GPU compute
# and takes no lock (the rule's clause 2); it still runs under choom.
gpu_run() {
  local o="$1" e="$2"; shift 2
  if [ "$BACKEND" = gpu ]; then
    flock -w "$LOCK_WAIT" -E 75 /tmp/apr-gpu.lock choom -n 1000 -- timeout "$TMO" "$@" > "$o" 2> "$e" < /dev/null
  else
    choom -n 1000 -- timeout "$TMO" "$@" > "$o" 2> "$e" < /dev/null
  fi
}

emit_gen() { # emit_gen <engine> <prompt_id> <rc> <stdout> <stderr> <refused>
  python3 - "$MANIFEST" "$1" "$2" "$3" "$4" "$5" "$6" "$SHA" "$HOST" "$VERB" "$THINK" "$BACKEND" <<'PY'
import json, sys
m, eng, pid, rc, o, e, ref, sha, host, verb, think, be = sys.argv[1:13]
row = {"kind": "gen", "engine": eng, "prompt_id": pid,
       "rc": int(rc) if rc.lstrip("-").isdigit() else None,
       "stdout": o or None, "stderr": e or None, "refused": ref or None,
       "model_sha256": sha, "host": host, "verb": verb, "thinking": think, "backend": be}
open(m, "a").write(json.dumps(row) + "\n")
PY
}

free_port() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }

# llama.cpp's OWN rendering and tokenization of each prompt: the reference apr's
# prompt ids are compared against. The server runs on the CPU (-ngl 0), only
# long enough to answer /props, /apply-template and /tokenize.
llama_tokenize_prompts() { # sets THINKING_CAPABLE; writes tok rows
  local port d="$WORK/$SHA12/tok" pid ok=0 i
  mkdir -p "$d"
  port=$(free_port) || return 1
  choom -n 1000 -- "$LLAMA_SERVER" -m "$M" -ngl 0 -c 512 --no-warmup --host 127.0.0.1 --port "$port" \
    > "$d/server.log" 2>&1 < /dev/null &
  SRV_PID=$!
  for i in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1 && { ok=1; break; }
    kill -0 "$SRV_PID" 2>/dev/null || break
    sleep 1
  done
  if [ "$ok" = 1 ]; then
    curl -sf "http://127.0.0.1:$port/props" > "$d/props.json" 2>/dev/null
    THINKING_CAPABLE=$(python3 -c 'import json,sys
t = json.load(open(sys.argv[1])).get("chat_template") or ""
print("true" if ("enable_thinking" in t or "<think>" in t) else "false")' "$d/props.json" 2>/dev/null || echo unknown)
    for pid in $PIDS; do
      curl -sf -X POST "http://127.0.0.1:$port/apply-template" -H 'Content-Type: application/json' \
        --data-binary @"$WORK/messages-$pid.json" > "$d/rendered-$pid.json" 2>/dev/null
      python3 -c 'import json,sys
r = json.load(open(sys.argv[1]))
json.dump({"content": r["prompt"], "add_special": True, "parse_special": True}, open(sys.argv[2], "w"))' \
        "$d/rendered-$pid.json" "$d/tokreq-$pid.json" 2>/dev/null || continue
      curl -sf -X POST "http://127.0.0.1:$port/tokenize" -H 'Content-Type: application/json' \
        --data-binary @"$d/tokreq-$pid.json" > "$d/ids-$pid.json" 2>/dev/null || continue
      python3 - "$MANIFEST" "$SHA" "$pid" "$d/rendered-$pid.json" "$d/ids-$pid.json" <<'PY'
import json, sys
m, sha, pid, rendered, ids = sys.argv[1:6]
open(m, "a").write(json.dumps({"kind": "tok", "engine": "llama.cpp", "model_sha256": sha,
                               "prompt_id": pid, "rendered": rendered, "ids": ids}) + "\n")
PY
    done
  fi
  kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null; SRV_PID=""
  [ "$ok" = 1 ]
}

# ---- the run ---------------------------------------------------------------------
printf -- '--- CRUX inference dogfood %s on %s (%s lane) ---\n' "$VERSION" "$HOST" "$BACKEND"
printf '  apr       %s\n' "$APR_VERSION_LINE"
printf '  llama.cpp %s\n' "$( [ "$LLAMA_OK" = 1 ] && printf '%s' "${LLAMA_BUILD:-?}" || printf 'UNAVAILABLE: %s' "${LLAMA_WHY:-not requested}")"
printf '  ollama    %s\n' "$( [ "$OLLAMA_OK" = 1 ] && printf 'server %s (client %s)' "$OLLAMA_SERVER" "${OLLAMA_CLIENT:-?}" || printf 'UNAVAILABLE: %s' "${OLLAMA_WHY:-not requested}")"

case "$BACKEND" in
  gpu) APR_BE="--gpu"; NGL=999 ;;
  cpu) APR_BE="--no-gpu"; NGL=0 ;;
esac
VERB=run
for M_IN in "${MODELS[@]}"; do
  M=$(readlink -f "$M_IN" 2>/dev/null) || M=""
  [ -n "$M" ] && [ -f "$M" ] || decline "model not found: $M_IN"
  SHA=$(sha256sum "$M" | cut -d' ' -f1)
  SHA12=${SHA:0:12}
  NAME=$(basename "$M")
  mkdir -p "$WORK/$SHA12"
  printf '  model %s  sha256 %s\n' "$NAME" "$SHA"

  THINKING_CAPABLE=unknown
  TOK_WHY=""
  if [ "$LLAMA_OK" = 1 ]; then
    llama_tokenize_prompts || TOK_WHY="llama.cpp server did not come up for tokenization (see its log)"
  fi

  # ollama: import THIS file, then read back what ollama made of it.
  OL_NAME="crux-$SHA12"
  OL_REFUSED=""
  OL_BLOB=""
  OL_TEMPLATE_SHA=""
  if [ "$OLLAMA_OK" = 1 ]; then
    {
      printf 'FROM %s\n' "$M"
      printf 'PARAMETER temperature %s\nPARAMETER seed %s\nPARAMETER num_predict %s\nPARAMETER num_ctx %s\n' "$TEMP" "$SEED" "$MAXTOK" "$CTX"
      [ "$BACKEND" = cpu ] && printf 'PARAMETER num_gpu 0\n'
    } > "$WORK/$SHA12/Modelfile"
    "$OLLAMA" create "$OL_NAME" -f "$WORK/$SHA12/Modelfile" > "$WORK/$SHA12/ollama-create.log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
      OL_REFUSED="ollama create failed (rc $rc): $(tail -1 "$WORK/$SHA12/ollama-create.log" | tr -d '\r' | cut -c1-160)"
    else
      "$OLLAMA" show "$OL_NAME" --template > "$WORK/$SHA12/ollama-template.txt" 2>/dev/null
      "$OLLAMA" show "$OL_NAME" --modelfile > "$WORK/$SHA12/ollama-modelfile.txt" 2>/dev/null
      OL_BLOB=$(sed -n 's/^FROM .*sha256[-:]\([0-9a-f]\{64\}\).*/\1/p' "$WORK/$SHA12/ollama-modelfile.txt" | head -1)
      OL_TEMPLATE_SHA=$(sha256sum "$WORK/$SHA12/ollama-template.txt" | cut -d' ' -f1)
      # ollama 0.5.7 imports a GGUF with TEMPLATE {{ .Prompt }}: no chat template
      # at all, so it would complete the raw prompt while the others chat. That is
      # not an answer to the same question, and it is refused by name.
      if [ "$(tr -d ' \n' < "$WORK/$SHA12/ollama-template.txt")" = "{{.Prompt}}" ]; then
        OL_REFUSED="ollama $OLLAMA_SERVER imported the GGUF with no chat template (TEMPLATE {{ .Prompt }}), so it would complete the raw text rather than chat"
      fi
    fi
  fi

  python3 - "$MODELS_JSONL" "$NAME" "$SHA" "$THINKING_CAPABLE" "$OL_BLOB" "$OL_TEMPLATE_SHA" "$OL_REFUSED" "$TOK_WHY" <<'PY'
import json, sys
f, name, sha, think, blob, tsha, oref, twhy = sys.argv[1:9]
open(f, "a").write(json.dumps({"name": name, "sha256": sha,
    "thinking_capable": {"true": True, "false": False}.get(think),
    "ollama": {"blob_sha256": blob or None, "blob_is_input": (blob == sha) if blob else None,
               "template_sha256": tsha or None, "refused": oref or None},
    "tokenization": {"refused": twhy or None}}) + "\n")
PY

  # Thinking OFF only in this slice. apr has no toggle yet (#3723), and a
  # thinking-capable model is told OFF explicitly by the comparators, which
  # otherwise default to thinking.
  THINK=off
  LLAMA_THINK=()
  OLLAMA_THINK=()
  if [ "$THINKING_CAPABLE" = true ]; then
    LLAMA_THINK=(--reasoning off)
    [ "$OLLAMA_OK" = 1 ] && "$OLLAMA" run --help 2>&1 | grep -q -- '--think' && OLLAMA_THINK=(--think=false)
  fi

  for pid in $PIDS; do
    content=$(cat "$WORK/prompt-$pid.txt")
    d="$WORK/$SHA12/$VERB"
    mkdir -p "$d"

    gpu_run "$d/apr-$pid.out" "$d/apr-$pid.err" "$APR" run "$M" --prompt "$content" --max-tokens "$MAXTOK" \
      --temperature "$TEMP" --seed "$SEED" --format json -v "$APR_BE"
    rc=$?
    ref=""; [ "$rc" = 75 ] && ref="the GPU lock was not had within ${LOCK_WAIT}s"
    emit_gen apr "$pid" "$rc" "$d/apr-$pid.out" "$d/apr-$pid.err" "$ref"

    if [ "$LLAMA_OK" = 1 ]; then
      gpu_run "$d/llama-$pid.out" "$d/llama-$pid.err" "$LLAMA_CLI" -m "$M" -p "$content" -st -n "$MAXTOK" \
        --temp "$TEMP" --seed "$SEED" -c "$CTX" -ngl "$NGL" "${LLAMA_THINK[@]}"
      rc=$?
      ref=""; [ "$rc" = 75 ] && ref="the GPU lock was not had within ${LOCK_WAIT}s"
      emit_gen llama.cpp "$pid" "$rc" "$d/llama-$pid.out" "$d/llama-$pid.err" "$ref"
    elif want llama.cpp; then
      emit_gen llama.cpp "$pid" "" "" "" "$LLAMA_WHY"
    fi

    if [ "$OLLAMA_OK" = 1 ] && [ -z "$OL_REFUSED" ]; then
      gpu_run "$d/ollama-$pid.out" "$d/ollama-$pid.err" "$OLLAMA" run "$OL_NAME" "$content" --verbose --nowordwrap \
        "${OLLAMA_THINK[@]}"
      rc=$?
      ref=""; [ "$rc" = 75 ] && ref="the GPU lock was not had within ${LOCK_WAIT}s"
      emit_gen ollama "$pid" "$rc" "$d/ollama-$pid.out" "$d/ollama-$pid.err" "$ref"
    elif want ollama; then
      emit_gen ollama "$pid" "" "" "" "${OL_REFUSED:-$OLLAMA_WHY}"
    fi
  done

  if [ "$OLLAMA_OK" = 1 ]; then
    "$OLLAMA" stop "$OL_NAME" > /dev/null 2>&1
    [ "$KEEP_OLLAMA" = 1 ] || "$OLLAMA" rm "$OL_NAME" > /dev/null 2>&1
  fi
done

# ---- judge --------------------------------------------------------------------------
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
python3 - "$WORK/meta.json" "$MODELS_JSONL" "$VERSION" "$HOST" "$BACKEND" "$ENGINES" "$VERBS" \
  "$APR_VERSION_LINE" "${LLAMA_BUILD:-}" "$(llama_pin_get build_commit 2>/dev/null)" "$LLAMA_WHY" \
  "$OLLAMA_SERVER" "$OLLAMA_CLIENT" "$OLLAMA_WHY" "$TEMP" "$SEED" "$CTX" "$MAXTOK" "${GPU_NAME:-}" <<'PY'
import json, platform, sys
(out, models, version, host, backend, engines, verbs, apr_line, lbuild, lpin, lwhy,
 osrv, ocli, owhy, temp, seed, ctx, maxtok, gpu) = sys.argv[1:20]
meta = {
    "version": version, "host": host, "backend": backend, "isa": platform.machine(), "gpu": gpu or None,
    "engines": engines.split(","), "verbs": verbs.split(","), "thinking": ["off"],
    "sampling": {"temperature": float(temp), "seed": int(seed), "context": int(ctx), "max_tokens": int(maxtok),
                 "source": "scripts/llama_pin.toml [protocol]"},
    "apr": {"version_line": apr_line},
    "llama_cpp": {"build": lbuild or None, "pin": lpin or None, "unavailable": lwhy or None},
    "ollama": {"server_version": osrv or None, "client_version": ocli or None, "unavailable": owhy or None},
    "models": [json.loads(l) for l in open(models) if l.strip()],
    "not_covered": [
        "verbs chat, serve and code (the next slices of #3739)",
        "thinking ON (apr has no toggle until #3723)",
        "consumer-brief context rungs (#3716) and each engine's max accepted context",
        "TTFT, which belongs to the serve verb's single OpenAI client",
    ],
}
json.dump(meta, open(out, "w"), indent=2)
PY
mkdir -p "$OUT_DIR" || decline "cannot create $OUT_DIR"
python3 scripts/lib/crux_inference_judge.py collect --manifest "$MANIFEST" --prompts "$PROMPTS" \
  --meta "$WORK/meta.json" --out-json "$OUT_DIR/$HOST-$BACKEND.json" --out-md "$OUT_DIR/$HOST-$BACKEND.md"
rc=$?
printf 'receipt: %s/%s-%s.json (judge rc %s)\n' "$OUT_DIR" "$HOST" "$BACKEND" "$rc"
exit "$rc"
