#!/usr/bin/env bash
# scripts/crux_engine_llamafile.sh — the llamafile CRUX engine (#3739, PMAT-3778).
#
#   crux_engine_llamafile.sh probe
#   crux_engine_llamafile.sh gen --model <gguf> --model-sha256 <hex> --verb run|chat|'serve run'|code \
#       --prompt-id <id> --messages <json> [--prompt-file <p>] --thinking on|off|unset \
#       --backend gpu|cpu --host <h> --max-tokens N --seed S --temperature T --context N
#
# Row contract v1 (aprender-76, #3739 comment 5765991210): `gen` appends exactly ONE JSONL row to
# $CRUX_MANIFEST and writes its artifacts under $CRUX_WORK/<model_sha256[:12]>/<verb>/. A cell llamafile
# cannot run is still a row — `rc: null`, `refused: "<llamafile's own words>"` — never an absence.
# `tok`, `tmpl` and `greedy` are `none` in llamafile's correspondence column, so they exit 2.
#
# THE BINARY IS DECLARED, NEVER FOUND. forjar (paiml/infra#906) installs one sha256-pinned release to
# ~/.local/lib/crux/llamafile. The pin has ONE owner, that forjar resource; `probe` MEASURES the file it
# will execute (version text + sha256) and the receipt carries both, so a replaced binary shows up there.
#
# TWO INTERFACES, chosen per verb as the correspondence maps them: `run`/`chat` through `--cli`,
# `serve run`/`code` through the OpenAI-compatible server. Measured 2026-09-21 on 0.10.6 with
# Qwen3.5-0.8B: `--cli` IGNORES both `-rea off` and `--chat-template-kwargs '{"enable_thinking":false}'`
# (byte-identical output, a <think> block every time), while the server honours `enable_thinking` per
# request and returns the reasoning SEPARATELY. So a `--cli` cell asked for thinking=off that thinks
# anyway is recorded as such (`reported.thinking_emitted: true`), and its <think> block is split into
# `reasoning` — the answer judged is only what follows `</think>`.
# `--cli` also does not bound its output with `-n` (measured: `-n 8` printed 100 numbers). `--interface
# server` runs `run`/`chat` through the server too, which honours both; the mapping is the correspondence
# owner's call (#3774), so the default stays the correspondence's.
set -euo pipefail

LLAMAFILE=${CRUX_LLAMAFILE:-$HOME/.local/lib/crux/llamafile}

die() { printf 'crux_engine_llamafile: %s\n' "$*" >&2; exit 2; }

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# The file the symlink names, so the hash is of what runs.
real_path() {
  local p=$1
  while [ -L "$p" ]; do
    local t
    t=$(readlink "$p")
    case "$t" in /*) p=$t ;; *) p=$(dirname "$p")/$t ;; esac
  done
  printf '%s' "$p"
}

probe() {
  if [ ! -x "$LLAMAFILE" ]; then
    echo "llamafile is not installed at $LLAMAFILE (forjar resource crux-llamafile, paiml/infra#906)" >&2
    exit 3
  fi
  local ver rc=0 real
  ver=$("$LLAMAFILE" --version 2>&1) || rc=$?
  if [ "$rc" != 0 ] || [ -z "$ver" ]; then
    echo "llamafile --version exited $rc: ${ver:-<no output>}" >&2
    exit 3
  fi
  real=$(real_path "$LLAMAFILE")
  printf '%s sha256=%s\n' "$ver" "$(sha256_of "$real")"
}

# ── gen ─────────────────────────────────────────────────────────────────
MODEL="" MODEL_SHA="" VERB="" PROMPT_ID="" MESSAGES="" PROMPT_FILE="" THINKING="" BACKEND="" HOST=""
MAX_TOKENS="" SEED="" TEMP="" CONTEXT="" INTERFACE=""

parse_gen() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --model) MODEL=${2:-}; shift 2 ;;
      --model-sha256) MODEL_SHA=${2:-}; shift 2 ;;
      --verb) VERB=${2:-}; shift 2 ;;
      --prompt-id) PROMPT_ID=${2:-}; shift 2 ;;
      --messages) MESSAGES=${2:-}; shift 2 ;;
      --prompt-file) PROMPT_FILE=${2:-}; shift 2 ;;
      --thinking) THINKING=${2:-}; shift 2 ;;
      --backend) BACKEND=${2:-}; shift 2 ;;
      --host) HOST=${2:-}; shift 2 ;;
      --max-tokens) MAX_TOKENS=${2:-}; shift 2 ;;
      --seed) SEED=${2:-}; shift 2 ;;
      --temperature) TEMP=${2:-}; shift 2 ;;
      --context) CONTEXT=${2:-}; shift 2 ;;
      --interface) INTERFACE=${2:-}; shift 2 ;;
      *) die "gen: unknown argument $1" ;;
    esac
  done
  local v
  for v in MODEL MODEL_SHA VERB PROMPT_ID MESSAGES THINKING BACKEND HOST MAX_TOKENS SEED TEMP CONTEXT; do
    [ -n "${!v}" ] || die "gen: --$(printf '%s' "$v" | tr 'A-Z_' 'a-z-') is required"
  done
  case "$VERB" in run|chat|"serve run"|code) ;; *) die "gen: --verb must be run|chat|'serve run'|code, got '$VERB'" ;; esac
  case "$THINKING" in on|off|unset) ;; *) die "gen: --thinking must be on|off|unset" ;; esac
  case "$BACKEND" in gpu|cpu) ;; *) die "gen: --backend must be gpu|cpu" ;; esac
  case "$INTERFACE" in ""|cli|server) ;; *) die "gen: --interface must be cli|server" ;; esac
  [ -f "$MODEL" ] || die "gen: --model $MODEL is not a file"
  [ -f "$MESSAGES" ] || die "gen: --messages $MESSAGES is not a file"
  if [ "$VERB" = chat ] && [ "$INTERFACE" != server ] \
     && [ "$(jq '[.messages[] | select(.role == "user")] | length' "$MESSAGES")" -gt 1 ]; then
    die "gen: a chat with more than one user turn needs --interface server — llamafile --cli answers one prompt"
  fi
  [ -n "${CRUX_MANIFEST:-}" ] || die "CRUX_MANIFEST is unset"
  [ -n "${CRUX_WORK:-}" ] || die "CRUX_WORK is unset"
  case "$MODEL_SHA" in *[!0-9a-f]*|"") die "gen: --model-sha256 must be lowercase hex" ;; esac
  [ "${#MODEL_SHA}" = 64 ] || die "gen: --model-sha256 must be 64 hex characters"
}

# The model file must BE the sha the row names. Hashed once per model per run: a GGUF is gigabytes.
verify_model() {
  local mark="$CRUX_WORK/${MODEL_SHA:0:12}/.llamafile-model-verified" got
  mkdir -p "$(dirname "$mark")"
  if [ -f "$mark" ] && [ "$(cat "$mark")" = "$MODEL" ]; then return 0; fi
  got=$(sha256_of "$MODEL")
  [ "$got" = "$MODEL_SHA" ] || die "gen: $MODEL hashes to $got, not the --model-sha256 $MODEL_SHA"
  printf '%s' "$MODEL" > "$mark"
}

gpu_args() {
  if [ "$BACKEND" = gpu ]; then printf '%s\n' --gpu auto -ngl 999; else printf '%s\n' --gpu disable; fi
}

# The text of the last message with a given role, from {"messages":[…]}.
message_of() { jq -r --arg r "$1" '[.messages[] | select(.role == $r) | .content] | last // ""' "$MESSAGES"; }

# Splits a raw completion into answer and reasoning. A <think> with no </think> is a generation that
# ran out inside its reasoning: the answer is EMPTY, never the reasoning passed off as one.
split_think() {  # <raw file> <answer out> <reasoning out>
  awk -v A="$2" -v R="$3" '
    { buf = buf $0 "\n" }
    END {
      s = index(buf, "<think>"); e = index(buf, "</think>")
      if (s == 0) { printf "%s", buf > A; printf "" > R; exit }
      if (e == 0) { printf "" > A; printf "%s", substr(buf, s + 7) > R; exit }
      printf "%s", substr(buf, e + 8) > A
      printf "%s", substr(buf, s + 7, e - s - 7) > R
    }' "$1"
}

run_cli() {  # <dir> <stem> -> writes <stem>.raw, <stem>.err; echoes rc
  local dir=$1 stem=$2 user system rc=0
  user=$(message_of user)
  system=$(message_of system)
  [ -n "$PROMPT_FILE" ] && [ -f "$PROMPT_FILE" ] && user=$(cat "$PROMPT_FILE")
  set -- -m "$MODEL" --cli -p "$user" -n "$MAX_TOKENS" --temp "$TEMP" --seed "$SEED" -c "$CONTEXT" --nologo
  [ -n "$system" ] && set -- "$@" -sys "$system"
  case "$THINKING" in
    on) set -- "$@" -rea on ;;
    off) set -- "$@" -rea off --chat-template-kwargs '{"enable_thinking":false}' ;;
  esac
  while IFS= read -r a; do set -- "$@" "$a"; done < <(gpu_args)
  "$LLAMAFILE" "$@" > "$dir/$stem.raw" 2> "$dir/$stem.err" || rc=$?
  printf '%s' "$rc"
}

free_port() {
  local p
  for p in $(seq 18800 18899); do
    if ! (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null; then printf '%s' "$p"; return 0; fi
  done
  return 1
}

# The server's own pid is not the one we launch: the APE re-execs itself (measured). So it is stopped by
# asking the kernel which process LISTENS on the port, and the port is waited on until it is free.
stop_server() {  # <port> <launched pid>
  local port=$1 pid=$2 p i
  kill "$pid" 2>/dev/null || true
  for p in $(ss -ltnp 2>/dev/null | grep ":$port " | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u); do kill "$p" 2>/dev/null || true; done
  for i in $(seq 1 20); do
    (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null || return 0
    sleep 0.5
  done
  for p in $(ss -ltnp 2>/dev/null | grep ":$port " | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u); do kill -9 "$p" 2>/dev/null || true; done
  kill -9 "$pid" 2>/dev/null || true
}

run_server() {  # <dir> <stem> -> writes <stem>.resp, <stem>.err; echoes rc (0 ok, 1 server failed)
  local dir=$1 stem=$2 port pid i up=0 body kwargs
  port=$(free_port) || { echo "no free port in 18800-18899" > "$dir/$stem.err"; printf 1; return 0; }
  set -- -m "$MODEL" --server --host 127.0.0.1 --port "$port" -c "$CONTEXT" --nologo
  while IFS= read -r a; do set -- "$@" "$a"; done < <(gpu_args)
  "$LLAMAFILE" "$@" > "$dir/$stem.server.out" 2> "$dir/$stem.err" < /dev/null &
  pid=$!
  for i in $(seq 1 240); do
    if curl -sf -o /dev/null "http://127.0.0.1:$port/health"; then up=1; break; fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.5
  done
  if [ "$up" != 1 ]; then stop_server "$port" "$pid"; printf 1; return 0; fi
  kwargs='{}'
  case "$THINKING" in on) kwargs='{"enable_thinking":true}' ;; off) kwargs='{"enable_thinking":false}' ;; esac
  # One request over a message list; the response lands in <stem>.resp.
  post() {  # <messages json array>
    body=$(jq -n -c --argjson m "$1" --argjson max "$MAX_TOKENS" --argjson t "$TEMP" --argjson s "$SEED" --argjson kw "$kwargs" \
             '{messages:$m, max_tokens:$max, temperature:$t, seed:$s} + (if $kw == {} then {} else {chat_template_kwargs:$kw} end)')
    curl -sf "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" > "$dir/$stem.resp" 2>> "$dir/$stem.err"
  }
  local ok=0
  if [ "$VERB" = chat ]; then
    # aprender-76 (#3739): `--messages` holds the USER turns; the engine drives the conversation. Each user
    # turn is answered with the conversation so far, and the ANSWER (the server returns reasoning separately)
    # is appended as the assistant turn. Every answer, in order, goes to <stem>.turns.json.
    local convo='[]' turns='[]' n j role reply
    n=$(jq '.messages | length' "$MESSAGES")
    for j in $(seq 0 $((n - 1))); do
      convo=$(jq -c --argjson c "$convo" --argjson j "$j" '$c + [.messages[$j]]' "$MESSAGES")
      role=$(jq -r --argjson j "$j" '.messages[$j].role' "$MESSAGES")
      [ "$role" = user ] || continue
      post "$convo" || { ok=1; break; }
      reply=$(jq -r '(.choices[0].message.content // "") | sub("^\\s+"; "") | sub("\\s+$"; "")' "$dir/$stem.resp")
      convo=$(jq -n -c --argjson c "$convo" --arg r "$reply" '$c + [{role: "assistant", content: $r}]')
      turns=$(jq -n -c --argjson t "$turns" --arg r "$reply" '$t + [$r]')
    done
    if [ "$ok" = 0 ] && [ "$turns" = '[]' ]; then echo "--messages holds no user turn to answer" >> "$dir/$stem.err"; ok=1; fi
    [ "$ok" = 0 ] && printf '%s' "$turns" > "$dir/$stem.turns.json"
  else
    post "$(jq -c '.messages' "$MESSAGES")" || ok=1
  fi
  stop_server "$port" "$pid"
  printf '%s' "$ok"
}

# The device the cell ran on, as the judge requires it (aprender-76: a row off its lane is no answer).
# cpu lane: `--gpu disable`, so cpu by construction. gpu lane: llama.cpp's own load line
# "offloaded N/M layers to GPU"; without it the label starts with `cpu`, so the judge REJECTS the cell on
# the gpu lane rather than trusting a GPU nobody measured.
device_label() {  # <stderr file>
  local line
  if [ "$BACKEND" = cpu ]; then printf 'cpu (llamafile --gpu disable)'; return 0; fi
  line=$(grep -o -E 'offloaded [0-9]+/[0-9]+ layers to GPU' "$1" 2>/dev/null | tail -n 1 || true)
  case "$line" in
    "offloaded 0/"*|"") printf 'cpu (llamafile reported no GPU offload)' ;;
    *) printf 'gpu (llamafile %s)' "$line" ;;
  esac
}

# llamafile's own words when it could not run the cell: its error lines, else the tail of stderr.
refusal_text() {  # <err file>
  local t
  t=$(grep -i -E 'error|unknown|unsupported|failed|cannot|not supported' "$1" 2>/dev/null | tail -n 5 || true)
  [ -n "$t" ] || t=$(tail -n 5 "$1" 2>/dev/null || true)
  printf '%s' "${t:-llamafile exited without output}"
}

gen() {
  parse_gen "$@"
  verify_model
  local slug dir stem rc answer reasoning out emitted
  slug=$(printf '%s' "$VERB" | tr ' ' '-')
  dir="$CRUX_WORK/${MODEL_SHA:0:12}/$slug"
  mkdir -p "$dir"
  stem="llamafile-$PROMPT_ID-$THINKING"
  out="$dir/$stem.json"
  local interface=cli refused="" row_rc
  if [ "$INTERFACE" != server ] && { [ "$VERB" = run ] || [ "$VERB" = chat ]; }; then
    rc=$(run_cli "$dir" "$stem")
    if [ "$rc" = 0 ]; then
      split_think "$dir/$stem.raw" "$dir/$stem.answer" "$dir/$stem.reasoning"
      emitted=false; grep -q '<think>' "$dir/$stem.raw" && emitted=true
      jq -n --rawfile a "$dir/$stem.answer" --rawfile r "$dir/$stem.reasoning" --argjson e "$emitted" --arg th "$THINKING" \
            --arg dev "$(device_label "$dir/$stem.err")" '
        {text: ($a | sub("^\\s+"; "") | sub("\\s+$"; ""))}
        + (if ($r | length) > 0 then {reasoning: ($r | sub("^\\s+"; "") | sub("\\s+$"; ""))} else {} end)
        + {reported: {interface: "cli", thinking_requested: $th, thinking_emitted: $e, device: $dev}}' > "$out"
    fi
  else
    interface=server
    rc=$(run_server "$dir" "$stem")
    if [ "$rc" = 0 ]; then
      local turns_json=null
      [ -f "$dir/$stem.turns.json" ] && turns_json=$(cat "$dir/$stem.turns.json")
      jq --arg th "$THINKING" --arg dev "$(device_label "$dir/$stem.err")" --argjson turns "$turns_json" '
        .choices[0].message as $m
        | {text: (($m.content // "") | sub("^\\s+"; "") | sub("\\s+$"; ""))}
          + (if $turns == null then {} else {turns: $turns} end)
          + (if (($m.reasoning_content // "") | length) > 0 then {reasoning: $m.reasoning_content} else {} end)
          + {reported: {interface: "server", thinking_requested: $th,
                        thinking_emitted: ((($m.reasoning_content // "") | length) > 0),
                        prompt_tokens: .usage.prompt_tokens, completion_tokens: .usage.completion_tokens,
                        timings: .timings, device: $dev}}' "$dir/$stem.resp" > "$out"
    fi
  fi
  if [ "$rc" = 0 ]; then row_rc=0; else row_rc=null; refused=$(refusal_text "$dir/$stem.err"); fi
  jq -n -c --arg model_sha256 "$MODEL_SHA" --arg host "$HOST" --arg verb "$VERB" --arg thinking "$THINKING" \
        --arg backend "$BACKEND" --arg prompt_id "$PROMPT_ID" --argjson rc "$row_rc" \
        --arg stdout "$out" --arg stderr "$dir/$stem.err" --arg refused "$refused" --arg interface "$interface" '
    {kind:"gen", engine:"llamafile", model_sha256:$model_sha256, host:$host, verb:$verb, thinking:$thinking,
     backend:$backend, prompt_id:$prompt_id, rc:$rc,
     stdout:(if $rc == null then null else $stdout end), stderr:$stderr,
     refused:(if $refused == "" then null else $refused end), interface:$interface}' >> "$CRUX_MANIFEST"
}

case "${1:-}" in
  probe) probe ;;
  gen) shift; gen "$@" ;;
  tok|tmpl|greedy) die "$1: not a llamafile interface — the correspondence column is none for it (#3774)" ;;
  *) die "usage: $0 probe | gen --model … (see the header)" ;;
esac
