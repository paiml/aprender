# shellcheck shell=bash
# crux_cells_serve_code.sh: the CRUX `serve` and `code` cells (#3962). Sourced by
# scripts/crux_inference_dogfood.sh and never run on its own. It is OPTION-NEUTRAL (no
# `set`), per the CLAUDE.md rule for sourced libraries, and it reads the driver's
# globals: WORK, M, SHA, SHA12, APR, APR_BE, BACKEND, HOST, THINK, TEMP, SEED, CTX,
# TMO, NGL, LLAMA_DEV, LLAMA_OK, LLAMA_SERVER, LLAMA_DEVICE, OLLAMA_OK, OL_REFUSED,
# OL_NAME, OL_DEVICE, OLLAMA_HOST_URL, MANIFEST, EXT_OK, EXT_SCRIPT, EXT_WHY, HF_SRC,
# HF_MODEL_WHY, CELL_WHY. It also uses the driver's cell_add, run_cell,
# cell_result, cell_add_ollama_unload, emit_gen, rows_for, free_port and
# serve_wait_line.
#
# The prompt files it reads come from the driver's parse step, under $WORK:
#   prompt-<id>.json    the whole prompt entry (v1 or v2)
#   maxtok-<id>.txt     max_tokens for this prompt at the lane's thinking mode
#   serve-prompts.jsonl [id, [verbs]] for every prompt that has a serve verb
#   code-prompts.txt    one id per line for every prompt that has the code verb
#
# SERVE, one cell per model, under one hold of the GPU lock:
#   apr        `apr serve run` loads the model once. crux_serve_routes.py `sweep`
#              then reads the server's OWN `GET /` route index, classifies it,
#              and drives every generation route x every mode its wire has x
#              every serve prompt. An unclassified route, or a missing index,
#              is a RED row per prompt, never an absence.
#   llama.cpp  llama-server on the same GGUF, through its OpenAI chat route in
#   ollama     both modes (the same-quant differential, quorum Q2a). Its
#              /apply-template is also the REFERENCE RENDERER for apr's
#              raw-prompt routes.
#   plugins    hf / llamafile / vllm `gen --verb 'serve run'` and `--verb 'serve stream'`
#              (their own server), the bf16 row every serve GREEN needs.
#
# CODE, one cell per model:
#   apr        `apr code -p` via crux_apr_code.py. apr code has no backend flag and
#              always spawns `apr serve --gpu`, so on the cpu lane it is refused
#              with that reason and the judge counts it as not green.
#   llama.cpp  the same prompt through llama-server's OpenAI chat route
#   ollama     (non-streaming), the ggml vote.
#   plugins    `gen --verb code`, the hf / vllm votes.
# The oracle for every code row is EXECUTION (crux_oracles.py code_tests). The
# producer never looks inside the reply.

crux_plugin_engines() { # the plugin engine list: the driver's, else the pre-#3952 pair
  if declare -p PLUGIN_ENGINES > /dev/null 2>&1; then printf '%s\n' "${PLUGIN_ENGINES[@]}"; else printf 'hf\nllamafile\n'; fi
}
crux_lib_batched() { # the driver runs this engine's items as a batch cell (#4036); a standalone caller never does
  declare -F crux_batched > /dev/null 2>&1 && crux_batched "$1"
}
crux_is_source_engine() { # engines that run the SOURCE weights and need an HF source declared
  if declare -F source_engine > /dev/null 2>&1; then source_engine "$1"; else [ "$1" = hf ]; fi
}

# crux_plugin_lines <cell> <dir> <verb> <prompt ids...>: one `gen` line per wanted plugin engine
crux_plugin_lines() {
  local cell="$1" d="$2" verb="$3" eng pid ext_run ext_extra
  shift 3
  for eng in $(crux_plugin_engines); do
    want "$eng" && [ "${EXT_OK[$eng]:-0}" = 1 ] || continue
    crux_is_source_engine "$eng" && [ -n "$HF_MODEL_WHY" ] && continue
    crux_lib_batched "$eng" && continue
    case "${EXT_SCRIPT[$eng]}" in *.py) ext_run=(python3) ;; *) ext_run=(bash) ;; esac
    ext_extra=()
    [ "$eng" = llamafile ] && ext_extra=(--interface server)
    crux_is_source_engine "$eng" && ext_extra=("${HF_SRC[@]}")
    for pid in "$@"; do
      cell_add "$cell" "$d/$eng-$pid.driver" "${ext_run[@]}" "${EXT_SCRIPT[$eng]}" gen \
        --model "$M" --model-sha256 "$SHA" --verb "$verb" --prompt-id "$pid" \
        --messages "$WORK/messages-$pid.json" --prompt-file "$WORK/prompt-$pid.txt" \
        --thinking "$THINK" --backend "$BACKEND" --host "$HOST" \
        --max-tokens "$(cat "$WORK/maxtok-$pid.txt")" --seed "$SEED" --temperature "$TEMP" --context "$CTX" \
        "${ext_extra[@]}"
    done
  done
}

# crux_plugin_rows <dir> <before-assoc-name> <prompt ids...>: a refused row for every plugin
# engine that was wanted and did not append its own row. Never an absence.
crux_plugin_rows() {
  local d="$1" bname="$2" eng pid b
  shift 2
  local -n before_ref="$bname"
  for pid in "$@"; do
    for eng in $(crux_plugin_engines); do
      want "$eng" || continue
      crux_lib_batched "$eng" && continue
      if [ "${EXT_OK[$eng]:-0}" != 1 ]; then emit_gen "$eng" "$pid" "" "" "" "${EXT_WHY[$eng]:-engine unavailable}"; continue; fi
      if crux_is_source_engine "$eng" && [ -n "$HF_MODEL_WHY" ]; then emit_gen "$eng" "$pid" "" "" "" "$HF_MODEL_WHY"; continue; fi
      b="${before_ref[$eng-$pid]:-0}"
      if [ -n "$CELL_WHY" ]; then emit_gen "$eng" "$pid" "" "" "" "$CELL_WHY"
      elif [ "$(rows_for "$eng" "$pid")" -le "$b" ]; then
        emit_gen "$eng" "$pid" "" "" "" "engine driver ${EXT_SCRIPT[$eng]} gen exited $(cat "$d/$eng-$pid.driver.rc" 2> /dev/null || echo '?') without appending a row: $(tail -c 200 "$d/$eng-$pid.driver.err" 2> /dev/null | tr '\n' ' ')"
      elif [ "$(rows_for "$eng" "$pid")" -gt $((b + 1)) ]; then
        emit_gen "$eng" "$pid" "" "" "" "engine driver ${EXT_SCRIPT[$eng]} gen: a driver that returned $(( $(rows_for "$eng" "$pid") - b )) rows for ONE item is refused: the judge would keep whichever came last, and the reference cache would replay both (quorum round 7, lane 2)"
      fi
    done
  done
}

# A llama-server for the cell: the comparator AND the reference renderer. Sets CRUX_PL.
crux_llama_server_lines() { # <cell> <dir>
  CRUX_PL=$(free_port)
  { printf '%q ' "$LLAMA_SERVER" -m "$M" --port "$CRUX_PL" --host 127.0.0.1 -c "$CTX" -ngl "$NGL" "${LLAMA_DEV[@]}"
    printf '> %q 2>&1 < /dev/null &\necho $! > %q\n' "$2/llama-serve.log" "$2/llama-serve.pid"; } >> "$1"
  serve_wait_line "$1" "$CRUX_PL" /health "$2/llama-serve.pid"
}

# The cell's teardown, installed as its FIRST lines so no path out of the cell skips
# it: EXIT runs scripts/lib/crux_cell_teardown.sh over every server pid file the cell
# may write, and TERM/INT become an exit, so a gpu-q or timeout kill runs it too. The
# teardown proves each server is gone from the process table AND from nvidia-smi
# compute-apps before the lock drops (cop, 2026-09-23: a surviving llama-server made
# the next lock holder's prio-1 run CONTENDED). crux_teardown_why reads its verdict.
crux_teardown_trap() { # <cell> <state file> <pid files...>
  local cell="$1" state="$2"
  shift 2
  { printf 'trap %q EXIT\n' "bash $(printf '%q' "$PWD/scripts/lib/crux_cell_teardown.sh") $(printf '%q ' "$state" "$@")"
    printf "trap 'exit 143' TERM INT\n"; } >> "$cell"
}
crux_teardown_why() { # <state file>: empty when clean, else why the cell is RED
  [ -n "$CELL_WHY" ] && return 0 # the cell never ran: no server to tear down
  local st
  st=$(cat "$1" 2> /dev/null) || st=""
  case "$st" in
    clean) ;;
    "") printf 'cell teardown left no verdict (%s): its servers are not proven gone' "$1" ;;
    *) printf 'cell teardown %s' "$st" ;;
  esac
}

# The comparators' thinking switch, sent in the request: llama-server honours
# chat_template_kwargs. apr gets nothing, because it reads no toggle and the row says so.
crux_think_extra() { printf '{"chat_template_kwargs": {"enable_thinking": %s}}' "$([ "$THINK" = on ] && echo true || echo false)"; }

crux_serve_pids() { # <verb>: the serve prompts that carry that verb
  python3 -c 'import json,sys
for l in open(sys.argv[1]):
    if l.strip():
        i, v = json.loads(l)
        if sys.argv[2] in v: print(i)' "$WORK/serve-prompts.jsonl" "$1"
}

serve_routes_cell() {
  local d="$WORK/$SHA12/serve" cell pa pids=() spids=() eng pid
  local -A before=() before_s=()
  mkdir -p "$d/apr" "$d/llama" "$d/ollama" "$d/stream" || return 1
  [ -s "$WORK/serve-prompts.jsonl" ] || return 0
  # The plugins answer both serve verbs themselves (#3952 drivers: `serve run` and
  # `serve stream`), so each apr stream cell has a bf16 row on the same verb (#3957 F6).
  while IFS= read -r pid; do pids+=("$pid"); done < <(crux_serve_pids "serve run")
  while IFS= read -r pid; do spids+=("$pid"); done < <(crux_serve_pids "serve stream")
  cell="$d/cell-serve.sh"
  pa=$(free_port)
  printf '#!/usr/bin/env bash\n# one CRUX serve cell (#3962): every route apr mounts x every mode x every serve prompt\n' > "$cell"
  crux_teardown_trap "$cell" "$d/teardown.state" "$d/apr-serve.pid" "$d/llama-serve.pid"
  { printf '%q ' "$APR" serve run "$M" --port "$pa" "$APR_BE"; printf '> %q 2>&1 < /dev/null &\necho $! > %q\n' "$d/apr-serve.log" "$d/apr-serve.pid"; } >> "$cell"
  serve_wait_line "$cell" "$pa" /health "$d/apr-serve.pid"
  local render=()
  if [ "$LLAMA_OK" = 1 ]; then
    crux_llama_server_lines "$cell" "$d"
    render=(--render-url "http://127.0.0.1:$CRUX_PL")
  fi
  local common=(--prompt-list "$WORK/serve-prompts.jsonl" --prompt-dir "$WORK" --temperature "$TEMP" --seed "$SEED"
    --thinking "$THINK" --timeout "$TMO")
  cell_add "$cell" "$d/apr-sweep" python3 scripts/lib/crux_serve_routes.py sweep --url "http://127.0.0.1:$pa" \
    --out-dir "$d/apr" --device "apr serve $APR_BE" "${render[@]}" "${common[@]}"
  if [ "$LLAMA_OK" = 1 ]; then
    # #3962 B4: every oracle route, not only chat -- apr's raw routes are judged against llama's
    # /v1/completions on the byte-identical rendered prompt (--render-url: llama renders for itself).
    cell_add "$cell" "$d/llama-sweep" python3 scripts/lib/crux_serve_routes.py sweep --url "http://127.0.0.1:$CRUX_PL" \
      --routes "$(python3 scripts/lib/crux_serve_routes.py oracle-routes)" --model gguf --extra "$(crux_think_extra)" \
      --out-dir "$d/llama" --device "$LLAMA_DEVICE" "${render[@]}" "${common[@]}"
  fi
  if [ "$OLLAMA_OK" = 1 ] && [ -z "$OL_REFUSED" ]; then
    cell_add "$cell" "$d/ollama-sweep" python3 scripts/lib/crux_serve_routes.py sweep --url "$OLLAMA_HOST_URL" \
      --routes "POST /v1/chat/completions" --model "$OL_NAME" --extra '{"keep_alive": 0}' \
      --out-dir "$d/ollama" --device "$OL_DEVICE" "${common[@]}"
  fi
  [ "$OLLAMA_OK" = 1 ] && [ -z "$OL_REFUSED" ] && cell_add_ollama_unload "$cell" "$d/ollama-serve" "$OL_NAME"
  [ "${#pids[@]}" -gt 0 ] && crux_plugin_lines "$cell" "$d" "serve run" "${pids[@]}"
  [ "${#spids[@]}" -gt 0 ] && crux_plugin_lines "$cell" "$d/stream" "serve stream" "${spids[@]}"
  printf 'exit 0\n' >> "$cell"

  for pid in "${pids[@]}"; do for eng in $(crux_plugin_engines); do before[$eng-$pid]=$(VERB_KEY="serve run" rows_for "$eng" "$pid"); done; done
  for pid in "${spids[@]}"; do for eng in $(crux_plugin_engines); do before_s[$eng-$pid]=$(VERB_KEY="serve stream" rows_for "$eng" "$pid"); done; done
  run_cell "$cell"
  local td_why
  td_why=$(crux_teardown_why "$d/teardown.state")
  local rowargs=(--prompt-list "$WORK/serve-prompts.jsonl" --manifest "$MANIFEST" --sha "$SHA" --host "$HOST"
    --backend "$BACKEND" --thinking "$THINK" --cell-why "$CELL_WHY" --cell-fault "$td_why")
  python3 scripts/lib/crux_serve_routes.py rows --out-dir "$d/apr" --engine apr "${rowargs[@]}"
  if [ "$LLAMA_OK" = 1 ]; then
    python3 scripts/lib/crux_serve_routes.py rows --out-dir "$d/llama" --engine llama.cpp "${rowargs[@]}"
  elif want llama.cpp; then
    # argparse keeps the LAST --cell-why, so the engine's own reason replaces the cell's
    python3 scripts/lib/crux_serve_routes.py rows --out-dir "$d/llama" --engine llama.cpp "${rowargs[@]}" \
      --cell-why "$LLAMA_WHY"
  fi
  if [ "$OLLAMA_OK" = 1 ] && [ -z "$OL_REFUSED" ]; then
    python3 scripts/lib/crux_serve_routes.py rows --out-dir "$d/ollama" --engine ollama "${rowargs[@]}"
  elif want ollama; then
    python3 scripts/lib/crux_serve_routes.py rows --out-dir "$d/ollama" --engine ollama "${rowargs[@]}" \
      --cell-why "${OL_REFUSED:-$OLLAMA_WHY}"
  fi
  [ -n "$td_why" ] && CELL_WHY="$td_why"
  VERB_KEY="serve run" crux_plugin_rows "$d" before "${pids[@]}"
  VERB_KEY="serve stream" crux_plugin_rows "$d/stream" before_s "${spids[@]}"
}

code_cell() {
  local d="$WORK/$SHA12/code" cell pids=() eng pid mt
  local -A before=()
  mkdir -p "$d" || return 1
  [ -s "$WORK/code-prompts.txt" ] || return 0
  while IFS= read -r pid; do [ -n "$pid" ] && pids+=("$pid"); done < "$WORK/code-prompts.txt"
  cell="$d/cell-code.sh"
  printf '#!/usr/bin/env bash\n# one CRUX code cell (#3962): apr code -p, and the same prompts on every comparator\n' > "$cell"
  crux_teardown_trap "$cell" "$d/teardown.state" "$d/llama-serve.pid" "$d/apr-code-serve.pids"
  # Without #3978 apr code cannot be put on the CPU; with it, the lane's backend is passed.
  local apr_why=""
  if [ "$BACKEND" != gpu ] && [ "$(python3 scripts/lib/crux_apr_code.py controls --apr "$APR")" != yes ]; then
    apr_why="apr code has no backend flag: it always spawns \`apr serve run --gpu\` (agent/driver/apr_serve.rs), so it cannot be run on the $BACKEND lane (#3978)"
  fi
  for pid in "${pids[@]}"; do
    mt=$(cat "$WORK/maxtok-$pid.txt")
    [ -z "$apr_why" ] && cell_add "$cell" "$d/apr-$pid" python3 scripts/lib/crux_apr_code.py --apr "$APR" --model "$M" \
      --prompt-file "$WORK/prompt-$pid.json" --max-tokens "$mt" --thinking "$THINK" --backend "$BACKEND" --timeout "$TMO" \
      --serve-pid-file "$d/apr-code-serve.pids" --out "$d/apr-$pid.json"
  done
  if [ "$LLAMA_OK" = 1 ]; then
    crux_llama_server_lines "$cell" "$d"
    for pid in "${pids[@]}"; do
      cell_add "$cell" "$d/llama-$pid" python3 scripts/lib/crux_serve_routes.py drive --url "http://127.0.0.1:$CRUX_PL" \
        --route "POST /v1/chat/completions" --mode nonstream --prompt-file "$WORK/prompt-$pid.json" \
        --max-tokens "$(cat "$WORK/maxtok-$pid.txt")" --temperature "$TEMP" --seed "$SEED" --thinking "$THINK" \
        --model gguf --extra "$(crux_think_extra)" --device "$LLAMA_DEVICE" --timeout "$TMO" --out "$d/llama-$pid.json"
    done
  fi
  if [ "$OLLAMA_OK" = 1 ] && [ -z "$OL_REFUSED" ]; then
    for pid in "${pids[@]}"; do
      cell_add "$cell" "$d/ollama-$pid" python3 scripts/lib/crux_serve_routes.py drive --url "$OLLAMA_HOST_URL" \
        --route "POST /v1/chat/completions" --mode nonstream --prompt-file "$WORK/prompt-$pid.json" \
        --max-tokens "$(cat "$WORK/maxtok-$pid.txt")" --temperature "$TEMP" --seed "$SEED" --thinking "$THINK" \
        --model "$OL_NAME" --extra '{"keep_alive": 0}' --device "$OL_DEVICE" --timeout "$TMO" --out "$d/ollama-$pid.json"
    done
  fi
  [ "$OLLAMA_OK" = 1 ] && [ -z "$OL_REFUSED" ] && cell_add_ollama_unload "$cell" "$d/ollama-code" "$OL_NAME"
  crux_plugin_lines "$cell" "$d" code "${pids[@]}"
  printf 'exit 0\n' >> "$cell"

  for pid in "${pids[@]}"; do for eng in $(crux_plugin_engines); do before[$eng-$pid]=$(VERB_KEY=code rows_for "$eng" "$pid"); done; done
  run_cell "$cell"
  local td_why
  td_why=$(crux_teardown_why "$d/teardown.state")
  [ -n "$td_why" ] && CELL_WHY="$td_why"
  for pid in "${pids[@]}"; do
    if [ -n "$apr_why" ]; then VERB_KEY=code emit_gen apr "$pid" "" "" "" "$apr_why"
    else VERB_KEY=code cell_result apr "$pid" "$d/apr-$pid" "$d/apr-$pid.json"; fi
    if [ "$LLAMA_OK" = 1 ]; then VERB_KEY=code cell_result llama.cpp "$pid" "$d/llama-$pid" "$d/llama-$pid.json"
    elif want llama.cpp; then VERB_KEY=code emit_gen llama.cpp "$pid" "" "" "" "$LLAMA_WHY"; fi
    if [ "$OLLAMA_OK" = 1 ] && [ -z "$OL_REFUSED" ]; then VERB_KEY=code cell_result ollama "$pid" "$d/ollama-$pid" "$d/ollama-$pid.json"
    elif want ollama; then VERB_KEY=code emit_gen ollama "$pid" "" "" "" "${OL_REFUSED:-$OLLAMA_WHY}"; fi
  done
  VERB_KEY=code crux_plugin_rows "$d" before "${pids[@]}"
}
