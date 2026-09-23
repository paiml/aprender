# shellcheck shell=bash
# scripts/lib/crux_cells_greedy.sh — greedy token rows for #3957 F9 (aprender-36), SOURCED by
# scripts/crux_inference_dogfood.sh when it is run with --greedy. Option-neutral: it sets no shell options
# (scripts/check_sourced_libs_option_neutral.sh) and defines one function.
#
# For the current model ($M, $SHA, $SHA12) and every greedy prompt ($GREEDY_PIDS), in ONE cell under the GPU
# lock (run_cell, the same hold every other cell takes):
#   llama.cpp  thinking ON and OFF: the model's own template rendered with enable_thinking EXPLICIT, greedy
#              (temperature 0, top_k 1), the generated ids returned, decoded with special tokens.
#   apr        thinking OFF: `apr run --chat --temperature 0 --format json`, whose "tokens" are its generated
#              ids, decoded with special tokens through the SAME llama-server (one tokenizer for both texts).
#   apr ON     REFUSED by name: #3723 (apr has no thinking toggle; realizar routes every Qwen3/3.5 to the
#              no-think template, and a pre-rendered ON prompt is escaped). Never an absence.
# Each is one `kind: "greedy"` manifest row, key (model_sha256, host, prompt_id, thinking), whose `tokens`
# file is the `raw` object: {generated_ids, generated_text, greedy, special, max_tokens}. Both engines use the
# same max_tokens ($GREEDY_MAXTOK).
greedy_cells() {
  local d="$WORK/$SHA12/greedy" cell port pid th content
  mkdir -p "$d"
  cell="$d/cell-greedy.sh"
  printf '#!/usr/bin/env bash\n# one CRUX greedy cell: apr + llama.cpp greedy ids on the identical GGUF\n' > "$cell"
  if [ "$LLAMA_OK" = 1 ]; then
    port=$(free_port)
    { printf '%q ' "$LLAMA_SERVER" -m "$M" --port "$port" --host 127.0.0.1 -c "$CTX" -ngl "$NGL" "${LLAMA_DEV[@]}" \
        --jinja --no-warmup
      printf '> %q 2>&1 < /dev/null &\necho $! > %q\n' "$d/llama-server.log" "$d/llama-server.pid"; } >> "$cell"
    serve_wait_line "$cell" "$port" /health "$d/llama-server.pid"
  fi
  for pid in $GREEDY_PIDS; do
    content=$(cat "$WORK/prompt-$pid.txt")
    for th in on off; do
      [ "$LLAMA_OK" = 1 ] && cell_add "$cell" "$d/llama-$pid-$th" python3 scripts/lib/crux_greedy_llama.py gen \
        --url "http://127.0.0.1:$port" --messages "$WORK/messages-$pid.json" --thinking "$th" \
        --max-tokens "$GREEDY_MAXTOK" --seed "$SEED" --out "$d/llama-$pid-$th.json"
    done
    cell_add "$cell" "$d/apr-$pid-off.run" "$APR" run "$M" --prompt "$content" --chat --max-tokens "$GREEDY_MAXTOK" \
      --temperature 0 --seed "$SEED" --format json "$APR_BE"
    [ "$LLAMA_OK" = 1 ] && cell_add "$cell" "$d/apr-$pid-off" python3 scripts/lib/crux_greedy_llama.py apr \
      --url "http://127.0.0.1:$port" --apr-json "$d/apr-$pid-off.run.out" --max-tokens "$GREEDY_MAXTOK" \
      --out "$d/apr-$pid-off.json"
  done
  [ "$LLAMA_OK" = 1 ] && printf 'kill "$(cat %q)" 2> /dev/null; wait "$(cat %q)" 2> /dev/null\n' \
    "$d/llama-server.pid" "$d/llama-server.pid" >> "$cell"
  printf 'exit 0\n' >> "$cell"
  run_cell "$cell"
  python3 - "$MANIFEST" "$SHA" "$HOST" "$BACKEND" "$d" "$GREEDY_MAXTOK" "$LLAMA_OK" "${LLAMA_WHY:-}" \
    "${CELL_WHY:-}" $GREEDY_PIDS <<'PY'
import json, os, sys
m, sha, host, backend, d, maxtok, llama_ok, llama_why, cell_why = sys.argv[1:10]
pids = sys.argv[10:]
NO_ON = ("#3723: apr has no thinking toggle — realizar routes every Qwen3/Qwen3.5 to the no-think template, and a "
         "pre-rendered thinking-ON prompt is escaped (zero-width space inside its special tokens), so apr cannot "
         "generate greedily with thinking ON")


def row(engine, pid, th, path, refused):
    r = {"kind": "greedy", "engine": engine, "model_sha256": sha, "host": host, "backend": backend, "prompt_id": pid,
         "thinking": th, "max_tokens": int(maxtok), "tokens": None, "logits": None, "refused": refused}
    if refused is None:
        r["tokens"] = path
    open(m, "a").write(json.dumps(r) + "\n")


def judged(path, rc_path):
    """None if the artifact is a good raw object; else the reason, by name."""
    if cell_why:
        return cell_why
    try:
        doc = json.load(open(path))
    except (OSError, ValueError) as e:
        rc = open(rc_path).read().strip() if os.path.exists(rc_path) else "?"
        return "no greedy artifact (exit %s): %s" % (rc, e)
    if doc.get("error"):
        return doc["error"]
    return None


for pid in pids:
    for th in ("on", "off"):
        p = os.path.join(d, "llama-%s-%s.json" % (pid, th))
        row("llama.cpp", pid, th, p, judged(p, p[:-5] + ".rc") if llama_ok == "1" else (llama_why or "llama.cpp unavailable"))
    row("apr", pid, "on", None, NO_ON)
    p = os.path.join(d, "apr-%s-off.json" % pid)
    if llama_ok != "1":
        why = "apr's ids are decoded through llama-server, which is unavailable here: " + (llama_why or "?")
    else:
        why = judged(os.path.join(d, "apr-%s-off.run.out" % pid), os.path.join(d, "apr-%s-off.run.rc" % pid))
        if why is None:
            why = judged(p, p[:-5] + ".rc")
    row("apr", pid, "off", p, why)
PY
}
