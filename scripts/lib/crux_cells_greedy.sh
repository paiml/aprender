# shellcheck shell=bash
# scripts/lib/crux_cells_greedy.sh — greedy token rows for #3957 F9 (aprender-36), SOURCED by
# scripts/crux_inference_dogfood.sh when it is run with --greedy. Option-neutral: it sets no shell options
# (scripts/check_sourced_libs_option_neutral.sh) and defines one function.
#
# For the current model ($M, $SHA, $SHA12) and every greedy prompt ($GREEDY_PIDS), in ONE cell under the GPU
# lock (run_cell, the same hold every other cell takes):
#   llama.cpp  thinking ON and OFF: the model's own template rendered with enable_thinking EXPLICIT, greedy
#              (temperature 0, top_k 1), the generated ids returned, decoded with special tokens.
#   apr        `apr run --chat --temperature 0 --format json -v [--thinking on|off]`: "tokens" are its generated
#              ids (decoded with special tokens through the SAME llama-server: one tokenizer for both texts), and
#              -v prints the prompt ids apr built — which llama.cpp then generates from, so both engines start
#              from IDENTICAL tokens (aprender-36). apr runs FIRST in each (prompt, thinking) pair for that reason.
#   apr ON     needs `apr run --thinking` (aprender-36, fix/3957-f9-f10). An apr without it is REFUSED by name for
#              ON: #3723 (realizar routes every Qwen3/3.5 to the no-think template). Never an absence.
#   llama.cpp  TWO rows per (prompt, thinking) (cop ruling on F9, #3990): prompt_source "apr" — greedy on apr's own
#              prompt ids (ENGINE PARITY) — and "official" — greedy on llama.cpp's own /apply-template rendering
#              (the MODEL's true behaviour; apr's rendering was measured to differ from Qwen3.5's official template
#              in both modes). RED-MODEL needs the official row to show the defect too.
# Each is one `kind: "greedy"` manifest row, key (model_sha256, host, prompt_id, thinking), whose `tokens`
# file is the `raw` object: {generated_ids, generated_text, greedy, special, max_tokens}. Both engines use the
# same max_tokens ($GREEDY_MAXTOK).
greedy_cells() {
  local d="$WORK/$SHA12/greedy" cell port pid th content think_flag=0 aprflag
  mkdir -p "$d"
  "$APR" run --help 2>/dev/null | grep -q -- '--thinking' && think_flag=1
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
      aprflag=()
      [ "$think_flag" = 1 ] && aprflag=(--thinking "$th")
      if [ "$think_flag" = 1 ] || [ "$th" = off ]; then
        cell_add "$cell" "$d/apr-$pid-$th.run" "$APR" run "$M" --prompt "$content" --chat --max-tokens "$GREEDY_MAXTOK" \
          --temperature 0 --seed "$SEED" --format json -v "$APR_BE" "${aprflag[@]}"
        [ "$LLAMA_OK" = 1 ] && cell_add "$cell" "$d/apr-$pid-$th" python3 scripts/lib/crux_greedy_llama.py apr \
          --url "http://127.0.0.1:$port" --apr-json "$d/apr-$pid-$th.run.out" --apr-stderr "$d/apr-$pid-$th.run.err" \
          --messages "$WORK/messages-$pid.json" --thinking "$th" --max-tokens "$GREEDY_MAXTOK" --out "$d/apr-$pid-$th.json"
      fi
      if [ "$LLAMA_OK" = 1 ]; then
        cell_add "$cell" "$d/llama-$pid-$th" python3 scripts/lib/crux_greedy_llama.py gen --prompt-source apr \
          --url "http://127.0.0.1:$port" --messages "$WORK/messages-$pid.json" --thinking "$th" \
          --max-tokens "$GREEDY_MAXTOK" --seed "$SEED" --out "$d/llama-$pid-$th.json" --apr-stderr "$d/apr-$pid-$th.run.err"
        cell_add "$cell" "$d/llama-$pid-$th-official" python3 scripts/lib/crux_greedy_llama.py gen --prompt-source official \
          --url "http://127.0.0.1:$port" --messages "$WORK/messages-$pid.json" --thinking "$th" \
          --max-tokens "$GREEDY_MAXTOK" --seed "$SEED" --out "$d/llama-$pid-$th-official.json"
      fi
    done
  done
  [ "$LLAMA_OK" = 1 ] && printf 'kill "$(cat %q)" 2> /dev/null; wait "$(cat %q)" 2> /dev/null\n' \
    "$d/llama-server.pid" "$d/llama-server.pid" >> "$cell"
  printf 'exit 0\n' >> "$cell"
  run_cell "$cell"
  python3 - "$MANIFEST" "$SHA" "$HOST" "$BACKEND" "$d" "$GREEDY_MAXTOK" "$LLAMA_OK" "${LLAMA_WHY:-}" \
    "${CELL_WHY:-}" "$think_flag" $GREEDY_PIDS <<'PY'
import json, os, sys
m, sha, host, backend, d, maxtok, llama_ok, llama_why, cell_why, think_flag = sys.argv[1:11]
pids = sys.argv[11:]
NO_ON = ("#3723: this apr has no `run --thinking` flag — realizar routes every Qwen3/Qwen3.5 to the no-think "
         "template, and a pre-rendered thinking-ON prompt is escaped (zero-width space inside its special tokens), "
         "so apr cannot generate greedily with thinking ON")


def row(engine, pid, th, path, refused, source="apr"):
    r = {"kind": "greedy", "engine": engine, "model_sha256": sha, "host": host, "backend": backend, "prompt_id": pid,
         "thinking": th, "prompt_source": source, "max_tokens": int(maxtok), "tokens": None, "logits": None,
         "refused": refused}
    if refused is None:
        r["tokens"] = path
    open(m, "a").write(json.dumps(r) + "\n")


def load_first_object(path):
    """The first JSON object in a file: apr's `-v` stdout has `verbose: ...` lines before it (measured)."""
    text = open(path, encoding="utf-8", errors="replace").read()
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object")
    return json.JSONDecoder().raw_decode(text[start:])[0]


def lane_mismatch(be):
    """None when apr RAN on this row's lane; else the refusal, by name. A gpu-lane cell that fell back to the CPU is
    no GPU row (aprender-36: 40ae453c7's cuda build fell back on Qwen3.5-0.8B-UD-IQ2_XXS, measured here too:
    requested gpu, ran cpu, fell_back true), and a missing backend record cannot say where it ran."""
    if not isinstance(be, dict) or "ran" not in be or "fell_back" not in be:
        return "apr's --format json carried no backend record {requested, ran, fell_back}: the lane is unverifiable"
    ran = str(be.get("ran"))
    ok = (ran in ("gpu", "cuda")) if backend == "gpu" else (ran == "cpu")
    if be.get("fell_back") is not False or not ok:
        return ("apr did not run on the %s lane: requested %s, ran %s, fell_back %s — refused, never recorded as a "
                "%s row" % (backend, be.get("requested"), ran, be.get("fell_back"), backend))
    return None


def judged(path, rc_path):
    """None if the artifact is a good raw object; else the reason, by name."""
    if cell_why:
        return cell_why
    try:
        doc = load_first_object(path)
    except (OSError, ValueError) as e:
        rc = open(rc_path).read().strip() if os.path.exists(rc_path) else "?"
        return "no greedy artifact (exit %s): %s" % (rc, e)
    if doc.get("error"):
        return doc["error"]
    return None


for pid in pids:
    for th in ("on", "off"):
        for source, suffix in (("apr", ""), ("official", "-official")):
            p = os.path.join(d, "llama-%s-%s%s.json" % (pid, th, suffix))
            row("llama.cpp", pid, th, p, judged(p, p[:-5] + ".rc") if llama_ok == "1" else
                (llama_why or "llama.cpp unavailable"), source)
    for th in ("on", "off"):
        if th == "on" and think_flag != "1":
            row("apr", pid, "on", None, NO_ON)
            continue
        p = os.path.join(d, "apr-%s-%s.json" % (pid, th))
        if llama_ok != "1":
            why = "apr's ids are decoded through llama-server, which is unavailable here: " + (llama_why or "?")
        else:
            why = judged(os.path.join(d, "apr-%s-%s.run.out" % (pid, th)), os.path.join(d, "apr-%s-%s.run.rc" % (pid, th)))
            if why is None:
                why = judged(p, p[:-5] + ".rc")
            if why is None:
                why = lane_mismatch(load_first_object(p).get("backend"))
        row("apr", pid, th, p, why)
PY
}
