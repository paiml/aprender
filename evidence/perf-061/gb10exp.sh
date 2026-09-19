#!/usr/bin/env bash
# PERF-061 / #2786 — warm-up vs real deficit on GB10.
# NOT the perf-matrix §4.4.2 protocol: the beat harness under test
# (beat_ollama_decode_throughput_speed.rs) has NO warmup step at all, so there is
# no "2 x c requests + 5 s quiesce" knob to enlarge. This measures the underlying
# physical claim directly.
set -uo pipefail
BIN="${1:?apr binary}"; GGUF="${2:?gguf}"; OLM="${3:?ollama model}"; OUT="${4:?outdir}"
mkdir -p "$OUT"
PROMPT="Write a detailed 1000-word essay about the history of computing, covering Babbage, Turing, von Neumann, transistors, integrated circuits, microprocessors, the internet, and modern AI. Be thorough and do not stop early."
LOG="$OUT/raw.tsv"; : > "$LOG"

# one apr invocation -> tokens, apr-reported ms, wall ms
apr_once() {
  local n="$1" t0 t1 txt tok ms
  t0=$(date +%s%3N)
  txt=$("$BIN" run "$GGUF" --prompt "$PROMPT" --max-tokens "$n" --gpu --benchmark 2>&1)
  t1=$(date +%s%3N)
  # harness-identical parse of "Generated <tok> tokens in <ms>ms"
  local line; line=$(printf '%s\n' "$txt" | grep -m1 -E "Generated .* tokens in ")
  tok=$(printf '%s' "$line" | sed -E 's/.*Generated ([0-9]+) tokens.*/\1/')
  ms=$(printf '%s' "$line" | sed -E 's/.*tokens in ([0-9.]+)ms.*/\1/')
  local cb; cb=$(printf '%s\n' "$txt" | grep -c "CB-006-OUT")
  printf '%s\t%s\t%s\t%s\n' "${tok:-NA}" "${ms:-NA}" "$((t1-t0))" "$cb"
}

emit() { printf '%s\n' "$*" >> "$LOG"; }

echo "### ARM 3 first: ollama reference, full verbose (what token count is it over?)"
for i in 1 2 3 4 5; do
  ov=$("$HOME/.local/bin/ollama" run "$OLM" "$PROMPT" --verbose 2>&1 | tr -d '\r')
  ec=$(printf '%s\n' "$ov" | grep -m1 "^eval count:"    | sed -E 's/[^0-9]*([0-9]+).*/\1/')
  ed=$(printf '%s\n' "$ov" | grep -m1 "^eval duration:" | sed -E 's/eval duration:[[:space:]]*//')
  er=$(printf '%s\n' "$ov" | grep -m1 "^eval rate:"     | sed -E 's/[^0-9.]*([0-9.]+).*/\1/')
  emit "ollama	$i	${ec:-NA}	${ed:-NA}	${er:-NA}"
  echo "ollama trial $i: eval_count=${ec:-NA} eval_duration=${ed:-NA} eval_rate=${er:-NA}"
done

echo
echo "### ARM 1: token ladder, 4 reps, N interleaved (fit t(N) = C + N/R)"
for rep in 1 2 3 4; do
  for n in 128 256 384 512 768 1024; do
    r=$(apr_once "$n")
    emit "ladder	$rep	$n	$r"
    echo "  ladder rep=$rep N=$n -> $r"
  done
done

echo
echo "### ARM 2: the gate's own 128/384 differential, COLD vs WARM, interleaved"
# COLD = exactly what the gate does (fresh process each, GPU falls to P8 between).
# WARM = a discard 384-token invocation immediately before the measured pair, so
#        the GPU is at P0 and the page cache is hot when the pair runs.
for trial in 1 2 3 4 5 6 7; do
  for arm in cold warm; do
    if [ "$arm" = warm ]; then apr_once 384 > /dev/null; fi
    lo=$(apr_once 128); hi=$(apr_once 384)
    emit "gate	$trial	$arm	LO	$lo"
    emit "gate	$trial	$arm	HI	$hi"
    echo "  gate trial=$trial arm=$arm LO=[$lo] HI=[$hi]"
  done
done
echo "DONE"
