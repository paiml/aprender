#!/usr/bin/env bash
# Matched-clock probe: sample SM clock / power / temp during an apr decode and an
# ollama decode, back to back, same session, same GPU.
set -uo pipefail
BIN="$1"; GGUF="$2"; OLM="$3"; OUT="$4"
mkdir -p "$OUT"
PROMPT="Write a detailed 1000-word essay about the history of computing, covering Babbage, Turing, von Neumann, transistors, integrated circuits, microprocessors, the internet, and modern AI. Be thorough and do not stop early."
sample() { # $1=tag  $2=seconds
  ( end=$((SECONDS+$2)); while [ $SECONDS -lt $end ]; do
      printf '%s %s %s\n' "$1" "$(date +%s%3N)" "$(nvidia-smi --query-gpu=utilization.gpu,clocks.sm,power.draw,temperature.gpu --format=csv,noheader,nounits|tr -d ' ')"
      sleep 0.2; done ) >> "$OUT/matched.txt" &
  echo $!
}
: > "$OUT/matched.txt"
echo "--- apr 1024 tok ---"
S=$(sample apr 40); "$BIN" run "$GGUF" --prompt "$PROMPT" --max-tokens 1024 --gpu --benchmark 2>&1 | grep -E "Generated .* tokens in "; kill "$S" 2>/dev/null; wait "$S" 2>/dev/null
sleep 5
echo "--- ollama ---"
S=$(sample ollama 40); timeout 60 "$HOME/.local/bin/ollama" run "$OLM" "$PROMPT" --verbose 2>&1 | tr -d '\r' | grep -E "^eval (count|duration|rate):"; kill "$S" 2>/dev/null; wait "$S" 2>/dev/null
echo "MATCHED_DONE"
