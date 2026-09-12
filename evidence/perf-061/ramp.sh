#!/usr/bin/env bash
# Observe the GB10 SM-clock ramp during ONE cold `apr run`, sampling at 100 ms.
# Option-neutral is irrelevant (executed, not sourced), but be explicit.
set -uo pipefail
BIN="${1:?apr binary}"
GGUF="${2:?gguf}"
NTOK="${3:-1024}"
OUT="${4:-/tmp/ramp}"
mkdir -p "$OUT"

PROMPT="Write a detailed 1000-word essay about the history of computing, covering Babbage, Turing, von Neumann, transistors, integrated circuits, microprocessors, the internet, and modern AI. Be thorough and do not stop early."

# sampler: monotonic-ish ms since start, sm clock, power, temp, pstate, util
( while :; do
    printf '%s %s\n' "$(date +%s%3N)" "$(nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,pstate,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
    sleep 0.1
  done ) > "$OUT/clocks.txt" 2>&1 &
SAMPLER=$!
sleep 1.5

T0=$(date +%s%3N)
echo "T0 $T0" > "$OUT/marks.txt"
"$BIN" run "$GGUF" --prompt "$PROMPT" --max-tokens "$NTOK" --gpu --benchmark > "$OUT/apr.stdout" 2> "$OUT/apr.stderr"
rc=$?
T1=$(date +%s%3N)
echo "T1 $T1" >> "$OUT/marks.txt"
echo "rc $rc" >> "$OUT/marks.txt"

sleep 1.5
kill "$SAMPLER" 2>/dev/null
wait "$SAMPLER" 2>/dev/null

echo "=== apr rc=$rc wall=$((T1-T0))ms ==="
grep -E "Generated|tok/s|GPU|CUDA|backend" "$OUT/apr.stdout" "$OUT/apr.stderr" 2>/dev/null | head -20
echo "=== CB-006-OUT lines (must be 0) ==="
cat "$OUT/apr.stdout" "$OUT/apr.stderr" | grep -c "CB-006-OUT"
