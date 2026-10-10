#!/usr/bin/env bash
# jetson_fast_verbs.sh: the E2 (#3598) fast-verbs infra test on the jetson-edge
# runner (Jetson Orin, sm_87, aarch64). Drives the four user verbs on ONE pinned
# GGUF with the GPU forced, times each, and writes one receipt row per verb.
#
#   bash scripts/jetson_fast_verbs.sh <apr-binary> <model.gguf> <out-dir>
#
# A verb PASSES when it exits 0 and answered:
#   run    `--gpu --json`, and the JSON says backend.ran == "gpu" (apr exits 14
#          when --gpu fell back, so a CPU answer is never a pass)
#   chat   two stdin turns, --gpu, exit 0 on EOF, a non-empty transcript
#   serve  `serve run --gpu`, /health up, one POST /v1/chat/completions that
#          returns HTTP 200 with a non-empty choices[0].message.content
#   bench  `bench --json` exits 0 with tokens_per_second > 0. bench has no --gpu
#          flag and its compute_class reads nvidia-smi, which Jetson lacks, so
#          its device is NOT proven here; the row says so.
#
# NO SPEED GATE. Rates and wall times are recorded, never judged: this host has
# no SILICON_FLOORS key (scripts/perf-matrix.yaml, jetson: NA). A throughput
# claim goes through scripts/perf_gate.sh.
#
# Exit: 0 every verb passed · 1 a verb failed · 2 decline (bad args, missing
# tool, model or binary). Every status is `cmd; rc=$?`, never through a pipe.
set -uo pipefail

decline() { printf 'decline: %s\n' "$*" >&2; exit 2; }

[ "$#" -eq 3 ] || decline "usage: jetson_fast_verbs.sh <apr-binary> <model.gguf> <out-dir>"
APR="$1"
MODEL="$2"
OUT="$3"
[ -x "$APR" ] || decline "no executable apr at $APR"
[ -f "$MODEL" ] || decline "no model at $MODEL"
for t in jq curl timeout; do
    command -v "$t" >/dev/null 2>&1 || decline "$t is not installed on this runner"
done
mkdir -p "$OUT" || decline "cannot create $OUT"

TMO=${FAST_VERBS_TIMEOUT:-900}
PORT=${FAST_VERBS_PORT:-18431}
PROMPT='What is 2+2? Answer with one number.'
RECEIPT="$OUT/fast-verbs.tsv"
printf 'verb\trc\tpass\twall_ms\tdetail\n' > "$RECEIPT"
failed=0

now_ms() { date +%s%3N; }

# row <verb> <rc> <pass:yes|no> <wall_ms> <detail>
row() {
    local d="${5//$'\t'/ }"
    d="${d//$'\n'/ }"
    printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "${d:0:300}" >> "$RECEIPT"
    [ "$3" = yes ] || failed=1
    printf '%-6s rc=%s pass=%s wall_ms=%s %s\n' "$1" "$2" "$3" "$4" "$5"
}

# ---- run ------------------------------------------------------------------
t0=$(now_ms)
timeout -k 10 "$TMO" "$APR" run "$MODEL" --prompt "$PROMPT" --max-tokens 32 --gpu --json \
    > "$OUT/run.json" 2> "$OUT/run.err"
rc=$?
wall=$(( $(now_ms) - t0 ))
ran=$(jq -r '.backend.ran // "absent"' "$OUT/run.json" 2>/dev/null) || ran=unparsed
tps=$(jq -r '.tok_per_sec // "absent"' "$OUT/run.json" 2>/dev/null) || tps=unparsed
setup=$(jq -r '.setup_ms // "absent"' "$OUT/run.json" 2>/dev/null) || setup=unparsed
pass=no
[ "$rc" -eq 0 ] && [ "$ran" = gpu ] && pass=yes
row run "$rc" "$pass" "$wall" "backend.ran=$ran tok_per_sec=$tps setup_ms=$setup"

# ---- chat -----------------------------------------------------------------
printf '%s\n%s\n' "$PROMPT" 'Now add 3 to it. One number.' > "$OUT/chat-turns.txt"
t0=$(now_ms)
timeout -k 10 "$TMO" "$APR" chat "$MODEL" --max-tokens 64 --gpu \
    < "$OUT/chat-turns.txt" > "$OUT/chat.out" 2> "$OUT/chat.err"
rc=$?
wall=$(( $(now_ms) - t0 ))
bytes=$(wc -c < "$OUT/chat.out")
pass=no
[ "$rc" -eq 0 ] && [ "$bytes" -gt 0 ] && pass=yes
row chat "$rc" "$pass" "$wall" "transcript_bytes=$bytes"

# ---- serve ----------------------------------------------------------------
t0=$(now_ms)
"$APR" serve run "$MODEL" --host 127.0.0.1 --port "$PORT" --gpu \
    > "$OUT/serve.log" 2>&1 < /dev/null &
spid=$!
up=no
i=0
while [ "$i" -lt "$TMO" ]; do
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then up=yes; break; fi
    kill -0 "$spid" 2>/dev/null || break
    sleep 1
    i=$((i + 1))
done
ready_ms=$(( $(now_ms) - t0 ))
code=none
content_bytes=0
if [ "$up" = yes ]; then
    jq -n --arg p "$PROMPT" '{model: "default", max_tokens: 32, messages: [{role: "user", content: $p}]}' \
        > "$OUT/serve-req.json"
    code=$(curl -s -o "$OUT/serve-resp.json" -w '%{http_code}' --max-time "$TMO" \
        -H 'Content-Type: application/json' --data @"$OUT/serve-req.json" \
        "http://127.0.0.1:$PORT/v1/chat/completions") || code=curl-failed
    jq -r '.choices[0].message.content // ""' "$OUT/serve-resp.json" > "$OUT/serve-content.txt" 2>/dev/null
    content_bytes=$(wc -c < "$OUT/serve-content.txt")
fi
wall=$(( $(now_ms) - t0 ))
kill -TERM "$spid" 2>/dev/null
j=0
while kill -0 "$spid" 2>/dev/null && [ "$j" -lt 20 ]; do sleep 1; j=$((j + 1)); done
kill -KILL "$spid" 2>/dev/null
wait "$spid" 2>/dev/null
src=$?
pass=no
[ "$up" = yes ] && [ "$code" = 200 ] && [ "$content_bytes" -gt 1 ] && pass=yes
row serve "$src" "$pass" "$wall" "health_up=$up ready_ms=$ready_ms http=$code content_bytes=$content_bytes"

# ---- bench ----------------------------------------------------------------
t0=$(now_ms)
timeout -k 10 "$TMO" "$APR" bench "$MODEL" --warmup 1 --iterations 3 --max-tokens 32 --json \
    > "$OUT/bench.json" 2> "$OUT/bench.err"
rc=$?
wall=$(( $(now_ms) - t0 ))
btps=$(jq -r '.tokens_per_second // 0' "$OUT/bench.json" 2>/dev/null) || btps=0
ttft=$(jq -r '.time_to_first_token_ms // "absent"' "$OUT/bench.json" 2>/dev/null) || ttft=unparsed
pass=no
[ "$rc" -eq 0 ] && awk -v v="$btps" 'BEGIN { exit !(v + 0 > 0) }' && pass=yes
row bench "$rc" "$pass" "$wall" "tokens_per_second=$btps ttft_ms=$ttft device=not-proven(no --gpu flag, no nvidia-smi)"

printf '\nreceipt: %s\n' "$RECEIPT"
exit "$failed"
