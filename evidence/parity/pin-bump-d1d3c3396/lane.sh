#!/usr/bin/env bash
# lane.sh <label> <llama-server> <runs>  — row-0a comparator lane, template argv, c=1, 7B, lambda
set -uo pipefail
label="$1"; server="$2"; runs="$3"
out="${LANE_OUT:?set LANE_OUT to the directory that collects this lane's artifacts}"
model="$HOME/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
port="${LANE_PORT:-8091}"
apr="$HOME/.cargo/bin/apr"
"$server" -m "$model" --port "$port" -ngl 999 -c 4096 -t 8 --no-warmup > "$out/$label.server.log" 2>&1 &
spid=$!
for _ in $(seq 1 120); do curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break; sleep 1; done
curl -s "http://127.0.0.1:$port/props" > "$out/$label.props.json"
date -u -d "@${SOURCE_DATE_EPOCH:-$(date -u +%s)}" +%FT%TZ > "$out/$label.started_utc"
nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$out/$label.vram_before"
"$apr" test llm bench --url "http://127.0.0.1:$port" --model qwen2.5-coder-7b-instruct-q4_k_m --profile medium --warmup 15 --duration 30 --runs "$runs" --cooldown 10 --concurrency 1 --stream --runtime-name "$label" --output "$out/$label.json" > "$out/$label.bench.log" 2>&1
rc=$?
echo "bench rc=$rc" >> "$out/$label.bench.log"
nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$out/$label.vram_during"
kill "$spid"; wait "$spid" 2>/dev/null; for _ in $(seq 1 30); do listening=$(ss -ltn); grep -q ":$port " <<<"$listening" || break; sleep 1; done; sleep 3
echo "rc=$rc"
