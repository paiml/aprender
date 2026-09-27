#!/usr/bin/env bash
# TR-13 (#4568) live K-08 discharge against a real `apr serve`.
set -uo pipefail
APR=/mnt/nvme-raid0/targets/m0694-3850/release/apr
MODEL=${MODEL:-/home/noah/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf}
D=/mnt/nvme-raid0/tmp/k08-live/out-$(date -u +%H%M%S); mkdir -p "$D"
SP=14318; AP=17850
TP=00-0af7651916cd43dd8448eb211c80319c-00f067aa0ba902b7-01
{ date -u +%FT%TZ; hostname; git -C /mnt/nvme-raid0/wt/m0694-4568 rev-parse HEAD; git -C /mnt/nvme-raid0/wt/m0694-4568 status --short | head; "$APR" --version; sha256sum "$APR"; sha256sum "$MODEL"; } > "$D/stamp.txt" 2>&1
python3 /mnt/nvme-raid0/tmp/k08-live/sink.py $SP "$D/sink" & SINK=$!
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:$SP OTEL_EXPORTER_OTLP_PROTOCOL=http/json OTEL_SERVICE_NAME=apr-test \
  CUDA_VISIBLE_DEVICES= nice "$APR" serve run "$MODEL" --port $AP --host 127.0.0.1 > "$D/serve.log" 2>&1 & SRV=$!
trap 'kill $SRV $SINK 2>/dev/null' EXIT INT TERM HUP
for i in $(seq 1 120); do curl -sf http://127.0.0.1:$AP/health >/dev/null && break; sleep 1; done
echo "ready_after=${i}s" >> "$D/stamp.txt"
curl -sS http://127.0.0.1:$AP/v1/chat/completions -H 'Content-Type: application/json' -H "traceparent: $TP" \
  -d '{"model":"test","messages":[{"role":"user","content":"hi"}],"max_tokens":8}' > "$D/chat.json"; echo "chat_rc=$?" >> "$D/stamp.txt"
curl -sS http://127.0.0.1:$AP/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"test","messages":[{"role":"user","content":"hi"}],"max_tokens":8}' > "$D/chat-noparent.json"; echo "chat2_rc=$?" >> "$D/stamp.txt"
curl -sS http://127.0.0.1:$AP/health > "$D/health.json"
sleep 3
kill $SRV; wait $SRV 2>/dev/null
for b in "$D"/sink/body-*.json; do
  for g in --require-apr-span --require-genai-attrs "--expect-trace-id 0af7651916cd43dd8448eb211c80319c"; do
    # rc captured BEFORE any $(...) in the echo can overwrite $?
    $APR otlp-lint --otlp-file "$b" $g > "$D/lint-$(basename "$b" .json)-${g%% *}.txt" 2>&1; rc=$?
    echo "lint $(basename "$b") $g rc=$rc" >> "$D/stamp.txt"
  done
done
echo "done $(date -u +%FT%TZ) dir=$D" >> "$D/stamp.txt"; echo "$D"
