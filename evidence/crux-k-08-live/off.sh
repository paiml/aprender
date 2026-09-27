#!/usr/bin/env bash
set -uo pipefail
APR=/mnt/nvme-raid0/targets/m0694-3850/release/apr; D=/mnt/nvme-raid0/tmp/k08-live/off; rm -rf "${D:?}"; mkdir -p "$D"
python3 /mnt/nvme-raid0/tmp/k08-live/sink.py 14319 "$D/sink" & SINK=$!
env -u OTEL_EXPORTER_OTLP_ENDPOINT -u OTEL_EXPORTER_OTLP_TRACES_ENDPOINT CUDA_VISIBLE_DEVICES= nice "$APR" serve run /home/noah/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf --port 17851 --host 127.0.0.1 > "$D/serve.log" 2>&1 & SRV=$!
trap 'kill $SRV $SINK 2>/dev/null' EXIT INT TERM HUP
for i in $(seq 1 120); do curl -sf http://127.0.0.1:17851/health >/dev/null && break; sleep 1; done
curl -sS http://127.0.0.1:17851/v1/chat/completions -H 'Content-Type: application/json' -H 'traceparent: 00-0af7651916cd43dd8448eb211c80319c-00f067aa0ba902b7-01' -d '{"model":"test","messages":[{"role":"user","content":"hi"}],"max_tokens":8}' > "$D/chat.json"; echo "chat_rc=$?"
sleep 3
echo "sink_bodies=$(find "$D/sink" -name 'body-*' 2>/dev/null | wc -l)"
echo "otlp_lines=$(grep -c OTLP "$D/serve.log")"
