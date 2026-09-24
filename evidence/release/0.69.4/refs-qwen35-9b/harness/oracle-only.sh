#!/usr/bin/env bash
set -u
cd /mnt/nvme-raid0/tmp/g1-9b || exit 2
M=/home/noah/models/Qwen3.5-9B-Q4_K_M.gguf
gpu-q --prio 1 -- bash -c '
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > out/llama.smi-inlock 2>&1
  llama-server -m '"$M"' -ngl 99 -c 40960 --jinja --port 18261 -np 1 > out/llama-server.log 2>&1 & sp=$!
  for _ in $(seq 1 120); do curl -sf http://127.0.0.1:18261/health >/dev/null && break; sleep 1; done
  python3 oracle.py p850 p4k p32k > out/oracle.log 2>&1; orc=$?
  kill "$sp"; wait "$sp"; exit $orc'
echo "llama rc=$?" >> out/runs.log
