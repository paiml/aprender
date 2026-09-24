#!/usr/bin/env bash
# #4261: Qwen3.5-9B temp-0 refs at 850/4k/32k. One gpu-q hold per cell; outputs keyed bin-model-prompt.
set -u
export TMPDIR=/mnt/nvme-raid0/tmp/g1-9b/tmp
mkdir -p "$TMPDIR"
cd /mnt/nvme-raid0/tmp/g1-9b || exit 2
B=/mnt/nvme-raid0/agent-wt/f5-bins
M=/home/noah/models/Qwen3.5-9B-Q4_K_M.gguf
declare -A BIN=([rc]=$B/apr-0.69.3-rc-7ff50ec2a [base]=$B/apr-0.69.1-base-eed4a959a)
one() {
  local k="$1-9b-$2"
  [ -s "out/$k.json" ] && return 0
  local t0; t0=$(date +%s)
  gpu-q --prio 1 -- bash -c "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > out/$k.smi-inlock 2>&1; exec timeout 2400 ${BIN[$1]} run $M -i $2.txt --chat --temperature 0 -n 64 --json --backend cuda" > "out/$k.json" 2> "out/$k.err"
  echo "$k rc=$? wall=$(( $(date +%s) - t0 ))" >> out/runs.log
}
for p in p850 p4k p32k; do one rc "$p"; one base "$p"; done
gpu-q --prio 1 -- bash -c '
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > out/llama.smi-inlock 2>&1
  llama-server -m '"$M"' -ngl 99 -c 40960 --jinja --port 18261 -np 1 > out/llama-server.log 2>&1 & sp=$!
  for _ in $(seq 1 120); do curl -sf http://127.0.0.1:18261/health >/dev/null && break; sleep 1; done
  python3 oracle.py p850 p4k p32k > out/oracle.log 2>&1; orc=$?
  kill "$sp"; wait "$sp"; exit $orc'
echo "llama rc=$?" >> out/runs.log
echo DONE >> out/runs.log
