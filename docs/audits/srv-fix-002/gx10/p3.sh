#!/usr/bin/env bash
# p3.sh: nsys kernel sums at the 30k prompt — apr flash, apr f32, llama.cpp d1d3c3396 (sequential)
set -uo pipefail
R=/mnt/nvme-raid0/scratch/84-run; M=$HOME/models/Qwen3.5-4B-Q4_K_M.gguf; P=$R/prompts/p5_30k.txt; O=$R/p3; mkdir -p "$O"
LS=/mnt/nvme-raid0/llama.cpp-d1d3c3396/build/bin/llama-server
exec 9>/tmp/apr-gpu.lock; flock 9
for path in flash f32; do
  APR_QWEN35_PREFILL_ATTENTION=$path nsys profile -f true -t cuda -o "$O/apr-$path" \
    "$R/apr-3b8fc876" run "$M" --prompt "$(cat "$P")" --max-tokens 64 > "$O/apr-$path.out" 2> "$O/apr-$path.err"
  echo "apr-$path rc=$?" >> "$O/rc.txt"
  nsys stats -q --report cuda_gpu_kern_sum --format csv "$O/apr-$path.nsys-rep" > "$O/apr-$path.kern.csv" 2>/dev/null
done
nsys profile -f true -t cuda -o "$O/llama" "$LS" -m "$M" -ngl 99 -c 32768 --port 18484 > "$O/llama.srv.log" 2>&1 &
NP=$!; echo "nsys pid $NP" >> "$O/rc.txt"
for i in $(seq 1 120); do curl -sf localhost:18484/health >/dev/null && break; sleep 2; done
jq -n --rawfile p "$P" '{prompt:$p, n_predict:64, temperature:0, cache_prompt:false}' \
  | curl -s localhost:18484/completion -H 'Content-Type: application/json' -d @- > "$O/llama.json"
echo "llama curl rc=$?" >> "$O/rc.txt"
SP=$(pgrep -P "$NP" -f llama-server | head -1); [ -z "$SP" ] && SP=$(pgrep -f "port 18484" | grep -v "^$NP$" | head -1)
echo "server pid $SP" >> "$O/rc.txt"; kill -INT "$SP"; wait "$NP"; echo "nsys llama rc=$?" >> "$O/rc.txt"
nsys stats -q --report cuda_gpu_kern_sum --format csv "$O/llama.nsys-rep" > "$O/llama.kern.csv" 2>/dev/null
jq -c '.timings' "$O/llama.json" >> "$O/rc.txt"
