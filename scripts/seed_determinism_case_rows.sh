#!/usr/bin/env bash
# #3720 done_when 2: same model, prompt, seed, sampling -> byte-identical output twice?
# usage: det.sh APR OUTDIR BACKENDFLAG MODEL...
set -u
APR=$1; OUT=$2; BF=$3; shift 3
mkdir -p "$OUT"; "$APR" --version > "$OUT/version.txt"; hostname > "$OUT/host.txt"
Q='Write two sentences about the moon.'
port=18841
for M in "$@"; do
  t=$(basename "$M" .gguf)
  for mode in greedy sampled; do
    case $mode in greedy) S="--temperature 0";; sampled) S="--temperature 0.7 --top-k 40 --top-p 0.9 --seed 42";; esac
    for i in 1 2; do
      # shellcheck disable=SC2086
      timeout 600 "$APR" run "$M" --prompt "$Q" --chat $BF --json --max-tokens 64 --thinking off $S > "$OUT/$t.run.$mode.$i.json" 2> "$OUT/$t.run.$mode.$i.err"
      echo "$t run $mode $i rc=$?" >> "$OUT/rc.txt"
    done
  done
  port=$((port+1))
  "$APR" serve run "$M" $BF --port "$port" --host 127.0.0.1 > "$OUT/$t.serve.log" 2>&1 & sp=$!
  for _ in $(seq 1 300); do curl -sf -o /dev/null --max-time 2 "http://127.0.0.1:$port/health" && break; kill -0 $sp 2>/dev/null || break; sleep 1; done
  for mode in greedy sampled; do
    case $mode in greedy) B='"temperature":0';; sampled) B='"temperature":0.7,"top_k":40,"top_p":0.9,"seed":42';; esac
    for i in 1 2; do
      curl -s -D "$OUT/$t.serve.$mode.$i.headers" -o "$OUT/$t.serve.$mode.$i.json" --max-time 600 -H 'Content-Type: application/json' \
        -d '{"model":"m","messages":[{"role":"user","content":"'"$Q"'"}],"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false},'"$B"'}' \
        "http://127.0.0.1:$port/v1/chat/completions"
    done
  done
  kill $sp 2>/dev/null; wait $sp 2>/dev/null
  sha256sum "$M" | cut -d' ' -f1 > "$OUT/$t.sha256sum"
done
echo done > "$OUT/DONE"
