#!/usr/bin/env bash
# PERF-000 — falsify the single-lock hypothesis.
# PREDICTION, recorded before the run: if generation holds an exclusive writer
# lock on the model for its whole duration, wall time for N concurrent identical
# requests is ~N x the single-request time, and aggregate throughput is flat.
set -uo pipefail
# NEVER a hardcoded absolute path and never a bare `apr`. Four apr binaries once
# coexisted on this box and a bare `apr` resolved to a 24-day-old one; the path
# this line used to hardcode is also wrong in any fresh worktree, because
# .cargo/config.toml redirects the target-dir and is gitignored. apr_bin.sh
# resolves it AND proves it was built from HEAD, which is what makes a
# measurement attributable to a commit.
. scripts/apr_bin.sh || exit 1
BIN="$APR"
# Same rule as the binary above: the model is resolved, never hardcoded.
# APR_MODELS is the convention check_hardcoded_paths.sh documents; the
# default keeps this working unchanged on a box that has ~/models.
MODEL="${APR_MODELS:-$HOME/models}/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
PORT=8407
"$BIN" serve run "$MODEL" --gpu --port $PORT --context-length 4096 > /tmp/p000h.log 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null' EXIT
for i in $(seq 1 300); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ] && break
  sleep 1
done
H='Content-Type: application/json'
mk(){ printf '{"model":"q","messages":[{"role":"user","content":"Write an essay on compilers."}],"max_tokens":64,"temperature":0.0,"stream":%s}' "$1"; }
hit(){ curl -s -o /dev/null --max-time 60 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H "$H" -d "$(mk "$1")"; }
hit false   # warm
echo "=== wall time vs concurrency ==="
for st in false true; do
  base=""
  for N in 1 2 4 8; do
    pids=(); s=$(date +%s.%N)
    for ((i=0;i<N;i++)); do hit "$st" & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p"; done
    e=$(date +%s.%N)
    w=$(python3 -c "print(f'{$e-$s:.3f}')")
    [ -z "$base" ] && base="$w"
    python3 -c "
w=$w; b=$base; r=w/b
verdict = 'SERIALIZED (~N)' if r > 0.7*$N else ('sublinear' if r > 1.15 else 'CONCURRENT (flat)')
print(f'  stream={\"$st\":<5} N={$N:<2} wall {w:6.3f}s  ratio {r:5.2f}x  {verdict}')"
  done
done
