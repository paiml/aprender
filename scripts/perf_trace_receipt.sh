#!/usr/bin/env bash
# CRUX perf history (crux-perf-receipt-v1, G2; Refs infra#1057, #4464): one receipt for one
# rc binary on one cell. Each pass is a separate process, and no pass overlaps another:
#   T0  apr serve, then llama-server, on the same GGUF with the PRM-S1 replay set, back to back
#       in this session. Tracing is off. This pass is the gate.
#   T1  apr profile. Its top-k kernel output is kept by sha. A failure leaves T1 absent, which
#       the G3 coverage check reports RED; the script never skips it quietly.
#   T2  apr run --trace-level layer. The blob is kept by sha only, and it stays under --out,
#       which must lie outside the repo.
#   OH  tracer overhead = median(traced wall) / median(untraced wall) - 1, over n runs each.
# Run it under gpu-q, on a quiet GPU. The first thing it does is refuse any binary whose sha is
# not the tag's asset sha.
#
#   perf_trace_receipt.sh --tag T --apr BIN --tag-asset-sha S --gguf F --model M --quant Q
#     --ctx N --host H --gpu G --driver-cuda D --llama-server BIN --llama-version V
#     --set SET --diffs DIR --tool BIN --out DIR [--port P] [--runs N]
#
# --tool is the built `crux_perf` example, and its `replay` sibling must sit in the same
# directory. Prints the receipt path. Exits non-zero when the receipt is refused.
set -euo pipefail

usage() { sed -n '2,21p' "$0" >&2; exit 2; }

port=18093
runs=5
while [ "$#" -gt 0 ]; do
  [ "$#" -ge 2 ] || usage
  case "$1" in
    --tag) tag=$2 ;;
    --apr) apr=$2 ;;
    --tag-asset-sha) asset_sha=$2 ;;
    --gguf) gguf=$2 ;;
    --model) model=$2 ;;
    --quant) quant=$2 ;;
    --ctx) ctx=$2 ;;
    --host) host=$2 ;;
    --gpu) gpu=$2 ;;
    --driver-cuda) driver=$2 ;;
    --llama-server) llama=$2 ;;
    --llama-version) llama_version=$2 ;;
    --set) set_file=$2 ;;
    --diffs) diffs=$2 ;;
    --tool) tool=$2 ;;
    --out) out=$2 ;;
    --port) port=$2 ;;
    --runs) runs=$2 ;;
    *) usage ;;
  esac
  shift 2
done
for v in tag apr asset_sha gguf model quant ctx host gpu driver llama llama_version set_file diffs tool out; do
  [ -n "${!v:-}" ] || { echo "perf_trace_receipt: missing --${v//_/-}" >&2; usage; }
done

sha() { sha256sum "$1" | cut -d' ' -f1; }

# F5 up front: never measure a binary the tag does not ship.
apr_sha=$(sha "$apr")
if [ "$apr_sha" != "$(printf '%s' "$asset_sha" | tr 'A-F' 'a-f')" ]; then
  echo "perf_trace_receipt: $apr is $apr_sha, not the $tag asset $asset_sha" >&2
  exit 3
fi

# T2 blobs are private: --out must not be inside a git work tree.
mkdir -p "$out"
if git -C "$out" rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  echo "perf_trace_receipt: --out $out is inside a git work tree; trace blobs are private" >&2
  exit 4
fi
out=$(cd "$out" && pwd -P)
replay="$(dirname "$tool")/replay"
gguf_sha=$(sha "$gguf")
url="http://127.0.0.1:$port"
spid=""

stop_server() {
  if [ -n "$spid" ]; then
    kill "$spid" 2> /dev/null || true
    wait "$spid" 2> /dev/null || true
    spid=""
  fi
}
trap stop_server EXIT

start_server() { # engine
  local log="$out/server-$1.log"
  if [ "$1" = apr ]; then
    "$apr" serve run "$gguf" --port "$port" --host 127.0.0.1 --gpu-layers all --context-length "$ctx" > "$log" 2>&1 &
  else
    "$llama" -m "$gguf" --port "$port" --host 127.0.0.1 -ngl 99 -c "$ctx" -np 1 > "$log" 2>&1 &
  fi
  spid=$!
  local i
  for i in $(seq 1 180); do
    if curl -sf "$url/health" > /dev/null 2>&1; then return 0; fi
    kill -0 "$spid" 2> /dev/null || { echo "perf_trace_receipt: $1 server died (try $i)" >&2; tail -n 20 "$log" >&2; return 1; }
    sleep 2
  done
  echo "perf_trace_receipt: $1 server never healthy" >&2
  return 1
}

# ---- T0: both engines, same session, same GGUF, same set; tracing off. ----
: > "$out/rows-both.jsonl"
for engine in apr llama_cpp; do
  if [ "$engine" = apr ]; then version="apr $tag ($apr_sha)"; else version=$llama_version; fi
  start_server "$engine"
  "$replay" prompt-ids --set "$set_file" --diffs "$diffs" --url "$url" --engine "$engine" \
    --model "$model" --out "$out/ids-$engine.jsonl"
  "$replay" run --set "$set_file" --diffs "$diffs" --url "$url/v1/chat/completions" \
    --engine "$engine" --engine-version "$version" --ids "$out/ids-$engine.jsonl" \
    --apr-tag "$tag" --cell "$host" --gguf-sha "$gguf_sha" --model "$model" \
    --server-pid "$spid" --out "$out/rows-$engine.jsonl"
  stop_server
  cat "$out/rows-$engine.jsonl" >> "$out/rows-both.jsonl"
done
# Comparability: both engines prefilled the same ids.
"$replay" ids-diff "$out/ids-apr.jsonl" "$out/ids-llama_cpp.jsonl"

# ---- T1: profile, in its own process. ----
t1=()
if "$apr" profile "$gguf" --json > "$out/t1-profile.json" 2> "$out/t1-profile.err"; then
  t1=(--t1-sha "$(sha "$out/t1-profile.json")")
else
  echo "perf_trace_receipt: T1 apr profile failed; T1 absent (G3 RED): $(tail -n 1 "$out/t1-profile.err")" >&2
fi

# ---- OH + T2: n untraced then n traced runs of one fixed prompt. ----
prompt="Review this change for correctness: fn add(a: i32, b: i32) -> i32 { a - b }"
wall_ms() { # args...
  local t0 t1
  t0=$(date +%s%N)
  "$@" > /dev/null 2>&1
  t1=$(date +%s%N)
  echo $(((t1 - t0) / 1000000))
}
median() { sort -n | awk '{a[NR]=$1} END {print a[int((NR + 1) / 2)]}'; }
run=("$apr" run "$gguf" --prompt "$prompt" --max-tokens 64)
plain=$(for _ in $(seq 1 "$runs"); do wall_ms "${run[@]}"; done | median)
traced=$(for i in $(seq 1 "$runs"); do
  wall_ms "${run[@]}" --trace --trace-level layer --trace-output "$out/t2-layer-$i.json"
done | median)
overhead=$(awk -v t="$traced" -v p="$plain" 'BEGIN { printf "%.2f", (t / p - 1) * 100 }')
t2=()
if [ -s "$out/t2-layer-1.json" ]; then
  t2=(--t2-sha "$(sha "$out/t2-layer-1.json")" --trace-overhead-pct "$overhead")
fi

# ---- the receipt ----
llama_sha=$(sha "$llama")
"$tool" receipt --rows "$out/rows-both.jsonl" --tag "$tag" --rc-sha "$apr_sha" \
  --host "$host" --gpu "$gpu" --driver-cuda "$driver" --model "$model" --gguf-sha "$gguf_sha" \
  --quant "$quant" --ctx "$ctx" --batch 1 --competitor llama.cpp \
  --competitor-version "$llama_version" --competitor-sha "$llama_sha" \
  "${t1[@]}" "${t2[@]}" --tag-asset-sha "$asset_sha" > "$out/crux-perf-receipt.json"
echo "$out/crux-perf-receipt.json"
