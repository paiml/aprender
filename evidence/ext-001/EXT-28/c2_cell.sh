#!/usr/bin/env bash
# EXT-28 (aprender#4410): every C2 speed arm on one cell, measured by ONE client
# (curl + bash $EPOCHREALTIME on the streamed /v1/completions lines).
#
#   c2_cell.sh <out dir> <arm>...
#
# arm: apr | llama.cpp | ollama | mistral.rs. Every arm is a server started here
# under `env -i` with the env recorded (its sha is the comparator block's
# env_sha256), pinned to the same CPUs and thread count, and sent the same prompt
# with temperature 0. All arms stay RESIDENT for the whole run and their measured
# iterations are INTERLEAVED (round i runs every arm once, the order rotated by i),
# so a change in host load lands on every arm alike instead of on whichever arm ran
# in that minute. Servers are killed by their recorded PIDs.
#
# Per arm, <out dir>/<arm>.json, over N measured iterations after one warmup. N=5 and
# the decode statistic are PRE-REGISTERED (cop ruling on EXT-022, 2026-09-26):
# decode_tok_s is the BEST of the N (a contention-robust statistic: host load only
# ever slows an iteration), with every sample kept in decode_tok_s_iters and the
# 1-minute load average read at each iteration start in loadavg_1m_iters. The other
# timings are medians:
#   load_ms      spawn -> first successful 1-token completion (includes model load;
#                arms load one after another, so no load overlaps another arm's)
#   ttft_ms      request sent -> first non-empty text chunk
#   itl_ms       median gap between consecutive non-empty text chunks
#   e2e_ms       request sent -> last text chunk
#   decode_tok_s (chunks - 1) / (t_last - t_first)
#   peak_rss_kb  max VmHWM over the server and its descendants, read before the kill
# The comparator block's artifact is the arm's raw measurement: its N timestamped
# SSE streams, concatenated in order.
# Required env: APR_BIN (also reads GGUF metadata for every arm), PROMPT_FILE, and
# per arm: MODEL (apr, llama.cpp, mistral.rs), LLAMA_SERVER, MISTRALRS_BIN,
# OLLAMA_BIN + OLLAMA_MODELS (a private store holding OLLAMA_TAG).
set -euo pipefail

outdir=${1:?out dir}
shift
[ "$#" -ge 1 ] || { echo "usage: c2_cell.sh <out dir> <arm>..." >&2; exit 2; }
arms=("$@")
CPUS=${CPUS:-0-15}
THREADS=${THREADS:-16}
N=5 # pre-registered; not an env knob
MAX_TOKENS=${MAX_TOKENS:-32}
BASE_PORT=${BASE_PORT:-18780}
OLLAMA_TAG=${OLLAMA_TAG:-qwen3.5:4b}
mkdir -p "$outdir"
outdir=$(cd "$outdir" && pwd)
logdir=$outdir/logs
mkdir -p "$logdir"
prompt=$(cat "${PROMPT_FILE:?}")
prompt_sha=$(sha256sum "$PROMPT_FILE" | cut -d' ' -f1)
utc() { date -u +%Y-%m-%dT%H:%M:%S.%3NZ; }

declare -A PORT SERVED FILE VERSION ENGINE ENVSHA CMDJSON PID LOAD STARTED FINISHED

body() { # $1 = arm, $2 = max_tokens
    jq -cn --arg p "$prompt" --arg m "${SERVED[$1]}" --argjson n "$2" \
        '{model:$m, prompt:$p, max_tokens:$n, temperature:0, stream:true}'
}
url() { echo "http://127.0.0.1:${PORT[$1]}/v1/completions"; }

cleanup() {
    for a in "${!PID[@]}"; do kill "${PID[$a]}" 2>/dev/null || true; done
    for a in "${!PID[@]}"; do wait "${PID[$a]}" 2>/dev/null || true; done
}
trap cleanup EXIT

start() { # $1 = arm, $2 = port
    local arm=$1 port=$2 home="$logdir/$1.home" served file version
    local -a env_arm argv cmd
    mkdir -p "$home"
    # CPU cells hide every GPU from every arm (ollama otherwise puts its vision
    # tower on CUDA).
    env_arm=("PATH=/usr/bin:/bin" "HOME=$home" "CUDA_VISIBLE_DEVICES=")
    case "$arm" in
    apr)
        served=ours
        file=${MODEL:?}
        version=$("${APR_BIN:?}" --version | head -1)
        env_arm+=("RAYON_NUM_THREADS=$THREADS")
        argv=("$APR_BIN" serve run "$file" --host 127.0.0.1 --port "$port" --no-gpu)
        ;;
    llama.cpp)
        served=ours
        file=${MODEL:?}
        version=$("${LLAMA_SERVER:?}" --version 2>&1 | grep -m1 '^version')
        argv=("$LLAMA_SERVER" -m "$file" -t "$THREADS" -c 4096 --host 127.0.0.1 --port "$port")
        ;;
    mistral.rs)
        served=default # mistral.rs routes an unnamed single model as `default`
        file=${MODEL:?}
        version=$("${MISTRALRS_BIN:?}" --version 2>&1 | head -1)
        env_arm+=("RAYON_NUM_THREADS=$THREADS")
        argv=("$MISTRALRS_BIN" serve --cpu -p "$port" --format gguf -m "$(dirname "$file")" -f "$(basename "$file")")
        ;;
    ollama)
        served=$OLLAMA_TAG
        file=
        version=$("${OLLAMA_BIN:?}" --version 2>&1 | tail -1)
        env_arm+=("OLLAMA_MODELS=${OLLAMA_MODELS:?}" "OLLAMA_HOST=127.0.0.1:$port" "OLLAMA_NUM_PARALLEL=1")
        argv=("$OLLAMA_BIN" serve)
        ;;
    *)
        echo "unknown arm: $arm" >&2
        exit 2
        ;;
    esac
    cmd=(env -i "${env_arm[@]}" taskset -c "$CPUS" "${argv[@]}")
    PORT[$arm]=$port
    SERVED[$arm]=$served
    FILE[$arm]=$file
    VERSION[$arm]=$version
    ENGINE[$arm]=$(sha256sum "${argv[0]}" | cut -d' ' -f1)
    ENVSHA[$arm]=$(printf '%s\n' "${env_arm[@]}" | LC_ALL=C sort | sha256sum | cut -d' ' -f1)
    CMDJSON[$arm]=$(printf '%s\0' "${cmd[@]}" | jq -Rs 'split("\u0000")[:-1]')
    local t_spawn=$EPOCHREALTIME loaded=
    "${cmd[@]}" >"$logdir/$arm.server.log" 2>&1 &
    PID[$arm]=$!
    # Load: the first 1-token completion that succeeds (ollama loads lazily, the
    # others at start; the same definition covers both).
    for _ in $(seq 1 1800); do
        if curl -sf -m 900 "$(url "$arm")" -H 'content-type: application/json' \
            -d "$(body "$arm" 1 | jq -c '.stream=false')" >/dev/null 2>&1; then
            loaded=$EPOCHREALTIME
            break
        fi
        kill -0 "${PID[$arm]}" 2>/dev/null || { echo "$arm: server exited, see $logdir/$arm.server.log" >&2; exit 1; }
        sleep 1
    done
    [ -n "$loaded" ] || { echo "$arm: never answered" >&2; exit 1; }
    LOAD[$arm]=$(awk -v a="$t_spawn" -v b="$loaded" 'BEGIN{printf "%.1f", (b-a)*1000}')
}

# One streamed request -> "ttft_ms itl_ms e2e_ms chunks decode_tok_s".
measure() { # $1 = arm, $2 = iteration
    local arm=$1 t0 raw req u rc
    raw="$logdir/$arm.iter$2.sse"
    req=$(body "$arm" "$MAX_TOKENS")
    u=$(url "$arm")
    t0=$EPOCHREALTIME
    # Every line is kept (an error body is evidence too); only data lines are parsed.
    curl -sSN -m 1800 "$u" -H 'content-type: application/json' -d "$req" 2>"$raw.err" |
        while IFS= read -r line; do
            printf '%s %s\n' "$EPOCHREALTIME" "$line"
        done >"$raw"
    rc=${PIPESTATUS[0]}
    [ "$rc" -eq 0 ] || echo "curl rc=$rc" >>"$raw.err"
    # A text chunk is a data line whose choices[0].text is non-empty.
    { grep -E '^[0-9.]+ data: ' "$raw" || true; } | while IFS= read -r l; do
        ts=${l%% *}
        j=${l#* data: }
        [ "$j" = "[DONE]" ] && continue
        txt=$(jq -r '.choices[0].text // ""' <<<"$j" 2>/dev/null || true)
        if [ -n "$txt" ]; then echo "$ts"; fi
    done | awk -v t0="$t0" '
        { t[NR] = $1 }
        END {
            if (NR < 2) { print "ERR"; exit }
            for (i = 2; i <= NR; i++) g[i-1] = t[i] - t[i-1]
            n = asort(g)
            med = (n % 2) ? g[(n+1)/2] : (g[n/2] + g[n/2+1]) / 2
            printf "%.1f %.1f %.1f %d %.4f\n", (t[1]-t0)*1000, med*1000, (t[NR]-t0)*1000, NR, (NR-1)/(t[NR]-t[1])
        }'
}

i=0
for arm in "${arms[@]}"; do
    start "$arm" $((BASE_PORT + i))
    i=$((i + 1))
done
for arm in "${arms[@]}"; do
    measure "$arm" 0 >/dev/null # warmup
    : >"$logdir/$arm.iters"
done
k=${#arms[@]}
for it in $(seq 1 "$N"); do
    for j in $(seq 0 $((k - 1))); do
        arm=${arms[$(((it - 1 + j) % k))]}
        [ -n "${STARTED[$arm]:-}" ] || STARTED[$arm]=$(utc)
        la=$(cut -d" " -f1 /proc/loadavg)
        r=$(measure "$arm" "$it")
        [ "$r" != ERR ] || { echo "$arm: iteration $it streamed < 2 text chunks: $(tail -c 400 "$logdir/$arm.iter$it.sse.err" 2>/dev/null) $(tail -c 400 "$logdir/$arm.iter$it.sse")" >&2; exit 1; }
        echo "$r $la" >>"$logdir/$arm.iters"
        FINISHED[$arm]=$(utc)
    done
done

# Peak RSS over one server tree, before the kill.
peak_rss() {
    local tree=$1 frontier=$1 next p c v peak=0
    while [ -n "$frontier" ]; do
        next=
        for p in $frontier; do
            for c in $(pgrep -P "$p" || true); do next="$next $c"; done
        done
        tree="$tree $next"
        frontier=${next# }
    done
    for p in $tree; do
        v=$(awk '/^VmHWM:/{print $2}' "/proc/$p/status" 2>/dev/null || echo 0)
        [ "${v:-0}" -gt "$peak" ] && peak=$v
    done
    echo "$peak"
}

med() { sort -n | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }
for arm in "${arms[@]}"; do
    peak=$(peak_rss "${PID[$arm]}")
    digest=
    file=${FILE[$arm]}
    if [ "$arm" = ollama ]; then
        digest=$(curl -sf "http://127.0.0.1:${PORT[$arm]}/api/tags" | jq -r --arg m "$OLLAMA_TAG" '.models[] | select(.name==$m) | .digest')
        file=$(find "$OLLAMA_MODELS/blobs" -size +1G -name 'sha256-*' | head -1)
    fi
    file_sha=$(sha256sum "$file" | cut -d' ' -f1)
    file_type=$("$APR_BIN" inspect "$file" --json 2>/dev/null | jq -r '.metadata["general.file_type"] | tonumber')
    vision=$("$APR_BIN" tensors "$file" --json 2>/dev/null |
        jq '[.tensors[].name | select(startswith("v.") or startswith("mm."))] | length')
    artifact_sha=$(for it in $(seq 1 "$N"); do cat "$logdir/$arm.iter$it.sse"; done | sha256sum | cut -d' ' -f1)
    col() { awk -v c="$1" '{print $c}' "$logdir/$arm.iters" | med; }
    jq -n --arg arm "$arm" --arg version "${VERSION[$arm]}" --arg file "$file_sha" \
        --arg engine "${ENGINE[$arm]}" --argjson ftype "$file_type" --argjson vision "$vision" \
        --arg served "${SERVED[$arm]}" --arg digest "$digest" --arg cpus "$CPUS" \
        --argjson threads "$THREADS" --argjson n "$N" --argjson max "$MAX_TOKENS" \
        --arg psha "$prompt_sha" \
        --argjson load "${LOAD[$arm]}" --argjson ttft "$(col 1)" --argjson itl "$(col 2)" \
        --argjson e2e "$(col 3)" --argjson rss "$peak" \
        --argjson tpsi "$(awk '{print $5}' "$logdir/$arm.iters" | jq -s .)" \
        --argjson lai "$(awk '{print $6}' "$logdir/$arm.iters" | jq -s .)" \
        --argjson command "${CMDJSON[$arm]}" \
        --arg envsha "${ENVSHA[$arm]}" --arg asha "$artifact_sha" --arg log "logs/$arm.server.log" \
        --arg started "${STARTED[$arm]}" --arg finished "${FINISHED[$arm]}" \
        '{arm:$arm, version:$version, engine_sha256:$engine, model_sha256:$file, file_type:$ftype,
          vision_tensors:$vision, served_model:$served,
          ollama_manifest_digest:(if $digest == "" then null else $digest end),
          conditions:{cpus:$cpus, threads:$threads, concurrency:1, iterations:$n, statistic:"best_of_n_decode",
                      max_tokens:$max, prompt_sha256:$psha},
          timing:{load_ms:$load, ttft_ms:$ttft, itl_ms:$itl, e2e_ms:$e2e,
                  decode_tok_s:($tpsi | max), decode_tok_s_iters:$tpsi,
                  loadavg_1m_iters:$lai, peak_rss_kb:$rss},
          comparator:{command:$command, version:$version, env_sha256:$envsha,
                      artifact_sha256:$asha, log_path:$log,
                      started_utc:$started, finished_utc:$finished}}' >"$outdir/$arm.json"
done
