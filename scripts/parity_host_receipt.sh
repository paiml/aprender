#!/usr/bin/env bash
# parity_host_receipt.sh — produce the parity block for ONE host (#2696).
#
# Runs `apr test llm bench` against apr and against llama.cpp with the SAME
# client, prompts and clock, once per compute class the host can actually
# reach, and emits a block that scripts/lib/bench_receipt.py --parity accepts.
#
# THREE THINGS IT REFUSES TO DO, each because the alternative already happened:
#
#   1. It will not write a block it cannot validate. A producer that emits
#      something the gate rejects has moved the failure to release day.
#
#   2. It will not label a lane by intent. compute_class comes from a line the
#      SERVER printed about itself, not from the flag it was handed — `apr
#      serve run --gpu` on the published binary prints no CUDA banner and holds
#      zero VRAM, which is the whole of #2696.
#
#   3. It will not compare across classes. A cpu-class apr is measured against
#      llama.cpp `-ngl 0`, never against a CUDA comparator. Cross-class is how
#      a 0.099x artifact defect reads as a kernel defect.
#
# Usage:
#   scripts/parity_host_receipt.sh --apr <path> --model <gguf> [--out <file>]
#                                  [--install-source crates.io|local-build]
#                                  [--runs N] [--duration S] [--warmup S]
set -euo pipefail

APR=""; MODEL=""; OUT="-"; SRC="crates.io"
RUNS=3; DURATION=30; WARMUP=15; COOLDOWN=10; PROFILE="medium"
while [ $# -gt 0 ]; do
    case "$1" in
        --apr) APR="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --install-source) SRC="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --profile) PROFILE="$2"; shift 2 ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[ -n "$APR" ] && [ -x "$APR" ] || { printf 'FAIL  --apr must name an executable\n' >&2; exit 2; }
[ -n "$MODEL" ] && [ -f "$MODEL" ] || { printf 'FAIL  --model must name a GGUF file\n' >&2; exit 2; }

# The comparator is resolved and PINNED, never taken from PATH.
# shellcheck source=scripts/llama_bin.sh
. "$(dirname "$0")/llama_bin.sh" || {
    printf 'FAIL  llama.cpp comparator is not resolved/pinned; see scripts/llama_pin.toml\n' >&2
    exit 1
}
[ -n "${LLAMA_SERVER:-}" ] || { printf 'FAIL  no llama-server beside the pinned llama-bench\n' >&2; exit 1; }

WORK=$(mktemp -d); trap 'rm -rf "${WORK:?}"; kill_servers' EXIT
SERVER_PIDS=""
kill_servers() { for p in $SERVER_PIDS; do kill "$p" 2>/dev/null || true; done; }

sha256_of() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1 || shasum -a 256 "$1" | cut -d' ' -f1; }

wait_healthy() { # wait_healthy <port> <seconds>
    local port="$1" limit="$2" i=0
    while [ "$i" -lt "$limit" ]; do
        if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health" 2>/dev/null)" = "200" ]; then
            return 0
        fi
        i=$((i + 1)); sleep 1
    done
    return 1
}

# THE CLASS IS READ FROM THE SERVER'S OWN OUTPUT. `--gpu` proves nothing.
apr_class_from_log() {
    if grep -qi "CUDA optimized model ready\|Enabling optimized CUDA acceleration" "$1"; then
        echo cuda
    elif grep -qi "metal.*ready\|Metal acceleration" "$1"; then
        echo metal
    elif grep -qi "wgpu" "$1"; then
        echo wgpu
    else
        echo cpu
    fi
}

run_lane() { # run_lane <class> <apr-flags> <llama-ngl> -> writes $WORK/<class>.json
    local klass="$1" apr_flags="$2" ngl="$3"
    local aport=8090 lport=8091

    # shellcheck disable=SC2086
    "$APR" serve run "$MODEL" $apr_flags --port "$aport" --context-length 4096 \
        > "$WORK/apr-$klass.log" 2>&1 &
    SERVER_PIDS="$SERVER_PIDS $!"
    wait_healthy "$aport" 300 || { printf 'FAIL  apr did not become healthy for lane %s\n' "$klass" >&2; return 1; }
    local taken; taken=$(apr_class_from_log "$WORK/apr-$klass.log")
    "$APR" test llm bench --url "http://127.0.0.1:$aport" --model "$(basename "$MODEL" .gguf)" \
        --profile "$PROFILE" --warmup "$WARMUP" --duration "$DURATION" --runs "$RUNS" \
        --cooldown "$COOLDOWN" --concurrency 1 --stream --runtime-name "apr-$klass" \
        --output "$WORK/apr-$klass.json" >/dev/null 2>&1 || true
    kill_servers; SERVER_PIDS=""; sleep 5

    "$LLAMA_SERVER" -m "$MODEL" --port "$lport" -ngl "$ngl" -c 4096 -t 8 -b 1 --no-warmup \
        > "$WORK/llama-$klass.log" 2>&1 &
    SERVER_PIDS="$SERVER_PIDS $!"
    wait_healthy "$lport" 300 || { printf 'FAIL  llama-server did not become healthy for lane %s\n' "$klass" >&2; return 1; }
    local ctaken=cpu
    if [ "$ngl" != "0" ] && grep -q "layers to GPU" "$WORK/llama-$klass.log"; then
        if grep -qi "CUDA" "$WORK/llama-$klass.log"; then ctaken=cuda; else ctaken=metal; fi
    fi
    "$APR" test llm bench --url "http://127.0.0.1:$lport" --model "$(basename "$MODEL" .gguf)" \
        --profile "$PROFILE" --warmup "$WARMUP" --duration "$DURATION" --runs "$RUNS" \
        --cooldown "$COOLDOWN" --concurrency 1 --stream --runtime-name "llamacpp-$LLAMA_BUILD" \
        --output "$WORK/llama-$klass.json" >/dev/null 2>&1 || true
    kill_servers; SERVER_PIDS=""; sleep 5

    printf '%s %s %s\n' "$klass" "$taken" "$ctaken" >> "$WORK/lanes.txt"
}

# WHICH LANES THIS HOST CAN ACTUALLY REACH — asked of the binary, not assumed.
# A published apr has no GPU path at all, so its only honest lane is cpu, and
# that is the finding rather than a gap in coverage.
run_lane cpu "--no-gpu" 0
if "$APR" serve run --help 2>&1 | grep -q -- '--gpu' && \
   strings -a "$APR" 2>/dev/null | grep -qE 'cudarc|libcuda\.so|libcublas|Metal'; then
    run_lane accel "--gpu" 999
else
    printf 'REPORT this apr has no accelerated path linked; cpu lane only (#2696)\n' >&2
fi

python3 "$(dirname "$0")/lib/parity_block.py" \
    --work "$WORK" --apr "$APR" --apr-sha "$(sha256_of "$APR")" \
    --llama "$LLAMA_SERVER" --llama-sha "$(sha256_of "$LLAMA_SERVER")" \
    --llama-build "$(printf '%s' "$LLAMA_BUILD" | sed 's/.*(\(.*\)).*/\1/')" \
    --model "$MODEL" --install-source "$SRC" --out "$OUT"
