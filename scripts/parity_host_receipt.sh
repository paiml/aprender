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

# The bands, read from the declaration rather than hardcoded here, so the
# protocol file stays the single source of truth (#2696).
BANDS=$(sed -n 's/^[[:space:]]*http_concurrency_bands[[:space:]]*=[[:space:]]*\[\(.*\)\].*/\1/p' \
        "$(dirname "$0")/llama_pin.toml" | tr -d ' ' | tr ',' ' ')
[ -n "$BANDS" ] || { printf 'FAIL  llama_pin.toml declares no http_concurrency_bands\n' >&2; exit 1; }

# Same rule on the apr side. This line read `--context-length 4096`, a literal
# copy of the declared context_length, so apr and the comparator could have been
# run at different context lengths with both "matching" a declaration neither
# read (#2737).
CTX=$(llama_pin_get_raw context_length "$(dirname "$0")/llama_pin.toml")
case "$CTX" in
    ''|*[!0-9]*) printf 'FAIL  llama_pin.toml declares no numeric context_length\n' >&2; exit 1 ;;
esac

run_lane() { # run_lane <class> <apr-flags> <llama-ngl> -> writes $WORK/<class>.json
    local klass="$1" apr_flags="$2" ngl="$3"
    local aport=8090 lport=8091

    # shellcheck disable=SC2086
    "$APR" serve run "$MODEL" $apr_flags --port "$aport" --context-length "$CTX" \
        > "$WORK/apr-$klass.log" 2>&1 &
    SERVER_PIDS="$SERVER_PIDS $!"
    wait_healthy "$aport" 300 || { printf 'FAIL  apr did not become healthy for lane %s\n' "$klass" >&2; return 1; }
    local taken; taken=$(apr_class_from_log "$WORK/apr-$klass.log")
    for c in $BANDS; do
        "$APR" test llm bench --url "http://127.0.0.1:$aport" --model "$(basename "$MODEL" .gguf)" \
            --profile "$PROFILE" --warmup "$WARMUP" --duration "$DURATION" --runs "$RUNS" \
            --cooldown "$COOLDOWN" --concurrency "$c" --stream --runtime-name "apr-$klass-c$c" \
            --output "$WORK/apr-$klass-c$c.json" >/dev/null 2>&1 || true
    done
    kill_servers; SERVER_PIDS=""; sleep 5

    # THE COMPARATOR'S FLAGS COME FROM THE DECLARATION, NOT FROM THIS LINE.
    #
    # This line used to read `-c 4096 -t 8 -b 1`: a third copy of values that
    # scripts/llama_pin.toml already declared as context_length, threads and
    # batch_size. Three copies, no joins, and `-b 1` alone inflated the c=16
    # aggregate ratio 2.03x -> 4.85x by switching llama.cpp's batching off
    # (#2737). llama_comparator_server_flags is now the only producer, and
    # scripts/check_comparator_flags.sh fails if this line grows a literal back.
    #
    # FAIL CLOSED. An unreadable declaration returns non-zero and prints
    # nothing, so the alternative to correct flags is no run — never a run with
    # a silently truncated flag list.
    local lflags
    lflags=$(llama_comparator_server_flags "$ngl" "$(dirname "$0")/llama_pin.toml") || {
        printf 'FAIL  cannot build the comparator invocation from scripts/llama_pin.toml\n' >&2
        return 1
    }
    # shellcheck disable=SC2086
    "$LLAMA_SERVER" -m "$MODEL" --port "$lport" $lflags \
        > "$WORK/llama-$klass.log" 2>&1 &
    SERVER_PIDS="$SERVER_PIDS $!"
    wait_healthy "$lport" 300 || { printf 'FAIL  llama-server did not become healthy for lane %s\n' "$klass" >&2; return 1; }
    local ctaken=cpu
    if [ "$ngl" != "0" ] && grep -q "layers to GPU" "$WORK/llama-$klass.log"; then
        if grep -qi "CUDA" "$WORK/llama-$klass.log"; then ctaken=cuda; else ctaken=metal; fi
    fi
    # RESOLVED, NOT REQUESTED (§4.4.9). `comparator_parallel = "default"` means
    # we pass no `-np`, so the slot count is whatever the pinned build chose.
    # Report the line the SERVER printed about itself rather than inferring it,
    # so a pin bump that changes the auto value is visible in the lane output
    # instead of silently moving every band above c=4.
    local nparallel
    nparallel=$(sed -n 's/.*n_parallel = \([0-9][0-9]*\).*/\1/p' "$WORK/llama-$klass.log" | head -1)
    # Same rule for flash attention (#2743). `flash_attention = "default"` means
    # we pass no `-fa`, so the resolved value is whatever the pinned build chose
    # -- and in the pinned era (7746) that default is `auto`, which MAY turn
    # flash attention on. The declaration used to say `false` while no
    # invocation carried the flag at all, so the receipt recorded a
    # configuration that had never run. Report what the server says about
    # itself, so a pin bump that flips the default is visible in the lane output
    # instead of silently moving prefill.
    local fattn
    fattn=$(sed -n 's/.*[Ff]lash[ _-]*[Aa]tt[a-z]* *[:=] *\([A-Za-z0-9]*\).*/\1/p' \
        "$WORK/llama-$klass.log" | head -1)
    printf 'REPORT lane %s: comparator flags [%s]; server-reported n_parallel=%s flash_attn=%s\n' \
        "$klass" "$lflags" "${nparallel:-unreported}" "${fattn:-unreported}" >&2
    for c in $BANDS; do
        "$APR" test llm bench --url "http://127.0.0.1:$lport" --model "$(basename "$MODEL" .gguf)" \
            --profile "$PROFILE" --warmup "$WARMUP" --duration "$DURATION" --runs "$RUNS" \
            --cooldown "$COOLDOWN" --concurrency "$c" --stream --runtime-name "llamacpp-$klass-c$c" \
            --output "$WORK/llama-$klass-c$c.json" >/dev/null 2>&1 || true
    done
    kill_servers; SERVER_PIDS=""; sleep 5

    printf '%s %s %s\n' "$klass" "$taken" "$ctaken" >> "$WORK/lanes.txt"
}

# WHICH LANES THIS HOST CAN ACTUALLY REACH — asked of the binary, not assumed.
# A published apr has no GPU path at all, so its only honest lane is cpu, and
# that is the finding rather than a gap in coverage.
run_lane cpu "--no-gpu" 0
# Both probes read a herestring, never a pipe. `grep -q` exits on its FIRST
# match, the producer takes SIGPIPE, and `set -o pipefail` hands the pipeline
# the producer's 141 -- so the condition was FALSE on a binary that does
# contain the markers. Measured on gx10 and reproduced on lambda: 141 on
# 10/10 repeats with pipefail, 0 without, while the pattern matches 5 times.
# Every receipt therefore recorded `accel_absent` for a CUDA-capable apr,
# which is a fabricated provenance field emitted by the provenance producer.
APR_HELP="$( "$APR" serve run --help 2>&1 )"
APR_STRINGS="$( strings -a "$APR" 2>/dev/null )"
if grep -q -- '--gpu' <<< "$APR_HELP" && \
   grep -qE 'cudarc|libcuda\.so|libcublas|Metal' <<< "$APR_STRINGS"; then
    run_lane accel "--gpu" 999
else
    # Say WHY, in the receipt, in a form the gate can read. A gate that demands
    # a lane its own producer cannot emit is unsatisfiable, and an unsatisfiable
    # gate gets bypassed for substance (#2696).
    printf 'no-accelerator-linked\n' > "$WORK/accel-absent.txt"
    printf 'REPORT this apr has no accelerated path linked; cpu lane only.\n' >&2
    printf '       The receipt records accel_absent so the gate reports rather\n' >&2
    printf '       than demanding a lane that cannot exist (#2696).\n' >&2
fi

python3 "$(dirname "$0")/lib/parity_block.py" \
    --work "$WORK" --apr "$APR" --apr-sha "$(sha256_of "$APR")" \
    --llama "$LLAMA_SERVER" --llama-sha "$(sha256_of "$LLAMA_SERVER")" \
    --llama-build "$(printf '%s' "$LLAMA_BUILD" | sed 's/.*(\(.*\)).*/\1/')" \
    --model "$MODEL" --install-source "$SRC" --out "$OUT"
