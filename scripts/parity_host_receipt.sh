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
#   4. It will not treat a failure of the measuring instrument as a
#      measurement. Every `apr test llm bench` invocation is checked for its
#      exit status AND for the report it was told to write; see measure_band.
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

# Every diagnosis names the machine it came from, because these runs are
# collected from four hosts into one ledger and "the harness failed" is not
# actionable without knowing whose harness.
HOST_TAG=$(uname -n 2>/dev/null) || HOST_TAG=""
[ -n "$HOST_TAG" ] || HOST_TAG="unknown-host"

# NO PIPE. `"$APR" --version | head -1` returns 141 rather than apr's status
# whenever head exits before apr has finished writing: head closes the pipe,
# apr takes SIGPIPE, and `set -o pipefail` hands the pipeline apr's death
# signal. That is input-size dependent, which is how it stays green locally and
# reds in CI. The first line is taken with parameter expansion instead.
APR_VERSION=$("$APR" --version 2>&1) || APR_VERSION="<--version exited non-zero>"
APR_VERSION=${APR_VERSION%%$'\n'*}

# The harness's OWN words, indented, whole. A failure diagnosis that truncates
# is a diagnosis you have to reproduce to read.
harness_said() { # harness_said <logfile>
    if [ -s "$1" ]; then
        printf '      the harness said:\n' >&2
        sed 's/^/        /' "$1" >&2
    else
        printf '      the harness printed nothing at all.\n' >&2
    fi
}

# ── THE APPARATUS IS PROVED BEFORE ANY SERVER STARTS ───────────────────────
#
# gx10 runs apr 0.64.0 (78d485eb), where `apr test llm bench` DOES NOT EXIST:
# clap answers `error: unrecognized subcommand 'llm'` and exits 2. Learning
# that after two 300-second health waits and two model loads is learning it in
# the most expensive place available, eight times over, once per band per side.
#
# BEHAVIOUR, NOT EXISTENCE — llama_bin.sh's second property, for the same
# reason: an exit status of 0 accepts `/bin/true`, which answers 0 to every
# question ever asked of it. Probed, not assumed: pointed at /bin/true, the
# status-only check passed preflight and then sat in wait_healthy for the full
# 300 seconds before saying something unrelated about a server. So the probe
# must also SPEAK, and say the one word that identifies the banded harness.
harness_preflight() {
    local log="$WORK/harness-preflight.log" rc=0 why=""
    "$APR" test llm bench --help > "$log" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then
        why="exit $rc"
    elif [ ! -s "$log" ]; then
        why="exit 0 but printed nothing — this binary answers, it does not respond"
    elif ! grep -q -- '--concurrency' "$log"; then
        # The flag that IS the banded protocol. It was `http_concurrency = 1`
        # until bands existed, and a harness without it can only re-measure the
        # worst band and call it the answer.
        why="exit 0 but its help does not mention --concurrency"
    else
        return 0
    fi
    printf 'FAIL  host=%s — the measurement harness is not usable on this apr.\n' "$HOST_TAG" >&2
    printf '      binary  %s\n' "$APR" >&2
    printf '      version %s\n' "$APR_VERSION" >&2
    printf '      probe   test llm bench --help  ->  %s\n' "$why" >&2
    harness_said "$log"
    printf '      This is the only instrument in the chain. Nothing on this host can\n' >&2
    printf '      be measured, so no lane is started and no receipt is written.\n' >&2
    return 1
}

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

# ── ONE BAND, ONE SIDE: A FAILURE IS A RESULT, NEVER A SHRUG ───────────────
#
# Both calls to `apr test llm bench` used to end `>/dev/null 2>&1 || true`,
# discarding stdout, stderr AND the exit status of the only instrument in this
# chain. One idiom erased three distinct faults:
#
#   · THE HARNESS IS NOT THERE — the gx10 case above. A silent no-op, eight
#     times, surfacing minutes later on another machine as an absent receipt
#     that says nothing about a subcommand.
#
#   · THE HARNESS REFUSED THE MEASUREMENT — and this one is a false GREEN, not
#     merely a late red. test_llm.rs writes `--output` BEFORE it validates, then
#     exits non-zero on a run with failed requests, on a run that generated zero
#     tokens, and on a regression past threshold. So a swallowed refusal leaves
#     a well-formed report on disk, and parity_block.py reads `tokens_per_sec`
#     and `decode_tok_per_sec` out of it while never looking at `failed`. A lane
#     whose requests failed publishes as a clean PASS — precisely the
#     survivors' throughput that test_llm.rs exists to refuse.
#
#   · THE HARNESS NEVER RAN — no file, no band, and the failure lands at the
#     gate as "bands [...] are declared in the protocol and absent from the
#     receipt": a true statement that names the wrong thing.
#
# The two kinds are told apart by a MECHANICAL signal rather than by guessing
# at exit codes, which drift: whether the harness produced its report file.
#
#   rc != 0, NO report      APPARATUS. It never measured — missing subcommand,
#                           rejected flag, no connection, panic. Every band
#                           would fail identically, so the lane aborts here
#                           rather than spending six more minutes proving it
#                           four more times.
#
#   rc != 0, report present VERDICT. It measured and refused. That is a fact
#                           about the runtime, and its shape ACROSS bands is
#                           the diagnosis — gx10 refuses two of four — so the
#                           sweep finishes and the lane fails at the end with
#                           the whole table. The report is deliberately LEFT
#                           WHERE IT IS: deleting it would turn the producer's
#                           honest refusal into a silently dropped band, which
#                           is this same defect wearing a different hat.
#
#   rc == 0, NO report      APPARATUS, and meant to be impossible. Asserted
#                           rather than assumed, because "cannot happen" is
#                           where this repo keeps finding things.
#
# There is no fourth case in which a missing band is tolerable. The protocol
# declares four bands and bench_receipt.py already rules that "an unmeasured
# band is not a passing band". A band that produces nothing is a failure to
# measure, never a measurement of nothing.
#
# Returns 0 measured · 1 refused (report on disk) · 2 apparatus fault.
measure_band() { # measure_band <side> <class> <port> <concurrency> <runtime-name>
    local side="$1" klass="$2" port="$3" c="$4" name="$5"
    local out="$WORK/$side-$klass-c$c.json"
    local log="$WORK/$side-$klass-c$c.harness.log"
    local rc=0

    # So that "the file is there" can only mean "THIS invocation wrote it".
    rm -f "$out"
    # Status captured directly, never through a pipe: `$?` after a pipeline is
    # the LAST command's status, which has shipped twice in this repo.
    "$APR" test llm bench --url "http://127.0.0.1:$port" \
        --model "$(basename "$MODEL" .gguf)" \
        --profile "$PROFILE" --warmup "$WARMUP" --duration "$DURATION" \
        --runs "$RUNS" --cooldown "$COOLDOWN" --concurrency "$c" --stream \
        --runtime-name "$name" --output "$out" > "$log" 2>&1 || rc=$?

    if [ "$rc" -eq 0 ] && [ -f "$out" ]; then
        return 0
    fi

    printf 'FAIL  host=%s lane=%s side=%s band=c%s — harness exited %s\n' \
        "$HOST_TAG" "$klass" "$side" "$c" "$rc" >&2
    printf '      binary   %s (%s)\n' "$APR" "$APR_VERSION" >&2
    printf '      endpoint http://127.0.0.1:%s   runtime-name %s\n' "$port" "$name" >&2
    harness_said "$log"

    if [ -f "$out" ]; then
        printf '      It measured and then REFUSED the result. %s is left in place so\n' "$out" >&2
        printf '      the producer can render its own refusal rather than see a dropped\n' >&2
        printf '      band; the sweep continues so the shape across bands is visible,\n' >&2
        printf '      and the lane FAILS at the end.\n' >&2
        return 1
    fi
    if [ "$rc" -eq 0 ]; then
        printf '      It exited 0 and wrote NO report. A harness that reports success\n' >&2
        printf '      without producing its measurement is broken, not successful.\n' >&2
    fi
    printf '      Nothing was measured. Every remaining band would fail the same way,\n' >&2
    printf '      so this lane stops here.\n' >&2
    return 2
}

run_lane() { # run_lane <class> <apr-flags> <llama-ngl> -> writes $WORK/<class>.json
    local klass="$1" apr_flags="$2" ngl="$3"
    local aport=8090 lport=8091
    local refused="" brc=0

    # shellcheck disable=SC2086
    "$APR" serve run "$MODEL" $apr_flags --port "$aport" --context-length 4096 \
        > "$WORK/apr-$klass.log" 2>&1 &
    SERVER_PIDS="$SERVER_PIDS $!"
    wait_healthy "$aport" 300 || { printf 'FAIL  apr did not become healthy for lane %s\n' "$klass" >&2; return 1; }
    local taken; taken=$(apr_class_from_log "$WORK/apr-$klass.log")
    for c in $BANDS; do
        brc=0
        measure_band apr "$klass" "$aport" "$c" "apr-$klass-c$c" || brc=$?
        case "$brc" in
            0) ;;
            1) refused="$refused apr/c$c" ;;
            *) return 1 ;;
        esac
    done
    kill_servers; SERVER_PIDS=""; sleep 5

    "$LLAMA_SERVER" -m "$MODEL" --port "$lport" -ngl "$ngl" -c 4096 -t 8 -b 1 --no-warmup \
        > "$WORK/llama-$klass.log" 2>&1 &
    SERVER_PIDS="$SERVER_PIDS $!"
    wait_healthy "$lport" 300 || { printf 'FAIL  llama-server did not become healthy for lane %s\n' "$klass" >&2; return 1; }
    local ctaken=cpu
    if [ "$ngl" != "0" ] && grep -q "layers to GPU" "$WORK/llama-$klass.log"; then
        if grep -qi "CUDA" "$WORK/llama-$klass.log"; then ctaken=cuda; else ctaken=metal; fi
    fi
    for c in $BANDS; do
        brc=0
        measure_band llama "$klass" "$lport" "$c" "llamacpp-$klass-c$c" || brc=$?
        case "$brc" in
            0) ;;
            1) refused="$refused llama/c$c" ;;
            *) return 1 ;;
        esac
    done
    kill_servers; SERVER_PIDS=""; sleep 5

    # BOTH sides are swept before a refusal is acted on, deliberately. Which
    # bands refuse, and on which side, is the diagnosis: a zero-token band on
    # apr alone is an apr defect, the same band refusing on both sides is the
    # model or the protocol. Stopping at the first refusal reports one of the
    # two zero-token bands gx10 actually has and hides the comparator entirely.
    if [ -n "$refused" ]; then
        printf 'FAIL  host=%s lane=%s — the harness refused these bands:%s\n' \
            "$HOST_TAG" "$klass" "$refused" >&2
        printf '      A refused band is not a measured band, and its report on disk is\n' >&2
        printf '      not a sample. Each refusal is printed above in the harness own\n' >&2
        printf '      words; this lane is not written to lanes.txt.\n' >&2
        return 1
    fi

    printf '%s %s %s\n' "$klass" "$taken" "$ctaken" >> "$WORK/lanes.txt"
}

harness_preflight || exit 1

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
