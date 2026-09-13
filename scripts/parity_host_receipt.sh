#!/usr/bin/env bash
# parity_host_receipt.sh — produce the parity block for ONE host (#2696).
#
# Runs `apr test llm bench` against apr and against llama.cpp with the SAME
# client, prompts and clock, once per compute class the host can actually
# reach, and emits a block that scripts/lib/bench_receipt.py --parity accepts.
#
# FIVE THINGS IT REFUSES TO DO, each because the alternative already happened:
#
#   1. It will not write a block it cannot validate. A producer that emits
#      something the gate rejects has moved the failure to release day.
#
#   2. It will not label a lane by intent. compute_class comes from a line the
#      SERVER printed about itself, not from the flag it was handed — `apr
#      serve run --gpu` on the published binary prints no CUDA banner and holds
#      zero VRAM, which is the whole of #2696. The accel lane is now taken iff
#      the loader's own `gpu-layers: requested=… resolved=N total=T` line
#      reports N > 0. The old probe grepped `--help` for `--gpu`, so the lane
#      guard was built out of the very flag PP-15 wants gone: removing the
#      boolean would have made the accel lane silently disappear.
#
#   3. It will not compare across classes. A cpu-class apr is measured against
#      llama.cpp `-ngl 0`, never against a CUDA comparator. Cross-class is how
#      a 0.099x artifact defect reads as a kernel defect.
#
#   4. It will not run a band the comparator cannot serve. `GET /props` is read
#      before the first request; `total_slots < c` or a per-slot `n_ctx` below
#      the workload's 640 refuses the band by name (§5.3, PP-24). The witness
#      this replaces was a `sed` over the server log for `n_parallel = N` — a
#      line the pinned build prints ONLY in its auto branch, so under the
#      decided `-np c` it read `unreported` exactly when it mattered.
#
#   5. It will not emit a sweep and call it interleaved. §4.3 requires n >= 5
#      PAIRED replicates alternating A,B,A,B within one invocation, because
#      thermal state, graph-capture warm state and free VRAM all drift across a
#      sweep and alternation is the only design that cancels the drift. This
#      script used to run apr for every band, kill it, then llama for every
#      band: A,A,A,A then B,B,B,B, with no `interleaved` field written anywhere.
#      No conformant receipt was reachable from it.
#
# Usage:
#   LLAMA_BENCH_PATH=/path/to/llama.cpp/build/bin/llama-bench \
#   scripts/parity_host_receipt.sh --apr <path> --model <gguf> [--out <file>]
#                                  [--install-source crates.io|local-build]
#                                  [--replicates N] [--duration S] [--warmup S]
#                                  [--cooldown S] [--profile P] [--dry-run]
#
# What it writes to --out: a parity block carrying `layout: "executor"` and a
# `run_id` both lanes of every band share. scripts/lib/parity_block.py builds it
# from $WORK -- one band-<class>-c<c>.json per band, N per-replicate reports per
# lane, the comparator's /props, the subject's /v1/effective-config and the two
# device records -- and scripts/lib/perf_receipt.py --from-parity turns that
# into a schema_version 3 receipt: per-band §7.4 status, a replicate t lower
# bound on the aggregate and prefill ratios, a paired bootstrap on the decode
# ratio, and a PP-22 join key per band. A block with no `layout` is the
# historical one-report-per-lane shape and converts to schema_version 2.
#
#   LLAMA_BENCH_PATH is REQUIRED and is the only input the pin resolver takes:
#   it never consults PATH (four `apr` binaries once coexisted here and a bare
#   `apr` resolved to a 26-day-old copy). Point it at the llama-bench inside the
#   pinned build tree; llama-cli/llama-server beside it are the build oracle and
#   the CMakeCache.txt one level up is the cmake witness.
#
# Exit codes:
#   0  a parity block was written
#   1  a lane or band refused: the comparator is not the declared build, its
#      CMakeCache disagrees with build_flags_<host>, a server never became
#      healthy, /props says the comparator cannot serve the band, or another
#      parity run already holds this host's exclusive lock (§5.4, PP-19)
#   2  usage: a missing or unusable --apr / --model / argument
#   3  the declaration is missing or incomplete (scripts/llama_pin.toml)
#   4  COMPARATOR_STALE (PP-20): the pin is past its expiry. The binary is the
#      declared build, so the remedy is a RE-PIN, not a rebuild; every ratio
#      measured now is COMPARATOR_STALE (§7.4) and may not be MEASURED
set -euo pipefail

APR=""; MODEL=""; OUT="-"; SRC="crates.io"
REPLICATES=5; DURATION=30; WARMUP=15; COOLDOWN=10; PROFILE="medium"; DRY_RUN=0
WITNESS_JSON=""   # PP-26: the perf041 witness for this host, attached per band by parity_block.py
while [ $# -gt 0 ]; do
    case "$1" in
        --apr) APR="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --install-source) SRC="$2"; shift 2 ;;
        --replicates) REPLICATES="$2"; shift 2 ;;
        --witness-json) WITNESS_JSON="$2"; shift 2 ;;
        # `--runs` used to mean "consecutive same-lane runs inside one bench
        # invocation", which is exactly the non-interleaved design §4.3
        # refuses. Accepted as an alias for the replicate count so an existing
        # caller does not silently get 3 consecutive runs of one lane.
        --runs) REPLICATES="$2"
                printf 'REPORT --runs is now --replicates (interleaved, §4.3); using %s\n' "$2" >&2
                shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --cooldown) COOLDOWN="$2"; shift 2 ;;
        --profile) PROFILE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
case "$REPLICATES" in ''|*[!0-9]*) printf 'FAIL  --replicates must be a number\n' >&2; exit 2 ;; esac
if [ "$DRY_RUN" -eq 0 ]; then
    [ -n "$APR" ] && [ -x "$APR" ] || { printf 'FAIL  --apr must name an executable\n' >&2; exit 2; }
    [ -n "$MODEL" ] && [ -f "$MODEL" ] || { printf 'FAIL  --model must name a GGUF file\n' >&2; exit 2; }
else
    [ -n "$APR" ] || APR="<apr>"
    [ -n "$MODEL" ] || MODEL="<model.gguf>"
fi

# n < 5 bounds no variance (§4.3), so a receipt built from fewer replicates is
# NONCONFORMANT-VALID and may be cited but never armed. WARN rather than refuse:
# a 3-replicate run is a legitimate smoke test, and refusing it would push
# people to a script with no conformance statement at all.
if [ "$REPLICATES" -lt 5 ]; then
    printf 'REPORT replicates=%s < 5. §4.3 sizes an effect at n=3 and bounds no\n' "$REPLICATES" >&2
    printf '       variance below 5, so the receipt will be NONCONFORMANT-VALID and\n' >&2
    printf '       may not arm a threshold.\n' >&2
fi

# The comparator is resolved and PINNED, never taken from PATH.
# shellcheck source=scripts/llama_bin.sh
. "$(dirname "$0")/llama_bin.sh" || pin_rc=$?
pin_rc=${pin_rc:-0}
case "$pin_rc" in
    0) : ;;
    1) printf 'FAIL  llama.cpp is not the declared build (%s).\n' "${LLAMA_PIN_REASON:-unknown}" >&2
       printf '      Set LLAMA_BENCH_PATH to the llama-bench inside the pinned build\n' >&2
       printf '      tree; see scripts/llama_pin.toml and scripts/check_llama_pin.sh.\n' >&2
       exit 1 ;;
    2) printf 'FAIL  the comparator is UNPINNED; a ratio measured now is\n' >&2
       printf '      EXISTENCE-ONLY and may not arm a threshold (#2676).\n' >&2
       exit 1 ;;
    3) printf 'FAIL  scripts/llama_pin.toml is missing or incomplete (%s).\n' "${LLAMA_PIN_REASON:-unknown}" >&2
       exit 3 ;;
    4) printf 'FAIL  COMPARATOR_STALE: the pin expired on %s (PP-20).\n' "${LLAMA_PIN_EXPIRY:-<unset>}" >&2
       printf '      Re-pin scripts/llama_pin.toml and record why; every ratio measured\n' >&2
       printf '      against an expired pin is COMPARATOR_STALE (§7.4).\n' >&2
       exit 4 ;;
    *) printf 'FAIL  unexpected pin resolution rc=%s\n' "$pin_rc" >&2; exit 1 ;;
esac
[ -n "${LLAMA_SERVER:-}" ] || { printf 'FAIL  no llama-server beside the pinned llama-bench\n' >&2; exit 1; }

PIN="$(dirname "$0")/llama_pin.toml"
ISOLATION="$(dirname "$0")/perf_isolation.sh"

# The three ports this harness binds. Declared BEFORE the EXIT trap, because
# kill_servers now waits on them and the trap can fire on any of the exits
# above the band loop.
APORT=8090
LPORT=8091
PROBE_PORT=8092

# ---------------------------------------------------------------------------
# §5.4 / PP-19 / §12 row 12: THE HARNESS IS THE PRODUCER OF ISOLATION ON A HOST
# WITH NO CI RUNNER.
#
# The spec's §12 row 12 says that on `lambda` and `mini` -- neither of which has
# a CI runner -- isolation is produced by "an exclusive `flock` around the whole
# cell plus an `nvidia-smi --query-compute-apps` record". The second half
# existed; the first half was a sentence. A GitHub `concurrency:` group cannot
# serialise a lane that never runs in GitHub, so without this the only thing
# standing between two concurrent parity runs on one 4090 was that nobody had
# started one yet -- and §5.4 makes any foreign compute PID fatal to the band,
# so the run that loses the race does not just measure badly, it invalidates
# both.
#
# REFUSE, NEVER QUEUE. `flock -n` fails immediately rather than blocking: a
# second run that waited would begin its warmup the instant the first released,
# on a device that has not cooled, which is the drift §4.3's interleaving exists
# to cancel. A named refusal is a finding; a queued run is a quiet measurement
# of the wrong thing.
#
# FD 9 is held for the life of the process, so the kernel releases the lock on
# exit -- including a crash, a SIGKILL, and the EXIT trap below.
#
# A MISSING `flock` IS ALSO A REFUSAL. An exclusivity claim nothing enforces is
# the theater class this document exists to refuse, so the absence of the tool
# is loud rather than silently skipped.
PERF_LOCK="${PERF_LOCK:-/tmp/perf-${PERF_HOST:-$(hostname 2>/dev/null || printf 'unknown')}.lock}"
if ! command -v flock >/dev/null 2>&1; then
    printf 'FAIL  no flock(1): this host cannot take the exclusive cell lock\n' >&2
    printf '      %s that §5.4/PP-19 requires, and an unenforced exclusivity\n' "$PERF_LOCK" >&2
    printf '      claim is worse than none. Install util-linux, or run the cell\n' >&2
    printf '      under a CI concurrency group (scripts/check_perf_concurrency_groups.sh).\n' >&2
    exit 1
fi
exec 9>"$PERF_LOCK" || { printf 'FAIL  cannot open the cell lock %s\n' "$PERF_LOCK" >&2; exit 1; }
if ! flock -n 9; then
    printf 'FAIL  another parity run holds %s. §5.4 makes any foreign compute PID\n' "$PERF_LOCK" >&2
    printf '      fatal to a band, so two runs on this device invalidate BOTH.\n' >&2
    printf '      Wait for it to finish; this refuses rather than queueing, because a\n' >&2
    printf '      queued run starts measuring on a device that has not cooled.\n' >&2
    exit 1
fi

WORK=$(mktemp -d)
# On a non-zero exit the work directory is KEPT and its path printed. The first
# W1 run on lambda (2026-09-03) refused its block because one band recorded a
# zero rate, then deleted the 40 per-replicate reports and the server logs that
# said why; three hours of measurement left no evidence to read. A refusal is
# the right verdict and the wrong moment to destroy the diagnosis.
cleanup_work() {
    local rc=$?
    kill_servers
    if [ "$rc" -eq 0 ]; then
        rm -rf "${WORK:?}"
    else
        printf 'REPORT exit %s: work directory kept for diagnosis: %s\n' "$rc" "$WORK" >&2
    fi
}
trap cleanup_work EXIT
SERVER_PIDS=""

# How long a server may take to honour SIGTERM before it is SIGKILLed. Not a
# gate threshold: nothing is compared against it and no verdict reads it.
KILL_GRACE_S=30

# Is nothing accepting connections on <port>? bash's own /dev/tcp, so this needs
# neither `ss` nor `lsof` and behaves the same on every host in §8.
port_closed() { # port_closed <port>
    ! (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
}

# TERM, THEN WAIT, THEN KILL — and the waiting is the whole point.
#
# `kill` REQUESTS an exit; it does not perform one. This function used to send
# SIGTERM and return in the same breath, and the next band relaunches on the
# SAME two ports within milliseconds. Three things follow from that, and the
# third is the one that produces a number:
#
#   1. the relaunch can fail to bind, because the old server still holds 8090;
#   2. the dead PID stays in SERVER_PIDS and is killed again next time;
#   3. `wait_healthy` gets a 200 from the PREVIOUS band's server and the band
#      is measured against a process this band did not launch.
#
# So: SIGTERM every PID, then block until each has actually gone AND every port
# is closed, escalating to SIGKILL after KILL_GRACE_S rather than returning
# early. `wait` reaps the child so its exit is not left to the shell.
kill_servers() {
    local p i port
    for p in $SERVER_PIDS; do kill "$p" 2>/dev/null || true; done
    for p in $SERVER_PIDS; do
        i=0
        while kill -0 "$p" 2>/dev/null && [ "$i" -lt "$KILL_GRACE_S" ]; do
            i=$((i + 1)); sleep 1
        done
        if kill -0 "$p" 2>/dev/null; then
            printf 'REPORT pid %s ignored SIGTERM for %ss; SIGKILL\n' "$p" "$KILL_GRACE_S" >&2
            kill -9 "$p" 2>/dev/null || true
        fi
        wait "$p" 2>/dev/null || true
    done
    SERVER_PIDS=""
    # The PIDs are gone; the SOCKETS may not be. A listener inherited by a
    # child, or a socket in the kernel's teardown, keeps answering /health after
    # its parent has been reaped — which is defect (3) above with no PID left to
    # see it.
    for port in "${APORT:-}" "${LPORT:-}" "${PROBE_PORT:-}"; do
        [ -n "$port" ] || continue
        i=0
        while ! port_closed "$port" && [ "$i" -lt "$KILL_GRACE_S" ]; do
            i=$((i + 1)); sleep 1
        done
        if ! port_closed "$port"; then
            printf 'REPORT port %s is still accepting connections %ss after the kill;\n' "$port" "$KILL_GRACE_S" >&2
            printf '       the next band may measure a server it did not launch.\n' >&2
        fi
    done
}

sha256_of() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1 || shasum -a 256 "$1" | cut -d' ' -f1; }
json_str() { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }

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

# THE CLASS IS READ FROM THE SERVER'S OWN OUTPUT. A flag proves nothing.
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

# `gpu-layers: requested=all resolved=28 total=28 (backend=cuda)` — printed on
# BOTH paths by the loader, including resolved=0, so it reports the failure it
# exists to make visible rather than only the success.
gpu_layers_field() { # gpu_layers_field <log> <requested|resolved|total>
    sed -n "s/.*gpu-layers: .*$2=\\([A-Za-z0-9]*\\).*/\\1/p" "$1" | head -1
}

# The bands, read from the declaration rather than hardcoded here, so the
# protocol file stays the single source of truth (#2696).
BANDS=$(sed -n 's/^[[:space:]]*http_concurrency_bands[[:space:]]*=[[:space:]]*\[\(.*\)\].*/\1/p' \
        "$PIN" | tr -d ' ' | tr ',' ' ')
[ -n "$BANDS" ] || { printf 'FAIL  llama_pin.toml declares no http_concurrency_bands\n' >&2; exit 3; }

# Same rule on the apr side. This line read `--context-length 4096`, a literal
# copy of the declared context_length, so apr and the comparator could have been
# run at different context lengths with both "matching" a declaration neither
# read (#2737).
CTX=$(llama_pin_get_raw context_length "$PIN")
case "$CTX" in
    ''|*[!0-9]*) printf 'FAIL  llama_pin.toml declares no numeric context_length\n' >&2; exit 3 ;;
esac
N_CTX_SLOT=$(llama_pin_get_raw n_ctx_slot "$PIN")
case "$N_CTX_SLOT" in
    ''|*[!0-9]*) printf 'FAIL  llama_pin.toml declares no numeric n_ctx_slot\n' >&2; exit 3 ;;
esac
MODEL_NAME=$(basename "$MODEL" .gguf)

# APORT / LPORT / PROBE_PORT are declared above the EXIT trap: kill_servers
# waits on them, and the trap can fire before this point.

# ---------------------------------------------------------------------------
# One BAND. Both servers are up at once and the two lanes alternate inside it,
# which is what makes the replicates paired (§4.3).
#
# Writes, per band:
#   $WORK/llama-<klass>-c<c>.props.json      GET /props, verbatim (§5.3)
#   $WORK/apr-<klass>-c<c>.config.json       GET /v1/effective-config, or absent
#   $WORK/{apr,llama}-<klass>-c<c>-r<k>.json one bench report per replicate
#   $WORK/iso-<klass>-c<c>-{before,after}.json  the device record (§5.4)
#   $WORK/band-<klass>-c<c>.json             what the band was, for the receipt
run_band() { # run_band <klass> <apr-gpu-layers> <llama-ngl> <c>
    local klass="$1" gl="$2" ngl="$3" c="$4"
    local tag="$klass-c$c"
    local lflags
    lflags=$(llama_comparator_server_flags "$ngl" "$c" "$PIN") || {
        printf 'FAIL  band %s: cannot build the comparator invocation from %s (rc=%s).\n' \
            "$tag" "$PIN" "$?" >&2
        printf '      A partial flag list would run a comparator nobody declared.\n' >&2
        return 1
    }

    if [ "$DRY_RUN" -eq 1 ]; then
        printf 'band %s\n' "$tag"
        printf '  isolation : %s before %s/iso-%s-before.json\n' "$ISOLATION" "$WORK" "$tag"
        # shellcheck disable=SC2086
        printf '  comparator: %s -m %s --port %s %s\n' "$LLAMA_SERVER" "$MODEL" "$LPORT" "$lflags"
        printf '  props     : curl -s http://127.0.0.1:%s/props  (refuse if total_slots < %s or n_ctx < 640)\n' "$LPORT" "$c"
        printf '  subject   : %s serve run %s --gpu-layers %s --port %s --context-length %s\n' \
            "$APR" "$MODEL" "$gl" "$APORT" "$CTX"
        printf '  config    : curl -s http://127.0.0.1:%s/v1/effective-config  (404 -> absent)\n' "$APORT"
        local k
        for k in $(seq 1 "$REPLICATES"); do
            printf '  r%-2s A    : %s test llm bench --url http://127.0.0.1:%s --concurrency %s --runtime-name apr-%s (r%s)\n' \
                "$k" "$APR" "$APORT" "$c" "$tag" "$k"
            printf '  r%-2s cool : %ss\n' "$k" "$COOLDOWN"
            printf '  r%-2s B    : %s test llm bench --url http://127.0.0.1:%s --concurrency %s --runtime-name llamacpp-%s (r%s)\n' \
                "$k" "$APR" "$LPORT" "$c" "$tag" "$k"
            printf '  r%-2s cool : %ss\n' "$k" "$COOLDOWN"
        done
        # THE ORDER IS PART OF THE PROTOCOL, so the dry run prints it: the
        # `after` record is taken while both servers are STILL RUNNING and
        # still declared as ours, and the shutdown comes after it. Reversed --
        # which is what this script used to do -- the run's own dying servers
        # are recorded as foreign compute PIDs and every clean band reads as a
        # §5.4 breach.
        printf '  isolation : %s after %s/iso-%s-after.json  (servers STILL UP; PERF_ISOLATION_OWN_PIDS = this band'"'"'s server PIDs)\n' \
            "$ISOLATION" "$WORK" "$tag"
        printf '  shutdown  : SIGTERM both servers, then WAIT up to %ss for each PID to exit and for ports %s/%s/%s to close, then SIGKILL\n' \
            "$KILL_GRACE_S" "$APORT" "$LPORT" "$PROBE_PORT"
        return 0
    fi

    bash "$ISOLATION" before "$WORK/iso-$tag-before.json" || true

    # THE COMPARATOR'S FLAGS COME FROM THE DECLARATION, NOT FROM THIS LINE.
    # This line used to read `-c 4096 -t 8 -b 1`: a third copy of values
    # scripts/llama_pin.toml already declared, and `-b 1` alone inflated the
    # c=16 aggregate ratio 2.03x -> 4.85x by switching llama.cpp's batching off
    # (#2737). llama_comparator_server_flags is the only producer, and
    # scripts/check_comparator_flags.sh fails if this line grows a literal back.
    # shellcheck disable=SC2086
    "$LLAMA_SERVER" -m "$MODEL" --port "$LPORT" $lflags \
        > "$WORK/llama-$tag.log" 2>&1 &
    SERVER_PIDS="$SERVER_PIDS $!"
    wait_healthy "$LPORT" 300 || {
        printf 'FAIL  band %s: llama-server did not become healthy\n' "$tag" >&2
        kill_servers; return 1
    }

    # SERVER-REPORTED ADMISSION, BEFORE THE FIRST REQUEST (§5.3, PP-24).
    curl -s "http://127.0.0.1:$LPORT/props" > "$WORK/llama-$tag.props.json" || true
    local slots nctx
    slots=$(sed -n 's/.*"total_slots"[[:space:]]*:[[:space:]]*\([0-9]*\).*/\1/p' "$WORK/llama-$tag.props.json" | head -1)
    nctx=$(sed -n 's/.*"n_ctx"[[:space:]]*:[[:space:]]*\([0-9]*\).*/\1/p' "$WORK/llama-$tag.props.json" | head -1)
    if [ -z "$slots" ] || [ -z "$nctx" ]; then
        printf 'FAIL  band %s: GET /props carried no total_slots/n_ctx; the band cannot\n' "$tag" >&2
        printf '      state what the comparator admitted, so its ratio is unreadable.\n' >&2
        kill_servers; return 1
    fi
    if [ "$slots" -lt "$c" ]; then
        printf 'FAIL  band %s: the comparator admits %s slots for %s clients. It is not\n' "$tag" "$slots" "$c" >&2
        printf '      serving the band, it is queueing %s of them (PP-24, §5.3).\n' "$((c - slots))" >&2
        kill_servers; return 1
    fi
    if [ "$nctx" -lt 640 ]; then
        printf 'FAIL  band %s: per-slot n_ctx is %s; W1 is 512 prompt + 128 generated,\n' "$tag" "$nctx" >&2
        printf '      so below 640 the slot truncates and the band measures another\n' >&2
        printf '      workload while reporting this one (§5.3).\n' >&2
        kill_servers; return 1
    fi
    # The comparator's resolved batch size, from the line the server prints
    # about itself. PP-22 carries n_batch in the join key precisely so a `-b 1`
    # lane can never be joined to a default one.
    local nbatch
    nbatch=$(sed -n 's/.*n_batch[[:space:]]*=[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$WORK/llama-$tag.log" | head -1)

    # shellcheck disable=SC2086
    "$APR" serve run "$MODEL" --gpu-layers "$gl" --port "$APORT" --context-length "$CTX" \
        > "$WORK/apr-$tag.log" 2>&1 &
    SERVER_PIDS="$SERVER_PIDS $!"
    wait_healthy "$APORT" 300 || {
        printf 'FAIL  band %s: apr did not become healthy\n' "$tag" >&2
        kill_servers; return 1
    }
    local taken glr glq glt
    taken=$(apr_class_from_log "$WORK/apr-$tag.log")
    glq=$(gpu_layers_field "$WORK/apr-$tag.log" requested)
    glr=$(gpu_layers_field "$WORK/apr-$tag.log" resolved)
    glt=$(gpu_layers_field "$WORK/apr-$tag.log" total)

    # §5.2 / PP-2: the subject's resolved configuration, verbatim, before the
    # first request. An older binary has no such route; `absent` is recorded so
    # the receipt says the field was not produced rather than omitting it.
    local cfg_code
    cfg_code=$(curl -s -o "$WORK/apr-$tag.config.json" -w '%{http_code}' \
        "http://127.0.0.1:$APORT/v1/effective-config" 2>/dev/null || printf '000')
    local cfg_state=present
    if [ "$cfg_code" != "200" ]; then
        cfg_state=absent
        printf 'absent\n' > "$WORK/apr-$tag.config.json"
        printf 'REPORT band %s: GET /v1/effective-config returned %s; recorded as absent\n' \
            "$tag" "$cfg_code" >&2
    fi

    # INTERLEAVED: A, B, A, B, … inside one invocation (§4.3). The cooldown sits
    # between LANES, not only between replicates, because the drift being
    # cancelled is between the two measurements being divided.
    local k
    for k in $(seq 1 "$REPLICATES"); do
        "$APR" test llm bench --url "http://127.0.0.1:$APORT" --model "$MODEL_NAME" \
            --profile "$PROFILE" --warmup "$WARMUP" --duration "$DURATION" --runs 1 \
            --cooldown "$COOLDOWN" --concurrency "$c" --stream --runtime-name "apr-$tag" \
            --output "$WORK/apr-$tag-r$k.json" >/dev/null 2>&1 || true
        sleep "$COOLDOWN"
        "$APR" test llm bench --url "http://127.0.0.1:$LPORT" --model "$MODEL_NAME" \
            --profile "$PROFILE" --warmup "$WARMUP" --duration "$DURATION" --runs 1 \
            --cooldown "$COOLDOWN" --concurrency "$c" --stream --runtime-name "llamacpp-$tag" \
            --output "$WORK/llama-$tag-r$k.json" >/dev/null 2>&1 || true
        sleep "$COOLDOWN"
    done

    # THE `after` RECORD IS TAKEN WHILE THIS BAND'S SERVERS ARE STILL UP, AND
    # STILL NAMED AS OURS.
    #
    # It used to run `PERF_ISOLATION_OWN_PIDS=""` immediately after the kill.
    # Both halves of that are wrong in the same direction. A SIGTERMed server
    # holds its CUDA context for as long as it takes to unwind, so it is still
    # in `nvidia-smi --query-compute-apps` when the record is taken; and with
    # OWN_PIDS blanked, every PID the probe sees is FOREIGN by construction. The
    # record therefore reported a §5.4 breach — "a foreign compute PID during
    # the window", which is fatal to the band — on every CLEAN band, naming this
    # run's own two dying servers as the intruders.
    #
    # Taking it BEFORE the kill answers the question §5.4 actually asks: who
    # else was on the device while we were measuring. `kill_servers` follows.
    PERF_ISOLATION_OWN_PIDS="$SERVER_PIDS" bash "$ISOLATION" after "$WORK/iso-$tag-after.json" || true
    kill_servers

    cat > "$WORK/band-$tag.json" <<JSON
{
  "class": "$(json_str "$klass")",
  "concurrency": $c,
  "interleaved": true,
  "replicates": $REPLICATES,
  "client_concurrency": {"subject": $c, "comparator": $c},
  "comparator_flags": "$(json_str "$lflags")",
  "comparator_slots_admitted": $slots,
  "comparator_n_ctx_slot": $nctx,
  "comparator_n_batch": ${nbatch:-null},
  "comparator_props_file": "llama-$tag.props.json",
  "subject_compute_class": "$(json_str "$taken")",
  "subject_effective_config": "$cfg_state",
  "subject_effective_config_file": "apr-$tag.config.json",
  "gpu_layers_requested": "$(json_str "${glq:-unreported}")",
  "gpu_layers_resolved": ${glr:-null},
  "gpu_layers_total": ${glt:-null},
  "isolation_before_file": "iso-$tag-before.json",
  "isolation_after_file": "iso-$tag-after.json",
  "replicate_files": {
    "subject": "apr-$tag-r{k}.json",
    "comparator": "llama-$tag-r{k}.json"
  }
}
JSON
    printf 'REPORT band %s: comparator [%s] slots=%s n_ctx=%s n_batch=%s; apr gpu-layers %s->%s/%s; class %s\n' \
        "$tag" "$lflags" "$slots" "$nctx" "${nbatch:-unreported}" \
        "${glq:-unreported}" "${glr:-unreported}" "${glt:-unreported}" "$taken" >&2
    return 0
}

run_lane() { # run_lane <klass> <apr-gpu-layers> <llama-ngl>
    local klass="$1" gl="$2" ngl="$3" c brc
    for c in $BANDS; do
        brc=0
        run_band "$klass" "$gl" "$ngl" "$c" || brc=$?
        [ "$brc" -eq 0 ] || return "$brc"
    done
    printf '%s %s\n' "$klass" "$gl" >> "$WORK/lanes.txt"
}

# WHICH LANES THIS HOST CAN ACTUALLY REACH — asked of the LOADER, not of
# `--help` and not of the flag. The accel lane is taken iff apr, started with
# `--gpu-layers all`, reports `resolved > 0` about itself.
probe_accel() {
    local pl="$PROBE_PORT" r
    "$APR" serve run "$MODEL" --gpu-layers all --port "$pl" --context-length "$CTX" \
        > "$WORK/apr-probe.log" 2>&1 &
    SERVER_PIDS="$SERVER_PIDS $!"
    wait_healthy "$pl" 300 || true
    r=$(gpu_layers_field "$WORK/apr-probe.log" resolved)
    kill_servers
    printf '%s' "${r:-0}"
}

if [ "$DRY_RUN" -eq 1 ]; then
    printf -- '--- dry run: the commands each band would issue ----------------------\n'
    printf 'comparator: %s (pin %s, expiry %s)\n' \
        "${LLAMA_SERVER:-<unresolved>}" "$(llama_pin_get build_commit "$PIN")" "${LLAMA_PIN_EXPIRY:-<unset>}"
    printf 'bands     : %s   replicates: %s   interleaved: true\n' "$BANDS" "$REPLICATES"
    printf 'cell lock : %s (exclusive, flock -n; a second run on this host REFUSES, §5.4/PP-19)\n' "$PERF_LOCK"
    printf 'accel probe: %s serve run %s --gpu-layers all --port 8092 (accel lane iff resolved > 0)\n' \
        "$APR" "$MODEL"
    for c in $BANDS; do
        run_band cpu 0 0 "$c"
    done
    for c in $BANDS; do
        run_band accel all 999 "$c"
    done
    printf -- '--- end dry run (nothing was launched) -------------------------------\n'
    exit 0
fi

# A cpu-class apr must be measured against llama.cpp at `-ngl 0`, so the
# quantity is 0 on both sides rather than a boolean absence on one.
run_lane cpu 0 0

ACCEL_RESOLVED=$(probe_accel)
case "$ACCEL_RESOLVED" in ''|*[!0-9]*) ACCEL_RESOLVED=0 ;; esac
if [ "$ACCEL_RESOLVED" -gt 0 ]; then
    run_lane accel all 999
else
    # Say WHY, in the receipt, in a form the gate can read. A gate that demands
    # a lane its own producer cannot emit is unsatisfiable, and an unsatisfiable
    # gate gets bypassed for substance (#2696).
    printf 'no-accelerator-resolved\n' > "$WORK/accel-absent.txt"
    printf 'REPORT this apr resolved %s gpu layers; cpu lane only. The receipt records\n' "$ACCEL_RESOLVED" >&2
    printf '       accel_absent so the gate reports rather than demanding a lane that\n' >&2
    printf '       cannot exist (#2696).\n' >&2
fi

# THE CONSUMER READS $WORK, NOT A RESTATED CONVENTION. Every file name it needs
# is written into band-<class>-c<c>.json above, so this line hands over the
# directory and the two facts that live outside it:
#
#   --pin-expiry  the declaration's own expiry. This script REFUSES to start
#                 against an expired pin (exit 4), so a block written here is
#                 fresh -- but a receipt read months later cannot tell that from
#                 silence, and §7.4's COMPARATOR_STALE has to be decidable from
#                 the artifact rather than from the fact that it exists.
#   --run-id      omitted: parity_block.py mints one per invocation, and both
#                 lanes of every band inherit it. Passing one from here would
#                 let two invocations share an id, which is the cross-run
#                 pairing PP-3 exists to make unwriteable.
python3 "$(dirname "$0")/lib/parity_block.py" \
    --work "$WORK" --apr "$APR" --apr-sha "$(sha256_of "$APR")" \
    --llama "$LLAMA_SERVER" --llama-sha "$(sha256_of "$LLAMA_SERVER")" \
    --llama-build "$(printf '%s' "$LLAMA_BUILD" | sed 's/.*(\(.*\)).*/\1/')" \
    --pin-expiry "${LLAMA_PIN_EXPIRY:-}" \
    --model "$MODEL" --install-source "$SRC" --out "$OUT" \
    ${WITNESS_JSON:+--witness-json "$WITNESS_JSON"}
