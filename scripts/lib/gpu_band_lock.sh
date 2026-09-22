# gpu_band_lock.sh -- hold the fleet GPU lock for ONE band, never for a whole parity run (#3731).
#
# WHY. host_receipt.sh used to wrap all of parity_host_receipt.sh in one GPU-rule call, so its
# cpu-c1/c4/c8/c16 bands (apr --gpu-layers 0 against llama.cpp -ngl 0) sat inside /tmp/apr-gpu.lock.
# Measured on lambda 2026-09-21: 54+ min holding the lock at 1% GPU utilisation while ~10
# release-blocking cells queued behind it. The cop's ruling (rule rev 5): the lock is taken PER
# BAND and ONLY for GPU work -- the accel probe and each accel band -- through the ordered queue,
# with the wait bounded (GPUQ_WAIT) and each band bounded (<= 20 min). CPU bands never touch it.
#
# HOW. gpu-q runs its command as `flock /tmp/apr-gpu.lock choom -n 1000 -- <cmd>`, and flock(1)
# holds the lock while ANY process holding its descriptor lives. So the band does not run inside
# gpu-q; a HOLDER does: it queues, writes a ready file once it holds the lock, and sleeps as a
# child under a TERM trap, so release (TERM to the holder) leaves no process holding the lock. At
# the band's time limit the holder kills the band's registered servers WHILE still holding the
# lock, then exits: a band can never run GPU work unlocked, and never hold the lock past its bound.
#
# Sourced by scripts/parity_host_receipt.sh. OPTION-NEUTRAL (no `set`): the caller runs set -e,
# so every command here that may fail is guarded (scripts/check_sourced_libs_option_neutral.sh).
#
#   GPU_BAND_Q          the queue prefix, e.g. "gpu-q --prio 8 --"; EMPTY = no GPU rule on this
#                       host, and every function is a no-op
#   GPU_BAND_TIMEOUT_S  the band's bound in seconds (default 1200; values above 1200 are cut to it)
#   GPUQ_WAIT           read by gpu-q v3: bounds the queue wait (exit 75 on timeout, command not run)
#
#   gpu_band_acquire TAG DIR   0 held (or no rule); 1 the queue gave up or the holder died
#   gpu_band_track PID         register a band server the holder must kill at the time limit
#   gpu_band_expired           0 when the band ran past its bound (the holder killed its servers)
#   gpu_band_release           release the lock; idempotent

GPU_BAND_HOLDER=""
GPU_BAND_DIR=""
GPU_BAND_TAG=""

gpu_band_acquire() { # gpu_band_acquire TAG DIR
    local tag="$1" dir="$2" t ready
    GPU_BAND_HOLDER=""; GPU_BAND_DIR="$dir"; GPU_BAND_TAG="$tag"
    [ -n "${GPU_BAND_Q:-}" ] || return 0
    t="${GPU_BAND_TIMEOUT_S:-1200}"
    case "$t" in ''|*[!0-9]*) printf 'FAIL  GPU_BAND_TIMEOUT_S=%s is not whole seconds\n' "$t" >&2; return 1 ;; esac
    [ "$t" -le 1200 ] || t=1200
    ready="$dir/gpu-held-$tag"
    rm -f -- "$ready" "$dir/gpu-expired-$tag"
    : > "$dir/gpu-pids-$tag" || return 1
    # The holder: queue, then (holding the lock) write the ready file, sleep as a CHILD, and on the
    # time limit kill the band's servers before exiting. TERM kills the sleep and exits at once.
    # shellcheck disable=SC2086,SC2016
    $GPU_BAND_Q bash -c '
        trap "kill \$s 2> /dev/null; exit 0" TERM
        printf "%s\n" "$$" > "$1"
        sleep "$2" & s=$!
        wait "$s"
        : > "$4"
        while read -r p; do [ -n "$p" ] && kill "$p" 2> /dev/null; done < "$3"
        while read -r p; do
            i=0; while [ -n "$p" ] && kill -0 "$p" 2> /dev/null && [ "$i" -lt 30 ]; do sleep 1; i=$((i + 1)); done
            [ -n "$p" ] && kill -9 "$p" 2> /dev/null
        done < "$3"
        exit 0' _ "$ready" "$t" "$dir/gpu-pids-$tag" "$dir/gpu-expired-$tag" &
    GPU_BAND_HOLDER=$!
    while [ ! -s "$ready" ]; do
        if ! kill -0 "$GPU_BAND_HOLDER" 2> /dev/null; then
            wait "$GPU_BAND_HOLDER" 2> /dev/null || true
            GPU_BAND_HOLDER=""
            return 1
        fi
        sleep 1
    done
    return 0
}

gpu_band_track() { # gpu_band_track PID
    [ -n "$GPU_BAND_HOLDER" ] || return 0
    printf '%s\n' "$1" >> "$GPU_BAND_DIR/gpu-pids-$GPU_BAND_TAG" || true
}

gpu_band_expired() {
    [ -n "$GPU_BAND_DIR" ] && [ -n "$GPU_BAND_TAG" ] && [ -e "$GPU_BAND_DIR/gpu-expired-$GPU_BAND_TAG" ]
}

gpu_band_release() {
    local holder_pid
    [ -n "$GPU_BAND_HOLDER" ] || return 0
    holder_pid=""
    read -r holder_pid < "$GPU_BAND_DIR/gpu-held-$GPU_BAND_TAG" 2> /dev/null || true
    [ -z "$holder_pid" ] || kill "$holder_pid" 2> /dev/null || true
    wait "$GPU_BAND_HOLDER" 2> /dev/null || true
    GPU_BAND_HOLDER=""
    return 0
}
