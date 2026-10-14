#!/usr/bin/env bash
# nextest_list_wrapper.sh — the `ci` profile's list-wrapper (.config/nextest.toml).
#
# WHY THIS EXISTS (#5294)
# -----------------------
# nextest runs every test binary with `--list` before any test starts, and has no
# timeout for that phase. Merge-group run 38534258555 sat 72 min in
# `realizar-<hash> --list --format terse` at 0% CPU; the process was gone before
# anyone read where it was blocked, so the hang has no cause yet.
#
# This bounds the listing (default 300 s, as `timeout` did) and, BEFORE killing
# the binary, writes to stderr where both processes are blocked: per-thread state
# and wchan, the blocked syscall, the kernel stack (when readable), and the fd 1
# target, for the binary and for its reader (this script's parent, nextest).
# nextest prints a failed listing's stderr, so the dump lands in the job log.
#
# The dump is read-only and changes no exit code: a timed-out listing exits 124,
# exactly like `timeout`; any other exit status passes through unchanged.
#
#   scripts/ci/nextest_list_wrapper.sh <test-binary> --list --format terse
#   NEXTEST_LIST_TIMEOUT_S=5 scripts/ci/nextest_list_wrapper.sh ...
#   bash scripts/ci/nextest_list_wrapper.sh --self-test   # case table

set -uo pipefail

dump_proc() {
    local label="$1" pid="$2" t
    printf '%s pid %s: %s\n' "$label" "$pid" "$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)"
    printf '  fd1 -> %s\n' "$(readlink "/proc/$pid/fd/1" 2>/dev/null || printf 'unreadable')"
    for t in /proc/"$pid"/task/*; do
        [ -d "$t" ] || continue
        printf '  tid %s state=%s wchan=%s syscall=%s\n' "${t##*/}" \
            "$(awk '{print $3}' "$t/stat" 2>/dev/null)" \
            "$(cat "$t/wchan" 2>/dev/null || printf '?')" \
            "$(cat "$t/syscall" 2>/dev/null || printf '?')"
    done
    if [ -r "/proc/$pid/stack" ]; then
        sed 's/^/  stack /' "/proc/$pid/stack" 2>/dev/null
    else
        printf '  stack unreadable\n'
    fi
}

run_wrapped() {
    local limit="${NEXTEST_LIST_TIMEOUT_S:-300}" pid ticks=0 grace=0
    # 0<&0: a background job in a non-interactive shell otherwise gets /dev/null
    # as stdin, and the binary must see the same stdin nextest gave this script.
    "$@" 0<&0 &
    pid=$!
    # Poll every 0.1 s rather than run a background watchdog: a watchdog `sleep`
    # would inherit the stdout/stderr pipes and hold them open after the binary
    # exits, so nextest would wait on EOF. The poll adds at most 0.1 s per listing.
    while kill -0 "$pid" 2>/dev/null; do
        if [ "$ticks" -ge $((limit * 10)) ]; then
            {
                printf '=== nextest list-phase timeout (#5294): %ss, dump before kill ===\n' "$limit"
                dump_proc binary "$pid"
                dump_proc reader "$PPID"
                printf '=== end dump ===\n'
            } >&2
            kill -TERM "$pid" 2>/dev/null
            while kill -0 "$pid" 2>/dev/null && [ "$grace" -lt 100 ]; do
                sleep 0.1
                grace=$((grace + 1))
            done
            kill -KILL "$pid" 2>/dev/null
            wait "$pid" 2>/dev/null
            return 124
        fi
        sleep 0.1
        ticks=$((ticks + 1))
    done
    wait "$pid"
}

if [ "${1:-}" = "--self-test" ]; then
    fails=0
    SD="$(mktemp -d)" || exit 1
    case "$SD" in /tmp/*) : ;; *) printf 'BADTMP %s\n' "$SD"; exit 1 ;; esac
    trap 'rm -rf "${SD:?}"' EXIT
    self="${BASH_SOURCE[0]}"

    check() {
        if [ "$2" = "$3" ]; then printf 'ok    %s\n' "$1"; else printf 'FAIL  %s: got [%s], expected [%s]\n' "$1" "$2" "$3"; fails=1; fi
    }

    out="$(bash "$self" printf 'a: test\n' 2>/dev/null)"; rc=$?
    check "row 1 a fast listing passes its stdout through" "$out" "a: test"
    check "row 1 rc 0" "$rc" "0"

    bash "$self" bash -c 'exit 3' 2>/dev/null; rc=$?
    check "row 2 a failing binary's rc passes through" "$rc" "3"

    NEXTEST_LIST_TIMEOUT_S=2 bash "$self" sleep 600 > "$SD/o" 2> "$SD/e"; rc=$?
    check "row 3 a hang exits 124 at the bound" "$rc" "124"
    check "row 3 the dump names the binary's threads" "$(grep -c '^binary pid' "$SD/e")" "1"
    check "row 3 the dump has a wchan line" "$(grep -c 'wchan=' "$SD/e" | awk '{print ($1>0)}')" "1"
    check "row 3 the dump names the reader" "$(grep -c '^reader pid' "$SD/e")" "1"
    check "row 3 nothing is written to stdout" "$(wc -c < "$SD/o" | tr -d ' ')" "0"

    bash "$self" bash -c 'head -c 2000000 /dev/zero | tr "\0" "x"' > "$SD/big" 2>/dev/null; rc=$?
    check "row 4 a 2 MB listing passes through intact" "$(wc -c < "$SD/big" | tr -d ' ')" "2000000"
    check "row 4 rc 0" "$rc" "0"

    printf 'in\n' | bash "$self" cat > "$SD/in" 2>/dev/null
    check "row 5 the binary sees the caller's stdin, not /dev/null" "$(cat "$SD/in")" "in"

    [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED\n'; exit 1; }
    printf '\nSELF-TEST PASSED\n'
    exit 0
fi

[ "$#" -ge 1 ] || { printf 'usage: %s <test-binary> [args...]\n' "${0##*/}" >&2; exit 2; }
run_wrapped "$@"
