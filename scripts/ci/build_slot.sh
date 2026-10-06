#!/usr/bin/env bash
# build_slot.sh -- the host build-slot pool (C313/10-06 item 3).
#
# A host has N heavy-build slots. Every cargo on the host, CI or agent, takes
# one before it starts and gives it back when it ends. A slot is a flock on
# $DIR/slot.<i>; the kernel drops it when the holder dies, so a crashed holder
# never needs a release. Queue runs (GITHUB_EVENT_NAME=merge_group) are tier 0
# and are admitted before tier 1 (everything else). Protocol v1:
#
#   $DIR = /run/lock/fleet-build/v1        lock dir (forjar tmpfiles, 1777)
#   $DIR/slot.<i>, i = 0..N-1              flock; held = admitted
#   $DIR/slot.<i>.owner                    "pid label tier t0", read only while slot.<i> is held
#   $DIR/prio                              tier-0 waiters hold it shared while they wait
#   /etc/fleet-build/slots                 N (absent: nproc/8, at least 1)
#   /var/lib/fleet-build/ledger.tsv        one row per release (best effort)
#
# The command runs with the slot fd CLOSED, so a daemon it spawns (an sccache
# server) cannot keep the slot after the command ends. A nested call passes
# straight through when the holder recorded in FLEET_BUILD_SLOT_PID is one of
# its ancestors AND that slot is still locked; an inherited env var alone
# admits nothing.
#
# Usage:
#   build_slot.sh run [--label L] -- cmd...   acquire (blocking, no timeout), run, release; rc of cmd
#   build_slot.sh status                      N, then one line per held slot
#   build_slot.sh --self-test                 case table
#
# cmd sees FLEET_BUILD_SLOT, FLEET_BUILD_SLOT_PID, FLEET_BUILD_SLOT_T0 (epoch s
# at admission), FLEET_BUILD_SLOT_WAIT_S, FLEET_BUILD_SLOT_N, FLEET_BUILD_SLOT_LABEL.
#
# No pool on the host (no $DIR): with FLEET_BUILD_SLOT_REQUIRED=1 that is an
# error (rc 1, the ci-target-admit rule); otherwise one notice and cmd runs
# unadmitted. FLEET_BUILD_SLOT_DIR, FLEET_BUILD_SLOTS_FILE, FLEET_BUILD_LEDGER,
# FLEET_BUILD_PRIO and FLEET_BUILD_SLOT_POLL_S exist for the self-test only;
# check_build_slot_call_sites.sh keeps them out of workflows.
#
# Sourceable: defines functions only, sets no shell options (see
# scripts/check_sourced_libs_option_neutral.sh).

bs_dir() { printf '%s' "${FLEET_BUILD_SLOT_DIR:-/run/lock/fleet-build/v1}"; }

bs_n() {
    local f="${FLEET_BUILD_SLOTS_FILE:-/etc/fleet-build/slots}" n=""
    if [ -r "$f" ]; then
        n=$(tr -d ' \t\n' < "$f")
    fi
    if ! [[ "$n" =~ ^[1-9][0-9]*$ ]]; then
        n=$(( $(nproc) / 8 ))
        [ "$n" -ge 1 ] || n=1
    fi
    printf '%s' "$n"
}

bs_tier() {
    case "${FLEET_BUILD_PRIO:-}" in
        0 | 1) printf '%s' "$FLEET_BUILD_PRIO"; return 0 ;;
    esac
    if [ "${GITHUB_EVENT_NAME:-}" = merge_group ]; then printf 0; else printf 1; fi
}

bs_default_label() {
    local who
    if [ -n "${FAT_SECTION_LOG:-}" ]; then
        who=$(basename "$FAT_SECTION_LOG" .log)
    elif [ -n "${GITHUB_JOB:-}" ]; then
        who=$GITHUB_JOB
    else
        who="agent:${USER:-unknown}@$$"
    fi
    printf '%s/%s' "${GITHUB_RUN_ID:-local}" "$who"
}

# 0 when pid $1 is $$ or one of its ancestors.
bs_is_ancestor() {
    local want="$1" p="$$" k v
    case "$want" in '' | *[!0-9]*) return 1 ;; esac
    while [ -n "$p" ] && [ "$p" != 0 ]; do
        [ "$p" = "$want" ] && return 0
        v=""
        while read -r k v; do
            [ "$k" = "PPid:" ] && break
            v=""
        done < "/proc/$p/status" 2> /dev/null
        p=$v
    done
    return 1
}

# 0 when slot $1 under $2 is locked by someone (probe takes and drops it at once).
bs_locked() {
    local fd rc=0
    exec {fd}<> "$2/slot.$1" || return 1
    if flock -n "$fd"; then rc=1; fi
    exec {fd}>&-
    return "$rc"
}

# 0 when this process already runs under a held slot (nested call).
bs_nested() {
    local dir="$1"
    [ -n "${FLEET_BUILD_SLOT:-}" ] && [ -n "${FLEET_BUILD_SLOT_PID:-}" ] || return 1
    bs_is_ancestor "$FLEET_BUILD_SLOT_PID" || return 1
    # the slot owner must be that same pid: an ancestor that holds nothing
    # (a login shell, pid 1) admits nothing.
    [ "$(cut -d" " -f1 "$dir/slot.$FLEET_BUILD_SLOT.owner" 2> /dev/null)" = "$FLEET_BUILD_SLOT_PID" ] || return 1
    bs_locked "$FLEET_BUILD_SLOT" "$dir"
}

# Blocking acquire. Sets BS_FD, BS_SLOT, BS_WAIT_S, BS_T0 in the caller.
bs_acquire() {
    local dir="$1" label="$2" tier n t0 now last pfd="" q fd i try held poll owner
    n=$(bs_n)
    tier=$(bs_tier)
    poll="${FLEET_BUILD_SLOT_POLL_S:-}"
    if [ -z "$poll" ]; then
        if [ "$tier" = 0 ]; then poll=1; else poll=3; fi
    fi
    t0=$(date +%s)
    last=$t0
    if [ "$tier" = 0 ]; then
        exec {pfd}<> "$dir/prio" || return 1
        flock -s "$pfd" || return 1
    fi
    while :; do
        try=1
        held=""
        if [ "$tier" != 0 ]; then
            exec {q}<> "$dir/prio" || return 1
            flock -n -x "$q" || try=0
            exec {q}>&-
        fi
        if [ "$try" = 1 ]; then
            i=0
            while [ "$i" -lt "$n" ]; do
                exec {fd}<> "$dir/slot.$i" || return 1
                if flock -n "$fd"; then
                    now=$(date +%s)
                    printf '%s %s %s %s\n' "$BASHPID" "$label" "$tier" "$now" > "$dir/slot.$i.owner"
                    [ -z "$pfd" ] || exec {pfd}>&-
                    BS_FD=$fd BS_SLOT=$i BS_T0=$now BS_WAIT_S=$((now - t0))
                    return 0
                fi
                exec {fd}>&-
                owner=$(cut -d' ' -f2 "$dir/slot.$i.owner" 2> /dev/null)
                held="$held $i=${owner:-?}"
                i=$((i + 1))
            done
        else
            held=" (a queue run is waiting first)"
        fi
        now=$(date +%s)
        if [ $((now - last)) -ge 60 ]; then
            printf 'build-slot: waiting %ss tier=%s N=%s; held by:%s\n' "$((now - t0))" "$tier" "$n" "$held" >&2
            last=$now
        fi
        sleep "$poll"
    done
}

bs_run() {
    local label="" dir rc t1 ledger
    while [ $# -gt 0 ]; do
        case "$1" in
            --label) label="${2:-}"; shift 2 ;;
            --) shift; break ;;
            *) printf 'build_slot: unknown argument %s\n' "$1" >&2; return 2 ;;
        esac
    done
    [ $# -gt 0 ] || { printf 'build_slot: run needs -- cmd\n' >&2; return 2; }
    [ -n "$label" ] || label=$(bs_default_label)
    label=$(printf '%s' "$label" | tr -c 'A-Za-z0-9._:/@=+-' '_')
    dir=$(bs_dir)
    if [ ! -d "$dir" ]; then
        if [ "${FLEET_BUILD_SLOT_REQUIRED:-0}" = 1 ]; then
            printf '::error::build-slot: FLEET_BUILD_SLOT_REQUIRED=1 and %s has no pool (%s)\n' "${RUNNER_NAME:-$(hostname)}" "$dir"
            return 1
        fi
        printf '::notice::build-slot: no pool on %s; unadmitted\n' "${RUNNER_NAME:-$(hostname)}"
        "$@"
        return
    fi
    if bs_nested "$dir"; then
        "$@"
        return
    fi
    bs_acquire "$dir" "$label" || { printf '::error::build-slot: cannot open the pool at %s\n' "$dir"; return 1; }
    [ "$BS_WAIT_S" -eq 0 ] || printf 'build-slot: admitted to slot %s after %ss (%s)\n' "$BS_SLOT" "$BS_WAIT_S" "$label" >&2
    FLEET_BUILD_SLOT=$BS_SLOT FLEET_BUILD_SLOT_PID=$BASHPID FLEET_BUILD_SLOT_T0=$BS_T0 \
        FLEET_BUILD_SLOT_WAIT_S=$BS_WAIT_S FLEET_BUILD_SLOT_N=$(bs_n) FLEET_BUILD_SLOT_LABEL=$label \
        "$@" {BS_FD}>&-
    rc=$?
    t1=$(date +%s)
    ledger="${FLEET_BUILD_LEDGER:-/var/lib/fleet-build/ledger.tsv}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$BS_T0" "$t1" "$BS_WAIT_S" "$((t1 - BS_T0))" \
        "$(bs_tier)" "$label" "$rc" "$(hostname)" "$(bs_n)" >> "$ledger" 2> /dev/null ||
        printf 'build-slot: ledger %s not writable; row dropped\n' "$ledger" >&2
    exec {BS_FD}>&-
    return "$rc"
}

bs_status() {
    local dir n i owner
    dir=$(bs_dir)
    [ -d "$dir" ] || { printf 'build-slot: no pool (%s)\n' "$dir"; return 1; }
    n=$(bs_n)
    printf 'N=%s dir=%s\n' "$n" "$dir"
    i=0
    while [ "$i" -lt "$n" ]; do
        if bs_locked "$i" "$dir"; then
            owner=$(cat "$dir/slot.$i.owner" 2> /dev/null)
            printf 'slot %s held: %s\n' "$i" "${owner:-?}"
        fi
        i=$((i + 1))
    done
}

bs_self_test() {
    local tmp fails=0 rows=0 me rc out t p1 p2 p3
    me="${BASH_SOURCE[0]}"
    tmp=$(mktemp -d) || return 1
    export FLEET_BUILD_SLOT_DIR="$tmp/v1" FLEET_BUILD_SLOTS_FILE="$tmp/slots" FLEET_BUILD_LEDGER="$tmp/ledger.tsv" FLEET_BUILD_SLOT_POLL_S=0.2
    unset FLEET_BUILD_SLOT FLEET_BUILD_SLOT_PID FLEET_BUILD_PRIO FLEET_BUILD_SLOT_REQUIRED GITHUB_EVENT_NAME
    mkdir -p "$tmp/v1"
    row() { # row <name> <got> <want>
        rows=$((rows + 1))
        if [ "$2" = "$3" ]; then printf '  ok    %s\n' "$1"; else printf '  FAIL  %s: got [%s] want [%s]\n' "$1" "$2" "$3"; fails=$((fails + 1)); fi
    }

    # (a) N+1-th waits; admitted after a release; wait recorded.
    echo 2 > "$tmp/slots"
    setsid bash "$me" run --label h1 -- sleep 2 & p1=$!
    setsid bash "$me" run --label h2 -- sleep 2 & p2=$!
    sleep 0.5
    out=$(bash "$me" status | grep -c ' held: ')
    row "a: two holders on N=2" "$out" 2
    out=$(bash "$me" run --label w -- sh -c 'echo "$FLEET_BUILD_SLOT_WAIT_S"' 2> /dev/null)
    row "a: third run waited for a release" "$([ "${out:-0}" -ge 1 ] && echo waited)" waited
    wait "$p1" "$p2"

    # (b) a killed holder frees its slot; its stale owner line is not a holder.
    echo 1 > "$tmp/slots"
    setsid bash "$me" run --label dead -- sleep 30 & p1=$!
    sleep 0.5
    kill -KILL -- "-$p1" 2> /dev/null; wait "$p1" 2> /dev/null
    row "b: stale owner file kept" "$(cut -d' ' -f2 "$tmp/v1/slot.0.owner")" dead
    row "b: status lists no holder" "$(bash "$me" status | grep -c ' held: ')" 0
    out=$(timeout 5 bash "$me" run -- sh -c 'echo "$FLEET_BUILD_SLOT_WAIT_S"' 2> /dev/null)
    row "b: next run admitted at once" "$out" 0

    # (c) tier 0 arriving after a tier 1 waiter is admitted first.
    : > "$tmp/order"
    setsid bash "$me" run --label h -- sleep 1.5 & p1=$!
    sleep 0.3
    FLEET_BUILD_PRIO=1 bash "$me" run --label pr -- sh -c "echo pr >> '$tmp/order'" & p2=$!
    sleep 0.5
    GITHUB_EVENT_NAME=merge_group bash "$me" run --label q -- sh -c "echo queue >> '$tmp/order'" & p3=$!
    wait "$p1" "$p2" "$p3"
    row "c: queue admitted before the earlier PR waiter" "$(head -1 "$tmp/order")" queue

    # (d) nested run under a held slot passes through; an env var alone does not.
    out=$(timeout 5 bash "$me" run --label outer -- bash "$me" run --label inner -- echo nested 2> /dev/null)
    row "d: nested run passes through on N=1" "$out" nested
    setsid bash "$me" run --label other -- sleep 4 & p1=$!
    sleep 0.3
    FLEET_BUILD_SLOT=0 FLEET_BUILD_SLOT_PID=$$ timeout 1 bash "$me" run -- echo leaked > /dev/null 2>&1; rc=$?
    row "d: env naming an ancestor that is not the holder waits" "$rc" 124
    FLEET_BUILD_SLOT=0 FLEET_BUILD_SLOT_PID=1 timeout 1 bash "$me" run -- echo leaked > /dev/null 2>&1; rc=$?
    row "d: env naming pid 1 waits" "$rc" 124
    wait "$p1"

    # (d2) the command runs with the slot fd closed: a daemon cannot keep the slot.
    setsid bash "$me" run --label d -- sh -c 'setsid sleep 3 < /dev/null > /dev/null 2>&1 &'
    sleep 0.3
    row "d2: a daemon left behind holds no slot" "$(bash "$me" status | grep -c ' held: ')" 0

    # (e) no pool.
    out=$(FLEET_BUILD_SLOT_DIR="$tmp/none" FLEET_BUILD_SLOT_REQUIRED=1 bash "$me" run -- echo ran); rc=$?
    row "e: REQUIRED=1 and no pool is red" "$rc/$(printf '%s' "$out" | grep -c '^ran$')" 1/0
    out=$(FLEET_BUILD_SLOT_DIR="$tmp/none" bash "$me" run -- echo ran); rc=$?
    row "e: no pool, not required, runs with a notice" "$rc/$(printf '%s' "$out" | grep -c '^::notice::build-slot: no pool')/$(printf '%s' "$out" | grep -c '^ran$')" 0/1/1

    # (f) rc propagates.
    for t in 0 1 124; do
        bash "$me" run -- sh -c "exit $t"; rc=$?
        row "f: rc $t propagates" "$rc" "$t"
    done
    bash "$me" run -- sh -c 'kill -KILL $$'; rc=$?
    row "f: a killed cmd reads 137" "$rc" 137

    # (g) label sanitised; ledger row written with the cmd rc.
    bash "$me" run --label 'a b;c' -- sh -c 'exit 3'
    row "g: label has no spaces, ledger keeps the rc" "$(tail -1 "$tmp/ledger.tsv" | cut -f6,7 | tr '\t' ' ')" "a_b_c 3"

    # (h) N from the file; absent or junk falls back to nproc/8, at least 1.
    echo 4 > "$tmp/slots"; row "h: N from file" "$(bs_n)" 4
    echo x > "$tmp/slots"; t=$(( $(nproc) / 8 )); [ "$t" -ge 1 ] || t=1
    row "h: junk N falls back to nproc/8" "$(bs_n)" "$t"

    rm -rf "${tmp:?}"
    printf 'build_slot self-test: %s rows, %s failed\n' "$rows" "$fails"
    if [ "$rows" -lt 17 ]; then printf 'build_slot self-test: only %s rows ran (want 19): NOT MEASURED\n' "$rows"; return 1; fi
    [ "$fails" = 0 ]
}

bs_main() {
    case "${1:-}" in
        run) shift; bs_run "$@" ;;
        status) bs_status ;;
        --self-test) bs_self_test ;;
        *) printf 'usage: build_slot.sh run [--label L] -- cmd... | status | --self-test\n' >&2; return 2 ;;
    esac
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    bs_main "$@"
    exit $?
fi
