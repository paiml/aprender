#!/usr/bin/env bash
# check_guard_lane.sh -- the cargo-free lane of ci_guards.sh keeps its three rules (#5920).
#
# scripts/lib/ci_guard_steps.sh runs a step listed in ci/guard-lane-steps.txt beside the
# serial steps. That only stays "the same guards ran" if a misfiled step is RED, never
# silently green. Each case builds a one-job fixture repo (ci/sections.yml + a lane list),
# runs the real runner on it and reads the row it printed:
#   lane-ok       a listed step with no cargo and no state handoff        -> PASS, 1 parallel
#   lane-off      the same list with CI_GUARDS_LANE=0                      -> PASS, 0 parallel
#   ci-default-off GITHUB_ACTIONS=true, CI_GUARDS_LANE unset               -> lane OFF (default-off on PR/MQ)
#   cargo-step    a listed step that calls cargo (the shim, rc 97)         -> FAIL
#   env-write     a listed step that writes GITHUB_ENV (rc 96)             -> FAIL
#   missing-list  CI_GUARDS_LANE=1 with no list file                       -> runner refuses, nonzero
#   bad-row       a list row with no step name                             -> runner refuses, nonzero
#
#   bash scripts/check_guard_lane.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB="${REPO_ROOT}/scripts/lib/ci_guard_steps.sh"
[ -r "$LIB" ] || { echo "check_guard_lane: $LIB unreadable" >&2; exit 2; }
work="$(mktemp -d "${TMPDIR:-/tmp}/guard-lane-XXXXXX")" || exit 2
trap 'rm -rf "${work:?}"' EXIT

fail=0 n=0
LIB_SHELL=(bash)
# mk_repo <dir> <run line of the step under test> -- a fixture repo with one guard job
mk_repo() {
    mkdir -p "$1/ci"
    {
        printf 'jobs:\n  fx:\n    runs-on: [self-hosted]\n    steps:\n      - name: run\n        run: bash scripts/ci_guards.sh fx\n'
        printf '  fx-steps:\n    if: false\n    runs-on: [self-hosted]\n    steps:\n      - name: lane step\n        run: %s\n' "$2"
    } > "$1/ci/sections.yml"
    git -C "$1" init -q
    git -C "$1" add -A
    git -C "$1" -c user.email=lane@example.invalid -c user.name=lane commit -q -m fixture > /dev/null 2>&1
}
# case_run <name> <run line> <lane value> <list content or NONE> <want rc class> <want pattern>
case_run() {
    local name=$1 run=$2 lane=$3 list=$4 want_rc=$5 want=$6 d out rc
    d="$work/$name"
    mk_repo "$d" "$run"
    if [ "$list" != NONE ]; then printf '%b' "$list" > "$d/ci/guard-lane-steps.txt"; fi
    if [ "$lane" = CI-UNSET ]; then # a CI run that sets nothing: the lane must be off (C324)
        out="$(cd "$d" && env -u CI_GUARDS_LANE GITHUB_ACTIONS=true CI_GUARDS_SCRATCH="$d/scratch" "${LIB_SHELL[@]}" "$LIB" run fx 2>&1)"
    else
        out="$(cd "$d" && CI_GUARDS_LANE="$lane" CI_GUARDS_SCRATCH="$d/scratch" "${LIB_SHELL[@]}" "$LIB" run fx 2>&1)"
    fi
    rc=$?
    n=$((n + 1))
    local rc_ok=0
    case "$want_rc" in
        zero) [ "$rc" -eq 0 ] && rc_ok=1 ;;
        nonzero) [ "$rc" -ne 0 ] && rc_ok=1 ;;
    esac
    if [ "$rc_ok" = 1 ] && grep -Eq -- "$want" <<< "$out"; then
        echo "ok   $name (rc=$rc)"
    else
        fail=$((fail + 1))
        echo "FAIL $name: want rc $want_rc and /$want/, got rc=$rc"
        printf '%s\n' "$out" | tail -n 8 | sed 's/^/     | /'
    fi
}

LIST='fx\tlane step\n'
case_run lane-ok 'echo hello' 1 "$LIST" zero '^PASS +fx#.*lane step'
case_run lane-off 'echo hello' 0 "$LIST" zero 'lane off: every step serial'
case_run ci-default-off 'echo hello' CI-UNSET "$LIST" zero 'lane off: every step serial'
case_run cargo-step 'cargo --version' 1 "$LIST" nonzero 'lane step  \[rc=97'
case_run env-write 'echo X=1 >> "$GITHUB_ENV"' 1 "$LIST" nonzero 'lane step  \[rc=96'
case_run missing-list 'echo hello' 1 NONE nonzero 'lane list .* is unreadable'
case_run bad-row 'echo hello' 1 'fx\n' nonzero 'a row needs job<TAB>step name'
case_run cargo-abs '$HOME/.cargo/bin/cargo --version' 1 "$LIST" nonzero 'lane step  \[rc=97'
case_run cargo-path-env 'PATH=$HOME/.cargo/bin:$PATH cargo --version' 1 "$LIST" nonzero 'lane step  \[rc=97'
case_run cargo-usrbin 'sh -c "/usr/bin/cargo"' 1 "$LIST" nonzero 'lane step  \[rc=97'
case_run cargo-usrlocal '/usr/local/cargo/bin/cargo --version' 1 "$LIST" nonzero 'lane step  \[rc=97'
case_run cargo-reprepend 'PATH=/x:$PATH cargo --version' 1 "$LIST" nonzero 'lane step  \[rc=97'
LIB_SHELL=(bash -e) # a caller under set -e: the step's real rc must still reach the row, not rc 99 "vanished"
case_run errexit-caller 'exit 3' 1 "$LIST" nonzero 'lane step  \[rc=3'
case_run errexit-cargo 'cargo --version' 1 "$LIST" nonzero 'lane step  \[rc=97'
LIB_SHELL=(bash)

# throttle: with width 1 two steps never overlap; with width 2 they do (S = step start, E = step end, in one shared file)
conc_case() {
    local w=$1 want=$2 d="$work/conc-$1" seq
    mkdir -p "$d/ci"
    {
        printf 'jobs:\n  fx:\n    runs-on: [self-hosted]\n    steps:\n      - name: run\n        run: bash scripts/ci_guards.sh fx\n'
        printf '  fx-steps:\n    if: false\n    runs-on: [self-hosted]\n    steps:\n'
        for i in 1 2; do printf '      - name: s%s\n        run: echo S >> "$CONC"; sleep 1; echo E >> "$CONC"\n' "$i"; done
    } > "$d/ci/sections.yml"
    printf 'fx\ts1\nfx\ts2\n' > "$d/ci/guard-lane-steps.txt"
    git -C "$d" init -q; git -C "$d" add -A
    git -C "$d" -c user.email=lane@example.invalid -c user.name=lane commit -q -m fixture > /dev/null 2>&1
    : > "$d/conc"
    (cd "$d" && CONC="$d/conc" CI_GUARDS_LANE=1 CI_GUARDS_LANE_WIDTH="$w" CI_GUARDS_SCRATCH="$d/scratch" bash "$LIB" run fx > /dev/null 2>&1)
    seq=$(tr -d '\n' < "$d/conc")
    n=$((n + 1))
    if [ "$seq" = "$want" ]; then echo "ok   throttle-width-$w ($seq)"
    else fail=$((fail + 1)); echo "FAIL throttle-width-$w: want $want, got '$seq'"; fi
}
conc_case 1 SESE
conc_case 2 SSEE

# every row of the real list names a step that exists in ci/sections.yml (a stale row would silently run nothing)
n=$((n + 1))
stale=""
while IFS=$'\t' read -r rj rs; do
    case "$rj" in ''|'#'*) continue ;; esac
    # the manifest quotes a name that holds a colon or a quote, so a row matches either spelling
    grep -qxF -e "      - name: $rs" -e "      - name: \"$rs\"" "$REPO_ROOT/ci/sections.yml" || stale="$stale [$rj: $rs]"
done < "$REPO_ROOT/ci/guard-lane-steps.txt"
if [ -z "$stale" ]; then echo "ok   list-rows-exist"; else fail=$((fail + 1)); echo "FAIL list-rows-exist: not in ci/sections.yml:$stale"; fi

# stop-step-error: the stop step dying with no output must SKIP the lane (C324 item 5: unmeasured is not a pass).
# Two halves: the script writes no decision on an env error, and the workflow runs the lane only on decision == run.
n=$((n + 1))
stop_out="$work/stop-out"; : > "$stop_out"
env -u GH_TOKEN GITHUB_REPOSITORY=o/r GITHUB_OUTPUT="$stop_out" bash "$REPO_ROOT/scripts/guard_lane_stop.sh" --run > /dev/null 2>&1
stop_rc=$?
wf="$REPO_ROOT/.github/workflows/guards-nightly.yml"
if [ "$stop_rc" -ne 0 ] && [ ! -s "$stop_out" ] && grep -q "steps.lab_guard_lane_stop.outputs.decision == 'run'" "$wf" &&
    ! grep -q "steps.lab_guard_lane_stop.outputs.decision != " "$wf"; then
    echo "ok   stop-step-error (rc=$stop_rc, no decision written, lane runs only on decision == run)"
else
    fail=$((fail + 1))
    echo "FAIL stop-step-error: rc=$stop_rc, output '$(cat "$stop_out")', or the workflow lets the lane run without decision == run"
fi

if [ "$fail" -ne 0 ]; then
    echo "check_guard_lane: FAILED $fail of $n cases"
    exit 1
fi
echo "check_guard_lane: PASS $n cases (lane on/off, CI default-off, stop-step error, cargo step and cargo path (abs, quoted, re-prepend), set -e caller, throttle, GITHUB_ENV write, missing list, bad row, list rows exist)"
