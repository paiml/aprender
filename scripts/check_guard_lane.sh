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
        out="$(cd "$d" && env -u CI_GUARDS_LANE GITHUB_ACTIONS=true CI_GUARDS_SCRATCH="$d/scratch" bash "$LIB" run fx 2>&1)"
    else
        out="$(cd "$d" && CI_GUARDS_LANE="$lane" CI_GUARDS_SCRATCH="$d/scratch" bash "$LIB" run fx 2>&1)"
    fi
    rc=$?
    n=$((n + 1))
    local rc_ok=0
    case "$want_rc" in
        zero) [ "$rc" -eq 0 ] && rc_ok=1 ;;
        nonzero) [ "$rc" -ne 0 ] && rc_ok=1 ;;
    esac
    if [ "$rc_ok" = 1 ] && printf '%s\n' "$out" | grep -Eq -- "$want"; then
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

if [ "$fail" -ne 0 ]; then
    echo "check_guard_lane: FAILED $fail of $n cases"
    exit 1
fi
echo "check_guard_lane: PASS $n cases (lane on/off, CI default-off, cargo step, GITHUB_ENV write, missing list, bad row)"
