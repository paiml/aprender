#!/usr/bin/env bash
# check_guard_steps_run_all.sh -- a guard step may not hide behind another
# guard's failure (#4415, lever 0 of the RC-speed report).
#
# WHY THIS EXISTS
# ---------------
# GitHub Actions skips every later step of a job once one step fails, unless
# the later step's `if:` names a status function (`!cancelled()`, `always()`,
# `failure()`). On the 0.69.4 RC PR #4318 every red guard job reported
# exactly ONE failure. guard-cargo skipped 15-71 of its 77 steps (median 55),
# so a fix-push-wait loop of ~35-50 min per cycle surfaced the guards one at
# a time: 02d46180e -> cef1b1241 -> 696576edb -> 6281d9fbe.
#
# The fail-fast count per job is a SHRINK-ONLY ratchet
# (scripts/guard_fail_fast_baseline.txt). A new guard step must be run-on-red,
# and the workflow half of #4415 takes both rows to 0.
#
# The step list is read from ci.yml by scripts/lib/ci_guard_steps.py, the same
# reader scripts/ci_guards_local.sh (#4416) uses to run those steps locally.
# There is no second list.
#
#   bash scripts/check_guard_steps_run_all.sh              # check
#   bash scripts/check_guard_steps_run_all.sh --self-test  # case table
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB="${REPO_ROOT}/scripts/lib/ci_guard_steps.py"

usage() {
    sed -n '2,23p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

fixture() { # fixture <file> <kind>... ; kinds: plain cancelled always event event_cancelled setup pass fail
    local f=$1 k
    shift
    {
        printf 'on: push\njobs:\n  guard-x:\n    runs-on: ubuntu-latest\n    steps:\n'
        printf '      - uses: actions/checkout@v7\n'
        printf '      - name: fetch\n        run: git fetch origin main\n'
        for k in "$@"; do
            printf '      - name: %s\n' "$k"
            case "$k" in
                cancelled) printf '        if: ${{ !cancelled() && steps.guard-setup.outcome == %s }}\n' "'success'" ;;
                always) printf '        if: always()\n' ;;
                event) printf '        if: github.event_name == %s\n' "'pull_request'" ;;
                event_cancelled) printf '        if: "!cancelled() && github.event_name == %s"\n' "'pull_request'" ;;
                setup) printf '        id: guard-setup\n' ;;
            esac
            case "$k" in
                fail) printf '        run: "exit 3"\n' ;;
                *) printf '        run: "true"\n' ;;
            esac
        done
    } > "$f"
}

self_test() {
    local d fail=0 n=0
    d=$(mktemp -d) || return 2
    # shellcheck disable=SC2064
    trap "rm -rf -- '${d:?}'" RETURN
    printf 'guard-x 2\n' > "$d/base2"
    printf 'guard-x 0\n' > "$d/base0"
    printf 'other 5\n' > "$d/base_missing"

    case_row() { # case_row <label> <want_rc> <workflow> <baseline> [job]
        local got
        python3 "$LIB" --workflow "$3" check-run-all --baseline "$4" "${5:-guard-x}" > "$d/out" 2>&1
        got=$?
        n=$((n + 1))
        if [ "$got" -eq "$2" ]; then
            printf 'ok   %-58s rc=%s\n' "$1" "$got"
        else
            printf 'FAIL %-58s rc=%s want %s\n' "$1" "$got" "$2"
            sed 's/^/     | /' "$d/out"
            fail=1
        fi
    }

    fixture "$d/w1.yml" plain plain
    case_row "2 fail-fast steps, baseline 2 -> pass" 0 "$d/w1.yml" "$d/base2"
    case_row "2 fail-fast steps, baseline 0 -> RED" 1 "$d/w1.yml" "$d/base0"
    fixture "$d/w2.yml" plain plain plain
    case_row "3rd fail-fast step over baseline 2 -> RED" 1 "$d/w2.yml" "$d/base2"
    fixture "$d/w3.yml" cancelled always event_cancelled
    case_row "!cancelled()/always()/!cancelled()&&event: 0 fail-fast" 0 "$d/w3.yml" "$d/base0"
    fixture "$d/w4.yml" event
    case_row "event-only if: is still fail-fast -> RED at baseline 0" 1 "$d/w4.yml" "$d/base0"
    fixture "$d/w5.yml" setup plain plain
    case_row "id: guard-setup is setup, the 2 after it count" 0 "$d/w5.yml" "$d/base2"
    case_row "missing job fails closed (rc 2)" 2 "$d/w1.yml" "$d/base2" no-such-job
    case_row "job with no baseline row -> RED" 1 "$d/w1.yml" "$d/base_missing"
    case_row "unreadable workflow fails closed (rc 2)" 2 "$d/absent.yml" "$d/base2"

    # The runner half (#4416) must RUN PAST a failure and report every one.
    local repo="$d/repo"
    mkdir -p "$repo" && git -C "$repo" init -q && mkdir -p "$repo/.github/workflows"
    fixture "$repo/.github/workflows/ci.yml" pass fail pass fail
    local got
    ( cd "$repo" && CI_GUARDS_SCRATCH="$d/scratch" python3 "$LIB" run guard-x ) > "$d/out" 2>/dev/null
    got=$?
    n=$((n + 1))
    if [ "$got" -eq 1 ] && grep -q '^SUMMARY: 2 failed / 4 ran / 0 skipped$' "$d/out"; then
        printf 'ok   %-58s rc=%s\n' "runner reports BOTH failures of 4 steps in one pass" "$got"
    else
        printf 'FAIL %-58s rc=%s\n' "runner reports BOTH failures of 4 steps in one pass" "$got"
        sed 's/^/     | /' "$d/out"
        fail=1
    fi
    fixture "$repo/.github/workflows/ci.yml" event
    ( cd "$repo" && CI_GUARDS_SCRATCH="$d/scratch" python3 "$LIB" run guard-x ) > "$d/out" 2>/dev/null
    got=$?
    n=$((n + 1))
    if [ "$got" -eq 2 ]; then
        printf 'ok   %-58s rc=%s\n' "runner with nothing runnable refuses a vacuous pass" "$got"
    else
        printf 'FAIL %-58s rc=%s want 2\n' "runner with nothing runnable refuses a vacuous pass" "$got"
        fail=1
    fi

    if [ "$fail" -ne 0 ]; then
        echo "check_guard_steps_run_all self-test: FAILED"
        return 1
    fi
    echo "check_guard_steps_run_all self-test: ${n}/${n} cases pass"
}

case "${1:-}" in
    --help|-h) usage; exit 0 ;;
    --self-test) self_test; exit $? ;;
    "") ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
esac

command -v python3 > /dev/null 2>&1 || { echo "check_guard_steps_run_all: python3 missing" >&2; exit 2; }
cd "$REPO_ROOT" || exit 2
python3 "$LIB" check-run-all
