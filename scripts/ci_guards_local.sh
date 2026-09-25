#!/usr/bin/env bash
# ci_guards_local.sh -- run CI's guard jobs locally, ALL of their steps, before
# you push (#4416, lever 2 of the RC-speed report).
#
# On RC PR #4318, 16 of 23 failing CI jobs were guard reds, and each one cost
# a ~35-50 min CI cycle to discover. This runs the SAME `run:` blocks that
# ci.yml's guard jobs run -- read from ci.yml by scripts/lib/ci_guard_steps.py,
# never copied (F1, #2640: two hand-kept runners drifted 86 lines) -- and it
# does not stop at the first red: every step runs and one table reports all.
#
# Steps that cannot run off-CI print a SKIP row with the reason (an `if:` on
# the CI event, a docker image this host lacks). A run in which NOTHING could
# execute exits 2, never 0.
#
#   bash scripts/ci_guards_local.sh                    # guard-cargo + guard-tree
#   bash scripts/ci_guards_local.sh guard-tree         # one job
#   bash scripts/ci_guards_local.sh --only 'bashrs|pinned' guard-cargo
#   bash scripts/ci_guards_local.sh --list             # steps, and which are fail-fast in CI
#   bash scripts/ci_guards_local.sh --check-coverage   # drift: guard scripts CI runs elsewhere
#   bash scripts/ci_guards_local.sh --step-timeout 300 # per step, seconds (default 900; TIMEOUT = red)
#
# The jobs are the `guard-*` jobs the `gate` job needs, read from ci.yml. Rows
# stream as each step finishes. `make guards-local` runs this; the pre-push
# hook runs it when APR_PREPUSH_GUARDS=1.
#
# Exit: 0 all ran steps passed, 1 a step failed, 2 usage / nothing ran.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB="${REPO_ROOT}/scripts/lib/ci_guard_steps.py"
command -v python3 > /dev/null 2>&1 || { echo "ci_guards_local: python3 missing" >&2; exit 2; }
cd "$REPO_ROOT" || exit 2

case "${1:-}" in
    --help|-h) sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --list) shift; exec python3 "$LIB" list "$@" ;;
    --check-coverage) shift; exec python3 "$LIB" check-coverage "$@" ;;
    --only) [ $# -ge 2 ] || { echo "--only needs a regex" >&2; exit 2; }
            re=$2; shift 2; exec python3 "$LIB" run --only "$re" "$@" ;;
esac
exec python3 "$LIB" run "$@"
