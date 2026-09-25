#!/usr/bin/env bash
# ci_guards_local_test.sh -- the case table for #4416's local guard runner
# (scripts/ci_guards_local.sh over scripts/lib/ci_guard_steps.py).
#
# Every row drives the SHIPPED library against a fixture workflow in a scratch
# git repo, so what is pinned is behaviour, not the YAML of the day:
#   - a step added to a guard job, and a whole new guard-* job in gate.needs,
#     run locally with no second edit (the issue's first falsifier);
#   - a planted red reports FAIL under the step name CI would show, and the
#     steps after it still run (CI stops there; the point is one list of all);
#   - a hung step is a TIMEOUT, which is red, and its children die with it;
#   - rows stream: a run killed half way has already printed its first rows;
#   - nothing runnable exits 2, never a vacuous 0;
#   - the drift check is red on a guard script run by a non-guard job with no
#     acknowledgement row, and on a stale row (the issue's second falsifier);
#   - the real ci.yml passes the drift check.
#
#   bash scripts/tests/ci_guards_local_test.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIB="${REPO_ROOT}/scripts/lib/ci_guard_steps.py"
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/ci-guards-local-test.XXXXXX")" || exit 2
trap 'rm -rf -- "${SCRATCH:?}"' EXIT
git -C "$SCRATCH" init -q || exit 2
export CI_GUARDS_SCRATCH="$SCRATCH/out"
mkdir -p "$CI_GUARDS_SCRATCH"

n=0
red=0
row() { # row <want-rc> <label> <grep -E pattern> <cmd...>
    local want=$1 label=$2 pat=$3 rc=0 out
    shift 3
    n=$((n + 1))
    out=$(cd "$SCRATCH" && "$@" 2>&1) || rc=$?
    if [ "$rc" = "$want" ] && grep -qE -- "$pat" <<< "$out"; then
        printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
    else
        printf 'FAIL  row %-2s rc=%s (wanted %s, must match /%s/)  %s\n' \
            "$n" "$rc" "$want" "$pat" "$label"
        printf '%s\n' "$out" | sed 's/^/        /'
        red=1
    fi
}
lib() { python3 "$LIB" --workflow "$SCRATCH/wf.yml" "$@"; }

# Base fixture: two guard jobs in gate.needs, one non-guard job, and a guard-*
# job the gate does NOT need (it must not run: CI would not block on it).
wf() {
    cat > "$SCRATCH/wf.yml" <<YAML
jobs:
  guard-tree:
    runs-on: x
    steps:
      - uses: actions/checkout@v4
      - name: tree ok
        run: echo tree-ok
$1
  guard-cargo:
    runs-on: x
    steps:
      - uses: actions/checkout@v4
      - name: cargo ok
        run: echo cargo-ok
$2
  guard-orphan:
    runs-on: x
    steps:
      - name: orphan step
        run: echo orphan
$3
  gate:
    needs: [guard-tree, guard-cargo$4]
    steps:
      - run: echo gate
  other:
    steps:
      - run: bash scripts/check_other_thing.sh$5
YAML
}

# 1. Base: both gate-needed guard jobs run, the orphan does not.
wf "" "" "" "" ""
row 0 "gate-needed guard jobs run; a guard-* job outside gate.needs does not" \
    '^guard-cargo +[0-9]+ PASS .*cargo ok' lib run
row 0 "no orphan row" '^SUMMARY: 0 failed / 2 ran / 0 skipped$' lib run

# 2. A new step in a guard job runs with no second edit.
wf "      - name: added later
        run: echo added" "" "" "" ""
row 0 "a step added to guard-tree runs locally, no second edit" \
    '^guard-tree +[0-9]+ PASS .*added later' lib run

# 3. A new guard job the gate needs runs with no second edit.
wf "" "" "" ", guard-orphan" ""
row 0 "a guard-* job added to gate.needs runs locally, no second edit" \
    '^guard-orphan +[0-9]+ PASS .*orphan step' lib run

# 4. Planted red: FAIL under CI's step name, and the later steps still run.
wf "      - name: planted violation
        run: exit 3
      - name: after the red
        run: echo still-runs" "" "" "" ""
row 1 "a planted red is FAIL under its CI step name" \
    '^guard-tree +[0-9]+ FAIL .*planted violation +\[rc=3' lib run
row 1 "the steps after a red still run" '^guard-tree +[0-9]+ PASS .*after the red' lib run
row 1 "and the other job still runs" '^SUMMARY: 1 failed / 4 ran' lib run

# 5. A hung step is a TIMEOUT (red), and its child is killed with it.
wf "      - name: hangs
        run: sleep 30 & echo \$! > \"\$CI_GUARDS_SCRATCH/child.pid\"; wait" "" "" "" ""
row 1 "a hung step is TIMEOUT, which is red" \
    '^guard-tree +[0-9]+ TIMEOUT .*hangs +\[rc=124' lib run --step-timeout 2
n=$((n + 1))
child=$(cat "$CI_GUARDS_SCRATCH/child.pid" 2>/dev/null)
if [ -n "$child" ] && ! kill -0 "$child" 2>/dev/null; then
    printf 'ok    row %-2s       the timed-out step'"'"'s child (pid %s) is dead\n' "$n" "$child"
else
    printf 'FAIL  row %-2s       the timed-out step'"'"'s child (pid %s) survived\n' "$n" "${child:-?}"
    red=1
fi

# 6. Rows stream: kill the run mid-way; the first row is already on stdout.
wf "      - name: slow
        run: sleep 30" "" "" "" ""
n=$((n + 1))
out=$(cd "$SCRATCH" && timeout 3 python3 "$LIB" --workflow wf.yml run guard-tree 2>/dev/null)
if grep -qE '^guard-tree +[0-9]+ PASS .*tree ok' <<< "$out"; then
    printf 'ok    row %-2s       a run killed mid-way has already printed its finished rows\n' "$n"
else
    printf 'FAIL  row %-2s       a killed run printed no rows (the table is not streamed)\n' "$n"
    printf '%s\n' "$out" | sed 's/^/        /'
    red=1
fi

# 7. Nothing runnable: exit 2.
wf "" "" "" "" ""
row 2 "--only matching nothing is exit 2, never a vacuous pass" \
    'nothing ran' lib run --only 'no-such-step'

# 8. Drift check.
printf 'other check_other_thing.sh  # fixture reason\n' > "$SCRATCH/ack.txt"
row 0 "drift: an acknowledged non-guard guard script passes" \
    '^ok +2 guard jobs' lib check-coverage --ack "$SCRATCH/ack.txt"
wf "" "" "" "" "
      - run: bash scripts/check_new_guard.sh"
row 1 "drift: a new guard script in a non-guard job with no row is red" \
    '^FAIL other runs scripts/check_new_guard.sh' lib check-coverage --ack "$SCRATCH/ack.txt"
printf 'other check_other_thing.sh  # r\nother check_new_guard.sh  # r\nother check_gone.sh  # r\n' \
    > "$SCRATCH/ack.txt"
row 1 "drift: a stale acknowledgement row is red" \
    '^FAIL stale row .*check_gone.sh' lib check-coverage --ack "$SCRATCH/ack.txt"
printf 'other check_other_thing.sh\n' > "$SCRATCH/ack.txt"
row 2 "drift: a row without a reason is refused" 'want `<job> <script>' \
    lib check-coverage --ack "$SCRATCH/ack.txt"

# 9. The real ci.yml: the drift check is green and there are guard steps to run.
row 0 "real ci.yml: every guard script outside a guard job is acknowledged" \
    '^ok +[0-9]+ guard jobs run locally' \
    bash -c "cd '$REPO_ROOT' && python3 '$LIB' check-coverage"

echo "---"
if [ "$red" = 0 ]; then
    echo "PASS: $n rows"
    exit 0
fi
echo "FAIL: at least one of $n rows is red"
exit 1
