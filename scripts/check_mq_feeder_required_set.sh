#!/usr/bin/env bash
# check_mq_feeder_required_set.sh -- the check the merge-queue feeder requires on a PR head must be a job a PR head runs (#6175).
# The feeder required x86-main, which runs only in merge_group/push, so it refused every CLEAN PR as "x86-main is skipped".
# Bare run: --self-test, then `mq_feeder.sh --check-required` against this tree's ci.yml. --self-test plants the incident
# (required x86-main, merge-queue-only, multi-line `if:`), an absent job, a missing file, one good workflow, and drives the
# feeder end to end to prove it refuses to act (exit 2 + inbox line) when the required check cannot exist on a PR head.
#   bash scripts/check_mq_feeder_required_set.sh [--self-test]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FEEDER="${MQ_FEEDER_UNDER_TEST:-$ROOT/scripts/mq_feeder.sh}"
if [ "${1:-}" = --self-test ]; then
    # hermetic: fixtures are judged against the feeder's default required check whatever the caller exports; only this branch pins it (it exits), so the bare run below still sees the real default
    export GUARD_CHECK_NAME=pr-light
    d="$(mktemp -d)"; fail=0
    printf 'on:\n  pull_request:\njobs:\n  pr-light:\n    if: ${{ github.event_name == '"'pull_request'"' }}\n' > "$d/good.yml"
    printf 'on:\n  pull_request:\njobs:\n  pr-light:\n    if: ${{ github.event_name != '"'pull_request'"' }}\n' > "$d/bad-mqonly.yml"
    printf 'on:\n  pull_request:\njobs:\n  other:\n    if: ${{ github.event_name == '"'pull_request'"' }}\n' > "$d/bad-absent.yml"
    printf 'on:\n  pull_request:\njobs:\n  x86-main:\n    name: x86-main\n    if: >-\n      github.event_name != '"'pull_request'"'\n    runs-on: x\n  pr-light:\n    if: ${{ github.event_name == '"'pull_request'"' }}\n' > "$d/incident.yml"
    bash "$FEEDER" --check-required "$d/good.yml" > /dev/null 2>&1 || { echo "FAIL good fixture refused"; fail=1; }
    bash "$FEEDER" --check-required "$d/bad-mqonly.yml" > /dev/null 2>&1 && { echo "FAIL mq-only required check accepted"; fail=1; }
    bash "$FEEDER" --check-required "$d/bad-absent.yml" > /dev/null 2>&1 && { echo "FAIL absent required check accepted"; fail=1; }
    bash "$FEEDER" --check-required "$d/missing.yml" > /dev/null 2>&1 && { echo "FAIL missing workflow accepted"; fail=1; }
    GUARD_CHECK_NAME=x86-main bash "$FEEDER" --check-required "$d/incident.yml" > /dev/null 2>&1 && { echo "FAIL incident (x86-main, multi-line if) accepted"; fail=1; }
    bash "$FEEDER" --check-required "$d/incident.yml" > /dev/null 2>&1 || { echo "FAIL incident workflow refused for pr-light"; fail=1; }
    # end to end: the deployed feeder judges the workflow it is handed, refuses to act, and says so in the inbox
    mkdir -p "$d/st"; : > "$d/inbox"
    # a stub gh with one CLEAN candidate: the guard sits after the idle checks, so it only fires when the feeder is about to act
    printf '#!/usr/bin/env bash\ncase "$*" in\n  *rate_limit*) echo 5000;;\n  *"mergeQueue(branch"*) printf %%s %s;;\nesac\n' "'"'{"data":{"repository":{"mergeQueue":{"entries":{"totalCount":0}},"pullRequests":{"nodes":[{"number":7,"isDraft":false,"baseRefName":"main","mergeStateStatus":"CLEAN","autoMergeRequest":null,"mergeQueueEntry":null}]}}}}'"'"'' > "$d/gh"; chmod +x "$d/gh"
    PATH="$d:$PATH" GUARD_CHECK_NAME=x86-main MQ_FEEDER_CI_FILE="$d/incident.yml" MQ_FEEDER_STATE="$d/st" MQ_FEEDER_INBOX="$d/inbox" bash "$FEEDER" > /dev/null 2>&1; rc=$?
    [ "$rc" -eq 2 ] || { echo "FAIL feeder acted on an unsatisfiable required check (rc=$rc)"; fail=1; }
    grep -q 'runs only in merge_group/push' "$d/inbox" || { echo "FAIL feeder refused silently (no inbox line)"; fail=1; }
    # the bare run needs BOTH legs: the feeder --self-test and --check-required on this tree's ci.yml. Each row turns exactly one
    # leg red (a stub feeder, steered by env, answers only the two bare-run calls; every other call goes to the real feeder),
    # so deleting either leg lets its row pass and fails this self-test. CHECK_MQ_IN_ROW stops the rows recursing.
    if [ -z "${CHECK_MQ_IN_ROW:-}" ]; then
        printf '#!/usr/bin/env bash\ncase "$*" in\n  *--self-test*) [ -z "${STUB_SELF:-}" ] || exit "$STUB_SELF" ;;\n  *%s*) [ -z "${STUB_CHK:-}" ] || exit "$STUB_CHK" ;;\nesac\nexec bash %s "$@"\n' '/.github/workflows/ci.yml' "$ROOT/scripts/mq_feeder.sh" > "$d/stub.sh"
        row() { # NAME WANT-RC STUB_SELF STUB_CHK
            CHECK_MQ_IN_ROW=1 MQ_FEEDER_UNDER_TEST="$d/stub.sh" STUB_SELF="$3" STUB_CHK="$4" bash "$0" > /dev/null 2>&1; got=$?
            [ "$got" -eq "$2" ] || { echo "FAIL bare run: $1 (rc=$got, want $2)"; fail=1; }
        }
        row "both legs green passes" 0 "" ""
        row "ci.yml without pr-light refused (feeder --self-test green)" 1 "" 1
        row "feeder --self-test red refused (ci.yml green)" 1 1 ""
    fi
    # mq_feeder_install.sh refuses to install a feeder whose required check main's ci.yml cannot satisfy (L18 of that script)
    for want in 0 1; do
        mkdir -p "$d/i$want"; cp "$ROOT/scripts/mq_feeder_install.sh" "$d/i$want/"; : > "$d/i$want/arm_pr_automerge.sh"
        printf '#!/usr/bin/env bash\nexit %s\n' "$want" > "$d/i$want/mq_feeder.sh"
        bash "$d/i$want/mq_feeder_install.sh" "$d/dest$want" > /dev/null 2>&1; got=$?
        if [ "$want" -eq 0 ]; then
            { [ "$got" -eq 0 ] && [ -f "$d/dest0/mq_feeder.sh" ]; } || { echo "FAIL installer refused a satisfiable feeder (rc=$got)"; fail=1; }
        else
            { [ "$got" -eq 2 ] && [ ! -e "$d/dest1" ]; } || { echo "FAIL installer installed a feeder whose required check cannot be met (rc=$got)"; fail=1; }
        fi
    done
    rm -rf "${d:?}"
    [ "$fail" -eq 0 ] && echo "check_mq_feeder_required_set: self-test ok (2 good, 4 planted bad, 1 end-to-end, 3 bare-run legs, 2 installer)"
    exit "$fail"
fi
bash "$0" --self-test || exit 1
# the feeder's own self-test holds the real-ci.yml assertion (x86-main is merge-queue-only, pr-light is not) and the runtime tests
bash "$FEEDER" --self-test > /dev/null || { echo "check_mq_feeder_required_set: FAIL the feeder --self-test is red"; exit 1; }
out="$(bash "$FEEDER" --check-required "$ROOT/.github/workflows/ci.yml" 2>&1)"; rc=$?
if [ "$rc" -eq 0 ]; then echo "check_mq_feeder_required_set: PASS ($out)"; exit 0; fi
printf '%s\n' "$out" | tail -n 4
echo "check_mq_feeder_required_set: FAIL rc=$rc"
exit 1
