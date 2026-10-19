#!/usr/bin/env bash
# rerun_on_75.sh -- run a CI step; exit 75 (EX_TEMPFAIL, "could not check", scripts/pv_bin.sh)
# means the probe failed, not the code, so rerun the step ONCE. A second 75 stays 75 and says
# COULD-NOT-CHECK: a check that could not run never passes (#5755, operator ruling 3).
#
#   rerun_on_75.sh CMD [ARG...]   run CMD; rerun once on rc 75; exit with the last rc
#   rerun_on_75.sh --self-test    case table
set -euo pipefail

INFRA_RC=75

run() {
    local rc=0
    "$@" || rc=$?
    if [ "$rc" -eq "$INFRA_RC" ]; then
        echo "::warning::step exited $INFRA_RC (could not check); rerunning once" >&2
        rc=0
        "$@" || rc=$?
        if [ "$rc" -eq "$INFRA_RC" ]; then
            echo "COULD-NOT-CHECK: step exited $INFRA_RC twice; failing, not passing" >&2
        fi
    fi
    return "$rc"
}

self_test() {
    local d bad=0
    d=$(mktemp -d)
    # shellcheck disable=SC2064
    trap "rm -rf -- '${d:?}'" RETURN
    # step.sh <counter-file> <rc1> [rc2 ...]: exits with the Nth rc on its Nth call.
    cat >"${d:?}/step.sh" <<'STEP'
#!/usr/bin/env bash
f=$1; shift
n=$(cat "$f" 2>/dev/null || echo 0); n=$((n + 1)); echo "$n" >"$f"
idx=$n; [ "$idx" -le "$#" ] || idx=$#
exit "${!idx}"
STEP
    case_() { # name want_rc want_calls rc-sequence...
        local name=$1 want=$2 calls=$3 rc=0 got
        shift 3
        rm -f "${d:?}/n"
        bash "$0" bash "$d/step.sh" "$d/n" "$@" >/dev/null 2>&1 || rc=$?
        got=$(cat "$d/n")
        if [ "$rc" -ne "$want" ] || [ "$got" -ne "$calls" ]; then
            echo "SELF-TEST FAIL: $name wanted rc=$want calls=$calls, got rc=$rc calls=$got" >&2
            bad=1
        fi
    }
    case_ '75 then pass: reran once, passes' 0 2 75 0
    case_ '75 then 75: second 75 fails, not rerun again' 75 2 75 75
    case_ 'wrong version (rc 1) is red and not rerun' 1 1 1
    case_ '75 then red: rerun result wins' 1 2 75 1
    case_ 'a pass is not rerun' 0 1 0
    if [ "$bad" -eq 0 ]; then
        echo "rerun_on_75 self-test: 5 cases ok"
    fi
    return "$bad"
}

case "${1:-}" in
    --self-test) self_test ;;
    "") echo "usage: $0 CMD [ARG...] | --self-test" >&2; exit 2 ;;
    *) run "$@" ;;
esac
