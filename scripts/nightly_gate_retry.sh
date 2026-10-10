#!/usr/bin/env bash
# Nightly gate with a retry on ci-pending, and the run's final verdict (#5053).
#
# Run 38016007449 (02:10Z) gated main 2 minutes after its push: workspace-test
# was still in progress, the gate said ci-pending, every build was skipped and
# the run ended GREEN. Nothing retried, so the nightly went 27h unbuilt while
# every signal said green.
#
#   decide <tries> <wait-seconds> -- <cmd...>
#       Runs <cmd> (prints one gate decision). While it prints ci-pending and
#       tries remain, sleeps <wait-seconds> and asks again -- the same night,
#       the same sha. Prints the last decision.
#   verdict <decision> [sha]
#       Exit status of the run: built|reused -> 0; anything else -> 1 with the
#       reason. ci-pending after the last retry is RED, never green.
#   --self-test
#       The case table, including the planted ci-pending head.
set -euo pipefail

decide() {
    local tries="$1" wait_s="$2"
    shift 3 # tries, wait-seconds, --
    local n=1 d
    d=$("$@") || return
    while [ "$d" = "ci-pending" ] && [ "$n" -lt "$tries" ]; do
        echo "nightly gate: ci-pending (try $n/$tries); asking again in ${wait_s}s" >&2
        sleep "$wait_s"
        n=$((n + 1))
        d=$("$@") || return
    done
    echo "nightly gate: $d after $n/$tries tries" >&2
    printf '%s\n' "$d"
}

verdict() {
    local d="$1" sha="${2:-HEAD}"
    case "$d" in
        built | reused) return 0 ;;
        ci-pending)
            echo "::error::required checks still not finished on ${sha:0:9} after the last retry; nothing built tonight (last green kept)"
            return 1
            ;;
        *)
            echo "::error::nightly ${d:-<empty>} at ${sha:0:9}; see nightly-manifest.json"
            return 1
            ;;
    esac
}

self_test() {
    local fails=0 total=0
    # Global, not local: the EXIT trap runs after this function has returned.
    tmp=$(mktemp -d)
    trap 'rm -rf "${tmp:?}"' EXIT
    # A stub gate that answers from a list of decisions, one per call.
    stub() {
        local first
        first=$(head -n 1 "$tmp/seq")
        tail -n +2 "$tmp/seq" > "$tmp/seq.next"
        mv "$tmp/seq.next" "$tmp/seq"
        printf '%s\n' "$first"
    }
    check() {
        total=$((total + 1))
        if [ "$2" = "$3" ]; then
            echo "ok   $1"
        else
            echo "FAIL $1: got '$2' want '$3'"
            fails=$((fails + 1))
        fi
    }
    rc() { if "$@" > /dev/null; then echo 0; else echo 1; fi; }

    printf 'ci-pending\nci-pending\nci-pending\n' > "$tmp/seq"
    check "planted: ci-pending head on every try ends ci-pending" "$(decide 3 0 -- stub 2> /dev/null)" "ci-pending"
    check "planted: that verdict is RED, never green" "$(rc verdict ci-pending abc)" "1"
    printf 'ci-pending\nbuild\n' > "$tmp/seq"
    check "ci-pending, then green on the retry -> build" "$(decide 3 0 -- stub 2> /dev/null)" "build"
    printf 'ci-pending\nci-pending\nbuild\n' > "$tmp/seq"
    check "tries are a cap: 2 tries never reach the 3rd answer" "$(decide 2 0 -- stub 2> /dev/null)" "ci-pending"
    printf 'red-ci\nbuild\n' > "$tmp/seq"
    check "red-ci is final, never retried" "$(decide 3 0 -- stub 2> /dev/null)" "red-ci"
    printf 'reused\n' > "$tmp/seq"
    check "reused needs no retry" "$(decide 3 0 -- stub 2> /dev/null)" "reused"
    # A gate that cannot answer (fetch error, rate limit) fails the step: no retry, no decision.
    check "failing gate command -> nonzero, never a decision" \
        "$( (decide 3 0 -- sh -c 'echo ci-pending; exit 3') 2> /dev/null; echo "rc=$?")" "rc=3"
    check "built -> green" "$(rc verdict built x)" "0"
    check "reused -> green" "$(rc verdict reused x)" "0"
    check "red-ci -> red" "$(rc verdict red-ci x)" "1"
    check "build-failed -> red" "$(rc verdict build-failed x)" "1"
    check "empty decision -> red" "$(rc verdict '' x)" "1"

    if [ "$fails" -eq 0 ]; then
        echo "nightly_gate_retry self-test: $total/$total ok"
        return 0
    fi
    echo "nightly_gate_retry self-test: $fails/$total FAILED"
    return 1
}

case "${1:-}" in
    decide) shift; decide "$@" ;;
    verdict) shift; verdict "$@" ;;
    --self-test) self_test ;;
    *)
        echo "usage: $0 decide <tries> <wait-s> -- <cmd...> | verdict <decision> [sha] | --self-test" >&2
        exit 2
        ;;
esac
