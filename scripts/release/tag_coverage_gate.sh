#!/usr/bin/env bash
# tag_coverage_gate.sh — did the release tag's own coverage run pass? (#3690)
#
#   bash scripts/release/tag_coverage_gate.sh TAG SHA   # wait for, then judge, the tag's coverage
#   bash scripts/release/tag_coverage_gate.sh --self-test
#
# WHY. #3676 took coverage off the PR/queue/push path (operator 2026-09-21: "YES, coverage
# on tags release only"). A push of a v* tag runs ci.yml, whose sovereign-ci `coverage_on:
# tag` job `ci / coverage` enforces COV_FLOOR. Nothing on the release path read that job:
# autopilot.sh went tag -> clean-room -> assets -> preflight -> cascade, so a floor breach
# turned the tag's run red and the crates.io cascade proceeded anyway. autopilot.sh now
# runs this before preflight, and a red or missing tag coverage stops it there.
#
# WHAT IS JUDGED. The `ci / coverage` JOB of the ci.yml push run on TAG whose head is SHA,
# not the run's overall conclusion: v0.69.1's tag run (35911380495) concluded failure on
# an unrelated job while `ci / coverage` was success. A run on another commit is ignored.
# A run that is absent, a job that never appears, or either still unfinished after the
# wait is a refusal: Unknown is not a pass.
#
# ENV  GH (default gh) · TCG_REPO (paiml/aprender) · TCG_TRIES (90) · TCG_SLEEP (60 s).
# EXIT 0 coverage green on the tag · 1 red, absent or unfinished · 2 usage.
set -uo pipefail
PROG=tag_coverage_gate
GH=${GH:-gh}
REPO=${TCG_REPO:-paiml/aprender}
JOB='ci / coverage'

# Pure. RUN is the matched run's status ('' = no run on this sha yet); JOB_STATE is
# "<status> <conclusion>" of the coverage job ('' = not present). Prints ok|wait|bad <why>.
tcg_decide() {
    local run=$1 job=$2
    if [ -z "$run" ]; then echo "wait no ci.yml push run on this commit yet"; return; fi
    case "$job" in
        'completed success') echo "ok"; return ;;
        completed\ *) echo "bad '$JOB' concluded '${job#completed }'"; return ;;
        '') ;;
        *) echo "wait '$JOB' is ${job%% *}"; return ;;
    esac
    if [ "$run" = completed ]; then echo "bad the run completed with no '$JOB' job"; return; fi
    echo "wait '$JOB' has not appeared"
}

# find_run TAG SHA -> "<id> <status>" of the newest ci.yml push run on TAG at SHA, or ''.
find_run() {
    "$GH" run list --repo "$REPO" --workflow ci.yml --event push --branch "$1" --limit 20 \
        --json databaseId,headSha,status 2>/dev/null \
    | jq -r --arg sha "$2" 'first(.[] | select(.headSha == $sha)) | "\(.databaseId) \(.status)"' 2>/dev/null || true
}

# job_state RUN_ID -> "<status> <conclusion>" of the coverage job, or ''.
job_state() {
    "$GH" api "repos/$REPO/actions/runs/$1/jobs?per_page=100" 2>/dev/null \
    | jq -r --arg name "$JOB" 'first((.jobs // [])[] | select(.name == $name)) | "\(.status) \(.conclusion // "")"' 2>/dev/null || true
}

gate() {
    local tag=$1 sha=$2 tries=${TCG_TRIES:-90} slp=${TCG_SLEEP:-60} i found id run job v=""
    for ((i = 1; i <= tries; i++)); do
        found=$(find_run "$tag" "$sha"); id=${found%% *}; run=${found#* }
        [ -n "$found" ] || run=""
        job=""; [ -n "$found" ] && job=$(job_state "$id")
        v=$(tcg_decide "$run" "$job")
        case "$v" in
            ok) echo "ok    $JOB green on $tag at $sha (run $id)"; return 0 ;;
            bad\ *) echo "FAIL  ${v#bad } on $tag (run $id) -- no preflight, no cascade"; return 1 ;;
        esac
        [ "$i" -lt "$tries" ] && sleep "$slp"
    done
    echo "FAIL  ${v#wait } on $tag after $tries check(s) -- Unknown is not a pass"
    return 1
}

self_test() {
    local fail=0 got want run job why d rc
    echo "$PROG self-test: case table"
    while IFS='|' read -r want run job why; do
        got=$(tcg_decide "$run" "$job")
        if [ "${got%% *}" = "$want" ]; then echo "  ok   $why"; else echo "  FAIL $why: wanted $want, got '$got'"; fail=1; fi
    done <<'EOF'
ok|completed|completed success|coverage green on a finished run
ok|in_progress|completed success|coverage green while other jobs still run
bad|completed|completed failure|a COV_FLOOR breach stops the release
bad|in_progress|completed cancelled|a cancelled coverage job is not a pass
bad|completed|completed skipped|a skipped coverage job is not a pass
bad|completed||a finished run with no coverage job is not a pass
wait|in_progress|in_progress |coverage still running
wait|queued||coverage not yet scheduled
wait|||no run on this commit yet
EOF
    # End to end through the real JSON filters, with a stub gh answering from fixtures.
    d=$(mktemp -d) || return 1
    cat > "$d/gh" <<'STUB'
#!/usr/bin/env bash
case "$1" in
    run) cat "$FIX/runs.json" ;;
    api) cat "$FIX/jobs.json" ;;
esac
STUB
    chmod +x "$d/gh"
    local S=1111111111111111111111111111111111111111 O=2222222222222222222222222222222222222222
    e2e() { # e2e WANT_RC WHY RUNS_JSON JOBS_JSON
        printf '%s' "$3" > "$d/runs.json"; printf '%s' "$4" > "$d/jobs.json"
        FIX=$d GH=$d/gh TCG_TRIES=2 TCG_SLEEP=0 gate v9.9.9 "$S" > "$d/out" 2>&1; rc=$?
        if [ "$rc" = "$1" ]; then echo "  ok   $2"; else echo "  FAIL $2: wanted rc $1, got $rc: $(cat "$d/out")"; fail=1; fi
    }
    e2e 0 "e2e: green coverage on the tag commit passes" \
        "[{\"databaseId\":7,\"headSha\":\"$S\",\"status\":\"completed\"}]" \
        '{"jobs":[{"name":"gate","status":"completed","conclusion":"failure"},{"name":"ci / coverage","status":"completed","conclusion":"success"}]}'
    e2e 1 "e2e: a failed tag coverage run stops the autopilot before the cascade" \
        "[{\"databaseId\":7,\"headSha\":\"$S\",\"status\":\"completed\"}]" \
        '{"jobs":[{"name":"ci / coverage","status":"completed","conclusion":"failure"}]}'
    e2e 1 "e2e: a missing run stops it too (Unknown is not a pass)" '[]' '{"jobs":[]}'
    e2e 1 "e2e: a green run on ANOTHER commit is not this tag's coverage" \
        "[{\"databaseId\":7,\"headSha\":\"$O\",\"status\":\"completed\"}]" \
        '{"jobs":[{"name":"ci / coverage","status":"completed","conclusion":"success"}]}'
    e2e 1 "e2e: coverage still running when the wait ends is a refusal" \
        "[{\"databaseId\":7,\"headSha\":\"$S\",\"status\":\"in_progress\"}]" \
        '{"jobs":[{"name":"ci / coverage","status":"in_progress","conclusion":null}]}'
    e2e 1 "e2e: gh answering garbage is a refusal, not a pass" 'rate limited' 'rate limited'
    rm -f -- "$d/gh" "$d/runs.json" "$d/jobs.json" "$d/out"; rmdir -- "$d"
    if [ "$fail" -eq 0 ]; then echo "$PROG self-test: PASS"; return 0; fi
    echo "$PROG self-test: FAIL"; return 1
}

main() {
    case "${1:-}" in
        --self-test) self_test; return ;;
        -h|--help) sed -n '2,23p' "${BASH_SOURCE[0]}"; return 0 ;;
    esac
    if [ "$#" -ne 2 ] || [[ ! $2 =~ ^[0-9a-f]{40}$ ]]; then
        echo "$PROG: usage: TAG SHA(40-hex) | --self-test" >&2; return 2
    fi
    gate "$1" "$2"
}

main "$@"
