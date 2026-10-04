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
# THE JOB MUST EXIST BEFORE THE TAG (#4691). ci.yml dropped its `ci:` sovereign-ci call, and
# `ci / coverage` with it, while this gate kept waiting for that name after every tag. On v0.70.1
# it waited for the whole 25-minute tag run and then refused, with the tag and the GitHub release
# already public. `tcg_job_name` is the ONE place the name lives. `--resolve SHA` reads
# .github/workflows/ci.yml at SHA and refuses, naming the job and the file, unless one of its jobs
# produces that check name. autopilot.sh cut_tag() runs it ahead of `git tag`, and gate() runs it
# again before its first gh call. A ci.yml that cannot be read is NOT_MEASURED, which refuses. A
# name that only a reusable workflow's inner job would produce (`<caller> / <job>`, the caller
# having `uses:`) cannot be resolved from ci.yml, and refuses the same way.
#
#   bash scripts/release/tag_coverage_gate.sh --resolve SHA   # before the tag: is the job declared?
#
# ENV  GH (default gh) · TCG_REPO (paiml/aprender) · TCG_TRIES (90) · TCG_SLEEP (60 s) ·
#      TCG_CI_YML (read this file instead of `git show SHA:.github/workflows/ci.yml`).
# EXIT 0 coverage green on the tag (with --resolve: the job is declared) · 1 red, absent,
#      unfinished, undeclared or not measured · 2 usage.
set -uo pipefail
PROG=tag_coverage_gate
GH=${GH:-gh}
REPO=${TCG_REPO:-paiml/aprender}
CI_YML_REL=.github/workflows/ci.yml

# The ONE place the coverage check name lives (#4691).
tcg_job_name() { echo 'ci / coverage'; }
JOB=$(tcg_job_name)

# ci_jobs FILE -> one "<id><TAB><check name><TAB><uses or ->" line per job under `jobs:`. The
# check name is the job's `name:` when it has one, else its id. Pure; no network.
ci_jobs() {
    awk '
        function flush() { if (id != "") printf "%s\t%s\t%s\n", id, (name == "" ? id : name), (uses == "" ? "-" : uses); id = ""; name = ""; uses = "" }
        function val(s) { sub(/^[^:]*:[ \t]*/, "", s); sub(/[ \t]+$/, "", s); if (s ~ /^".*"$/ || s ~ /^\047.*\047$/) s = substr(s, 2, length(s) - 2); return s }
        /^jobs:[ \t]*$/ { inj = 1; next }
        inj && /^[^ \t#]/ { flush(); inj = 0; next }
        inj && /^  [A-Za-z0-9_-]+:[ \t]*$/ { flush(); id = $0; sub(/^  /, "", id); sub(/:.*/, "", id); next }
        inj && id != "" && /^    name:/ { name = val($0); next }
        inj && id != "" && /^    uses:/ { uses = val($0); next }
        END { flush() }
    ' "$1"
}

# tcg_resolve FILE -> prints ok|bad|nm <why>. Pure: is $JOB a check name some job in FILE produces?
tcg_resolve() {
    local f=$1 jobs caller
    [ -s "$f" ] || { echo "nm $CI_YML_REL could not be read"; return; }
    jobs=$(ci_jobs "$f")
    [ -n "$jobs" ] || { echo "nm $CI_YML_REL declares no jobs that could be parsed"; return; }
    if awk -F'\t' -v j="$JOB" '$2 == j && $3 == "-" { f = 1 } END { exit !f }' <<< "$jobs"; then echo ok; return; fi
    caller=${JOB%% / *}
    if [ "$caller" != "$JOB" ] && awk -F'\t' -v c="$caller" '$2 == c && $3 != "-" { f = 1 } END { exit !f }' <<< "$jobs"; then
        echo "nm '$JOB' would come from inside the reusable workflow job '$caller' calls, which $CI_YML_REL does not show"; return
    fi
    echo "bad no job in $CI_YML_REL produces '$JOB' (its jobs: $(cut -f2 <<< "$jobs" | paste -sd, -))"
}

# resolve SHA -> 0 when the job is declared in ci.yml at SHA; 1 (refuse) otherwise. Prints one line.
resolve() {
    local sha=$1 f v d=""
    if [ -n "${TCG_CI_YML:-}" ]; then f=$TCG_CI_YML
    else
        d=$(mktemp -d) || { echo "NOT_MEASURED mktemp failed -- refused before the tag"; return 1; }
        f=$d/ci.yml
        git show "$sha:$CI_YML_REL" > "$f" 2>/dev/null || : > "$f"
    fi
    v=$(tcg_resolve "$f")
    [ -n "$d" ] && { rm -f -- "$d/ci.yml"; rmdir -- "$d"; }
    case "$v" in
        ok) echo "ok    '$JOB' is declared in $CI_YML_REL at $sha"; return 0 ;;
        nm\ *) echo "NOT_MEASURED ${v#nm } at $sha -- refused; Unknown is not a pass" ;;
        *) echo "FAIL  ${v#bad } at $sha -- refused before the tag" ;;
    esac
    return 1
}

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
    # #4691: a job ci.yml does not declare can never appear; refuse now, not after the wait.
    resolve "$sha" || return 1
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
    # End to end through the real JSON filters, with a stub gh answering from fixtures. The stub
    # logs every call, so a refusal that must come before any gh call (#4691) can be told apart
    # from one that waited for a job that can never appear.
    d=$(mktemp -d) || return 1
    cat > "$d/gh" <<'STUB'
#!/usr/bin/env bash
echo "$1" >> "$FIX/calls"
[ -e "$FIX/gh-fails" ] && { echo "HTTP 502" >&2; exit 1; }
case "$1" in
    run) cat "$FIX/runs.json" ;;
    api) cat "$FIX/jobs.json" ;;
esac
STUB
    chmod +x "$d/gh"
    # ci.yml fixtures: the job declared (in the `ci / gate` shape ci.yml uses today), absent,
    # renamed, and reachable only through a reusable workflow call (the shape #3676 waited on).
    printf 'name: CI\non: push\njobs:\n  x86-main:\n    runs-on: x\n  ci-coverage:\n    name: "ci / coverage"\n    runs-on: x\n' > "$d/present.yml"
    printf 'name: CI\non: push\njobs:\n  x86-main:\n    runs-on: x\n  ci-gate:\n    name: ci / gate\n    runs-on: x\n' > "$d/absent.yml"
    printf 'name: CI\njobs:\n  ci-coverage:\n    name: ci / coverage-tag\n    runs-on: x\n' > "$d/renamed.yml"
    printf 'name: CI\njobs:\n  ci:\n    uses: paiml/.github/.github/workflows/sovereign-ci.yml@main\n' > "$d/reusable.yml"
    printf 'name: CI\non: push\n' > "$d/nojobs.yml"
    : > "$d/empty.yml"
    echo "$PROG self-test: ci.yml resolution (#4691)"
    local fx
    while IFS='|' read -r want fx why; do
        got=$(tcg_resolve "$d/$fx.yml")
        if [ "${got%% *}" = "$want" ]; then echo "  ok   $why"; else echo "  FAIL $why: wanted $want, got '$got'"; fail=1; fi
    done <<'EOF'
ok|present|a job whose name: is the check name resolves
bad|absent|no job produces the check name: refused
bad|renamed|a renamed job (ci / coverage-tag) is not the job waited for: refused
nm|reusable|a name only a reusable workflow's inner job would produce cannot be resolved: refused
nm|nojobs|a ci.yml with no parsable jobs is not measured: refused
nm|empty|an unreadable (empty) ci.yml is not measured: refused
EOF
    local S=1111111111111111111111111111111111111111 O=2222222222222222222222222222222222222222
    e2e() { # e2e WANT_RC WHY RUNS_JSON JOBS_JSON [CI_YML_FIXTURE [WANT_GH_CALLS]]
        local calls
        printf '%s' "$3" > "$d/runs.json"; printf '%s' "$4" > "$d/jobs.json"; : > "$d/calls"
        FIX=$d GH=$d/gh TCG_TRIES=2 TCG_SLEEP=0 TCG_CI_YML="$d/${5:-present}.yml" gate v9.9.9 "$S" > "$d/out" 2>&1; rc=$?
        calls=$(wc -l < "$d/calls")
        if [ "$rc" != "$1" ]; then echo "  FAIL $2: wanted rc $1, got $rc: $(cat "$d/out")"; fail=1
        elif [ -n "${6:-}" ] && [ "$calls" != "$6" ]; then echo "  FAIL $2: wanted $6 gh call(s), got $calls: $(cat "$d/out")"; fail=1
        else echo "  ok   $2"; fi
    }
    local GREEN='{"jobs":[{"name":"ci / coverage","status":"completed","conclusion":"success"}]}'
    local RUN_DONE="[{\"databaseId\":7,\"headSha\":\"$S\",\"status\":\"completed\"}]"
    local RUN_LIVE="[{\"databaseId\":7,\"headSha\":\"$S\",\"status\":\"in_progress\"}]"
    e2e 1 "e2e #4691: job absent from ci.yml refuses before any gh call (no wait on a live run)" \
        "$RUN_LIVE" '{"jobs":[{"name":"x86-main","status":"in_progress","conclusion":null}]}' absent 0
    e2e 1 "e2e #4691: job renamed in ci.yml refuses before any gh call" "$RUN_DONE" "$GREEN" renamed 0
    e2e 1 "e2e #4691: job behind a reusable workflow refuses before any gh call" "$RUN_DONE" "$GREEN" reusable 0
    e2e 0 "e2e #4691: job declared and green on the tag passes" "$RUN_DONE" "$GREEN" present
    e2e 1 "e2e #4691: job declared and red on the tag refuses" \
        "$RUN_DONE" '{"jobs":[{"name":"ci / coverage","status":"completed","conclusion":"failure"}]}' present
    : > "$d/gh-fails"
    e2e 1 "e2e #4691: job declared, gh failing refuses (a failed read is not a pass)" "$RUN_DONE" "$GREEN" present
    rm -f -- "${d:?}/gh-fails"
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
    rm -f -- "${d:?}/gh" "$d/runs.json" "$d/jobs.json" "$d/out" "$d/calls" "$d"/*.yml; rmdir -- "$d"
    if [ "$fail" -eq 0 ]; then echo "$PROG self-test: PASS"; return 0; fi
    echo "$PROG self-test: FAIL"; return 1
}

main() {
    case "${1:-}" in
        --self-test) self_test; return ;;
        -h|--help) sed -n '2,36p' "${BASH_SOURCE[0]}"; return 0 ;;
        --resolve)
            if [ "$#" -ne 2 ] || [[ ! $2 =~ ^[0-9a-f]{40}$ ]]; then
                echo "$PROG: usage: --resolve SHA(40-hex)" >&2; return 2
            fi
            resolve "$2"; return ;;
    esac
    if [ "$#" -ne 2 ] || [[ ! $2 =~ ^[0-9a-f]{40}$ ]]; then
        echo "$PROG: usage: TAG SHA(40-hex) | --resolve SHA(40-hex) | --self-test" >&2; return 2
    fi
    gate "$1" "$2"
}

main "$@"
