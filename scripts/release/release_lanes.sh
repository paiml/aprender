#!/usr/bin/env bash
# release_lanes.sh — the nightly producers for the clean-room release lanes (#4720, 0702-P3; BLD-002).
#
#   bash scripts/release/release_lanes.sh measure cleanroom-gpu         # judge this run's b2-gpu job
#   bash scripts/release/release_lanes.sh measure cleanroom-cpu         # judge paiml/infra's clean-room run on C
#   bash scripts/release/release_lanes.sh measure publish-dryrun ROOT   # rc_publish_gate.sh --verify ROOT
#   bash scripts/release/release_lanes.sh measure assets                # judge this run's dry binary-release jobs
#   bash scripts/release/release_lanes.sh measured-cpu --cache DIR      # every commit infra's clean-room measured
#   bash scripts/release/release_lanes.sh assert-head REF               # a lane job's proof that it ran on REF
#   bash scripts/release/release_lanes.sh --self-test | --mutants
#
# WHO READS THIS. nightly_train.sh reads a lane as the newest completed run of its producer on C (main's head):
# green when every job whose name matches the lane ran to success at attempt 1, red when one failed, and
# not_measured when the matched job was skipped or cancelled, or there is no run on C. This script is the
# producer half, run by .github/workflows/release-lanes-nightly.yml, where each lane has two jobs:
#   `measure: <lane>`  runs `measure <lane>`, which prints one verdict and writes `verdict` and `reason` to
#                      GITHUB_OUTPUT. It exits 0 whatever the verdict: the verdict travels in the outputs.
#   `<lane>`           the job the train matches. It runs only when the verdict is green or red, and exits 0 or
#                      1. On not_measured it is SKIPPED, which the train reads as not_measured. A measure job
#                      that dies writes no output, so its lane job is skipped too. No path from a crash to a pass.
#
# THE LANES. C is the workflow's `inputs.ref || github.sha`, passed in as LANES_REF: the rehearsal's pick when
# release-rehearsal-nightly.yml calls the workflow (B1 Q1, Q3), else the head of main when the scheduled run
# started. GITHUB_SHA is never C in a called run (it is the caller's commit), so no lane reads it. A called
# run's job names start with the caller job's name: LANES_CALLER names it, and the judged prefixes carry it.
# Every lane job checks out C and runs `assert-head C` first, so a lane job that ran on another commit fails.
#   cleanroom-gpu   b2-gpu.yml called with ref=C. green: every b2-gpu job succeeded. red: a step that is not a
#                   precondition failed, in a job that ended failure or timed_out (a step cancelled inside such
#                   a job counts as failed). not_measured: a precondition failed (checkout, the commit assert,
#                   no CUDA device), or the job ended cancelled or skipped. A job GitHub reports as cancelled
#                   when it hits timeout-minutes therefore reads not_measured, never green.
#   cleanroom-cpu   paiml/infra clean-room.yml, job `clean-room (aprender)`: the newest one that tested C, read
#                   the way the release driver reads it (cascade-publish.sh clean_room_gate): the log carries
#                   exactly one structured `tested-sha:` record, and `Assert the commit under test` succeeded.
#                   green: success at attempt 1. red: failure, timed_out or startup_failure, or success only at
#                   attempt 2 or later. not_measured: no read token, no infra run tested C, still running.
#                   paiml/infra is private and GITHUB_TOKEN cannot read it: INFRA_TOKEN comes from the
#                   repository secret INFRA_ACTIONS_READ (Actions: read on paiml/infra), and without it the
#                   lane says so every night instead of guessing.
#   publish-dryrun  rc_publish_gate.sh --verify ROOT (graph, pathonly, `cargo package --locked` with verify).
#                   Its exits map 0 green, 1 red, 2 not_measured. The gate is only meaningful on a BUMPED tree:
#                   at a version crates.io already has, `^X` resolves the registry's newer copy (the gate's own
#                   header calls it a false red). So while main's version is at or below crates.io's newest the
#                   lane is not_measured and the gate is not run. `cascade-publish.sh --check` is printed as a
#                   receipt and never decides.
#   assets          (wired with the binary-release.yml dry mode, #4720 P3b) binary-release.yml called dry at C:
#                   every asset built and smoke-run. Same judge as cleanroom-gpu, with that workflow's
#                   precondition steps.
#
# THE MEASURED COMMITS (B1 Q1: C is the newest main commit both the models bundle and an infra clean-room run
# measured). `measured-cpu` prints one `sha TAB green|red TAB run-id` line per commit the newest infra runs
# measured, newest run first, each judged exactly as `measure cleanroom-cpu` judges C. It reads with gh as
# INFRA_TOKEN, every read sent with the ETag cached in DIR (a 304 reuses DIR's copy), and keeps the rate floor
# (a fifth of the token's limit, capped at 1000). No token, the floor reached, or any read that failed, a job
# log's included, is exit 2: not_measured, never an empty list. An empty list at exit 0 is a read that found none.
#
# EXIT  measure: 0 when a verdict was written · 3 a caller error. measured-cpu: 0 read · 2 not_measured · 3.
#       assert-head: 0 HEAD is REF (prints `tested-sha: REF`) · 1 it is not. --self-test / --mutants: 0 green, 1 red.
set -uo pipefail
PROG=release_lanes
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$(cd -- "$HERE/.." && pwd)"
SCRIPT_PATH="$HERE/$(basename -- "${BASH_SOURCE[0]}")"
WORKFLOW="$HERE/../../.github/workflows/release-lanes-nightly.yml"
TAB=$'\t'
API="${GITHUB_API_URL:-https://api.github.com}"
INFRA_REPO=paiml/infra
INFRA_WORKFLOW=clean-room.yml
INFRA_JOB='clean-room (aprender)'
INFRA_ASSERT='Assert the commit under test'   # the step whose success makes the log's tested-sha count
INFRA_RUNS=6            # newest infra runs searched for the one that tested C
INFRA_MAX_LOGS=3        # job logs read at most (each is a whole clean-room log)
RATE_FLOOR=1000         # measured-cpu reads nothing below min(this, a fifth of the limit), as nightly_train.sh
INDEX_URL="${RELEASE_LANES_INDEX_URL:-https://index.crates.io/ap/re/aprender}"
# the lanes release-lanes-nightly.yml wires (the self-test checks each one's two jobs)
WIRED_LANES='cleanroom-gpu cleanroom-cpu publish-dryrun'
# the caller job whose called jobs a lane judges, and the steps whose failure means "could not run here"
GPU_PREFIX='b2-gpu at C'
GPU_ENV='^(Set up job|Complete job|Set up runner|Run actions/checkout@.*|Assert the commit under test|A CUDA device is visible|Post .*)$'
ASSETS_PREFIX='binary-release dry at C'
ASSETS_ENV='^(Set up job|Complete job|Set up runner|Run actions/checkout@.*|Checkout.*|Resolve release tag|Preflight .*|Post .*)$'

caller_error() { printf '%s: caller error: %s\n' "$PROG" "$*" >&2; exit 3; }
tsv() { local IFS="$TAB"; printf '%s\n' "$*"; }
# called PREFIX -> PREFIX as this run's job names carry it: after the caller job's name when the workflow was called
called() { printf '%s%s\n' "${LANES_CALLER:+$LANES_CALLER / }" "$1"; }

# ---------------------------------------------------------------- pure judges: print VERDICT, TAB, REASON --------
# judge_jobs PREFIX ENV_RE (a run's jobs JSON on stdin). The jobs a lane judges are the ones named `PREFIX / ...`
# (a called workflow's jobs carry the caller job's name). A failed job is red unless the step that failed is a
# precondition (ENV_RE); a job that failed with no failed step recorded is not_measured. One red outranks any
# number of preconditions that could not run. A step cancelled inside a failed job is a timeout: it failed.
judge_jobs() {
    jq -r --arg p "$1" --arg env "$2" '
        def cut: (.conclusion == "failure" or .conclusion == "timed_out");
        def failed_step: [.steps[]? | select(.conclusion == "failure" or .conclusion == "timed_out" or .conclusion == "cancelled") | .name][0] // "";
        [.jobs[]? | select(.name | startswith($p + " / "))] as $js
        | if ($js | length) == 0 then "not_measured\tno job named \($p) / ... in this run"
          elif any($js[]; .status != "completed") then
            ($js | map(select(.status != "completed"))[0]) as $j | "not_measured\t\($j.name) is still \($j.status)"
          else
            [$js[] | select(cut) | {name, step: failed_step}] as $f
            | [$f[] | select(.step != "" and (.step | test($env) | not))] as $red
            | if ($red | length) > 0 then
                "red\t\($red[0].name): step \"\($red[0].step)\" failed" + (if ($red | length) > 1 then " (+\(($red | length) - 1) more job(s))" else "" end)
              elif ($f | length) > 0 then
                if $f[0].step == "" then "not_measured\t\($f[0].name) failed with no failed step recorded (runner lost?)"
                else "not_measured\t\($f[0].name): precondition step \"\($f[0].step)\" failed, so the code was not measured" end
              elif any($js[]; .conclusion != "success") then
                ($js | map(select(.conclusion != "success"))[0]) as $j | "not_measured\t\($j.name) ended \($j.conclusion)"
              else "green\tall \($js | length) job(s) of \($p) succeeded" end
          end' 2>/dev/null || printf 'not_measured\tthe job list did not parse\n'
}

# judge_cpu C (TSV on stdin, newest run first: run_id attempt status conclusion tested_sha). The newest row that
# tested C decides. A row still running has tested_sha "?" (its log is not readable yet). A row that tested another
# commit is named in the reason only when its sha is as long as C's.
judge_cpu() {
    awk -F '\t' -v c="$1" -v shalen="${#1}" '
        NF >= 5 && $5 == c { found = 1
            if ($3 != "completed") { printf "not_measured\tinfra run %s on C is still %s\n", $1, $3; exit }
            if ($4 == "success" && $2 == 1) { printf "green\tinfra run %s tested C: success at attempt 1\n", $1; exit }
            if ($4 == "success") { printf "red\tinfra run %s tested C: success only at attempt %s, a retried green is not a green\n", $1, $2; exit }
            if ($4 == "failure" || $4 == "timed_out" || $4 == "startup_failure") { printf "red\tinfra run %s tested C: %s\n", $1, $4; exit }
            printf "not_measured\tinfra run %s tested C and ended %s\n", $1, $4; exit }
        NF >= 5 && $3 != "completed" && run == "" { run = $1 }
        NF >= 5 && $3 == "completed" && $5 ~ /^[0-9a-f]+$/ && length($5) == shalen && newest == "" { newest = substr($5, 1, 12) }
        END { if (found) exit
            if (run != "") printf "not_measured\tinfra run %s is still running; no finished run tested C yet\n", run
            else if (newest != "") printf "not_measured\tno infra clean-room run tested C; the newest tested %s\n", newest
            else printf "not_measured\tno finished clean-room (aprender) job in the newest infra runs\n" }'
}

# is_bumped MAIN REGISTRY_NEWEST -> 0 when main's release (x.y.z, prerelease and build cut off) is above crates.io's
# newest (sort -V, not strings). A prerelease of a published version (0.70.1-rc.1 vs 0.70.1) is NOT bumped: `^0.70.1-rc.1`
# matches the published 0.70.1, the same false red.
is_bumped() {
    local m="${1%%[-+]*}" r="${2%%[-+]*}"
    [ -n "$m" ] && [ -n "$r" ] && [ "$m" != "$r" ] && [ "$(printf '%s\n%s\n' "$r" "$m" | sort -V | tail -n 1)" = "$m" ]
}

# registry_newest (the crates.io sparse index lines on stdin) -> the newest version that is not yanked
registry_newest() {
    jq -r 'select(.yanked | not) | .vers' 2>/dev/null | sort -V | tail -n 1
}

# judge_dryrun RC MAIN_VERSION REGISTRY_NEWEST. RC is `-` when the gate was not run.
judge_dryrun() {
    local rc=$1 main=$2 reg=$3
    [ -n "$main" ] || { printf 'not_measured\tcargo metadata named no version for aprender at C\n'; return; }
    [ -n "$reg" ] || { printf 'not_measured\tthe crates.io index for aprender was not read\n'; return; }
    if ! is_bumped "$main" "$reg"; then
        printf 'not_measured\tmain is %s and crates.io already has %s: --verify would resolve the registry copy (the gate'"'"'s documented false red); main must be bumped first (BLD-002 R3)\n' "$main" "$reg"
        return
    fi
    case "$rc" in
        0) printf 'green\trc_publish_gate --verify green at %s (crates.io newest %s)\n' "$main" "$reg" ;;
        1) printf 'red\trc_publish_gate --verify found a publish defect at %s\n' "$main" ;;
        -) printf 'not_measured\trc_publish_gate was not run\n' ;;
        *) printf 'not_measured\trc_publish_gate could not measure (exit %s)\n' "$rc" ;;
    esac
}

# tested_sha (a clean-room job log on stdin) -> the tested commit, or nothing. The release driver's rule
# (cascade-publish.sh clean_room_tested_shas/_abbrevs + clean_room_gate): the DISTINCT structured `tested-sha:`
# records, and only exactly one full lowercase 40-hex sha counts; beside it, at most one legacy `commit:` record,
# and that one a prefix of the sha. Anything else is a log that proves no commit. A log with only the legacy record
# predates infra#621 and reads as no commit too (the driver resolves it with git; a lane need not).
# It reads the whole log, so the download is never cut off mid-stream.
tested_sha() {
    local log s a
    log="$(tr -d '\r')"
    s="$(printf '%s\n' "$log" | { grep -E '^([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z )?[[:space:]]*tested-sha: .+$' || true; } \
        | sed -E 's/^.*tested-sha: //; s/[[:space:]]+$//' | sort -u)"
    a="$(printf '%s\n' "$log" | { grep -E '^([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z )?    commit:  [0-9a-f]{7,40}$' || true; } \
        | sed -E 's/^.*    commit:  //' | sort -u)"
    [[ "$s" =~ ^[0-9a-f]{40}$ ]] || return 0
    case "$a" in
        "") ;;
        *$'\n'*) return 0 ;;
        *) case "$s" in "$a"*) ;; *) return 0 ;; esac ;;
    esac
    printf '%s\n' "$s"
}

# emit LANE (one judge line on stdin) -> GITHUB_OUTPUT verdict/reason, a summary row, one printed line. Anything
# but green/red/not_measured is not_measured: a judge that printed nonsense measured nothing.
emit() {
    local lane=$1 line v r
    IFS= read -r line || line=""
    v="${line%%"$TAB"*}"; r="${line#*"$TAB"}"
    [ "$r" != "$line" ] || r=""
    case "$v" in
        green|red|not_measured) ;;
        *) r="the judge printed no verdict (got: ${line:0:80})"; v=not_measured ;;
    esac
    r="$(printf '%s' "$r" | tr -d '\r\n')"
    if [ -n "${GITHUB_OUTPUT:-}" ]; then printf 'verdict=%s\nreason=%s\n' "$v" "$r" >> "$GITHUB_OUTPUT" || return 3; fi
    if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then printf '**%s: %s** — %s\n' "$lane" "$v" "$r" >> "$GITHUB_STEP_SUMMARY"; fi
    printf '%s %s: %s — %s\n' "$PROG" "$lane" "$v" "$r"
}

# ---------------------------------------------------------------- the reads (network) -----------------------------
# api URL TOKEN -> the body; non-zero on any HTTP error. A log download redirects to a signed blob URL on another
# host, and curl drops the Authorization header on that hop. The token goes in through a header file (printf is a
# builtin), so it never sits in curl's argv where any process on the runner could read it.
api() {
    curl -fsSL --max-time 120 -H @<(printf 'Authorization: Bearer %s\n' "$2") -H "Accept: application/vnd.github+json" \
        -H "X-GitHub-Api-Version: 2022-11-28" "$1"
}

# this run's jobs (GITHUB_TOKEN, actions: read)
run_jobs() {
    [ -n "${GH_TOKEN:-}" ] && [ -n "${GITHUB_REPOSITORY:-}" ] && [ -n "${GITHUB_RUN_ID:-}" ] || return 1
    api "$API/repos/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID/attempts/${GITHUB_RUN_ATTEMPT:-1}/jobs?per_page=100" "$GH_TOKEN"
}

measure_jobs() {   # measure_jobs PREFIX ENV_RE
    local js
    js="$(run_jobs)" || { printf 'not_measured\tthis run'"'"'s job list was not read\n'; return; }
    printf '%s\n' "$js" | judge_jobs "$1" "$2"
}

# infra_job (a run's jobs JSON on stdin) -> "id status conclusion assert-status/assert-conclusion" of the
# `clean-room (aprender)` job, or nothing; a missing assert step reads absent/-
infra_job() {
    jq -r --arg n "$INFRA_JOB" --arg a "$INFRA_ASSERT" \
        '[.jobs[] | select(.name == $n)][0] // empty
         | ([.steps[]? | select(.name == $a)][0] // {status: "absent", conclusion: null}) as $s
         | "\(.id) \(.status) \(.conclusion // "-") \($s.status)/\($s.conclusion // "-")"' 2>/dev/null
}

# log_proves STATUS ASSERT -> 0 when the job's log may name its commit: the job completed and its commit assert
# succeeded. A log whose assert did not succeed proves no commit, whatever it prints (the release driver's rule).
log_proves() {
    [ "$1" = completed ] && [ "$2" = completed/success ]
}

# infra_curl PATH -> the body of API PATH, read with INFRA_TOKEN (measure's read)
infra_curl() { api "$API/$1" "$INFRA_TOKEN"; }

# infra_gh PATH -> the body of API PATH, read with gh as INFRA_TOKEN and sent with the ETag cached in INFRA_CACHE
# (measured-cpu's read). 200 refreshes the cache; 304 reuses it, and a 304 with nothing cached is a failed read, as
# is any other status. gh's own exit is not read: on a 304 it is non-zero, so the status line decides.
infra_gh() {
    local k st hdr=()
    k="$INFRA_CACHE/$(printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_')"
    [ -s "$k.etag" ] && [ -f "$k.body" ] && hdr=(-H "If-None-Match: $(cat -- "$k.etag")")
    GH_TOKEN="$INFRA_TOKEN" gh api -i "${hdr[@]}" "$1" > "$k.http" 2>/dev/null
    st="$(awk 'NR == 1 { print $2; exit }' "$k.http")"
    case "$st" in
        200) awk 'f { print } /^\r?$/ { f = 1 }' "$k.http" > "$k.body" || return 1
             awk 'tolower($1) == "etag:" { sub(/\r$/, ""); sub(/^[^:]*: */, ""); print; exit }' "$k.http" > "$k.etag" ;;
        304) [ -f "$k.body" ] || return 1 ;;
        *) return 1 ;;
    esac
    cat -- "$k.body"
}

# infra_rows C [GET] -> the TSV judge_cpu reads, newest run first; stops at the first run that tested C (an empty C
# reads them all). GET PATH prints an API path's body (infra_curl by default). Any read that fails, a job log's
# included, fails the whole read: a log that could not be fetched is never a run that tested nothing.
infra_rows() {
    local get=${2:-infra_curl} runs ids id att jobs jid jst jcon jassert sha log logs=0
    runs="$("$get" "repos/$INFRA_REPO/actions/workflows/$INFRA_WORKFLOW/runs?per_page=$INFRA_RUNS")" || return 1
    ids="$(printf '%s\n' "$runs" | jq -r '.workflow_runs[] | "\(.id) \(.run_attempt)"')" || return 1
    while read -r id att; do
        [ -n "$id" ] || continue
        jobs="$("$get" "repos/$INFRA_REPO/actions/runs/$id/jobs?per_page=100")" || return 1
        jid=""; jst=""; jcon=""; jassert=""
        read -r jid jst jcon jassert < <(printf '%s\n' "$jobs" | infra_job)
        [ -n "$jid" ] || continue
        sha="?"
        if log_proves "$jst" "$jassert" && [ "$logs" -lt "$INFRA_MAX_LOGS" ]; then
            logs=$((logs + 1))
            log="$("$get" "repos/$INFRA_REPO/actions/jobs/$jid/logs")" || return 1
            sha="$(printf '%s\n' "$log" | tested_sha)"
            [ -n "$sha" ] || sha="?"
        fi
        tsv "$id" "$att" "$jst" "$jcon" "$sha"
        [ "$sha" != "$1" ] || return 0
    done <<< "$ids"
}

measure_cpu() {
    local c=$1 rows
    [ -n "${INFRA_TOKEN:-}" ] || { printf 'not_measured\tno infra read token: repository secret INFRA_ACTIONS_READ is unset, and paiml/infra is private\n'; return; }
    rows="$(infra_rows "$c")" || { printf 'not_measured\tpaiml/infra clean-room runs were not read\n'; return; }
    printf '%s\n' "$rows" | judge_cpu "$c"
}

# measured_cpu CACHE -> "sha TAB green|red TAB run" for each commit the newest infra clean-room runs measured, newest
# run first (THE MEASURED COMMITS). rc 2 with the reason on stderr: no token, the floor reached, a read failed.
measured_cpu() {
    local INFRA_CACHE=$1 lim rem fl rows sha v
    [ -n "${INFRA_TOKEN:-}" ] || { echo "not_measured: no infra read token (INFRA_ACTIONS_READ); paiml/infra is private" >&2; return 2; }
    mkdir -p -- "$INFRA_CACHE" || return 2
    read -r lim rem <<< "$(GH_TOKEN="$INFRA_TOKEN" gh api rate_limit --jq '"\(.resources.core.limit) \(.resources.core.remaining)"' 2>/dev/null)"
    case "$lim:$rem" in *[!0-9:]*|:*|*:) echo "not_measured: rate_limit unreadable" >&2; return 2 ;; esac
    fl=$((lim / 5)); [ "$fl" -le "$RATE_FLOOR" ] || fl=$RATE_FLOOR
    [ "$rem" -ge "$fl" ] || { echo "not_measured: core remaining $rem under $fl" >&2; return 2; }
    rows="$(infra_rows "" infra_gh)" || { echo "not_measured: paiml/infra clean-room runs were not read" >&2; return 2; }
    while read -r sha; do
        [[ "$sha" =~ ^[0-9a-f]{40}$ ]] || continue
        v="$(printf '%s\n' "$rows" | judge_cpu "$sha")"
        case "${v%%"$TAB"*}" in
            green|red) tsv "$sha" "${v%%"$TAB"*}" "$(printf '%s\n' "$rows" | awk -F '\t' -v s="$sha" '$5 == s { print $1; exit }')" ;;
        esac
    done < <(printf '%s\n' "$rows" | awk -F '\t' 'NF >= 5 && !seen[$5]++ { print $5 }')
}

# assert_head REF -> 0 when this checkout's HEAD is REF, a 40-hex sha; prints the tested-sha record. A lane job runs it
# before anything else, so a lane job that ran on any other commit fails (nightly_train.sh reads a lane on C only
# from a job whose assert step passed).
assert_head() {
    local ref=${1:-} head
    [[ "$ref" =~ ^[0-9a-f]{40}$ ]] || { printf '%s: assert-head: "%s" is not a 40-hex sha\n' "$PROG" "$ref" >&2; return 1; }
    head="$(git rev-parse --verify -q HEAD 2>/dev/null)" || { printf '%s: assert-head: no HEAD here\n' "$PROG" >&2; return 1; }
    [ "$head" = "$ref" ] || { printf '%s: assert-head: HEAD is %s, not %s\n' "$PROG" "$head" "$ref" >&2; return 1; }
    printf 'tested-sha: %s\n' "$head"
}

measure_dryrun() {
    local root=$1 main reg rc=-
    [ -f "$root/Cargo.toml" ] || { printf 'not_measured\t%s has no Cargo.toml\n' "$root"; return; }
    main="$(cargo metadata --no-deps --format-version 1 --manifest-path "$root/Cargo.toml" 2>/dev/null \
        | jq -r '[.packages[] | select(.name == "aprender") | .version][0] // empty')"
    reg="$(curl -fsSL --max-time 60 "$INDEX_URL" 2>/dev/null | registry_newest)"
    if is_bumped "$main" "$reg"; then
        bash "$HERE/rc_publish_gate.sh" --verify "$root" >&2; rc=$?
        # a receipt, never a stop: which crates the tree is ahead of crates.io on
        (cd -- "$root" && timeout 300 bash "$SCRIPTS/cascade-publish.sh" --check) 2>&1 | tail -n 20 >&2
    fi
    judge_dryrun "$rc" "$main" "$reg"
}

measure() {
    local lane=${1:-} line
    [[ "${LANES_CALLER:-}" =~ ^[A-Za-z0-9._-]*$ ]] || caller_error "LANES_CALLER '${LANES_CALLER:-}' is not a caller job name"
    case "$lane" in
        cleanroom-gpu) line="$(measure_jobs "$(called "$GPU_PREFIX")" "$GPU_ENV")" ;;
        assets) line="$(measure_jobs "$(called "$ASSETS_PREFIX")" "$ASSETS_ENV")" ;;
        cleanroom-cpu)
            [[ "${LANES_REF:-}" =~ ^[0-9a-f]{40}$ ]] || caller_error "LANES_REF is not a 40-hex sha (the workflow passes inputs.ref || github.sha)"
            line="$(measure_cpu "$LANES_REF")" ;;
        publish-dryrun)
            [ -n "${2:-}" ] || caller_error "measure publish-dryrun ROOT"
            line="$(measure_dryrun "$2")" ;;
        *) caller_error "unknown lane '$lane' (cleanroom-gpu | cleanroom-cpu | publish-dryrun | assets)" ;;
    esac
    emit "$lane" <<< "$line"
}

# wiring WORKFLOW -> 0 when every wired lane has its two jobs: a measure job that runs this script for the lane,
# and a lane job named exactly the lane, gated on the measure job's green/red, that exits 0 only on green. The
# train matches job NAMES, so a renamed job is a lane that reads not_measured forever, and a lane job that ran on
# not_measured would turn it into a pass.
# Every job measures at C (THE LANES): both jobs of a lane check out `inputs.ref || github.sha`, the lane job asserts
# its HEAD in a step named for C before anything else, the cpu read gets C as LANES_REF and the gpu judge its caller
# as LANES_CALLER. A called run's github.sha is the caller's commit, so nothing reads it (or GITHUB_SHA) but as that
# expression's fallback.
wiring() {
    local wf=$1 lane b bad="" n=0 at='\$\{\{ inputs\.ref \|\| github\.sha \}\}'
    [ -f "$wf" ] || { printf '%s is missing' "$wf"; return 1; }
    for lane in $WIRED_LANES; do
        n=$((n + 1))
        grep -qE "^    name: $lane\$" "$wf" || bad="$bad $lane(name)"
        grep -qE "^    if: always\(\) && contains\(fromJSON\('\[\"green\",\"red\"\]'\), needs\.measure-$lane\.outputs\.verdict\)\$" "$wf" || bad="$bad $lane(if)"
        grep -qE "^        run: bash scripts/release/release_lanes.sh measure $lane( |\$)" "$wf" || bad="$bad $lane(measure)"
        b="$(job_block "$wf" "$lane")"
        grep -qE "^          ref: $at\$" <<< "$b" || bad="$bad $lane(checkout-at-C)"
        { grep -qE "^      - name: Assert HEAD is $at\$" <<< "$b" && grep -qE "^          REF: $at\$" <<< "$b" \
            && grep -qE '^        run: bash scripts/release/release_lanes\.sh assert-head "\$REF"$' <<< "$b"; } || bad="$bad $lane(assert-head)"
        grep -qE "^          ref: $at\$" <<< "$(job_block "$wf" "measure-$lane")" || bad="$bad measure-$lane(checkout-at-C)"
    done
    [ "$(grep -cE '^      - uses: actions/checkout@' "$wf")" -eq "$(grep -cE "^          ref: $at\$" "$wf")" ] || bad="$bad checkout-at-C"
    grep -qE "^      ref: $at\$" <<< "$(job_block "$wf" b2-gpu-at-c)" || bad="$bad b2-gpu-at-c(ref)"
    grep -qE "^          LANES_REF: $at\$" <<< "$(job_block "$wf" measure-cleanroom-cpu)" || bad="$bad cleanroom-cpu(LANES_REF)"
    grep -qE '^          LANES_CALLER: \$\{\{ inputs\.caller \}\}$' <<< "$(job_block "$wf" measure-cleanroom-gpu)" || bad="$bad cleanroom-gpu(LANES_CALLER)"
    ! grep -qE 'github\.sha|GITHUB_SHA' <<< "$(sed -E "/^ *#/d; s/$at//g" "$wf")" || bad="$bad caller-sha"
    [ "$(grep -cE '^      - run: test "\$VERDICT" = green$' "$wf")" -eq "$n" ] || bad="$bad exit(green-only)"
    if [ -n "$bad" ]; then printf 'miswired:%s' "$bad"; return 1; fi
    printf '%s lanes' "$n"
}

# job_block WORKFLOW JOB -> the lines of job JOB, from `  JOB:` to the next job
job_block() {
    awk -v j="  $2:" '$0 == j { f = 1; next } f && /^  [^ #]/ { exit } f' "$1"
}

# ---------------------------------------------------------------- the case table ---------------------------------
# fx_job NAME STATUS CONCLUSION [STEP:CONCLUSION]... -> one job object ("-" conclusion = null, still running)
fx_job() {
    local name=$1 st=$2 con=$3 s steps='[]'; shift 3
    for s in "$@"; do steps="$(jq -c --arg n "${s%:*}" --arg c "${s##*:}" '. + [{name: $n, conclusion: $c}]' <<< "$steps")"; done
    jq -cn --arg n "$name" --arg s "$st" --arg c "$con" --argjson st "$steps" \
        '{name: $n, status: $s, conclusion: (if $c == "-" then null else $c end), steps: $st}'
}
fx_run() { jq -cs '{jobs: .}'; }
# fxgit ARG... -> git with no inherited repository: a self-test run from a hook must never touch the real one
fxgit() { env -u GIT_DIR -u GIT_WORK_TREE -u GIT_INDEX_FILE -u GIT_COMMON_DIR -u GIT_OBJECT_DIRECTORY git "$@"; }

# rl_rows_b1: the rehearsal's rows (B1 Q1, Q3). The measured-commits reader against a fixture gh, the lane job's HEAD
# assert against a fixture repo, and the caller prefix. Called by self_test, inside it (its ok, broke, tmp, c, d, g, gt).
rl_rows_b1() {
    local e h fg rc want
    exact() {   # exact NAME WANT-RC WANT-STDOUT -- CMD... : CMD exits WANT-RC and prints exactly WANT-STDOUT
        local name=$1 wrc=$2 wout=$3; shift 4
        o="$("$@" 2>/dev/null)"; rc=$?
        if [ "$rc" != "$wrc" ]; then broke "$name" "want exit $wrc, got $rc: $(printf '%s' "$o" | tr '\t\n' '^|')"
        elif [ "$o" != "$wout" ]; then broke "$name" "printed: $(printf '%s' "$o" | tr '\t\n' '^|')"
        else ok "$name" "exit $rc"; fi
    }
    fgh() {     # fgh DIR PATH STATUS [BODY] : the fixture gh answers PATH with STATUS, and BODY on a 200
        local k; k="$1/$(printf '%s' "$2" | tr -c 'A-Za-z0-9._-' '_')"
        printf '%s\n' "$3" > "$k.status"; printf '%s\n' "${4:-}" > "$k.body"
    }
    infra() {   # infra DIR RUN JOB STATUS CONCLUSION : run RUN's clean-room (aprender) job JOB, its commit assert passed
        fgh "$1" "repos/$INFRA_REPO/actions/runs/$2/jobs?per_page=100" 200 "$(jq -cn --argjson id "$3" --arg s "$4" --arg c "$5" \
            --arg n "$INFRA_JOB" --arg a "$INFRA_ASSERT" '{jobs: [{id: $id, name: $n, status: $s, conclusion: (if $c == "-" then null else $c end),
             steps: [{name: $a, status: "completed", conclusion: "success"}]}]}')"
    }
    runs() { fgh "$1" "repos/$INFRA_REPO/actions/workflows/$INFRA_WORKFLOW/runs?per_page=$INFRA_RUNS" "$2" "${3:-}"; }
    logs() { fgh "$1" "repos/$INFRA_REPO/actions/jobs/$2/logs" "$3" "${4:+2026-10-08T02:28:41Z     tested-sha: $4}"; }
    cpu() {     # cpu FIXTURE CACHE : measured-cpu against the fixture gh
        env -u GITHUB_OUTPUT -u GH_TOKEN PATH="$tmp/ghbin:$PATH" FGH="$1" INFRA_TOKEN=fixture-infra-token bash "$SCRIPT_PATH" measured-cpu --cache "$2"
    }
    fx_get() {  # fx_get PATH : infra_rows' getter over FGH_DIR, as a read that fails on any status but 200
        local k; k="$FGH_DIR/$(printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_')"
        [ "$(cat -- "$k.status" 2>/dev/null)" = 200 ] && cat -- "$k.body"
    }
    in_dir() { (cd -- "$1" && env -u GIT_DIR -u GIT_WORK_TREE -u GIT_INDEX_FILE -u GIT_COMMON_DIR "${@:2}"); }
    mkdir -p "$tmp/ghbin" "$tmp/fa" "$tmp/fb" "$tmp/fc" "$tmp/jobsbin"
    cat > "$tmp/ghbin/gh" <<'FAKE'
#!/usr/bin/env bash
# the self-test's gh, for INFRA_TOKEN only. `api rate_limit` prints $FGH/rate. `api -i [-H 'If-None-Match: E'] PATH`
# answers $FGH/<key>.status (404 when absent), and on a 200 the Etag "<key>" and $FGH/<key>.body; a request whose E is
# that Etag gets a 304, and every request does while $FGH/all304 exists. Each request is logged to $FGH/calls.
[ "${GH_TOKEN:-}" = fixture-infra-token ] || exit 4
[ "${1:-}" = api ] || exit 9
shift
if [ "${1:-}" = rate_limit ]; then [ -f "$FGH/rate" ] || exit 1; cat -- "$FGH/rate"; exit 0; fi
inm="" path=""
while [ $# -gt 0 ]; do
    case $1 in
        -i) shift ;;
        -H) case $2 in If-None-Match:*) inm=${2#If-None-Match: } ;; esac; shift 2 ;;
        *) path=$1; shift ;;
    esac
done
k=$(printf '%s' "$path" | tr -c 'A-Za-z0-9._-' '_')
printf '%s %s\n' "$k" "${inm:--}" >> "$FGH/calls"
st=$(cat -- "$FGH/$k.status" 2>/dev/null) || st=404
if [ -f "$FGH/all304" ] || { [ "$st" = 200 ] && [ "$inm" = "\"$k\"" ]; }; then printf 'HTTP/2.0 304 Not Modified\r\n\r\n'; exit 1; fi
printf 'HTTP/2.0 %s X\r\nEtag: "%s"\r\nContent-Type: application/json\r\n\r\n' "$st" "$k"
[ "$st" = 200 ] || exit 1
cat -- "$FGH/$k.body"
FAKE
    chmod +x "$tmp/ghbin/gh"
    # fixture a: four runs, newest first. 40 still running (no log), 30 tested d and failed, 25 tested e and was cancelled,
    # 20 tested c and passed at attempt 1. The measured commits are d (red) and c (green); e and the running one are not.
    e=1111111111111111111111111111111111111111
    fg="$tmp/fa"
    printf '5000 4000\n' > "$fg/rate"
    runs "$fg" 200 '{"workflow_runs":[{"id":40,"run_attempt":1},{"id":30,"run_attempt":1},{"id":25,"run_attempt":1},{"id":20,"run_attempt":1}]}'
    infra "$fg" 40 400 in_progress -; infra "$fg" 30 300 completed failure; infra "$fg" 25 250 completed cancelled; infra "$fg" 20 200 completed success
    logs "$fg" 300 200 "$d"; logs "$fg" 250 200 "$e"; logs "$fg" 200 200 "$c"
    want="$(tsv "$d" red 30; tsv "$c" green 20)"
    exact measured_cpu_lists_each_measured_commit_newest_first 0 "$want" -- cpu "$fg" "$tmp/ca"
    : > "$fg/calls"
    exact measured_cpu_second_read_is_the_same_read 0 "$want" -- cpu "$fg" "$tmp/ca"
    if [ "$(grep -c '' "$fg/calls")" -gt 0 ] && ! grep -q ' -$' "$fg/calls"; then ok measured_cpu_resends_every_cached_etag "$(grep -c '' "$fg/calls") reads"
    else broke measured_cpu_resends_every_cached_etag "calls: $(tr '\n' '|' < "$fg/calls")"; fi
    : > "$fg/all304"
    exact measured_cpu_a_304_with_nothing_cached_is_not_measured 2 "" -- cpu "$fg" "$tmp/cb"
    rm -f -- "$fg/all304"
    runs "$fg" 403
    exact measured_cpu_a_refused_read_is_not_measured_cache_or_not 2 "" -- cpu "$fg" "$tmp/ca"
    runs "$fg" 200 '{"workflow_runs":[{"id":40,"run_attempt":1},{"id":30,"run_attempt":1},{"id":25,"run_attempt":1},{"id":20,"run_attempt":1}]}'
    logs "$fg" 250 502
    exact measured_cpu_a_failed_log_read_is_not_measured 2 "" -- cpu "$fg" "$tmp/cc"
    if FGH_DIR=$fg infra_rows "$c" fx_get > /dev/null; then broke infra_rows_a_failed_log_read_fails_the_read "read as a run that tested nothing"
    else ok infra_rows_a_failed_log_read_fails_the_read "rc 1"; fi
    logs "$fg" 250 200 "$e"
    o="$(FGH_DIR=$fg infra_rows "$d" fx_get | cut -f1 | tr '\n' ' ')"
    if [ "$o" = "40 30 " ]; then ok infra_rows_stops_at_the_run_on_c "$o"; else broke infra_rows_stops_at_the_run_on_c "rows: $o"; fi
    infra "$fg" 25 250 completed cancelled; fgh "$fg" "repos/$INFRA_REPO/actions/runs/25/jobs?per_page=100" 500
    exact measured_cpu_a_failed_jobs_read_is_not_measured 2 "" -- cpu "$fg" "$tmp/cd"
    infra "$fg" 25 250 completed cancelled
    printf '5000 999\n' > "$fg/rate"
    exact measured_cpu_under_the_floor_is_not_measured 2 "" -- cpu "$fg" "$tmp/ce"
    printf '15000 1000\n' > "$fg/rate"
    exact measured_cpu_floor_is_capped_at_1000 0 "$want" -- cpu "$fg" "$tmp/cm"
    printf '1000 250\n' > "$fg/rate"
    exact measured_cpu_floor_is_a_fifth_of_a_small_limit 0 "$want" -- cpu "$fg" "$tmp/cf"
    printf '1000 199\n' > "$fg/rate"
    exact measured_cpu_floor_holds_on_a_small_limit 2 "" -- cpu "$fg" "$tmp/cg"
    rm -f -- "$fg/rate"
    exact measured_cpu_unreadable_rate_limit_is_not_measured 2 "" -- cpu "$fg" "$tmp/ch"
    printf '5000 4000\n' > "$fg/rate"
    exact measured_cpu_without_a_token_is_not_measured 2 "" -- env -u INFRA_TOKEN PATH="$tmp/ghbin:$PATH" FGH="$fg" bash "$SCRIPT_PATH" measured-cpu --cache "$tmp/ci"
    exact measured_cpu_needs_its_cache 3 "" -- bash "$SCRIPT_PATH" measured-cpu
    # fixture b: c tested twice, the newest run (green) decides and c is listed once
    printf '5000 4000\n' > "$tmp/fb/rate"
    runs "$tmp/fb" 200 '{"workflow_runs":[{"id":30,"run_attempt":1},{"id":20,"run_attempt":1}]}'
    infra "$tmp/fb" 30 300 completed success; infra "$tmp/fb" 20 200 completed failure
    logs "$tmp/fb" 300 200 "$c"; logs "$tmp/fb" 200 200 "$c"
    exact measured_cpu_lists_a_commit_once_as_its_newest_run_judged_it 0 "$(tsv "$c" green 30)" -- cpu "$tmp/fb" "$tmp/cj"
    # fixture c: an empty run list is a read that found none; a body that is no run list is a failed read
    printf '5000 4000\n' > "$tmp/fc/rate"
    runs "$tmp/fc" 200 '{"workflow_runs":[]}'
    exact measured_cpu_no_runs_is_a_measured_none 0 "" -- cpu "$tmp/fc" "$tmp/ck"
    runs "$tmp/fc" 200 '{"message":"Not Found"}'
    exact measured_cpu_a_body_that_is_no_run_list_is_not_measured 2 "" -- cpu "$tmp/fc" "$tmp/cl"
    # assert-head, in a fixture repo (one empty commit, made without a hook) and in a repo with no commit
    fxgit init -q "$tmp/repo" && fxgit init -q "$tmp/empty"
    h="$(fxgit -C "$tmp/repo" mktree < /dev/null)" \
        && h="$(GIT_AUTHOR_NAME=fixture GIT_AUTHOR_EMAIL=fixture@invalid GIT_COMMITTER_NAME=fixture GIT_COMMITTER_EMAIL=fixture@invalid \
                fxgit -C "$tmp/repo" commit-tree "$h" -m fixture)" && fxgit -C "$tmp/repo" update-ref HEAD "$h" || h=""
    exact assert_head_passes_on_its_own_commit 0 "tested-sha: $h" -- in_dir "$tmp/repo" bash "$SCRIPT_PATH" assert-head "$h"
    exact assert_head_fails_on_another_commit 1 "" -- in_dir "$tmp/repo" bash "$SCRIPT_PATH" assert-head "$c"
    exact assert_head_refuses_a_ref_name 1 "" -- in_dir "$tmp/repo" bash "$SCRIPT_PATH" assert-head HEAD
    says assert_head_names_a_ref_that_is_not_a_sha "is not a 40-hex sha" -- in_dir "$tmp/repo" bash "$SCRIPT_PATH" assert-head "${h^^}"
    exact assert_head_fails_with_no_commit 1 "" -- in_dir "$tmp/empty" bash "$SCRIPT_PATH" assert-head "$c"
    # the caller prefix: a called run's jobs are `<caller> / b2-gpu at C / ...`, and only LANES_CALLER finds them
    o="$(LANES_CALLER=lanes called "$GPU_PREFIX")"
    if [ "$o" = "lanes / $GPU_PREFIX" ]; then ok called_puts_the_caller_job_first "$o"; else broke called_puts_the_caller_job_first "got: $o"; fi
    cat > "$tmp/jobsbin/curl" <<'FAKE'
#!/usr/bin/env bash
# the self-test's curl for a run's job list: prints $FCURL, whatever was asked
cat -- "$FCURL"
FAKE
    chmod +x "$tmp/jobsbin/curl"
    fx_job "lanes / $g" completed success "$gt:success" | fx_run > "$tmp/called.json"
    gpu() { env -u GITHUB_OUTPUT -u GITHUB_STEP_SUMMARY -u LANES_CALLER PATH="$tmp/jobsbin:$PATH" FCURL="$tmp/called.json" GH_TOKEN=fixture \
                GITHUB_REPOSITORY=o/r GITHUB_RUN_ID=1 "$@" bash "$SCRIPT_PATH" measure cleanroom-gpu; }
    says measure_judges_the_callers_called_jobs "cleanroom-gpu: green" -- gpu LANES_CALLER=lanes
    says measure_without_its_caller_reads_no_called_job "no job named $GPU_PREFIX / ..." -- gpu
    says measure_refuses_a_caller_that_is_not_a_job_name "caller error" -- gpu "LANES_CALLER=lanes / x"
    says measure_cpu_reads_c_from_lanes_ref "cleanroom-cpu: not_measured — no infra read token" -- \
        env -u INFRA_TOKEN -u GITHUB_OUTPUT -u GITHUB_STEP_SUMMARY LANES_REF="$c" bash "$SCRIPT_PATH" measure cleanroom-cpu
    says measure_cpu_refuses_a_ref_that_is_not_a_sha "caller error" -- env -u INFRA_TOKEN LANES_REF=main bash "$SCRIPT_PATH" measure cleanroom-cpu
    says measure_cpu_never_reads_the_callers_sha "caller error" -- env -u INFRA_TOKEN -u LANES_REF GITHUB_SHA="$c" bash "$SCRIPT_PATH" measure cleanroom-cpu
}

self_test() {
    local pass=0 fail=0 o tmp c d g a gt st as
    tmp="$(mktemp -d)" || return 1
    ok() { printf '  ok    %-46s %s\n' "$1" "$2"; pass=$((pass + 1)); }
    broke() { printf '  BROKE %-46s %s\n' "$1" "$2"; fail=$((fail + 1)); }
    row() {   # row NAME WANT-VERDICT NEEDLE -- CMD... : the judge line's first field is WANT and it says NEEDLE
        local name=$1 want=$2 needle=$3; shift 4
        o="$("$@" 2>&1)"
        if [ "${o%%"$TAB"*}" != "$want" ]; then broke "$name" "want $want, got: $o"; return 0; fi
        case "$o" in *"$needle"*) ok "$name" "$want" ;; *) broke "$name" "never said: $needle (got: $o)" ;; esac
    }
    says() {  # says NAME NEEDLE -- CMD... : the printed output carries NEEDLE
        local name=$1 needle=$2; shift 3
        o="$("$@" 2>&1)"
        case "$o" in *"$needle"*) ok "$name" "$needle" ;; *) broke "$name" "never said: $needle (got: $o)" ;; esac
    }
    refused() {   # refused NAME SED-EXPR : wiring() refuses the workflow with SED-EXPR applied
        sed -e "$2" "$WORKFLOW" > "$tmp/wf.yml" 2>/dev/null
        if cmp -s "$WORKFLOW" "$tmp/wf.yml"; then broke "$1" "the fixture edit changed nothing"
        elif o="$(wiring "$tmp/wf.yml")"; then broke "$1" "accepted: $o"
        else ok "$1" refused; fi
    }
    on() { printf '%s\n' "$1" | "${@:2}"; }
    echo "$PROG self-test: case table"
    g="b2-gpu at C / b2-gpu (aprender-gpu, yoga sm_89)"
    a="binary-release dry at C"
    gt='cargo test -p aprender-gpu --lib --features cuda'
    # cleanroom-gpu
    d="$(fx_job "$g" completed success 'Set up job:success' "$gt:success" | fx_run)"
    row gpu_all_success_is_green green "succeeded" -- on "$d" judge_jobs "$GPU_PREFIX" "$GPU_ENV"
    d="$(fx_job "$g" completed failure 'A CUDA device is visible:success' "$gt:failure" | fx_run)"
    row gpu_test_step_failure_is_red red "$gt" -- on "$d" judge_jobs "$GPU_PREFIX" "$GPU_ENV"
    d="$(fx_job "$g" completed failure 'A CUDA device is visible:success' "$gt:cancelled" | fx_run)"
    row gpu_timeout_inside_the_tests_is_red red "$gt" -- on "$d" judge_jobs "$GPU_PREFIX" "$GPU_ENV"
    d="$(fx_job "$g" completed failure 'A CUDA device is visible:failure' "$gt:skipped" | fx_run)"
    row gpu_no_cuda_device_is_not_measured not_measured "precondition" -- on "$d" judge_jobs "$GPU_PREFIX" "$GPU_ENV"
    d="$(fx_job "$g" completed failure 'Assert the commit under test:failure' | fx_run)"
    row gpu_other_commit_is_not_measured not_measured "Assert the commit" -- on "$d" judge_jobs "$GPU_PREFIX" "$GPU_ENV"
    d="$(fx_job "$g" completed failure 'Run actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1:failure' | fx_run)"
    row gpu_checkout_failure_is_not_measured not_measured "precondition" -- on "$d" judge_jobs "$GPU_PREFIX" "$GPU_ENV"
    d="$(fx_job "$g" completed failure 'Set up runner:failure' | fx_run)"
    row gpu_runner_setup_failure_is_not_measured not_measured "Set up runner" -- on "$d" judge_jobs "$GPU_PREFIX" "$GPU_ENV"
    d="$(fx_job "$g" completed cancelled "$gt:cancelled" | fx_run)"
    row gpu_cancelled_is_not_measured not_measured "ended cancelled" -- on "$d" judge_jobs "$GPU_PREFIX" "$GPU_ENV"
    d="$(fx_job "$g" completed skipped | fx_run)"
    row gpu_skipped_is_not_measured not_measured "ended skipped" -- on "$d" judge_jobs "$GPU_PREFIX" "$GPU_ENV"
    d="$(fx_job "$g" completed failure 'Set up job:success' | fx_run)"
    row gpu_failure_without_a_step_is_not_measured not_measured "no failed step" -- on "$d" judge_jobs "$GPU_PREFIX" "$GPU_ENV"
    d="$(fx_job "$g" in_progress - | fx_run)"
    row gpu_in_progress_is_not_measured not_measured "still in_progress" -- on "$d" judge_jobs "$GPU_PREFIX" "$GPU_ENV"
    d="$(fx_job "cleanroom-gpu" completed success "$gt:success" | fx_run)"
    row gpu_no_called_job_is_not_measured not_measured "no job named" -- on "$d" judge_jobs "$GPU_PREFIX" "$GPU_ENV"
    row gpu_unparsable_list_is_not_measured not_measured "did not parse" -- on "not json" judge_jobs "$GPU_PREFIX" "$GPU_ENV"
    d="$( { fx_job "$g" completed success "$gt:success"
            fx_job "b2-gpu at Cx / b2-gpu (aprender-gpu, yoga sm_89)" completed failure "$gt:failure"; } | fx_run)"
    row gpu_judges_only_its_own_caller_job green "all 1 job(s)" -- on "$d" judge_jobs "$GPU_PREFIX" "$GPU_ENV"
    # several called jobs (the assets shape): one red outranks a precondition, and the rest are counted
    d="$( { fx_job "$a / pv x86_64-unknown-linux-gnu on yoga" completed success 'Package archive:success'
            fx_job "$a / smoke apr (cpu) on yoga" completed success 'Download, verify checksum, run:success'; } | fx_run)"
    row multi_all_success_is_green green "all 2 job(s)" -- on "$d" judge_jobs "$ASSETS_PREFIX" "$ASSETS_ENV"
    d="$( { fx_job "$a / pv x86_64-unknown-linux-gnu on yoga" completed failure 'Preflight — this box has the tools this job assumes:failure'
            fx_job "$a / smoke apr (cpu) on yoga" completed skipped; } | fx_run)"
    row multi_preflight_env_is_not_measured not_measured "precondition" -- on "$d" judge_jobs "$ASSETS_PREFIX" "$ASSETS_ENV"
    d="$( { fx_job "$a / pv x86_64-unknown-linux-gnu on yoga" completed failure 'Preflight - this box has the tools this job assumes:failure'
            fx_job "$a / apr (cpu) x86_64 on yoga" completed failure 'Build apr:failure'; } | fx_run)"
    row multi_red_outranks_precondition red "Build apr" -- on "$d" judge_jobs "$ASSETS_PREFIX" "$ASSETS_ENV"
    d="$( { fx_job "$a / apr (cpu) x86_64 on yoga" completed timed_out 'Build apr:cancelled'
            fx_job "$a / apr (cuda) x86_64 on yoga" completed failure 'Prove the cuda feature is in the artifact:failure'; } | fx_run)"
    row multi_counts_more_reds red "+1 more" -- on "$d" judge_jobs "$ASSETS_PREFIX" "$ASSETS_ENV"
    d="$( { fx_job "$a / pv x86_64-unknown-linux-gnu on yoga" completed success
            fx_job "$a / smoke apr (cpu) on yoga" completed skipped; } | fx_run)"
    row multi_skipped_job_is_not_measured not_measured "ended skipped" -- on "$d" judge_jobs "$ASSETS_PREFIX" "$ASSETS_ENV"
    # cleanroom-cpu
    c=0123456789abcdef0123456789abcdef01234567
    d=fedcba9876543210fedcba9876543210fedcba98
    row cpu_success_attempt_1_is_green green "attempt 1" -- on "$(tsv 9 1 completed success "$c")" judge_cpu "$c"
    row cpu_retried_green_is_red red "attempt 2" -- on "$(tsv 9 2 completed success "$c")" judge_cpu "$c"
    row cpu_failure_is_red red "failure" -- on "$(tsv 9 1 completed failure "$c")" judge_cpu "$c"
    row cpu_timed_out_is_red red "timed_out" -- on "$(tsv 9 1 completed timed_out "$c")" judge_cpu "$c"
    row cpu_startup_failure_is_red red "startup_failure" -- on "$(tsv 9 1 completed startup_failure "$c")" judge_cpu "$c"
    row cpu_cancelled_is_not_measured not_measured "ended cancelled" -- on "$(tsv 9 1 completed cancelled "$c")" judge_cpu "$c"
    row cpu_other_sha_is_not_measured not_measured "newest tested ${d:0:12}" -- on "$(tsv 9 1 completed success "$d")" judge_cpu "$c"
    row cpu_running_sweep_is_not_measured not_measured "still running" -- on "$(tsv 10 1 in_progress - '?'; tsv 9 1 completed success "$d")" judge_cpu "$c"
    row cpu_newest_row_on_c_decides red "run 10" -- on "$(tsv 10 1 completed failure "$c"; tsv 9 1 completed success "$c")" judge_cpu "$c"
    row cpu_older_run_on_c_is_read green "run 9" -- on "$(tsv 10 1 completed success "$d"; tsv 9 1 completed success "$c")" judge_cpu "$c"
    row cpu_no_rows_is_not_measured not_measured "no finished" -- on "" judge_cpu "$c"
    row cpu_no_token_is_not_measured not_measured "INFRA_ACTIONS_READ" -- env -u INFRA_TOKEN bash "$SCRIPT_PATH" judge-cpu-measure "$c"
    o="$(printf '2026-10-04T02:28:41Z     repo:       aprender\n2026-10-04T02:28:41Z     tested-sha: %s\r\n' "$c" | tested_sha)"
    if [ "$o" = "$c" ]; then ok tested_sha_is_read_from_the_log "${c:0:12}"; else broke tested_sha_is_read_from_the_log "got: $o"; fi
    o="$(printf '2026-10-04T02:28:41Z     tested-sha: %s\n2026-10-04T02:28:42Z     tested-sha: %s\n' "$c" "$c" | tested_sha)"
    if [ "$o" = "$c" ]; then ok tested_sha_repeated_record_is_one "${c:0:12}"; else broke tested_sha_repeated_record_is_one "got: $o"; fi
    o="$(printf '2026-10-04T02:28:41Z     tested-sha: %s\n2026-10-04T02:28:42Z     tested-sha: %s\n' "$c" "$d" | tested_sha)"
    if [ -z "$o" ]; then ok tested_sha_two_commits_prove_none empty; else broke tested_sha_two_commits_prove_none "got: $o"; fi
    o="$(printf '2026-10-04T02:28:41Z     tested-sha: %s\n' "${c:0:12}" | tested_sha)"
    if [ -z "$o" ]; then ok tested_sha_short_sha_proves_none empty; else broke tested_sha_short_sha_proves_none "got: $o"; fi
    o="$(printf '2026-10-04T02:28:41Z     commit:  %s\n2026-10-04T02:28:41Z     tested-sha: %s\n' "${c:0:9}" "$c" | tested_sha)"
    if [ "$o" = "$c" ]; then ok tested_sha_agreeing_legacy_record "${c:0:12}"; else broke tested_sha_agreeing_legacy_record "got: $o"; fi
    o="$(printf '2026-10-04T02:28:41Z     commit:  %s\n2026-10-04T02:28:41Z     tested-sha: %s\n' "${d:0:9}" "$c" | tested_sha)"
    if [ -z "$o" ]; then ok tested_sha_disagreeing_legacy_record_proves_none empty; else broke tested_sha_disagreeing_legacy_record_proves_none "got: $o"; fi
    o="$(printf '2026-10-04T02:28:41Z     commit:  %s\n2026-10-04T02:28:42Z     commit:  %s\n2026-10-04T02:28:42Z     tested-sha: %s\n' "${c:0:9}" "${d:0:9}" "$c" | tested_sha)"
    if [ -z "$o" ]; then ok tested_sha_two_legacy_records_prove_none empty; else broke tested_sha_two_legacy_records_prove_none "got: $o"; fi
    o="$(printf '2026-10-04T02:28:41Z     commit:  %s\n' "${c:0:9}" | tested_sha)"
    if [ -z "$o" ]; then ok tested_sha_legacy_record_alone_proves_none empty; else broke tested_sha_legacy_record_alone_proves_none "got: $o"; fi
    # infra_rows' seams: which job, its commit assert, and whether its log may name a commit
    o="$(jq -cn --arg a "$INFRA_ASSERT" '{jobs: [{id: 7, name: "clean-room (trueno)", status: "completed", conclusion: "success", steps: []},
        {id: 9, name: "clean-room (aprender)", status: "completed", conclusion: "failure",
         steps: [{name: $a, status: "completed", conclusion: "success"}]}]}' | infra_job)"
    if [ "$o" = "9 completed failure completed/success" ]; then ok infra_job_reads_its_job_and_assert "$o"; else broke infra_job_reads_its_job_and_assert "got: $o"; fi
    o="$(jq -cn '{jobs: [{id: 9, name: "clean-room (aprender)", status: "in_progress", conclusion: null, steps: []}]}' | infra_job)"
    if [ "$o" = "9 in_progress - absent/-" ]; then ok infra_job_missing_assert_is_absent "$o"; else broke infra_job_missing_assert_is_absent "got: $o"; fi
    if log_proves completed completed/success; then ok log_proves_after_a_passed_assert yes; else broke log_proves_after_a_passed_assert no; fi
    for o in "completed completed/failure" "completed completed/skipped" "completed absent/-" "in_progress completed/success"; do
        read -r st as <<< "$o"
        if log_proves "$st" "$as"; then broke log_proves_nothing_without_a_passed_assert "accepted: $o"; else ok log_proves_nothing_without_a_passed_assert "$o"; fi
    done
    mkdir -p "$tmp/bin"
    cat > "$tmp/bin/curl" <<'FAKE'
#!/usr/bin/env bash
# the self-test's curl: prints the headers it was handed by file, then its argv
for a in "$@"; do case "$a" in @*) printf 'HDR %s\n' "$(cat "${a#@}")" ;; esac; done
printf 'ARGV %s\n' "$*"
FAKE
    chmod +x "$tmp/bin/curl"
    o="$(PATH="$tmp/bin:$PATH" api https://example.invalid/x fixture-token-41)"
    if [[ "$o" == *"HDR Authorization: Bearer fixture-token-41"* ]] && ! grep -q '^ARGV .*fixture-token-41' <<< "$o"; then
        ok api_keeps_the_token_out_of_argv "header by fd"
    else broke api_keeps_the_token_out_of_argv "got: $o"; fi
    # publish-dryrun
    row dryrun_green_on_a_bumped_tree green "green at 0.70.2" -- judge_dryrun 0 0.70.2 0.70.1
    row dryrun_defect_is_red red "publish defect" -- judge_dryrun 1 0.70.2 0.70.1
    row dryrun_gate_could_not_measure not_measured "exit 2" -- judge_dryrun 2 0.70.2 0.70.1
    row dryrun_unbumped_main_is_not_measured not_measured "must be bumped" -- judge_dryrun 0 0.70.0 0.70.1
    row dryrun_same_version_is_not_measured not_measured "must be bumped" -- judge_dryrun 0 0.70.1 0.70.1
    row dryrun_compares_versions_not_strings green "0.70.10" -- judge_dryrun 0 0.70.10 0.70.9
    row dryrun_unread_registry_is_not_measured not_measured "index" -- judge_dryrun 0 0.70.2 ""
    row dryrun_not_run_is_not_measured not_measured "not run" -- judge_dryrun - 0.70.2 0.70.1
    row dryrun_prerelease_of_a_published_version_is_not_measured not_measured "must be bumped" -- judge_dryrun 0 0.70.1-rc.1 0.70.1
    row dryrun_prerelease_of_the_next_version_is_measured green "green at 0.70.2-rc.1" -- judge_dryrun 0 0.70.2-rc.1 0.70.1
    row dryrun_build_metadata_is_not_a_bump not_measured "must be bumped" -- judge_dryrun 0 0.70.1+build.5 0.70.1
    o="$(printf '%s\n' '{"vers":"0.70.0","yanked":false}' '{"vers":"0.70.2","yanked":true}' '{"vers":"0.70.1","yanked":false}' | registry_newest)"
    if [ "$o" = 0.70.1 ]; then ok registry_newest_skips_a_yanked_version 0.70.1; else broke registry_newest_skips_a_yanked_version "got: $o"; fi
    # emit: the only door to the outputs
    says emit_prints_the_verdict "cleanroom-gpu: green — why" -- on "$(tsv green why)" emit cleanroom-gpu
    says emit_turns_nonsense_into_not_measured "assets: not_measured — the judge printed no verdict" -- on "$(tsv pass 'looks fine')" emit assets
    says emit_turns_silence_into_not_measured "assets: not_measured — the judge printed no verdict" -- on "" emit assets
    tsv red defect | GITHUB_OUTPUT="$tmp/out" emit publish-dryrun > /dev/null
    if grep -qx 'verdict=red' "$tmp/out" 2>/dev/null && grep -qx 'reason=defect' "$tmp/out"; then ok emit_fills_github_output verdict=red
    else broke emit_fills_github_output "wrote: $(cat "$tmp/out" 2>/dev/null)"; fi
    # a reason carrying a CR must not reach GITHUB_OUTPUT as a second record
    tsv red "$(printf 'defect\rverdict=green')" | GITHUB_OUTPUT="$tmp/out2" emit publish-dryrun > /dev/null
    if [ "$(grep -c $'\r' "$tmp/out2" 2>/dev/null)" = 0 ] && [ "$(grep -c '' "$tmp/out2")" = 2 ] && grep -qx 'verdict=red' "$tmp/out2"; then
        ok emit_strips_a_cr_from_the_reason "2 records"
    else broke emit_strips_a_cr_from_the_reason "wrote: $(tr '\r' '^' < "$tmp/out2" 2>/dev/null)"; fi
    says measure_refuses_an_unknown_lane "caller error" -- bash "$SCRIPT_PATH" measure preflight
    # the wiring of the real workflow, and three miswirings it must refuse
    if o="$(wiring "$WORKFLOW")"; then ok workflow_wiring "$o"; else broke workflow_wiring "$o"; fi
    refused wiring_refuses_a_renamed_lane_job 's/^    name: cleanroom-cpu$/    name: cleanroom cpu/'
    refused wiring_refuses_a_lane_job_run_on_any_verdict 's/^\(    if: always() && \)contains(fromJSON(.\["green","red"\].), \(needs\.measure-publish-dryrun\.outputs\.verdict\))$/\1\2 != '"''"'/'
    refused wiring_refuses_a_lane_job_that_passes_red '0,/^      - run: test "\$VERDICT" = green$/s//      - run: true/'
    # every job at C (B1 Q3): the miswirings that would measure the caller's commit, or a lane job that never proved its own
    refused wiring_refuses_a_lane_job_without_its_head_assert '/^  cleanroom-gpu:$/,/^  [a-z]/s/release_lanes\.sh assert-head "\$REF"$/release_lanes.sh --self-test/'
    refused wiring_refuses_a_lane_job_checked_out_elsewhere '/^  publish-dryrun:$/,$s/^          ref: .*$/          ref: main/'
    refused wiring_refuses_a_measure_job_checked_out_elsewhere '/^  measure-cleanroom-cpu:$/,/^  cleanroom-cpu:$/s/^          ref: .*$/          ref: main/'
    refused wiring_refuses_a_lane_job_with_no_checkout '/^  cleanroom-cpu:$/,/^  [a-z]/{/actions\/checkout@/,/fetch-depth/d}'
    refused wiring_refuses_a_measure_job_with_no_checkout '/^  measure-cleanroom-gpu:$/,/^  [a-z]/{/actions\/checkout@/,/fetch-depth/d}'
    refused wiring_refuses_a_case_table_of_another_commit '/^  case-table:$/,/^  [a-z]/{/^          ref: /d}'
    refused wiring_refuses_the_callers_sha 's/^      ref: \${{ inputs\.ref || github\.sha }}$/      ref: ${{ github.sha }}/'
    refused wiring_refuses_b2_gpu_on_a_branch 's/^      ref: \${{ inputs\.ref || github\.sha }}$/      ref: main/'
    refused wiring_refuses_a_worktree_of_the_callers_commit 's/"\$RUNNER_TEMP\/lanes-c" "\$REF"$/"$RUNNER_TEMP\/lanes-c" "$GITHUB_SHA"/'
    refused wiring_refuses_a_cpu_read_without_lanes_ref '/^          LANES_REF: /d'
    refused wiring_refuses_a_gpu_judge_without_its_caller '/^          LANES_CALLER: /d'
    rl_rows_b1
    rm -rf -- "${tmp:?}"
    printf -- '--- %s ok, %s broke ---\n' "$pass" "$fail"
    [ "$fail" -eq 0 ]
}

# the planted mutants: one per line, NAME then a sed expression applied to this file
RL_MUTANTS='m01_precondition_counts_as_red s/select(.step != "" and (.step | test($env) | not))/select(.step != "")/
m02_red_step_counts_as_precondition s/select(.step != "" and (.step | test($env) | not))/select(false)/
m03_in_progress_job_is_judged s/elif any($js\[\]; .status != "completed") then/elif false then/
m04_skipped_job_is_green s/elif any($js\[\]; .conclusion != "success") then/elif false then/
m05_retried_infra_green_is_green s/if ($4 == "success" && $2 == 1)/if ($4 == "success")/
m06_infra_run_on_any_commit s/NF >= 5 && $5 == c { found = 1/NF >= 5 { found = 1/
m07_unbumped_main_is_measured s/    if ! is_bumped "$main" "$reg"; then/    if false; then/
m08_gate_env_is_green s/        0) printf .green\\trc_publish_gate/        0|2) printf '"'"'green\\trc_publish_gate/
m09_emit_passes_any_word s/        green|red|not_measured) ;;/        *) ;;/
m10_timeout_in_tests_is_not_a_failure s/ or .conclusion == "cancelled")/)/
m11_versions_compared_as_strings s/| sort -V | tail -n 1)" = "$m" \]/| sort | tail -n 1)" = "$m" ]/
m12_wiring_ignores_the_lane_gate s/ || bad="$bad $lane(if)"/ || true/
m13_wiring_ignores_green_only_exit s/ || bad="$bad exit(green-only)"/ || true/
m14_wiring_ignores_the_job_name s/ || bad="$bad $lane(name)"/ || true/
m15_prerelease_counts_as_bumped s/%%\[-+\]\*}" r=/}" r=/
m16_yanked_version_counts s/select(.yanked | not) | //
m17_startup_failure_is_not_red s/ || $4 == "startup_failure")/)/
m18_prefix_without_its_separator s/startswith($p + " \/ ")/startswith($p)/
m19_reason_keeps_a_cr s/tr -d .\\r\\n.)"/tr -d "\\n")"/
m20_first_of_two_tested_shas_counts s/| sort -u)"/| sort -u | head -n 1)"/
m21_build_metadata_counts_as_bumped s/%%\[-+\]\*}" r=/%%[-]*}" r=/
m22_log_proves_without_the_assert s/\[ "\$2" = completed\/success \]/true/
m23_two_legacy_records_count s/\(\*\$.\\n.\*)\) return 0 ;;/\1 ;;/
m24_disagreeing_legacy_record_counts s/\*) case "\$s" in "\$a"\*) ;; \*) return 0 ;; esac ;;/*) ;;/
m25_token_in_argv s/-H @<(printf .Authorization: Bearer %s\\n. "\$2")/-H "Authorization: Bearer $2"/
m26_failed_infra_read_is_none s#rows="\$(infra_rows "" infra_gh)" || {#rows="$(infra_rows "" infra_gh)" || true || {#
m27_failed_log_read_is_no_commit s#/logs")" || return 1#/logs")" || log=""#
m28_304_with_no_cache_is_empty s#304) \[ -f "\$k.body" \] || return 1 ;;#304) touch -- "$k.body" ;;#
m29_rate_floor_ignored s#\[ "\$rem" -ge "\$fl" \] || {#true || {#
m30_floor_is_never_a_fifth s#fl=\$((lim / 5)); \[ "\$fl" -le "\$RATE_FLOOR" \] || fl=\$RATE_FLOOR#fl=$RATE_FLOOR#
m31_floor_is_never_capped s#\] || fl=\$RATE_FLOOR#] || true#
m32_unmeasured_commit_listed s#            green|red) tsv "\$sha"#            *) tsv "$sha"#
m33_assert_head_passes_another_commit s#\[ "\$head" = "\$ref" \] || {#true || {#
m34_caller_prefix_dropped s#"\${LANES_CALLER:+\$LANES_CALLER / }" "\$1"#"" "$1"#
m35_wiring_ignores_the_head_assert s# || bad="\$bad \$lane(assert-head)"# || true#
m36_wiring_ignores_a_lane_jobs_checkout s# || bad="\$bad \$lane(checkout-at-C)"# || true#
m37_wiring_ignores_a_measure_jobs_checkout s# || bad="\$bad measure-\$lane(checkout-at-C)"# || true#
m38_wiring_ignores_a_checkout_of_another_commit s# || bad="\$bad checkout-at-C"# || true#
m39_wiring_ignores_the_callers_sha s# || bad="\$bad caller-sha"# || true#
m40_wiring_ignores_b2_gpus_ref s# || bad="\$bad b2-gpu-at-c(ref)"# || true#
m41_wiring_ignores_lanes_ref s# || bad="\$bad cleanroom-cpu(LANES_REF)"# || true#
m42_wiring_ignores_lanes_caller s# || bad="\$bad cleanroom-gpu(LANES_CALLER)"# || true#
m43_gh_read_with_the_ambient_token s#"\$INFRA_TOKEN" gh api -i#"${GH_TOKEN:-}" gh api -i#
m44_any_status_is_a_200 s#        200) awk#        [0-9]*) awk#
m45_commit_listed_once_per_run s#&& !seen\[\$5\]++ { print \$5 }#{ print $5 }#
m46_cached_etag_never_sent s#hdr=(-H "If-None-Match: \$(cat -- "\$k.etag")")#hdr=()#
m47_assert_head_takes_any_word s#\[\[ "\$ref" =~ #[[ -n "$ref" ]] || [[ "$ref" =~ #
m48_lanes_ref_takes_any_word s#"\${LANES_REF:-}" .~ \^\[0-9a-f\]{40}\$ \]\]#-n "${LANES_REF:-}" ]]#
m49_lanes_caller_takes_any_word s#\[\[ "\${LANES_CALLER:-}" .~ \^\[A-Za-z0-9._-\]\*\$ \]\]#true#'

mutants() {
    local tmp pass=0 fail=0 name expr o rc m
    tmp="$(mktemp -d)" || return 1
    mkdir -p "$tmp/scripts/release" "$tmp/.github/workflows"
    cp "$WORKFLOW" "$tmp/.github/workflows/" 2>/dev/null
    m="$tmp/scripts/release/release_lanes.sh"
    cp "$SCRIPT_PATH" "$m"
    if ! bash "$m" --self-test > /dev/null 2>&1; then printf '  BROKE %-46s the unmutated script is not green in the mutant dir\n' baseline; rm -rf -- "${tmp:?}"; return 1; fi
    while read -r name expr; do
        [ -n "$name" ] || continue
        sed -e "$expr" "$SCRIPT_PATH" > "$m"
        if cmp -s "$SCRIPT_PATH" "$m"; then printf '  BROKE %-46s changed nothing: its pattern no longer matches\n' "$name"; fail=$((fail + 1)); continue; fi
        if ! bash -n "$m" 2>/dev/null; then printf '  BROKE %-46s does not parse\n' "$name"; fail=$((fail + 1)); continue; fi
        o="$(bash "$m" --self-test 2>&1)"; rc=$?
        case "$rc:$o" in
            0:*) printf '  BROKE %-46s SURVIVED: the case table stayed green\n' "$name"; fail=$((fail + 1)) ;;
            *"  BROKE "*) printf '  ok    %-46s killed, %s row(s) broke\n' "$name" "$(printf '%s\n' "$o" | grep -c '^  BROKE ')"; pass=$((pass + 1)) ;;
            *) printf '  BROKE %-46s exit %s with no broken row: not a kill\n' "$name" "$rc"; fail=$((fail + 1)) ;;
        esac
    done < <(printf '%s\n' "$RL_MUTANTS")
    printf -- '--- %s/%s mutants killed ---\n' "$pass" "$((pass + fail))"
    rm -rf -- "${tmp:?}"
    [ "$fail" -eq 0 ]
}

case "${1:-}" in
    --self-test) self_test; exit $? ;;
    --mutants) mutants; exit $? ;;
    measure) shift; measure "$@"; exit $? ;;
    judge-cpu-measure) measure_cpu "${2:-}"; exit 0 ;;   # the self-test's door to the no-token row
    measured-cpu)
        [ "${2:-}" = --cache ] && [ -n "${3:-}" ] || caller_error "measured-cpu --cache DIR"
        measured_cpu "$3"; exit $? ;;
    assert-head) assert_head "${2:-}"; exit $? ;;
    -h|--help) sed -n '2,58p' "$SCRIPT_PATH"; exit 0 ;;
    *) caller_error "usage: measure LANE [ROOT] | measured-cpu --cache DIR | assert-head REF | --self-test | --mutants" ;;
esac
