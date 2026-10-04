#!/usr/bin/env bash
# release_lanes.sh — the nightly producers for the clean-room release lanes (#4720, 0702-P3; BLD-002).
#
#   bash scripts/release/release_lanes.sh measure cleanroom-gpu         # judge this run's b2-gpu job
#   bash scripts/release/release_lanes.sh measure cleanroom-cpu         # judge paiml/infra's clean-room run on C
#   bash scripts/release/release_lanes.sh measure publish-dryrun ROOT   # rc_publish_gate.sh --verify ROOT
#   bash scripts/release/release_lanes.sh measure assets                # judge this run's dry binary-release jobs
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
# THE LANES. C is GITHUB_SHA: the head of main when the scheduled run started.
#   cleanroom-gpu   b2-gpu.yml called with ref=C. green: every b2-gpu job succeeded. red: a step that is not a
#                   precondition failed (the cargo test step, a timeout inside it). not_measured: a
#                   precondition failed (checkout, the commit assert, no CUDA device), or the job was
#                   cancelled or skipped.
#   cleanroom-cpu   paiml/infra clean-room.yml, job `clean-room (aprender)`: the newest one whose log names
#                   `tested-sha: C`. green: success at attempt 1. red: failure or timed_out, or success only at
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
# EXIT  measure: 0 when a verdict was written · 3 a caller error. --self-test / --mutants: 0 green, 1 red.
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
INFRA_RUNS=6            # newest infra runs searched for the one that tested C
INFRA_MAX_LOGS=3        # job logs read at most (each is a whole clean-room log)
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
# tested C decides. A row still running has tested_sha "?" (its log is not readable yet).
judge_cpu() {
    awk -F '\t' -v c="$1" '
        NF >= 5 && $5 == c { found = 1
            if ($3 != "completed") { printf "not_measured\tinfra run %s on C is still %s\n", $1, $3; exit }
            if ($4 == "success" && $2 == 1) { printf "green\tinfra run %s tested C: success at attempt 1\n", $1; exit }
            if ($4 == "success") { printf "red\tinfra run %s tested C: success only at attempt %s, a retried green is not a green\n", $1, $2; exit }
            if ($4 == "failure" || $4 == "timed_out" || $4 == "startup_failure") { printf "red\tinfra run %s tested C: %s\n", $1, $4; exit }
            printf "not_measured\tinfra run %s tested C and ended %s\n", $1, $4; exit }
        NF >= 5 && $3 != "completed" && run == "" { run = $1 }
        NF >= 5 && $3 == "completed" && $5 ~ /^[0-9a-f]+$/ && length($5) == 40 && newest == "" { newest = substr($5, 1, 12) }
        END { if (found) exit
            if (run != "") printf "not_measured\tinfra run %s is still running; no finished run tested C yet\n", run
            else if (newest != "") printf "not_measured\tno infra clean-room run tested C; the newest tested %s\n", newest
            else printf "not_measured\tno finished clean-room (aprender) job in the newest infra runs\n" }'
}

# is_bumped MAIN REGISTRY_NEWEST -> 0 when main's version is above everything crates.io has (sort -V, not strings)
is_bumped() {
    [ -n "$1" ] && [ -n "$2" ] && [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | tail -n 1)" = "$1" ]
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

# tested_sha (a clean-room job log on stdin) -> the sha after `repo: aprender`, or nothing
tested_sha() {
    sed 's/\r$//' | awk '$2 == "repo:" && $3 == "aprender" { r = 1; next }
        r && $2 == "tested-sha:" { if ($3 ~ /^[0-9a-f]+$/ && length($3) == 40) print $3; exit }'
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
# host, and curl drops the Authorization header on that hop.
api() {
    curl -fsSL --max-time 120 -H "Authorization: Bearer $2" -H "Accept: application/vnd.github+json" \
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

# infra_rows C -> the TSV judge_cpu reads, newest run first; stops at the first run that tested C
infra_rows() {
    local runs ids id att jobs jid jst jcon sha logs=0
    runs="$(api "$API/repos/$INFRA_REPO/actions/workflows/$INFRA_WORKFLOW/runs?per_page=$INFRA_RUNS" "$INFRA_TOKEN")" || return 1
    ids="$(printf '%s\n' "$runs" | jq -r '.workflow_runs[] | "\(.id) \(.run_attempt)"')" || return 1
    while read -r id att; do
        [ -n "$id" ] || continue
        jobs="$(api "$API/repos/$INFRA_REPO/actions/runs/$id/jobs?per_page=100" "$INFRA_TOKEN")" || return 1
        jid=""; jst=""; jcon=""
        read -r jid jst jcon < <(printf '%s\n' "$jobs" | jq -r --arg n "$INFRA_JOB" \
            '[.jobs[] | select(.name == $n)][0] // empty | "\(.id) \(.status) \(.conclusion // "-")"')
        [ -n "$jid" ] || continue
        sha="?"
        if [ "$jst" = completed ] && [ "$logs" -lt "$INFRA_MAX_LOGS" ]; then
            logs=$((logs + 1))
            # stderr dropped: tested_sha stops reading at the sha, and curl reports the closed pipe
            sha="$(api "$API/repos/$INFRA_REPO/actions/jobs/$jid/logs" "$INFRA_TOKEN" 2>/dev/null | tested_sha)"
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

measure_dryrun() {
    local root=$1 main reg rc=-
    [ -f "$root/Cargo.toml" ] || { printf 'not_measured\t%s has no Cargo.toml\n' "$root"; return; }
    main="$(cargo metadata --no-deps --format-version 1 --manifest-path "$root/Cargo.toml" 2>/dev/null \
        | jq -r '[.packages[] | select(.name == "aprender") | .version][0] // empty')"
    reg="$(curl -fsSL --max-time 60 "$INDEX_URL" 2>/dev/null | jq -r 'select(.yanked | not) | .vers' | sort -V | tail -n 1)"
    if is_bumped "$main" "$reg"; then
        bash "$HERE/rc_publish_gate.sh" --verify "$root" >&2; rc=$?
        # a receipt, never a stop: which crates the tree is ahead of crates.io on
        (cd -- "$root" && timeout 300 bash "$SCRIPTS/cascade-publish.sh" --check) 2>&1 | tail -n 20 >&2
    fi
    judge_dryrun "$rc" "$main" "$reg"
}

measure() {
    local lane=${1:-} line
    case "$lane" in
        cleanroom-gpu) line="$(measure_jobs "$GPU_PREFIX" "$GPU_ENV")" ;;
        assets) line="$(measure_jobs "$ASSETS_PREFIX" "$ASSETS_ENV")" ;;
        cleanroom-cpu)
            [[ "${GITHUB_SHA:-}" =~ ^[0-9a-f]{40}$ ]] || caller_error "GITHUB_SHA is not a 40-hex sha"
            line="$(measure_cpu "$GITHUB_SHA")" ;;
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
wiring() {
    local wf=$1 lane bad="" n=0
    [ -f "$wf" ] || { printf '%s is missing' "$wf"; return 1; }
    for lane in $WIRED_LANES; do
        n=$((n + 1))
        grep -qE "^    name: $lane\$" "$wf" || bad="$bad $lane(name)"
        grep -qE "^    if: always\(\) && contains\(fromJSON\('\[\"green\",\"red\"\]'\), needs\.measure-$lane\.outputs\.verdict\)\$" "$wf" || bad="$bad $lane(if)"
        grep -qE "^        run: bash scripts/release/release_lanes.sh measure $lane( |\$)" "$wf" || bad="$bad $lane(measure)"
    done
    [ "$(grep -cE '^      - run: test "\$VERDICT" = green$' "$wf")" -eq "$n" ] || bad="$bad exit(green-only)"
    if [ -n "$bad" ]; then printf 'miswired:%s' "$bad"; return 1; fi
    printf '%s lanes' "$n"
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

self_test() {
    local pass=0 fail=0 o tmp c d g a T
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
    T='cargo test -p aprender-gpu --lib --features cuda'
    # cleanroom-gpu
    d="$(fx_job "$g" completed success 'Set up job:success' "$T:success" | fx_run)"
    row gpu_all_success_is_green green "succeeded" -- on "$d" judge_jobs "$GPU_PREFIX" "$GPU_ENV"
    d="$(fx_job "$g" completed failure 'A CUDA device is visible:success' "$T:failure" | fx_run)"
    row gpu_test_step_failure_is_red red "$T" -- on "$d" judge_jobs "$GPU_PREFIX" "$GPU_ENV"
    d="$(fx_job "$g" completed failure 'A CUDA device is visible:success' "$T:cancelled" | fx_run)"
    row gpu_timeout_inside_the_tests_is_red red "$T" -- on "$d" judge_jobs "$GPU_PREFIX" "$GPU_ENV"
    d="$(fx_job "$g" completed failure 'A CUDA device is visible:failure' "$T:skipped" | fx_run)"
    row gpu_no_cuda_device_is_not_measured not_measured "precondition" -- on "$d" judge_jobs "$GPU_PREFIX" "$GPU_ENV"
    d="$(fx_job "$g" completed failure 'Assert the commit under test:failure' | fx_run)"
    row gpu_other_commit_is_not_measured not_measured "Assert the commit" -- on "$d" judge_jobs "$GPU_PREFIX" "$GPU_ENV"
    d="$(fx_job "$g" completed failure 'Run actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1:failure' | fx_run)"
    row gpu_checkout_failure_is_not_measured not_measured "precondition" -- on "$d" judge_jobs "$GPU_PREFIX" "$GPU_ENV"
    d="$(fx_job "$g" completed failure 'Set up runner:failure' | fx_run)"
    row gpu_runner_setup_failure_is_not_measured not_measured "Set up runner" -- on "$d" judge_jobs "$GPU_PREFIX" "$GPU_ENV"
    d="$(fx_job "$g" completed cancelled "$T:cancelled" | fx_run)"
    row gpu_cancelled_is_not_measured not_measured "ended cancelled" -- on "$d" judge_jobs "$GPU_PREFIX" "$GPU_ENV"
    d="$(fx_job "$g" completed skipped | fx_run)"
    row gpu_skipped_is_not_measured not_measured "ended skipped" -- on "$d" judge_jobs "$GPU_PREFIX" "$GPU_ENV"
    d="$(fx_job "$g" completed failure 'Set up job:success' | fx_run)"
    row gpu_failure_without_a_step_is_not_measured not_measured "no failed step" -- on "$d" judge_jobs "$GPU_PREFIX" "$GPU_ENV"
    d="$(fx_job "$g" in_progress - | fx_run)"
    row gpu_in_progress_is_not_measured not_measured "still in_progress" -- on "$d" judge_jobs "$GPU_PREFIX" "$GPU_ENV"
    d="$(fx_job "cleanroom-gpu" completed success "$T:success" | fx_run)"
    row gpu_no_called_job_is_not_measured not_measured "no job named" -- on "$d" judge_jobs "$GPU_PREFIX" "$GPU_ENV"
    row gpu_unparsable_list_is_not_measured not_measured "did not parse" -- on "not json" judge_jobs "$GPU_PREFIX" "$GPU_ENV"
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
    row cpu_cancelled_is_not_measured not_measured "ended cancelled" -- on "$(tsv 9 1 completed cancelled "$c")" judge_cpu "$c"
    row cpu_other_sha_is_not_measured not_measured "newest tested ${d:0:12}" -- on "$(tsv 9 1 completed success "$d")" judge_cpu "$c"
    row cpu_running_sweep_is_not_measured not_measured "still running" -- on "$(tsv 10 1 in_progress - '?'; tsv 9 1 completed success "$d")" judge_cpu "$c"
    row cpu_newest_row_on_c_decides red "run 10" -- on "$(tsv 10 1 completed failure "$c"; tsv 9 1 completed success "$c")" judge_cpu "$c"
    row cpu_older_run_on_c_is_read green "run 9" -- on "$(tsv 10 1 completed success "$d"; tsv 9 1 completed success "$c")" judge_cpu "$c"
    row cpu_no_rows_is_not_measured not_measured "no finished" -- on "" judge_cpu "$c"
    row cpu_no_token_is_not_measured not_measured "INFRA_ACTIONS_READ" -- env -u INFRA_TOKEN bash "$SCRIPT_PATH" judge-cpu-measure "$c"
    o="$(printf '2026-10-04T02:28:41Z     repo:       aprender\n2026-10-04T02:28:41Z     tested-sha: %s\r\n' "$c" | tested_sha)"
    if [ "$o" = "$c" ]; then ok tested_sha_is_read_from_the_log "${c:0:12}"; else broke tested_sha_is_read_from_the_log "got: $o"; fi
    o="$(printf '2026-10-04T02:28:41Z     repo:       trueno\n2026-10-04T02:28:41Z     tested-sha: %s\n' "$c" | tested_sha)"
    if [ -z "$o" ]; then ok tested_sha_of_another_repo_is_ignored empty; else broke tested_sha_of_another_repo_is_ignored "got: $o"; fi
    # publish-dryrun
    row dryrun_green_on_a_bumped_tree green "green at 0.70.2" -- judge_dryrun 0 0.70.2 0.70.1
    row dryrun_defect_is_red red "publish defect" -- judge_dryrun 1 0.70.2 0.70.1
    row dryrun_gate_could_not_measure not_measured "exit 2" -- judge_dryrun 2 0.70.2 0.70.1
    row dryrun_unbumped_main_is_not_measured not_measured "must be bumped" -- judge_dryrun 0 0.70.0 0.70.1
    row dryrun_same_version_is_not_measured not_measured "must be bumped" -- judge_dryrun 0 0.70.1 0.70.1
    row dryrun_compares_versions_not_strings green "0.70.10" -- judge_dryrun 0 0.70.10 0.70.9
    row dryrun_unread_registry_is_not_measured not_measured "index" -- judge_dryrun 0 0.70.2 ""
    row dryrun_not_run_is_not_measured not_measured "not run" -- judge_dryrun - 0.70.2 0.70.1
    # emit: the only door to the outputs
    says emit_prints_the_verdict "cleanroom-gpu: green — why" -- on "$(tsv green why)" emit cleanroom-gpu
    says emit_turns_nonsense_into_not_measured "assets: not_measured — the judge printed no verdict" -- on "$(tsv pass 'looks fine')" emit assets
    says emit_turns_silence_into_not_measured "assets: not_measured — the judge printed no verdict" -- on "" emit assets
    tsv red defect | GITHUB_OUTPUT="$tmp/out" emit publish-dryrun > /dev/null
    if grep -qx 'verdict=red' "$tmp/out" 2>/dev/null && grep -qx 'reason=defect' "$tmp/out"; then ok emit_fills_github_output verdict=red
    else broke emit_fills_github_output "wrote: $(cat "$tmp/out" 2>/dev/null)"; fi
    says measure_refuses_an_unknown_lane "caller error" -- bash "$SCRIPT_PATH" measure preflight
    # the wiring of the real workflow, and three miswirings it must refuse
    if o="$(wiring "$WORKFLOW")"; then ok workflow_wiring "$o"; else broke workflow_wiring "$o"; fi
    refused wiring_refuses_a_renamed_lane_job 's/^    name: cleanroom-cpu$/    name: cleanroom cpu/'
    refused wiring_refuses_a_lane_job_run_on_any_verdict 's/^\(    if: always() && \)contains(fromJSON(.\["green","red"\].), \(needs\.measure-publish-dryrun\.outputs\.verdict\))$/\1\2 != '"''"'/'
    refused wiring_refuses_a_lane_job_that_passes_red '0,/^      - run: test "\$VERDICT" = green$/s//      - run: true/'
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
m11_versions_compared_as_strings s/| sort -V | tail -n 1)" = "$1" \]/| sort | tail -n 1)" = "$1" ]/
m12_wiring_ignores_the_lane_gate s/ || bad="$bad $lane(if)"/ || true/
m13_wiring_ignores_green_only_exit s/ || bad="$bad exit(green-only)"/ || true/
m14_wiring_ignores_the_job_name s/ || bad="$bad $lane(name)"/ || true/'

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
    -h|--help) sed -n '2,42p' "$SCRIPT_PATH"; exit 0 ;;
    *) caller_error "usage: measure LANE [ROOT] | --self-test | --mutants" ;;
esac
