#!/usr/bin/env bash
# ci_guard_steps.sh -- read CI's guard jobs from ci.yml; check them, list them,
# and run them ALL (#4415, #4416). bash + awk + jq: no interpreter, no PyYAML, no yq
# (the clean-room fleet pins jq and carries no yq -- infra readme-lock-check.sh).
#
# ci.yml is read by yaml_json below: an awk reader for the block-YAML subset
# ci.yml is written in (mappings, `- ` sequences, plain / 'single' / "double"
# scalars, `|` block scalars, one-line [flow] lists, {} and []). Anything else
# -- anchors, aliases, tags, folded `>`, multi-line plain or quoted scalars,
# flow maps, duplicate keys, tab indentation -- is REFUSED with rc 2. A reader
# that guessed would be a guard that checks something other than ci.yml.
#
#   bash scripts/lib/ci_guard_steps.sh [--workflow F] list|sha|check-run-all|check-coverage|run [opts] [job...]
#     check-run-all  [--baseline F]  (default scripts/guard_fail_fast_baseline.txt)
#     check-coverage [--ack F]       (default scripts/ci_guards_uncovered.txt)
#     run [--only ERE] [--step-timeout S] [--stream|--no-stream]
#
# Exit: 0 ok, 1 a finding / a failed step, 2 usage, unreadable input, nothing ran.
# Callers: scripts/ci_guards.sh (CI + `make guards-local` + pre-push) and
# scripts/check_guard_steps_run_all.sh (the case table).
set -uo pipefail


LIBDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$LIBDIR/$(basename "${BASH_SOURCE[0]}")"
RUNNER_SH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/ci_guards.sh"
RUNNER="scripts/ci_guards.sh"
MANIFEST_SUFFIX="-steps"
# jq regexes (Oniguruma): what the runner resolves and what counts as a status function.
STATUS_FN='\b(cancelled|always|failure)\s*\(\s*\)'
EXPR='\$\{\{\s*(.*?)\s*\}\}'
GUARD_SCRIPT='scripts/((?:check|guard)_[A-Za-z0-9_]+\.sh)'
# ${{ e }} -> the env names that carry its value at run time.
declare -A KNOWN_EXPR=([github.token]="GITHUB_TOKEN GH_TOKEN" [runner.temp]="RUNNER_TEMP")
KNOWN_LIST="github.token, runner.temp"

die() { echo "ci_guard_steps: $*" >&2; exit 2; }
in_ci() { [ "${GITHUB_ACTIONS:-}" = true ]; }
command -v jq > /dev/null 2>&1 || die "jq missing"

# ── the YAML reader ─────────────────────────────────────────────────────────
# yaml_json <file> -- the document as JSON on stdout; rc 2 + a line-numbered
# reason on stderr for anything outside the subset.
yaml_json() {
    awk -f "$LIBDIR/ci_guard_yaml.awk" "$1"
}

# ── shared reads ────────────────────────────────────────────────────────────
WORKFLOW=".github/workflows/ci.yml"
JOBS=""   # a file holding the jobs map as JSON

load_jobs() {
    local tmp err
    tmp="$(mktemp)" && err="$(mktemp)" || die "mktemp failed"
    CLEAN+=("$tmp" "$err")
    [ -r "$WORKFLOW" ] || die "cannot read $WORKFLOW: no such file"
    if ! yaml_json "$WORKFLOW" 2> "$err" | jq -c '.jobs // {} | if type == "object" then . else error("jobs: is not a mapping") end' > "$tmp" 2>> "$err"; then
        die "cannot read $WORKFLOW: $(tr '\n' ' ' < "$err")"
    fi
    JOBS="$tmp"
}

jqj() { jq "$@" "$JOBS"; }

# guard_jobs -- the guard-* jobs the gate needs, in ci.yml order. Never empty.
guard_jobs() {
    local out
    out="$(jqj -r '((.gate // {}).needs // []) as $n | ($n | if type == "string" then [.] else . end) as $n
        | keys_unsorted[] | select(startswith("guard-") and (. as $j | $n | index([$j]) != null))')"
    [ -n "$out" ] || die "no guard-* job in the gate's needs in $WORKFLOW"
    printf '%s\n' "$out"
}

job_steps_ok() { # job_steps_ok <job> -- rc 2 (via die) when missing or stepless
    local t
    t="$(jqj -r --arg j "$1" 'if has($j) then ((.[$j].steps // []) | length) else "missing" end')"
    [ "$t" = missing ] && die "job '$1' not in $WORKFLOW"
    [ "$t" -gt 0 ] || die "job '$1' has no steps"
}

# jq definitions shared by every query below.
JQDEFS="$(cat "$LIBDIR/ci_guard_steps.jq")" || die "cannot read $LIBDIR/ci_guard_steps.jq"
jqd() { jqj --arg runner "$RUNNER" --arg sfx "$MANIFEST_SUFFIX" --arg status "$STATUS_FN" --arg expr "$EXPR" "$@"; }

sha_line() { # sha_line <job>...
    local h m
    h="$(cat "$RUNNER_SH" "$SELF" "$LIBDIR/ci_guard_yaml.awk" "$LIBDIR/ci_guard_steps.jq" | sha256sum | cut -c1-16)"
    m="$(for j in "$@"; do jqj -S -c --arg j "$j$MANIFEST_SUFFIX" '.[$j]'; done | sha256sum | cut -c1-16)"
    echo "ci_guards: sha256 $h manifest $m ($*)"
}

# ── check-run-all ───────────────────────────────────────────────────────────
declare -A BASE=()
read_baseline() {
    local line job n x nr=0
    [ -r "$1" ] || die "bad baseline $1: cannot read"
    while IFS= read -r line || [ -n "$line" ]; do
        nr=$((nr + 1))
        line="${line%%#*}"
        read -r job n x <<< "$line" || true
        [ -z "${job:-}" ] && continue
        { [ -n "${n:-}" ] && [ -z "${x:-}" ] && [[ $n =~ ^[+-]?[0-9]+$ ]]; } || die "bad baseline $1: line $nr: want \`<job> <n>\`"
        BASE[$job]=$((10#${n#+}))
    done < "$1"
}

manifest_findings() { # manifest_findings <job> -- one finding per line
    jqd -r --arg job "$1" --arg known "$KNOWN_LIST" "$JQDEFS manifest_findings(\$job; \$known)"
}

cmd_check() {
    local baseline="scripts/guard_fail_fast_baseline.txt" rc=0 job end n f
    local -a names=()
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --baseline) [ "$#" -ge 2 ] || die "--baseline needs a value"; baseline=$2; shift 2 ;;
            -*) die "check-run-all: unknown flag $1" ;;
            *) names+=("$1"); shift ;;
        esac
    done
    load_jobs
    read_baseline "$baseline"
    [ "${#names[@]}" -gt 0 ] || mapfile -t names < <(guard_jobs) || exit 2
    [ "${#names[@]}" -gt 0 ] || exit 2
    for job in "${names[@]}"; do
        job_steps_ok "$job"
        read -r end n < <(jqd -r --arg job "$job" "$JQDEFS fail_fast_count(\$job)")
        [[ ${end:-} =~ ^-?[0-9]+$ && ${n:-} =~ ^[0-9]+$ ]] || die "cannot read the steps of $job in $WORKFLOW"
        if [ "$end" -lt 0 ]; then
            echo "FAIL $job: no step has \`id: guard-setup\` -- the runner step's if: reads steps.guard-setup.outcome, so without it the guards are skipped and $job is green"
            rc=1
        fi
        if [ -z "${BASE[$job]+x}" ]; then
            echo "FAIL $job: no baseline row in $baseline"
            rc=1
        elif [ "$n" -gt "${BASE[$job]}" ]; then
            echo "FAIL $job: $n fail-fast steps after the setup step > baseline ${BASE[$job]} -- a guard step goes in $job$MANIFEST_SUFFIX; a step that must stay in $job needs \`if: \${{ !cancelled() && steps.guard-setup.outcome == 'success' }}\`"
            rc=1
        else
            f=""
            [ "$n" -lt "${BASE[$job]}" ] && f=" -- tighten $baseline to $n"
            echo "ok   $job: $n fail-fast steps after setup (baseline ${BASE[$job]})$f"
        fi
        f="$(manifest_findings "$job")" || die "cannot read the manifest of $job in $WORKFLOW"
        if [ -n "$f" ]; then
            printf '%s\n' "$f" | sed 's/^/FAIL /'
            rc=1
        fi
    done
    while IFS= read -r f; do # an orphan manifest runs nowhere
        [ -n "$f" ] && { echo "FAIL $f"; rc=1; }
    done < <(jqd -r "$JQDEFS orphans")
    return "$rc"
}

# ── check-coverage (#4416) ──────────────────────────────────────────────────
cmd_coverage() {
    local ack="scripts/ci_guards_uncovered.txt" rc=0 line body reason nr=0 seen acked job script x
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --ack) [ "$#" -ge 2 ] || die "--ack needs a value"; ack=$2; shift 2 ;;
            -*) die "check-coverage: unknown flag $1" ;;
            *) shift ;;
        esac
    done
    load_jobs
    local -a g=()
    mapfile -t g < <(guard_jobs) || exit 2
    [ "${#g[@]}" -gt 0 ] || exit 2
    seen="$(jqd -r --argjson local "$(printf '%s\n' "${g[@]}" | jq -R . | jq -sc --arg sfx "$MANIFEST_SUFFIX" '. + map(. + $sfx)')" \
        --arg gs "$GUARD_SCRIPT" 'to_entries[] | select(.key as $k | $local | index([$k]) | not) | .key as $j
        | (.value.steps // [])[] | ((.run // "") | tostring) | match($gs; "g").captures[0].string | "\($j) \(.)"' | LC_ALL=C sort -u)" || die "cannot read the steps of $WORKFLOW"
    [ -r "$ack" ] || die "cannot read $ack"
    acked=""
    while IFS= read -r line || [ -n "$line" ]; do
        nr=$((nr + 1))
        body="${line%%#*}"
        reason=""
        [ "$body" != "$line" ] && reason="${line#*#}"
        [[ $body =~ ^[[:space:]]*$ ]] && continue
        read -r job script x <<< "$body" || true
        if [ -z "${script:-}" ] || [ -n "${x:-}" ] || [[ $reason =~ ^[[:space:]]*$ ]]; then
            echo "ci_guard_steps: $ack:$nr: want \`<job> <script>  # <reason>\`" >&2
            exit 2
        fi
        acked+="$job $script"$'\n'
    done < "$ack"
    acked="$(printf '%s' "$acked" | LC_ALL=C sort -u)" || die "cannot read the steps of $WORKFLOW"
    while IFS=' ' read -r job script; do
        [ -n "$job" ] || continue
        echo "FAIL $job runs scripts/$script, which \`make guards-local\` cannot reach -- move it into a guard manifest, or acknowledge it in $ack with a reason"
        rc=1
    done < <(LC_ALL=C comm -23 <(printf '%s\n' "$seen") <(printf '%s\n' "$acked"))
    while IFS=' ' read -r job script; do
        [ -n "$job" ] || continue
        echo "FAIL stale row in $ack: $job no longer runs scripts/$script -- delete it"
        rc=1
    done < <(LC_ALL=C comm -13 <(printf '%s\n' "$seen") <(printf '%s\n' "$acked"))
    if [ "$rc" -eq 0 ]; then
        echo "ok   guard scripts outside the guard jobs: $(printf '%s' "$seen" | grep -c .), all acknowledged"
    fi
    return "$rc"
}

# ── run ─────────────────────────────────────────────────────────────────────
# The run environment is the process environment (what os.environ was) plus
# ENV overrides; ORIG names what the process environment holds.
declare -A ORIG=() ENV=()
while IFS= read -r x; do ORIG[$x]=1; done < <(compgen -e)
getenv() { # getenv <name> -- ENV override, else the process environment
    if [ -n "${ENV[$1]+x}" ]; then printf '%s' "${ENV[$1]}"
    elif [ -n "${ORIG[$1]+x}" ]; then printf '%s' "${!1}"
    fi
}

# resolve <value> -- ${{ e }} -> its runtime value; rc 1 when one cannot be resolved
resolve() {
    local v=$1 out="" e name val hit
    local -a carriers=()
    local re='\$\{\{[[:space:]]*([^}]*[^}[:space:]])[[:space:]]*\}\}'
    while [[ $v =~ $re ]]; do
        e="${BASH_REMATCH[1]}"
        out+="${v%%"${BASH_REMATCH[0]}"*}"
        v="${v#*"${BASH_REMATCH[0]}"}"
        hit=""
        read -ra carriers <<< "${KNOWN_EXPR[$e]:-}"
        for name in "${carriers[@]}"; do
            val="$(getenv "$name")"
            [ -n "$val" ] && { hit=1; out+="$val"; break; }
        done
        [ -n "$hit" ] || return 1
    done
    printf '%s' "$out$v"
}

apply_github_env() { # apply_github_env <file> -- KEY=VAL and KEY<<DELIM lines, as the runner reads them
    local line key delim body
    local -a lines=()
    [ -r "$1" ] || return 0
    mapfile -t lines < "$1"
    local i=0
    while [ "$i" -lt "${#lines[@]}" ]; do
        line="${lines[$i]}"
        if [[ $line == *"<<"* ]] && { [[ $line != *=* ]] || [ "${#line%%<<*}" -lt "${#line%%=*}" ]; }; then
            key="${line%%<<*}"; delim="${line#*<<}"; body=""
            local first=1
            i=$((i + 1))
            while [ "$i" -lt "${#lines[@]}" ] && [ "${lines[$i]}" != "$delim" ]; do
                if [ -n "$first" ]; then body="${lines[$i]}"; first=""; else body+=$'\n'"${lines[$i]}"; fi
                i=$((i + 1))
            done
            ENV[$key]="$body"
        elif [[ $line == *=* ]]; then
            ENV[${line%%=*}]="${line#*=}"
        fi
        i=$((i + 1))
    done
}

now() { date +%s.%N; }

cmd_run() {
    local only="" stream="" timeout="${CI_GUARDS_STEP_TIMEOUT:-1200}" root scratch job
    local -a names=()
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --only) [ "$#" -ge 2 ] || die "--only needs a value"; only=$2; shift 2 ;;
            --step-timeout) [ "$#" -ge 2 ] || die "--step-timeout needs a value"; timeout=$2; shift 2 ;;
            --stream) stream=1; shift ;;
            --no-stream) stream=0; shift ;;
            -*) die "run: unknown flag $1" ;;
            *) names+=("$1"); shift ;;
        esac
    done
    [[ $timeout =~ ^[0-9]+$ ]] || die "--step-timeout wants whole seconds, got '$timeout'"
    if [ -z "$stream" ]; then if in_ci; then stream=1; else stream=0; fi; fi
    root="$(git rev-parse --show-toplevel 2> /dev/null)" || true
    [ -n "$root" ] || die "not in a git work tree"
    load_jobs
    [ "${#names[@]}" -gt 0 ] || mapfile -t names < <(guard_jobs) || exit 2
    [ "${#names[@]}" -gt 0 ] || exit 2
    sha_line "${names[@]}"
    if in_ci; then
        local -a missing=()
        for job in "${names[@]}"; do
            [ "$(jqj --arg m "$job$MANIFEST_SUFFIX" 'has($m)')" = true ] || missing+=("$job")
        done
        if [ "${#missing[@]}" -gt 0 ]; then
            die "no manifest job for $(printf '%s, ' "${missing[@]}" | sed 's/, $//') -- nothing to run"
        fi
    fi
    if [ -n "${CI_GUARDS_SCRATCH:-}" ]; then scratch=$CI_GUARDS_SCRATCH
    elif in_ci; then scratch="$(mktemp -d "${TMPDIR:-/tmp}/ci-guards-XXXXXX")" || die "mktemp failed"
    else scratch="$root/target/ci-guards-local"; fi
    mkdir -p "$scratch" || die "cannot create $scratch"

    local -a rows=()
    emit() { # emit <job> <idx> <status> <sec> <name>
        local disp round
        # awk formats the seconds: bash's printf %f reads the locale's radix
        read -r disp round < <(awk -v s="$4" 'BEGIN { printf "%6.1f %.0f\n", s, s }')
        rows+=("$1"$'\x1f'"$2"$'\x1f'"$3"$'\x1f'"$round"$'\x1f'"$5")
        printf '%-7s %s#%s %6ss  %s\n' "$3" "$1" "$2" "$disp" "$5"
        if { [ "$3" = FAIL ] || [ "$3" = TIMEOUT ]; } && in_ci; then
            printf '::error title=%s: guard step %s::%s\n' "$1" "${3,,}" "$5"
        fi
    }

    local idx name run cond has_uses has_run n i reason k v r tag log t0 t1 rc status detail p ci=false
    local -a ek=() ev=() senv=() paths=()
    in_ci && ci=true
    for job in "${names[@]}"; do
        ENV=()
        if ! in_ci; then
            [ -n "${ORIG[GITHUB_WORKSPACE]+x}" ] || ENV[GITHUB_WORKSPACE]=$root
            [ -n "${ORIG[GITHUB_EVENT_NAME]+x}" ] || ENV[GITHUB_EVENT_NAME]=local
            [ -n "${ORIG[PR_OR_REF]+x}" ] || ENV[PR_OR_REF]=local
            [ -n "${ORIG[RUNNER_TEMP]+x}" ] || ENV[RUNNER_TEMP]=$scratch/runner_temp
            mkdir -p "$(getenv RUNNER_TEMP)"
            # the job's own env: block (CI already exported it)
            while IFS= read -r -d '' k && IFS= read -r -d '' v; do
                [ -n "${ORIG[$k]+x}" ] && continue
                case "$k" in
                    GUARD_TARGET_DIR|GUARD_CARGO_HOME) ENV[$k]="$scratch/${k,,}" ;;
                    *) ENV[$k]="$(printf '%s' "$v" | sed -E 's/\$\{\{[^}]*\}\}/local/g')" ;;
                esac
            done < <(jqj -j --arg j "$job" '(.[$j].env // {}) | to_entries[] | "\(.key)\u0000\(.value | tostring)\u0000"')
        fi
        [ "$ci" = true ] || job_steps_ok "$job"
        # jq writes the records to a file first: a jq that dies mid-stream behind <(...)
        # would end the loop early with no error, and the unread steps would vanish.
        jqd -j --arg job "$job" --argjson ci "$ci" "$JQDEFS steps_for(\$job; \$ci)[] | step_record" \
            > "$scratch/steps.$job" || die "cannot read the steps of $job in $WORKFLOW (jq rc=$?)"
        while IFS= read -r -u 3 -d '' idx && IFS= read -r -u 3 -d '' name \
            && IFS= read -r -u 3 -d '' run && IFS= read -r -u 3 -d '' cond \
            && IFS= read -r -u 3 -d '' has_uses && IFS= read -r -u 3 -d '' has_run \
            && IFS= read -r -u 3 -d '' n; do
            ek=(); ev=()
            [[ $n =~ ^[0-9]+$ ]] || die "cannot read the steps of $job in $WORKFLOW"
            for ((i = 0; i < n; i++)); do
                IFS= read -r -u 3 -d '' k && IFS= read -r -u 3 -d '' v || die "cannot read the steps of $job in $WORKFLOW"
                ek+=("$k"); ev+=("$v")
            done
            if [ -n "$only" ] && ! [[ $name =~ $only ]]; then continue; fi
            reason=""
            if ! in_ci; then
                if [[ $cond == *event_name* ]]; then reason="if: needs a CI event"
                elif [ -n "$has_uses" ]; then reason="uses: action"
                elif [[ $run =~ \$\{\{.*\}\} ]]; then reason="run: has a \${{ }} expression"
                elif [[ $run == *"docker run"* || $run == *'$IMAGE'* || $run == *'${IMAGE}'* ]]; then
                    v="$(getenv IMAGE)"
                    if [ -z "$v" ] || ! docker image inspect "$v" > /dev/null 2>&1; then reason="needs docker image ${v:-\$IMAGE}"; fi
                fi
            fi
            [ -z "$reason" ] && [ -z "$has_run" ] && reason="FAIL: no run:"
            senv=()
            for p in "${!ENV[@]}"; do senv+=("$p=${ENV[$p]}"); done
            for i in "${!ek[@]}"; do
                k="${ek[$i]}"
                if r="$(resolve "${ev[$i]}")"; then
                    if [ -z "${ORIG[$k]+x}" ] || in_ci; then senv+=("$k=$r"); fi
                elif in_ci; then
                    reason="${reason:-FAIL: env $k cannot be resolved}"
                fi
            done
            if [[ $reason == FAIL* ]]; then emit "$job" "$idx" FAIL 0 "$name  [$reason]"; continue; fi
            if [ -n "$reason" ]; then emit "$job" "$idx" SKIP 0 "$name  [$reason]"; continue; fi
            tag="$job.$idx"
            for p in GITHUB_PATH GITHUB_ENV GITHUB_OUTPUT; do
                : > "$scratch/${p,,}.$tag"
                senv+=("$p=$scratch/${p,,}.$tag")
            done
            [ -n "$(getenv GITHUB_STEP_SUMMARY)" ] || senv+=("GITHUB_STEP_SUMMARY=$scratch/step_summary")
            log="$scratch/$tag.log"
            [ "$stream" = 1 ] && echo "::group::$job#$idx $name"
            t0="$(now)"
            # timeout puts the step in its own process group and SIGKILLs the
            # group, so its children (cargo, docker, sleep) die with it.
            local -a tcmd=()
            [ "$timeout" -gt 0 ] && tcmd=(timeout -s KILL "$timeout")
            if [ "$stream" = 1 ]; then
                (cd "$root" && env "${senv[@]}" "${tcmd[@]}" bash --noprofile --norc -eo pipefail -c "$run" < /dev/null 3<&-)
            else
                (cd "$root" && env "${senv[@]}" "${tcmd[@]}" bash --noprofile --norc -eo pipefail -c "$run" < /dev/null > "$log" 2>&1 3<&-)
            fi
            rc=$?
            t1="$(now)"
            [ "$stream" = 1 ] && echo "::endgroup::"
            status=PASS
            if [ "$rc" -ne 0 ]; then
                status=FAIL
                if [ "$timeout" -gt 0 ] && [ "$rc" -eq 137 ] && awk -v a="$t0" -v b="$t1" -v t="$timeout" 'BEGIN { exit !(b - a >= t) }'; then
                    status=TIMEOUT; rc=124
                fi
            fi
            # what the step handed to later steps, as the Actions runner applies it
            mapfile -t paths < <(sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' "$scratch/github_path.$tag" | grep -v '^$')
            for ((i = ${#paths[@]} - 1; i >= 0; i--)); do ENV[PATH]="${paths[$i]}:$(getenv PATH)"; done
            apply_github_env "$scratch/github_env.$tag"
            detail=$name
            if [ "$rc" -ne 0 ]; then
                if [ "$stream" = 1 ]; then detail="$name  [rc=$rc]"; else detail="$name  [rc=$rc, log $log]"; fi
            fi
            emit "$job" "$idx" "$status" "$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.3f", b - a }')" "$detail"
        done 3< "$scratch/steps.$job"
    done

    local failed=0 ran=0 skipped=0 row
    local -a fl=()
    for row in "${rows[@]}"; do
        IFS=$'\x1f' read -r job idx status t0 name <<< "$row"
        if [ "$status" = SKIP ]; then skipped=$((skipped + 1)); else ran=$((ran + 1)); fi
        if [ "$status" = FAIL ] || [ "$status" = TIMEOUT ]; then failed=$((failed + 1)); fl+=("$(printf '  %-7s %s#%s %s' "$status" "$job" "$idx" "$name")"); fi
    done
    echo "SUMMARY: $failed failed / $ran ran / $skipped skipped"
    [ "${#fl[@]}" -gt 0 ] && printf '%s\n' "${fl[@]}"
    if in_ci && [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
        {
            printf '### Guard steps: %s failed / %s ran\n\n| job | step | result | sec |\n|---|---|---|---|\n' "$failed" "$ran"
            for row in "${rows[@]}"; do
                IFS=$'\x1f' read -r job idx status t0 name <<< "$row"
                printf '| %s | %s | %s | %s |\n' "$job" "${name%%  \[*}" "$status" "$t0"
            done
        } >> "$GITHUB_STEP_SUMMARY"
    fi
    if [ "$ran" -eq 0 ]; then
        echo "ci_guard_steps: nothing ran -- refusing a vacuous pass" >&2
        return 2
    fi
    [ "$failed" -eq 0 ]
}

cmd_list() {
    local job ci=false
    in_ci && ci=true
    load_jobs
    local -a names=("$@")
    [ "${#names[@]}" -gt 0 ] || mapfile -t names < <(guard_jobs) || exit 2
    [ "${#names[@]}" -gt 0 ] || exit 2
    for job in "${names[@]}"; do
        echo "== $job"
        [ "$ci" = true ] && [ "$(jqj --arg m "$job$MANIFEST_SUFFIX" 'has($m)')" = true ] || job_steps_ok "$job"
        jqd -r --arg job "$job" --argjson ci "$ci" "$JQDEFS"'steps_for($job; $ci)[] | "\(.idx | if length < 4 then (" " * (4 - length)) + . else . end)  \(.step | name_of | tostring)"'
    done
}

cmd_sha() {
    load_jobs
    local -a names=("$@")
    [ "${#names[@]}" -gt 0 ] || mapfile -t names < <(guard_jobs) || exit 2
    [ "${#names[@]}" -gt 0 ] || exit 2
    sha_line "${names[@]}"
}

CLEAN=()
cleanup() { local f; for f in "${CLEAN[@]}"; do rm -f -- "${f:?}"; done; }
trap cleanup EXIT

while [ "$#" -gt 0 ]; do
    case "$1" in
        --workflow) [ "$#" -ge 2 ] || die "--workflow needs a value"; WORKFLOW=$2; shift 2 ;;
        *) break ;;
    esac
done
cmd="${1:-}"
[ "$#" -gt 0 ] && shift
case "$cmd" in
    list) cmd_list "$@" ;;
    sha) cmd_sha "$@" ;;
    check-run-all) cmd_check "$@" ;;
    check-coverage) cmd_coverage "$@" ;;
    run) cmd_run "$@" ;;
    *) die "usage: ci_guard_steps.sh [--workflow F] list|sha|check-run-all|check-coverage|run [job...]" ;;
esac
