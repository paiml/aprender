#!/usr/bin/env bash
# check_jetson_gpu_jobs.sh -- what keeps the Jetson Orin GPU lane safe and honest is a few lines of workflow YAML,
# and this guard pins each of them (#4879).
#
# WHY THIS EXISTS
#   The Jetson lane runs the aprender-gpu and aprender-serve CUDA tests on ONE persistent runner: a Jetson Orin
#   (sm_87) whose 7.4 GB of memory is shared by the CPU and the GPU, with a workspace and a build directory that
#   outlive every job and no container around the job. Everything that makes that safe, and everything that makes a
#   green mean "the tests ran on the GPU", is a handful of lines that no compiler, linter or test reads:
#
#     env pins   CARGO_BUILD_JOBS and RUST_TEST_THREADS bound the memory the build and the test threads can take (an
#                unpinned job builds at -j<cores> and the OOM killer ends the job, the runner or the box: trap C);
#                APR_REQUIRE_GPU=1 turns every "no CUDA device, skip" path into a panic (trap B: without it a missing
#                device is a GREEN run of zero GPU tests).
#     labels     the jobs must reach this box and only this box, and nothing else may be sent to it.
#     if:        a fork's pull request must never run its code on a persistent bare-metal box.
#     needs:     the lane is advisory. One edge into gate.needs and one box's availability gates every merge.
#     steps      all logic lives in scripts/ci_jetson_gpu.sh, which judges the LOG (skips, floors, a missing result
#                line). An inline test command, a step-level `if:` or a continue-on-error is a way round it.
#
#   A pin nothing checks is a comment. Each of these is deleted, raised or widened by an edit that looks harmless in
#   review, and the run that follows is either red for a reason that costs a night to find (an OOM-killed runner) or
#   GREEN for a reason that is worse (zero GPU tests, "success"). The signature failure of this fleet is a check that
#   reports a result it did not measure, so the guard also refuses to pass over zero jobs (R1) and refuses a verdict
#   when it cannot read what it must judge (exit 2).
#
# THE RULES (every failure names its rule:  RED  R<n>  file:line  [job]  what is wrong)
#   R1  the lane exists: gpu-failfast-jetson and gpu-e2e-jetson in ci.yml, jetson-gpu in jetson-gpu-nightly.yml, and
#       scripts/ci_jetson_gpu.sh.
#   R2  runs-on is exactly [self-hosted, Linux, ARM64, jetson]: never clean-room (the ARM64 pool's jobs must not land
#       on a 7.4 GB box), never cuda or gpu (check_perf_concurrency_groups.sh reads those as perf-sensitive).
#   R3  job-level env pins: APR_REQUIRE_GPU=1; CARGO_BUILD_JOBS in 1..MAX_BUILD_JOBS and RUST_TEST_THREADS in
#       1..MAX_TEST_THREADS, both bounds READ from scripts/ci_jetson_gpu.sh so the two lists cannot drift apart;
#       CARGO_TERM_COLOR=never; env is a block mapping; no step overrides a pin.
#   R4  timeout-minutes is an integer no larger than the lane's cap (15 / 25 / 45): a hung build must not hold the
#       one box for GitHub's six-hour default.
#   R5  permissions are exactly contents: read, at the job or, when the job declares none, at the workflow.
#   R6  the only action is actions/checkout, with persist-credentials: false (a persistent workspace must not keep
#       the token in .git/config).
#   R7  no secret and no token: the word `secrets`, github.token and GITHUB_TOKEN appear nowhere in the lane.
#   R8  who may start it. ci.yml jobs: `if` carries !cancelled() (without it a workflow_dispatch skips the job GREEN,
#       because gpu-touched is skipped on dispatch), the same-repository test (no fork code on this box) and the
#       gpu-touched selector, and names neither merge_group, push nor pull_request_target. The nightly: `on` is
#       exactly schedule + workflow_dispatch, and the schedule has a cron.
#   R9  needs: failfast needs exactly gpu-touched; e2e needs exactly failfast; the nightly needs nothing; no OTHER
#       job in ci.yml (gate, gx10, yoga, ...) names either jetson job. A needs it cannot read is exit 2, not a pass.
#   R10 steps are `bash scripts/ci_jetson_gpu.sh STAGE` and nothing else, in the lane's stage order (failfast:
#       preflight s1 s2; e2e: preflight s3; nightly: preflight s1 s2 s3). No step `if:`, no continue-on-error on a
#       step or the job: each turns a red stage green or a skipped one into a pass.
#   R11 concurrency: in ci.yml a job-level group that is neither the workflow's `ci-...` group nor shared with another
#       lane job, and never cancel-in-progress: true; the nightly's group exists and cancels nothing (a cancelled
#       cold build is the most expensive thing the lane does).
#   R12 scripts/ci_jetson_gpu.sh --self-test passes and ran something (N > 0 passed, 0 failed).
#   R13 the box is named by the lane and by nothing else: no other non-comment line of any workflow under
#       .github/workflows or ci/ contains the word jetson (a runs-on in any form, a matrix label, a needs edge).
#
# WHAT IT DOES NOT JUDGE (stated so that a green here is not read as more than it is)
#   Whether the jetson label is declared on a runner, whether the box has the model and the toolchain (S0 asks the
#   box at run time), and whether either job is a REQUIRED check: that is branch protection, which no file in the
#   tree shows. The tree-side half is R9: neither job is in gate.needs.
#
# USAGE
#   bash scripts/check_jetson_gpu_jobs.sh              # judge this tree
#   bash scripts/check_jetson_gpu_jobs.sh --root DIR   # judge another tree (fixtures)
#   bash scripts/check_jetson_gpu_jobs.sh --self-test  # parser cases + fixture trees: one GOOD, every broken one RED
#   bash scripts/check_jetson_gpu_jobs.sh --facts FILE # print the rows the parser reads from FILE (debugging)
# EXIT: 0 every rule holds; 1 a violation; 2 cannot judge (an unreadable needs, an unreadable bound, no tool).
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2

CI=.github/workflows/ci.yml
NIGHTLY=.github/workflows/jetson-gpu-nightly.yml
SCRIPT=scripts/ci_jetson_gpu.sh
# file | job | timeout cap in minutes | stage sequence
LANE_JOBS=(
    "$CI|gpu-failfast-jetson|15|preflight s1 s2"
    "$CI|gpu-e2e-jetson|25|preflight s3"
    "$NIGHTLY|jetson-gpu|45|preflight s1 s2 s3"
)
BOX_LABELS="arm64 jetson linux self-hosted"
US=$'\037'
WORK=""
declare -A VAL=() LINE=()
NRED=0
NCANT=0
NROWS=0
TRIMMED=""
UNQ=""
SCONST=""
PERMDECL=0
ITEMS=()
PERMS=()

cleanup() { if [ -n "$WORK" ]; then rm -rf "${WORK:?}"; fi; }
trap cleanup EXIT

usage() {
    cat <<'EOF'
usage: check_jetson_gpu_jobs.sh [--root DIR | --self-test | --facts FILE | --help]
  (no argument)  judge this tree: the three Jetson lane jobs, their pins, and nothing else naming the box
  --root DIR     judge the tree at DIR instead
  --self-test    parser cases for every YAML form, then fixture trees: the good one must pass, every broken one RED
  --facts FILE   print the rows the parser reads from FILE (debugging)
exit: 0 holds, 1 violation, 2 cannot judge
EOF
}

# facts.awk -- flatten a GitHub workflow (the block-style YAML subset these files use) into rows
#   F US JOB US KEY US VALUE US LINE        (US is octal 037, so an empty value keeps its column)
# JOB is "-" outside a job; KEY is relative to the job (env.NAME, steps[2].run, needs[1], on.schedule[1].cron).
# Scalars are unquoted and comment-free; a flow list or flow map keeps its brackets. Extra rows: @job (a job's
# first line), @secrets-ref:N and @jetson-word:N (a non-comment line naming the thing), @odd:N (a line it could not
# place). A key with an inline value owns the deeper-indented lines after it (a multi-line flow list); a block
# scalar (| or >) owns them and they are not read as keys.
write_facts_awk() {
    cat > "$1" <<'AWK'
function trim(s) { sub(/^[[:space:]]+/, "", s); sub(/[[:space:]]+$/, "", s); return s }
function indent(s,   t) { t = s; sub(/[^ ].*$/, "", t); return length(t) }
function nocomment(s,   i, c, q, p, o) {
    q = ""; p = " "; o = ""
    for (i = 1; i <= length(s); i++) {
        c = substr(s, i, 1)
        if (q == "") {
            if (c == "'" || c == "\"") {
                if (p == " " || p == "[" || p == "," || p == "{") q = c
            } else if (c == "#" && p == " ") break
        } else if (c == q) q = ""
        o = o c; p = c
    }
    return trim(o)
}
function unquote(s,   n, a, b) {
    n = length(s)
    if (n >= 2) {
        a = substr(s, 1, 1); b = substr(s, n, 1)
        if ((a == "\"" || a == "'") && a == b) return substr(s, 2, n - 2)
    }
    return s
}
function emit(job, key, val, ln) { printf "%s\037%s\037%s\037%s\037%d\n", F, job, key, val, ln; nrows++ }
function scan(b, ln,   l) {
    l = tolower(b)
    if (index(l, "secrets") > 0 || index(l, "github.token") > 0 || index(l, "github_token") > 0) emit(curjob, "@secrets-ref:" ln, b, ln)
    if (index(l, "jetson") > 0) emit(curjob, "@jetson-word:" ln, b, ln)
}
function flush(   v) {
    if (!pact) return
    v = pv
    if (substr(v, 1, 1) != "[" && substr(v, 1, 1) != "{") v = unquote(v)
    emit(pjob, prel, v, pln)
    pact = 0
}
function pushframe(ind, kind, job, rel) { depth++; fi[depth] = ind; fk[depth] = kind; fj[depth] = job; fr[depth] = rel }
function childrel(prel, key) { return (prel == "") ? key : prel "." key }
function keyof(b) {
    if (match(b, /^[A-Za-z0-9_.-]+:/) == 0) return ""
    if (length(b) > RLENGTH && substr(b, RLENGTH + 1, 1) != " ") return ""
    return substr(b, 1, RLENGTH - 1)
}
function handle_key(k, rest, kind_ind, ln,   nj, nr, ek, c) {
    if (depth == 0) { nj = "-"; nr = k }
    else if (depth == 1 && fr[1] == "jobs" && fj[1] == "-") { nj = k; nr = "" }
    else { nj = fj[depth]; nr = childrel(fr[depth], k) }
    curjob = nj
    ek = (nr == "") ? "@job" : nr
    if (rest == "") { emit(nj, ek, "", ln); pushframe(kind_ind, "K", nj, nr); return }
    c = substr(rest, 1, 1)
    if (c == "|" || c == ">") { emit(nj, ek, rest, ln); skipind = kind_ind; return }
    pact = 1; pjob = nj; prel = ek; pv = rest; pln = ln; pc = kind_ind
}
BEGIN { depth = 0; curjob = "-"; pact = 0; skipind = -1; nrows = 0 }
{
    ln = NR
    if ($0 ~ /^[[:space:]]*$/ || $0 ~ /^[[:space:]]*#/) next
    n = indent($0)
    body = nocomment(substr($0, n + 1))
    if (body == "") next
    if (pact && n > pc) { pv = pv " " body; scan(body, ln); next }
    flush()
    if (skipind >= 0) {
        if (n > skipind) { scan(body, ln); next }
        skipind = -1
    }
    if (body ~ /^"on":/ || body ~ /^'on':/) body = "on:" substr(body, 6)
    isdash = (body == "-" || substr(body, 1, 2) == "- ")
    while (depth > 0) {
        if (isdash) {
            if (fi[depth] > n || (fi[depth] == n && fk[depth] == "I")) depth--
            else break
        } else {
            if (fi[depth] >= n) depth--
            else break
        }
    }
    if (isdash) {
        if (depth > 0) {
            pj = fj[depth]; pr = fr[depth]
            idx = ++cnt[pj "|" pr]
            irel = pr "[" idx "]"
            curjob = pj
            rest = trim(substr(body, 2))
            m = n + 1 + (length(body) - 1 - length(rest))
            pushframe(n, "I", pj, irel)
            k = keyof(rest)
            if (rest == "") emit(pj, irel, "", ln)
            else if (k != "") { emit(pj, irel, "", ln); handle_key(k, trim(substr(rest, length(k) + 2)), m, ln) }
            else { pact = 1; pjob = pj; prel = irel; pv = rest; pln = ln; pc = n }
        }
    } else {
        k = keyof(body)
        if (k == "") emit(curjob, "@odd:" ln, body, ln)
        else handle_key(k, trim(substr(body, length(k) + 2)), n, ln)
    }
    scan(body, ln)
}
END { flush() }
AWK
}

setup_work() {
    if [ -z "$WORK" ]; then
        WORK=$(mktemp -d) || return 2
        case $WORK in /?*) ;; *) echo "ENV   mktemp gave an unusable directory: '$WORK'" >&2; return 2 ;; esac
        mkdir -p "$WORK/fx"
        write_facts_awk "$WORK/facts.awk"
    fi
}

red() { NRED=$((NRED + 1)); printf 'RED   %s  %s  %s\n' "$1" "$2" "$3" >&2; }
cant() { NCANT=$((NCANT + 1)); printf 'CANT  %s  %s\n' "$1" "$2" >&2; }

# jred RULE FILE JOB KEY MSG -- a violation located at the row KEY of the job (or at the job's first line)
jred() {
    local ln=${LINE["$2|$3|$4"]-}
    [[ -n $ln ]] || ln=${LINE["$2|$3|@job"]-0}
    red "$1" "$2:$ln" "[$3] $5"
}

trim() { TRIMMED=$1; TRIMMED=${TRIMMED#"${TRIMMED%%[![:space:]]*}"}; TRIMMED=${TRIMMED%"${TRIMMED##*[![:space:]]}"}; }
unq() {
    UNQ=$1
    local n=${#UNQ}
    if ((n >= 2)); then
        case ${UNQ:0:1}${UNQ: -1} in
            '""' | "''") UNQ=${UNQ:1:n-2} ;;
        esac
    fi
}

# load_rows ROOT REL -- flatten one workflow into VAL / LINE
load_rows() {
    local root=$1 rel=$2 f j k v l rows="$WORK/rows"
    if ! awk -v F="$rel" -f "$WORK/facts.awk" "$root/$rel" > "$rows"; then
        cant "$rel" "the row reader failed on this file"
        return 1
    fi
    while IFS=$US read -r f j k v l; do
        VAL["$f|$j|$k"]=$v
        LINE["$f|$j|$k"]=$l
        NROWS=$((NROWS + 1))
    done < "$rows"
}

# items FILE JOB KEY -- ITEMS: the elements of a scalar, a (possibly multi-line) flow list or a block list
items() {
    ITEMS=()
    local base="$1|$2|$3" v part kk i n
    local -a parts=()
    v=${VAL["$base"]-}
    if [[ -n $v ]]; then
        # the bracket strip below would also eat the braces of an expression opener, and then the CANT test on
        # `${{` in r_needs and r_no_other_needs could never fire for a flow list: park the opener for the strip
        v=${v//\$\{\{/$'\001'}
        v=${v//[\[\]\{\}]/ }
        v=${v//$'\001'/\$\{\{}
        IFS=',' read -ra parts <<<"$v" || true
        n=${#parts[@]}
        for ((i = 0; i < n; i++)); do
            trim "${parts[i]}"; unq "$TRIMMED"; trim "$UNQ"
            if [[ -n $TRIMMED ]]; then ITEMS+=("$TRIMMED"); fi
        done
    fi
    for kk in "${!VAL[@]}"; do
        if [[ $kk == "${base}["*"]" ]]; then
            trim "${VAL[$kk]}"; unq "$TRIMMED"
            if [[ -n $UNQ ]]; then ITEMS+=("$UNQ"); fi
        fi
    done
}

# same_set "a b c" -- ITEMS equals the words, as lower-cased sets
same_set() {
    local -a words=()
    local it w ok
    read -ra words <<<"$1" || true
    for it in "${ITEMS[@]}"; do
        ok=0
        for w in "${words[@]}"; do
            if [[ ${it,,} == "$w" ]]; then ok=1; fi
        done
        ((ok)) || return 1
    done
    for w in "${words[@]}"; do
        ok=0
        for it in "${ITEMS[@]}"; do
            if [[ ${it,,} == "$w" ]]; then ok=1; fi
        done
        ((ok)) || return 1
    done
    return 0
}

# perm_rows FILE JOB -- PERMS ("scope=level" ...) and PERMDECL=1 when that level declares permissions at all
perm_rows() {
    PERMS=(); PERMDECL=0
    local pp="$1|$2|" kk v part i n
    local -a parts=()
    if [[ -n ${VAL["${pp}permissions"]+x} ]]; then PERMDECL=1; fi
    for kk in "${!VAL[@]}"; do
        if [[ $kk == "${pp}permissions."* ]]; then
            PERMDECL=1
            PERMS+=("${kk#"${pp}permissions."}=${VAL[$kk]}")
        fi
    done
    v=${VAL["${pp}permissions"]-}
    if [[ -n $v ]]; then
        v=${v//[\{\}]/ }
        IFS=',' read -ra parts <<<"$v" || true
        n=${#parts[@]}
        for ((i = 0; i < n; i++)); do
            trim "${parts[i]}"
            [[ -n $TRIMMED ]] || continue
            if [[ $TRIMMED == *:* ]]; then
                part=${TRIMMED%%:*}; trim "${TRIMMED#*:}"; PERMS+=("$part=$TRIMMED")
            else
                PERMS+=("*=$TRIMMED")
            fi
        done
    fi
}

# script_const NAME -- SCONST: the integer scripts/ci_jetson_gpu.sh assigns to NAME (empty when it does not)
script_const() {
    SCONST=$(sed -n -e "s/^readonly $1=\([0-9][0-9]*\).*/\1/p" -e "s/^$1=\([0-9][0-9]*\).*/\1/p" "$2" 2>/dev/null | head -n 1)
}

# ---- the rules; each takes FILE JOB (and what it needs) and reports with jred ----

r_labels() {
    local f=$1 j=$2
    items "$f" "$j" runs-on
    same_set "$BOX_LABELS" || jred R2 "$f" "$j" runs-on "runs-on is [${ITEMS[*]-}] but must be exactly [self-hosted, Linux, ARM64, jetson]: the lane reaches this box and only this box, never the clean-room pool or a cuda/gpu-labelled host"
}

r_env() {
    local f=$1 j=$2 mj=$3 mt=$4 v name max kk
    if [[ -n ${VAL["$f|$j|env"]-} ]]; then
        jred R3 "$f" "$j" env "env is the inline value '${VAL["$f|$j|env"]}', not a block mapping, so the pins cannot be read"
        return
    fi
    v=${VAL["$f|$j|env.APR_REQUIRE_GPU"]-}
    [[ $v == 1 ]] || jred R3 "$f" "$j" env "APR_REQUIRE_GPU is '${v:-unset}', must be 1: without it a CUDA test that finds no device skips and the run is green with zero GPU tests (trap B)"
    for name in CARGO_BUILD_JOBS RUST_TEST_THREADS; do
        if [[ $name == CARGO_BUILD_JOBS ]]; then max=$mj; else max=$mt; fi
        v=${VAL["$f|$j|env.$name"]-}
        if [[ ! $v =~ ^[0-9]+$ ]] || ((10#$v < 1)); then
            jred R3 "$f" "$j" env "$name is '${v:-unset}', must be an integer of at least 1: 7.4 GB is the whole box, shared with the GPU (trap C)"
        elif ((max > 0 && 10#$v > max)); then
            jred R3 "$f" "$j" "env.$name" "$name is $v, above the $max that $SCRIPT allows: 7.4 GB is the whole box, shared with the GPU (trap C)"
        fi
    done
    v=${VAL["$f|$j|env.CARGO_TERM_COLOR"]-}
    [[ $v == never ]] || jred R3 "$f" "$j" env "CARGO_TERM_COLOR is '${v:-unset}', must be never: colour escapes split the 'test result:' line the log judge reads"
    for kk in "${!VAL[@]}"; do
        if [[ $kk == "$f|$j|steps["*"].env."* ]]; then
            name=${kk##*.env.}
            case $name in
                APR_REQUIRE_GPU | CARGO_BUILD_JOBS | RUST_TEST_THREADS | CARGO_TERM_COLOR)
                    jred R3 "$f" "$j" "${kk#"$f|$j|"}" "a step sets $name again, which overrides the job-level pin" ;;
            esac
        fi
    done
}

r_timeout() {
    local f=$1 j=$2 cap=$3 v=${VAL["$1|$2|timeout-minutes"]-}
    if [[ ! $v =~ ^[0-9]+$ ]] || ((10#$v < 1 || 10#$v > cap)); then
        jred R4 "$f" "$j" timeout-minutes "timeout-minutes is '${v:-unset}', must be an integer in 1..$cap: a hung build must not hold the one box for GitHub's six-hour default"
    fi
}

r_perms() {
    local f=$1 j=$2 where=job
    perm_rows "$f" "$j"
    if ((!PERMDECL)); then perm_rows "$f" "-"; where=workflow; fi
    if ((${#PERMS[@]} != 1)) || [[ ${PERMS[0]-} != "contents=read" ]]; then
        jred R5 "$f" "$j" permissions "permissions are {${PERMS[*]-none declared}} (at the $where level), must be exactly {contents=read}: a persistent box runs this job"
    fi
}

r_secrets() {
    local f=$1 j=$2 kk
    for kk in "${!VAL[@]}"; do
        if [[ $kk == "$f|$j|@secrets-ref:"* ]]; then
            jred R7 "$f" "$j" "${kk#"$f|$j|"}" "names a secret or a token: ${VAL[$kk]}"
        fi
    done
}

# r_steps FILE JOB SEQ -- R6 (the only action is a credential-free checkout) and R10 (the steps are the stage calls)
r_steps() {
    local f=$1 j=$2 seq=$3 n=0 i sp u r st k v checkouts=0
    local -a stages=() forbidden=()
    local re='^bash scripts/ci_jetson_gpu\.sh (preflight|s1|s2|s3)$'
    while [[ -n ${VAL["$f|$j|steps[$((n + 1))]"]+x} ]]; do n=$((n + 1)); done
    if ((n == 0)); then jred R10 "$f" "$j" steps "the job has no steps"; fi
    for ((i = 1; i <= n; i++)); do
        sp="$f|$j|steps[$i]"
        u=${VAL["$sp.uses"]-}
        r=${VAL["$sp.run"]-}
        forbidden=("$sp.if" "$sp.continue-on-error")
        for k in "${forbidden[@]}"; do
            if [[ -n ${VAL["$k"]+x} ]]; then
                jred R10 "$f" "$j" "${k#"$f|$j|"}" "step $i carries '${k##*.}': a conditional stage is a skipped stage, and a tolerated failure is a green one"
            fi
        done
        if [[ -n $u ]]; then
            if [[ $u == actions/checkout@* ]]; then
                checkouts=$((checkouts + 1))
                [[ ${VAL["$sp.with.persist-credentials"]-} == false ]] || jred R6 "$f" "$j" "steps[$i]" "step $i is a checkout without persist-credentials: false, so the token stays in the persistent workspace's .git/config"
            else
                jred R6 "$f" "$j" "steps[$i].uses" "step $i uses '$u': the lane runs actions/checkout and the stage script, nothing else"
            fi
        elif [[ -n $r ]]; then
            if [[ $r =~ $re ]]; then
                stages+=("${BASH_REMATCH[1]}")
            else
                jred R10 "$f" "$j" "steps[$i].run" "step $i runs '$r', not 'bash scripts/ci_jetson_gpu.sh STAGE': the log judge, the floors and the pins live in the script"
            fi
        else
            jred R10 "$f" "$j" "steps[$i]" "step $i has neither uses nor run"
        fi
    done
    ((checkouts >= 1)) || jred R6 "$f" "$j" steps "no actions/checkout step"
    st="${stages[*]-}"
    [[ $st == "$seq" ]] || jred R10 "$f" "$j" steps "the stage calls are '${st:-none}' in order, must be '$seq'"
    v=${VAL["$f|$j|continue-on-error"]-}
    [[ -z $v || $v == false ]] || jred R10 "$f" "$j" "continue-on-error" "continue-on-error is '$v': a red job must stay red"
}

# r_ci_if FILE JOB -- R8 for the two ci.yml jobs
r_ci_if() {
    local f=$1 j=$2 v=${VAL["$1|$2|if"]-} frag
    local -a need=()
    if [[ $j == gpu-e2e-jetson ]]; then
        need=("!cancelled()" "needs.gpu-failfast-jetson.result == 'success'")
    else
        need=("!cancelled()" "github.event.pull_request.head.repo.full_name == github.repository" "github.event_name == 'pull_request'" "needs.gpu-touched.outputs.gpu_touched == '1'" "github.event_name == 'workflow_dispatch'")
    fi
    if [[ -z $v ]]; then
        jred R8 "$f" "$j" if "no if: the job would start on every event, the merge queue and a push to main among them"
        return
    fi
    for frag in "${need[@]}"; do
        [[ $v == *"$frag"* ]] || jred R8 "$f" "$j" if "if: lacks '$frag'"
    done
    for frag in merge_group "'push'" pull_request_target; do
        if [[ $v == *"$frag"* ]]; then jred R8 "$f" "$j" if "if: names $frag: the lane is advisory and starts on a same-repository pull request or a dispatch only"; fi
    done
}

# r_nightly_on FILE -- R8 for the nightly: exactly schedule + workflow_dispatch, and the schedule has a cron
r_nightly_on() {
    local f=$1 kk t cron=0 job=jetson-gpu
    local -a trig=()
    items "$f" "-" on
    trig=("${ITEMS[@]}")
    for kk in "${!VAL[@]}"; do
        [[ $kk == "$f|-|on."* ]] || continue
        t=${kk#"$f|-|on."}
        t=${t%%[.[]*}
        case " ${trig[*]-} " in *" $t "*) ;; *) trig+=("$t") ;; esac
        if [[ $kk == "$f|-|on.schedule["*"].cron" && -n ${VAL[$kk]} ]]; then cron=1; fi
    done
    ITEMS=("${trig[@]}")
    same_set "schedule workflow_dispatch" || jred R8 "$f" "$job" on "the triggers are [${trig[*]-}], must be exactly schedule + workflow_dispatch: the lane must not be reachable from a pull request, a push or another workflow"
    ((cron)) || jred R8 "$f" "$job" on "the schedule has no cron: a nightly that never fires is the dead lane the liveness switch exists to catch"
}

r_concurrency() {
    local f=$1 j=$2 mode=$3 g c
    if [[ $mode == nightly ]]; then
        g=${VAL["$f|-|concurrency.group"]-}
        [[ -n $g ]] || g=${VAL["$f|-|concurrency"]-}
        c=${VAL["$f|-|concurrency.cancel-in-progress"]-}
        [[ -n $g ]] || jred R11 "$f" "$j" concurrency "the nightly has no concurrency group: a dispatch during the scheduled run would start a second cold build on the one box"
        [[ $c == false ]] || jred R11 "$f" "$j" concurrency "cancel-in-progress is '${c:-unset}', must be false: a cancelled cold build is the most expensive thing the lane does"
        return
    fi
    g=${VAL["$f|$j|concurrency.group"]-}
    [[ -n $g ]] || g=${VAL["$f|$j|concurrency"]-}
    c=${VAL["$f|$j|concurrency.cancel-in-progress"]-}
    if [[ -z $g ]]; then
        jred R11 "$f" "$j" concurrency "no job-level concurrency group: two runs would queue on the one box unordered"
    else
        if [[ $g == ci-* ]]; then jred R11 "$f" "$j" concurrency.group "the group '$g' is the workflow's own: a job in its parent's group deadlocks against it"; fi
        if [[ ${GROUPS_SEEN-} == *"|$g|"* ]]; then jred R11 "$f" "$j" concurrency.group "the group '$g' is shared with another lane job"; fi
        GROUPS_SEEN="${GROUPS_SEEN-}|$g|"
    fi
    if [[ $c == true ]]; then jred R11 "$f" "$j" concurrency.cancel-in-progress "cancel-in-progress is the literal true: a dispatch would cancel a running cold build; derive it from the event"; fi
}

r_needs() {
    local f=$1 j=$2 want=$3 it
    items "$f" "$j" needs
    for it in "${ITEMS[@]}"; do
        case $it in
            \** | \&* | *\$\{\{*) cant "$f" "[$j] needs has an element I cannot read ('$it'): an alias or an expression hides the edges" ;;
        esac
    done
    same_set "$want" || jred R9 "$f" "$j" needs "needs is [${ITEMS[*]-}], must be exactly [${want:-nothing}]"
}

r_no_other_needs() {
    local kk oj it
    for kk in "${!VAL[@]}"; do
        [[ $kk == "$CI|"*"|@job" ]] || continue
        oj=${kk#"$CI|"}
        oj=${oj%"|@job"}
        if [[ $oj == gpu-failfast-jetson || $oj == gpu-e2e-jetson ]]; then continue; fi
        items "$CI" "$oj" needs
        for it in "${ITEMS[@]}"; do
            case $it in
                gpu-failfast-jetson | gpu-e2e-jetson)
                    jred R9 "$CI" "$oj" needs "needs the advisory jetson job '$it': one edge here and one box's availability gates the merge" ;;
                \** | \&* | *\$\{\{*)
                    cant "$CI" "[$oj] needs has an element I cannot read ('$it'): an alias or an expression could hide a jetson edge" ;;
            esac
        done
    done
}

r_selftest() {
    local root=$1 out rc last
    local re='^self-test: ([0-9]+) passed, ([0-9]+) failed$'
    [[ -f $root/$SCRIPT ]] || return 0
    out=$(cd "$root" && bash "$SCRIPT" --self-test 2>&1)
    rc=$?
    last=$(printf '%s\n' "$out" | tail -n 1)
    if [[ ! $last =~ $re ]]; then
        red R12 "$SCRIPT" "--self-test printed no 'self-test: N passed, M failed' line (rc=$rc, last line: '$last')"
    elif ((rc != 0 || BASH_REMATCH[2] != 0)); then
        red R12 "$SCRIPT" "--self-test failed (rc=$rc): $last"
    elif ((BASH_REMATCH[1] == 0)); then
        red R12 "$SCRIPT" "--self-test passed nothing ('$last'): a self-test over zero cases proves nothing"
    fi
}

r_box_named() {
    local kk rest f j
    for kk in "${!VAL[@]}"; do
        [[ $kk == *"|@jetson-word:"* ]] || continue
        f=${kk%%|*}
        rest=${kk#*|}
        j=${rest%%|*}
        case "$f|$j" in
            "$CI|gpu-failfast-jetson" | "$CI|gpu-e2e-jetson" | "$NIGHTLY|jetson-gpu" | "$NIGHTLY|-") continue ;;
        esac
        red R13 "$f:${LINE[$kk]}" "[$j] names the Jetson box outside the lane: ${VAL[$kk]}"
    done
}

# judge ROOT -- RED / CANT lines on stderr; returns 0 holds, 1 a violation, 2 cannot judge
judge() {
    local root=$1 spec file job cap seq rel f mj="" mt="" lane_loaded=" "
    VAL=(); LINE=(); NRED=0; NCANT=0; NROWS=0; GROUPS_SEEN=""
    for rel in "$CI" "$NIGHTLY"; do
        if [[ -r $root/$rel ]]; then load_rows "$root" "$rel"; lane_loaded+="$rel "; fi
    done
    # any other workflow or ci section file that mentions the box is read too (R13)
    for f in "$root"/.github/workflows/*.yml "$root"/.github/workflows/*.yaml "$root"/ci/*.yml "$root"/ci/*.yaml; do
        [[ -f $f ]] || continue
        rel=${f#"$root"/}
        [[ $lane_loaded == *" $rel "* ]] && continue
        if grep -qi jetson "$f" 2>/dev/null; then load_rows "$root" "$rel"; fi
    done
    if [[ -f $root/$SCRIPT ]]; then
        script_const MAX_BUILD_JOBS "$root/$SCRIPT"; mj=$SCONST
        script_const MAX_TEST_THREADS "$root/$SCRIPT"; mt=$SCONST
        if [[ -z $mj || -z $mt ]]; then
            cant "$SCRIPT" "cannot read MAX_BUILD_JOBS / MAX_TEST_THREADS from the script: the env pins have no bound to be checked against"
        fi
    else
        red R1 "$SCRIPT" "the stage script is missing: every step of the lane runs it"
    fi
    mj=${mj:-0}; mt=${mt:-0}
    for spec in "${LANE_JOBS[@]}"; do
        IFS='|' read -r file job cap seq <<<"$spec"
        if [[ -z ${VAL["$file|$job|@job"]+x} ]]; then
            if [[ -r $root/$file ]]; then red R1 "$file" "job '$job' is not in the file"
            else red R1 "$file" "the workflow file is missing, so job '$job' does not exist"; fi
            continue
        fi
        r_labels "$file" "$job"
        r_env "$file" "$job" "$mj" "$mt"
        r_timeout "$file" "$job" "$cap"
        r_perms "$file" "$job"
        r_secrets "$file" "$job"
        r_steps "$file" "$job" "$seq"
        case $job in
            gpu-failfast-jetson) r_ci_if "$file" "$job"; r_needs "$file" "$job" "gpu-touched"; r_concurrency "$file" "$job" ci ;;
            gpu-e2e-jetson) r_ci_if "$file" "$job"; r_needs "$file" "$job" "gpu-failfast-jetson"; r_concurrency "$file" "$job" ci ;;
            jetson-gpu) r_nightly_on "$file"; r_needs "$file" "$job" ""; r_concurrency "$file" "$job" nightly; r_secrets "$file" "-" ;;
        esac
    done
    r_no_other_needs
    r_selftest "$root"
    r_box_named
    if ((NRED > 0)); then return 1; fi
    if ((NCANT > 0)); then return 2; fi
    return 0
}

# ---------------------------------------------------------------------------------------------------------------
# --self-test: the parser on every YAML form the rules read, then fixture trees
# ---------------------------------------------------------------------------------------------------------------
ST_PASS=0
ST_FAIL=0
st_ok() { ST_PASS=$((ST_PASS + 1)); printf '  ok    %s\n' "$1"; }
st_bad() { ST_FAIL=$((ST_FAIL + 1)); printf '  FAIL  %s\n' "$1"; }

# st_parse NAME YAML WANT -- WANT is the rows "job|key|value" (without the @ rows), one per line, in file order
st_parse() {
    local name=$1 yaml=$2 want=$3 got
    printf '%s\n' "$yaml" > "$WORK/p.yml"
    got=$(awk -v F=p.yml -f "$WORK/facts.awk" "$WORK/p.yml" | awk -F"$US" '$3 !~ /^@/ && !($2 == "-" && $3 == "jobs") { print $2 "|" $3 "|" $4 }')
    if [[ $got == "$want" ]]; then st_ok "parse: $name"; else st_bad "parse: $name"; printf '        want:\n%s\n        got:\n%s\n' "${want//$'\n'/$'\n'          }" "${got//$'\n'/$'\n'          }"; fi
}

# st_marks NAME YAML WANT -- the @ rows only ("job|@kind:line"), to pin the scanners and the job markers
st_marks() {
    local name=$1 yaml=$2 want=$3 got
    printf '%s\n' "$yaml" > "$WORK/p.yml"
    got=$(awk -v F=p.yml -f "$WORK/facts.awk" "$WORK/p.yml" | awk -F"$US" '$3 ~ /^@/ { print $2 "|" $3 }')
    if [[ $got == "$want" ]]; then st_ok "marks: $name"; else st_bad "marks: $name"; printf '        want:\n%s\n        got:\n%s\n' "${want//$'\n'/$'\n'          }" "${got//$'\n'/$'\n'          }"; fi
}

# shellcheck disable=SC2016 # the YAML under test holds literal expression openers; nothing here may expand
st_parser_cases() {
    st_parse "needs: scalar, flow, block, indentless block, multi-line flow" \
'jobs:
  a:
    needs: x
  b:
    needs: [x, "y"]
  c:
    needs:
      - x
      - y
  d:
    needs:
    - x
    - y
  e:
    needs: [x, y,
            z]' \
'a|needs|x
b|needs|[x, "y"]
c|needs|
c|needs[1]|x
c|needs[2]|y
d|needs|
d|needs[1]|x
d|needs[2]|y
e|needs|[x, y, z]'

    st_parse "runs-on, trailing comments, quotes keep their #" \
'jobs:
  a:
    runs-on: [self-hosted, Linux, ARM64, jetson]   # the box
    timeout-minutes: 15   # cap
    concurrency:
      group: "g # not a comment"
      cancel-in-progress: ${{ github.event_name == '"'"'pull_request'"'"' }}' \
'a|runs-on|[self-hosted, Linux, ARM64, jetson]
a|timeout-minutes|15
a|concurrency|
a|concurrency.group|g # not a comment
a|concurrency.cancel-in-progress|${{ github.event_name == '"'"'pull_request'"'"' }}'

    st_parse "steps: first key on the dash line, with-map, quoted name holding ::, block scalar holding key: value lines" \
'jobs:
  a:
    steps:
      - uses: actions/checkout@v7
        with:
          persist-credentials: false
      - name: "S1 under driver:: (the layer)"
        run: bash scripts/ci_jetson_gpu.sh s1
      - run: |
          echo "k: v"
          other: thing
      - run: bash scripts/ci_jetson_gpu.sh s2' \
'a|steps|
a|steps[1]|
a|steps[1].uses|actions/checkout@v7
a|steps[1].with|
a|steps[1].with.persist-credentials|false
a|steps[2]|
a|steps[2].name|S1 under driver:: (the layer)
a|steps[2].run|bash scripts/ci_jetson_gpu.sh s1
a|steps[3]|
a|steps[3].run||
a|steps[4]|
a|steps[4].run|bash scripts/ci_jetson_gpu.sh s2'

    st_parse "permissions: block, flow map, scalar; env block" \
'permissions:
  contents: read
jobs:
  a:
    permissions: {contents: read}
    env:
      A: "1"
      B: never
  b:
    permissions: read-all' \
'-|permissions|
-|permissions.contents|read
a|permissions|{contents: read}
a|env|
a|env.A|1
a|env.B|never
b|permissions|read-all'

    st_parse "on: block form with an empty trigger and a cron; quoted key; inline list" \
'"on":
  schedule:
    - cron: "50 21 * * *"   # nightly
  workflow_dispatch:
concurrency:
  group: g
  cancel-in-progress: false' \
'-|on|
-|on.schedule|
-|on.schedule[1]|
-|on.schedule[1].cron|50 21 * * *
-|on.workflow_dispatch|
-|concurrency|
-|concurrency.group|g
-|concurrency.cancel-in-progress|false'

    st_parse "on: inline flow list and scalar" \
'on: [push, pull_request]' \
'-|on|[push, pull_request]'

    st_parse "a plain multi-line scalar belongs to its key, a sibling key does not" \
'jobs:
  a:
    if: ${{ a
            && b }}
    needs: x' \
'a|if|${{ a && b }}
a|needs|x'

    st_marks "job markers and the two scanners; comments are not read" \
'# a jetson comment
jobs:
  a:
    runs-on: jetson   # jetson in a comment after a value
    env:
      X: ${{ secrets.FOO }}
  b:
    steps:
      - run: echo "Jetson"' \
'a|@job
a|@jetson-word:4
a|@secrets-ref:6
b|@job
b|@jetson-word:9'
}

# mut FILE JOB OP A B -- edit a fixture in place. JOB "*" is the whole file, else that job's block. OP: del (drop the
# lines holding the literal A), sub (replace the literal A by B), ins (add B after the first line holding A).
# @@ in B is a line break. Literals, not patterns: no escape sequence is ever interpreted.
mut() {
    local file=$1 job=$2 op=$3 a=$4 b=${5-}
    awk -v J="$job" -v OP="$op" -v A="$a" -v B="$b" '
        BEGIN { gsub(/@@/, "\n", B) }
        /^  [A-Za-z0-9_-]+:/ { cur = $0; sub(/^  /, "", cur); sub(/:.*$/, "", cur) }
        J == "*" || cur == J {
            if (index($0, A) > 0 && !(OP == "ins" && done)) {
                if (OP == "del") next
                i = index($0, A)
                if (OP == "sub") { print substr($0, 1, i - 1) B substr($0, i + length(A)); next }
                if (OP == "ins") { print; print B; done = 1; next }
            }
        }
        { print }
    ' "$file" > "$file.new" && mv "$file.new" "$file"
}

write_good() {
    local d=$1
    mkdir -p "$d/.github/workflows" "$d/scripts"
    cat > "$d/.github/workflows/ci.yml" <<'YML'
name: CI

on:
  push:
    branches: [main]
  pull_request:
  merge_group:
  workflow_dispatch:

concurrency:
  group: ci-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}

jobs:
  gpu-touched:
    runs-on: ubuntu-latest
    outputs:
      gpu_touched: ${{ steps.sel.outputs.gpu_touched }}
    steps:
      - uses: actions/checkout@v7
      - id: sel
        run: bash scripts/ci_gpu_touched.sh

  gx10:
    runs-on: [self-hosted, Linux, ARM64, gx10]
    needs: gpu-touched
    steps:
      - run: |
          echo "multi-line: with colons: and a #hash"
          echo done

  yoga:
    runs-on:
      - self-hosted
      - yoga
    needs:
      - gpu-touched
    steps:
      - run: echo yoga

  # the lane: advisory, one persistent box
  gpu-failfast-jetson:
    runs-on: [self-hosted, Linux, ARM64, jetson]
    timeout-minutes: 15
    needs: [gpu-touched]
    if: ${{ !cancelled() && ((github.event_name == 'pull_request' && github.event.pull_request.head.repo.full_name == github.repository && needs.gpu-touched.outputs.gpu_touched == '1') || github.event_name == 'workflow_dispatch') }}
    permissions:
      contents: read
    concurrency:
      group: jetson-failfast-${{ github.event.pull_request.number || github.ref }}
      cancel-in-progress: ${{ github.event_name == 'pull_request' }}
    env:
      APR_REQUIRE_GPU: "1"
      CARGO_BUILD_JOBS: "4"
      RUST_TEST_THREADS: "2"
      CARGO_TERM_COLOR: never
    steps:
      - uses: actions/checkout@v7
        with:
          persist-credentials: false
      - name: S0 the box (seconds, no compile)
        run: bash scripts/ci_jetson_gpu.sh preflight
      - name: "S1 aprender-gpu cuda lib tests under driver:: (the layer every other GPU test stands on)"
        run: bash scripts/ci_jetson_gpu.sh s1
      - name: S2 the rest of the aprender-gpu cuda lib tests
        run: bash scripts/ci_jetson_gpu.sh s2

  gpu-e2e-jetson:
    runs-on: [self-hosted, Linux, ARM64, jetson]
    timeout-minutes: 25
    needs: [gpu-failfast-jetson]
    if: ${{ !cancelled() && needs.gpu-failfast-jetson.result == 'success' }}
    permissions:
      contents: read
    concurrency:
      group: jetson-e2e-${{ github.event.pull_request.number || github.ref }}
      cancel-in-progress: ${{ github.event_name == 'pull_request' }}
    env:
      APR_REQUIRE_GPU: "1"
      CARGO_BUILD_JOBS: "4"
      RUST_TEST_THREADS: "2"
      CARGO_TERM_COLOR: never
    steps:
      - uses: actions/checkout@v7
        with:
          persist-credentials: false
      - name: S0 the box (seconds, no compile)
        run: bash scripts/ci_jetson_gpu.sh preflight
      - name: "S3 FALSIFY-CB-008 on the resident q4_k_m GGUF, then the aprender-serve gguf::cuda:: lib tests"
        run: bash scripts/ci_jetson_gpu.sh s3

  determinism:
    runs-on: [self-hosted, Linux, ARM64, clean-room]
    steps:
      - run: echo determinism

  gate:
    runs-on: [self-hosted, Linux, clean-room]   # arch-neutral
    needs: [determinism, yoga,
            gx10, gpu-touched]
    if: always()
    steps:
      - run: echo gate
YML
    cat > "$d/.github/workflows/jetson-gpu-nightly.yml" <<'YML'
name: Jetson GPU nightly

# a jetson comment, which no rule reads

on:
  schedule:
    - cron: "50 21 * * *"   # 21:50 UTC -> lands ~02:50 UTC
  workflow_dispatch:

permissions:
  contents: read

concurrency:
  group: jetson-gpu-nightly
  cancel-in-progress: false

jobs:
  jetson-gpu:
    runs-on: [self-hosted, Linux, ARM64, jetson]
    timeout-minutes: 45
    env:
      APR_REQUIRE_GPU: "1"
      CARGO_BUILD_JOBS: "4"
      RUST_TEST_THREADS: "2"
      CARGO_TERM_COLOR: never
    steps:
      - uses: actions/checkout@v7
        with:
          persist-credentials: false
      - name: S0 the box (seconds, no compile)
        run: bash scripts/ci_jetson_gpu.sh preflight
      - name: "S1 aprender-gpu cuda lib tests under driver:: (the layer every other GPU test stands on)"
        run: bash scripts/ci_jetson_gpu.sh s1
      - name: S2 the rest of the aprender-gpu cuda lib tests
        run: bash scripts/ci_jetson_gpu.sh s2
      - name: "S3 FALSIFY-CB-008 on the resident q4_k_m GGUF, then the aprender-serve gguf::cuda:: lib tests"
        run: bash scripts/ci_jetson_gpu.sh s3
YML
    cat > "$d/scripts/ci_jetson_gpu.sh" <<'SH'
#!/usr/bin/env bash
readonly MAX_BUILD_JOBS=4   # a bound
readonly MAX_TEST_THREADS=2
if [ "${1:-}" = "--self-test" ]; then
    echo "  ok    something"
    echo "self-test: 7 passed, 0 failed"
    exit 0
fi
exit 0
SH
}

# st_case NAME WANT_RC TAG [FILEKEY JOB OP A B]... -- copy the good tree, apply the edits, judge it.
# TAG: a rule (R3) the RED output must name, CANT for exit 2, - for none. FILEKEY: ci | nightly | script | a path.
st_case() {
    local name=$1 want_rc=$2 tag=$3 dir="$WORK/fx/$1" out="$WORK/fx/$1.out" rc path pat shown
    shift 3
    rm -rf "${dir:?}"; mkdir -p "$dir"; cp -R "$WORK/good/." "$dir/"
    while (($# >= 5)); do
        case $1 in ci) path=$CI ;; nightly) path=$NIGHTLY ;; script) path=$SCRIPT ;; *) path=$1 ;; esac
        if [[ $3 == rm ]]; then rm -f "${dir:?}/$path"; else mut "$dir/$path" "$2" "$3" "$4" "$5"; fi
        shift 5
    done
    st_judge "$name" "$want_rc" "$tag" "$dir" "$out"
}

# st_case_fn NAME WANT_RC TAG FUNCTION -- the edit is a function of the copied tree (adds files)
st_case_fn() {
    local name=$1 want_rc=$2 tag=$3 fn=$4 dir="$WORK/fx/$1" out="$WORK/fx/$1.out"
    rm -rf "${dir:?}"; mkdir -p "$dir"; cp -R "$WORK/good/." "$dir/"
    "$fn" "$dir"
    st_judge "$name" "$want_rc" "$tag" "$dir" "$out"
}

st_judge() {
    local name=$1 want_rc=$2 tag=$3 dir=$4 out=$5 rc pat shown
    judge "$dir" > /dev/null 2> "$out"
    rc=$?
    case $tag in
        -) pat='^$' ;;
        CANT) pat='^CANT ' ;;
        *) pat="^RED +$tag " ;;
    esac
    shown=$(head -n 3 "$out" | cut -c1-200)
    if ((rc != want_rc)); then
        st_bad "$name: exit $rc, wanted $want_rc"; printf '%s\n' "$shown"
    elif [[ $tag != - ]] && ! grep -Eq "$pat" "$out"; then
        st_bad "$name: exit $rc but no $tag line"; printf '%s\n' "$shown"
    elif [[ $tag == - && -s $out ]]; then
        st_bad "$name: a good tree printed output"; printf '%s\n' "$shown"
    else
        st_ok "$name (exit $rc${tag:+, $tag})"
    fi
}

fn_other_workflow() { mkdir -p "$1/.github/workflows"; printf 'name: other\njobs:\n  x:\n    runs-on: [self-hosted, jetson]\n    steps:\n      - run: echo x\n' > "$1/.github/workflows/other.yml"; }
fn_other_comment() { mkdir -p "$1/.github/workflows"; printf 'name: other\n# runs on the jetson box? no\njobs:\n  x:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo x\n' > "$1/.github/workflows/other.yml"; }
fn_ci_section() { mkdir -p "$1/ci"; printf 'sections:\n  gpu:\n    labels: [jetson]\n' > "$1/ci/sections.yml"; }
fn_empty_tree() { rm -rf "${1:?}/.github" "${1:?}/scripts"; }

# shellcheck disable=SC2016 # the YAML under test holds literal expression openers; nothing here may expand
st_fixture_cases() {
    local FF=gpu-failfast-jetson E2E=gpu-e2e-jetson
    local CANCEL="cancel-in-progress: \${{ github.event_name == 'pull_request' }}"
    write_good "$WORK/good"
    st_case good 0 -
    st_case_fn empty-tree 1 R1 fn_empty_tree
    st_case job-renamed 1 R1 ci '*' sub "$FF:" "$FF-x:"
    st_case nightly-missing 1 R1 nightly '*' rm '' ''
    st_case script-missing 1 R1 script '*' rm '' ''

    st_case labels-extra-cuda 1 R2 ci $FF sub 'ARM64, jetson]' 'ARM64, jetson, cuda]'
    st_case labels-clean-room 1 R2 ci $E2E sub 'ARM64, jetson]' 'ARM64, clean-room]'
    st_case labels-missing-arm64 1 R2 nightly '*' sub 'ARM64, ' ''
    st_case labels-block-form-is-green 0 - nightly '*' sub '    runs-on: [self-hosted, Linux, ARM64, jetson]' '    runs-on:@@      - self-hosted@@      - Linux@@      - ARM64@@      - jetson'

    st_case env-jobs-missing 1 R3 ci $FF del CARGO_BUILD_JOBS ''
    st_case env-jobs-above-the-script-bound 1 R3 ci $FF sub 'CARGO_BUILD_JOBS: "4"' 'CARGO_BUILD_JOBS: "16"'
    st_case env-jobs-zero 1 R3 ci $FF sub 'CARGO_BUILD_JOBS: "4"' 'CARGO_BUILD_JOBS: "0"'
    st_case env-threads-missing-in-e2e 1 R3 ci $E2E del RUST_TEST_THREADS ''
    st_case env-threads-above-the-script-bound 1 R3 ci $FF sub 'RUST_TEST_THREADS: "2"' 'RUST_TEST_THREADS: "8"'
    st_case env-threads-follow-the-script-bound 1 R3 script '*' sub 'MAX_TEST_THREADS=2' 'MAX_TEST_THREADS=1'
    st_case env-require-gpu-missing-in-nightly 1 R3 nightly '*' del APR_REQUIRE_GPU ''
    st_case env-require-gpu-zero 1 R3 ci $FF sub 'APR_REQUIRE_GPU: "1"' 'APR_REQUIRE_GPU: "0"'
    st_case env-color-missing 1 R3 ci $FF del CARGO_TERM_COLOR ''
    st_case env-inline-flow-map 1 R3 ci $E2E del 'APR_REQUIRE_GPU: "1"' '' ci $E2E del 'CARGO_BUILD_JOBS: "4"' '' ci $E2E del 'RUST_TEST_THREADS: "2"' '' ci $E2E del 'CARGO_TERM_COLOR: never' '' ci $E2E sub '    env:' '    env: {APR_REQUIRE_GPU: "1", CARGO_BUILD_JOBS: "4"}'
    st_case env-overridden-by-a-step 1 R3 ci $FF ins 'run: bash scripts/ci_jetson_gpu.sh s1' '        env:@@          CARGO_BUILD_JOBS: "16"'

    st_case timeout-missing 1 R4 ci $FF del timeout-minutes ''
    st_case timeout-default-six-hours 1 R4 ci $E2E sub 'timeout-minutes: 25' 'timeout-minutes: 360'
    st_case timeout-expression 1 R4 nightly '*' sub 'timeout-minutes: 45' 'timeout-minutes: ${{ vars.T }}'

    st_case perm-write 1 R5 ci $E2E sub 'contents: read' 'contents: write'
    st_case perm-extra-scope 1 R5 ci $FF ins 'contents: read' '      id-token: write'
    st_case perm-gone-from-the-nightly 1 R5 nightly '*' del 'contents: read' ''
    st_case perm-read-all 1 R5 ci $FF del 'contents: read' '' ci $FF sub '    permissions:' '    permissions: read-all'
    st_case perm-flow-map-is-green 0 - ci $E2E del 'contents: read' '' ci $E2E sub '    permissions:' '    permissions: {contents: read}'

    st_case checkout-keeps-the-token 1 R6 ci $FF del persist-credentials ''
    st_case checkout-persist-true 1 R6 nightly '*' sub 'persist-credentials: false' 'persist-credentials: true'
    st_case another-action 1 R6 ci $E2E ins '          persist-credentials: false' '      - uses: actions/cache@v4'
    st_case no-checkout 1 R6 nightly '*' del 'uses: actions/checkout@v7' '' nightly '*' del '        with:' '' nightly '*' del persist-credentials ''

    st_case secret-in-env 1 R7 ci $FF ins 'CARGO_TERM_COLOR: never' '      X: ${{ secrets.FOO }}'
    st_case token-in-the-nightly 1 R7 nightly '*' ins 'CARGO_TERM_COLOR: never' '      GH_TOKEN: ${{ github.token }}'

    st_case if-missing 1 R8 ci $FF del '    if: ${{' ''
    st_case if-no-fork-test 1 R8 ci $FF sub ' && github.event.pull_request.head.repo.full_name == github.repository' ''
    st_case if-no-selector 1 R8 ci $FF sub " && needs.gpu-touched.outputs.gpu_touched == '1'" ''
    st_case if-no-status-function 1 R8 ci $FF sub '!cancelled() && ' ''
    st_case if-merge-queue 1 R8 ci $FF sub "|| github.event_name == 'workflow_dispatch')" "|| github.event_name == 'merge_group' || github.event_name == 'workflow_dispatch')"
    st_case if-e2e-ignores-failfast 1 R8 ci $E2E sub "needs.gpu-failfast-jetson.result == 'success'" 'true'
    st_case nightly-on-pull-request 1 R8 nightly '*' ins '  workflow_dispatch:' '  pull_request:'
    st_case nightly-on-push 1 R8 nightly '*' ins '  workflow_dispatch:' '  push:'
    st_case nightly-without-cron 1 R8 nightly '*' del 'cron:' ''
    st_case nightly-without-schedule 1 R8 nightly '*' del 'cron:' '' nightly '*' del '  schedule:' ''

    st_case needs-flow-in-gate 1 R9 ci gate sub 'needs: [determinism, yoga,' 'needs: [gpu-failfast-jetson, determinism, yoga,'
    st_case needs-multi-line-flow-in-gate 1 R9 ci gate sub 'gx10, gpu-touched]' 'gx10, gpu-e2e-jetson]'
    st_case needs-block-in-yoga 1 R9 ci yoga sub '      - gpu-touched' '      - gpu-failfast-jetson'
    st_case needs-scalar-in-gx10 1 R9 ci gx10 sub 'needs: gpu-touched' 'needs: gpu-e2e-jetson'
    st_case needs-alias-cannot-be-read 2 CANT ci gx10 sub 'needs: gpu-touched' 'needs: *anchor'
    st_case needs-expression-in-a-flow-list-cannot-be-read 2 CANT ci gx10 sub 'needs: gpu-touched' 'needs: [gpu-touched, ${{ vars.N }}]'
    st_case needs-expression-in-a-block-list-cannot-be-read 2 CANT ci gx10 sub 'needs: gpu-touched' 'needs:@@      - ${{ vars.N }}'
    st_case needs-expression-in-the-lane-is-red 1 R9 ci $FF sub 'needs: [gpu-touched]' 'needs: [gpu-touched, ${{ vars.N }}]'
    st_case needs-failfast-on-e2e 1 R9 ci $FF sub 'needs: [gpu-touched]' 'needs: [gpu-touched, gpu-e2e-jetson]'
    st_case needs-e2e-lost-its-edge 1 R9 ci $E2E del 'needs: [gpu-failfast-jetson]' ''
    st_case needs-nightly-has-one 1 R9 nightly '*' ins 'timeout-minutes: 45' '    needs: [x]'
    st_case needs-scalar-is-green 0 - ci $FF sub 'needs: [gpu-touched]' 'needs: gpu-touched'
    st_case needs-block-is-green 0 - ci $FF sub 'needs: [gpu-touched]' 'needs:@@      - gpu-touched'

    st_case step-is-a-test-command 1 R10 ci $FF ins 'ci_jetson_gpu.sh s2' '      - run: make gpu-tests'
    st_case step-lost-its-run 1 R10 ci $FF del 'ci_jetson_gpu.sh s2' ''
    st_case stage-order 1 R10 ci $FF sub 'ci_jetson_gpu.sh s1' 'ci_jetson_gpu.sh sX' ci $FF sub 'ci_jetson_gpu.sh s2' 'ci_jetson_gpu.sh s1' ci $FF sub 'ci_jetson_gpu.sh sX' 'ci_jetson_gpu.sh s2'
    st_case stage-unknown 1 R10 ci $FF sub 'ci_jetson_gpu.sh s2' 'ci_jetson_gpu.sh s9'
    st_case step-if 1 R10 ci $FF ins 'ci_jetson_gpu.sh s1' '        if: false'
    st_case step-tolerated-failure 1 R10 ci $FF ins 'ci_jetson_gpu.sh s2' '        continue-on-error: true'
    st_case job-tolerated-failure 1 R10 nightly '*' ins 'timeout-minutes: 45' '    continue-on-error: true'
    st_case run-block-scalar 1 R10 ci $FF sub 'run: bash scripts/ci_jetson_gpu.sh s1' 'run: |@@          bash scripts/ci_jetson_gpu.sh s1'

    st_case concurrency-parents-group 1 R11 ci $FF sub 'group: jetson-failfast-' 'group: ci-'
    st_case concurrency-shared-group 1 R11 ci $E2E sub 'group: jetson-e2e-' 'group: jetson-failfast-'
    st_case concurrency-cancel-true 1 R11 ci $FF sub "$CANCEL" 'cancel-in-progress: true'
    st_case concurrency-missing 1 R11 ci $FF del 'group: jetson-failfast-' '' ci $FF del 'cancel-in-progress' ''
    st_case nightly-cancel-true 1 R11 nightly '*' sub 'cancel-in-progress: false' 'cancel-in-progress: true'
    st_case nightly-no-group 1 R11 nightly '*' del 'group: jetson-gpu-nightly' ''

    st_case selftest-ran-nothing 1 R12 script '*' sub '7 passed' '0 passed'
    st_case selftest-failed 1 R12 script '*' sub '0 failed' '2 failed'
    st_case selftest-exit-code 1 R12 script '*' sub 'exit 0' 'exit 3'
    st_case selftest-silent 1 R12 script '*' del 'self-test:' ''
    st_case bound-unreadable 2 CANT script '*' del MAX_BUILD_JOBS ''

    st_case box-in-another-job-flow 1 R13 ci determinism sub 'clean-room]' 'jetson]'
    st_case box-in-another-job-block 1 R13 ci yoga sub '      - yoga' '      - jetson'
    st_case box-behind-a-matrix 1 R13 ci determinism sub 'runs-on: [self-hosted, Linux, ARM64, clean-room]' 'runs-on: ${{ fromJSON(matrix.labels) }}@@    strategy:@@      matrix:@@        labels: [jetson-box]'
    st_case_fn box-in-another-workflow 1 R13 fn_other_workflow
    st_case_fn box-in-a-ci-section-file 1 R13 fn_ci_section
    st_case_fn box-in-a-comment-is-green 0 - fn_other_comment
}

self_test() {
    setup_work || return 2
    st_parser_cases
    st_fixture_cases
    printf 'self-test: %d passed, %d failed\n' "$ST_PASS" "$ST_FAIL"
    ((ST_FAIL == 0))
}

main() {
    local mode=judge root=$REPO_ROOT facts_file="" rc njobs=${#LANE_JOBS[@]}
    while (($# > 0)); do
        case $1 in
            --root) root=${2:?--root needs a directory}; shift 2 ;;
            --self-test) mode=selftest; shift ;;
            --facts) mode=facts; facts_file=${2:?--facts needs a file}; shift 2 ;;
            -h | --help) usage; exit 0 ;;
            *) usage >&2; exit 2 ;;
        esac
    done
    setup_work || exit 2
    case $mode in
        selftest) self_test; exit $? ;;
        facts)
            awk -v F="$facts_file" -f "$WORK/facts.awk" "$facts_file" | tr "$US" '\t'
            exit "${PIPESTATUS[0]}" ;;
    esac
    judge "$root"
    rc=$?
    if ((rc == 0)); then
        printf 'check_jetson_gpu_jobs: OK -- %d jobs judged over %d rows; R1..R13 hold (env pins, labels, gating, needs, steps, concurrency, script self-test)\n' "$njobs" "$NROWS"
    elif ((rc == 1)); then
        printf 'check_jetson_gpu_jobs: FAIL -- %d violation(s); see the RED lines above\n' "$NRED" >&2
    else
        printf 'check_jetson_gpu_jobs: CANNOT JUDGE -- %d thing(s) unreadable, no verdict\n' "$NCANT" >&2
    fi
    exit "$rc"
}

main "$@"
