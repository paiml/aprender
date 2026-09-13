#!/usr/bin/env bash
# check_receipt_gate_base_owned.sh — the check that judges a PR's own review
# receipt must be DEFINED FROM THE BASE, never from the PR's own HEAD.
#
# WHY THIS EXISTS
# ---------------
# Measured 2026-09-05 (PP-066 driver STEP A1). Ruleset "Green Main" requires
# the bare `gate` context (.github/workflows/ci.yml, job `gate`). `gate` has
#
#     needs: [ci, workspace-test, mutants, guard-runner-labels, pr-review-present]
#
# and exits 1 when `pr-review-present`'s result is `failure`. `pr-review-present`
# carries `if: github.event_name == 'pull_request'` — and GitHub reads a
# `pull_request`-triggered workflow's DEFINITION from the PR's own HEAD, not
# from the base branch. So the check that judges whether a PR shipped a valid
# review receipt is defined by the very PR under review: delete the step,
# comment it out, or `if: false` it in your own branch, and `gate` sees
# `skipped` and passes. This is the phantom-required-check deadlock's twin —
# not a check nothing runs, but a check whose verdict the subject controls.
#
# THE FIX (PRQ-013)
# ------------------
# The job that invokes `check_pr_review_arm4.sh` (without `--self-test` — the
# "this PR's own receipt" line, not the case-table run) must live in its OWN
# workflow, triggered by `pull_request_target` (base-defined: GitHub reads a
# `pull_request_target` workflow from the DEFAULT branch, regardless of what
# the PR's head carries) and `merge_group` (so the merge queue is not
# deadlocked waiting on an event the workflow never fires for), and must NOT
# also be triggered by `pull_request` (which would reintroduce the head-defined
# reading for the same job). `gate`, in turn, must not `needs:` that job, nor
# any job literally named `pr-review-present` — a `needs:` entry only resolves
# within the SAME workflow file, so `gate` depending on a base-owned job in a
# DIFFERENT file is structurally impossible; the only way `gate` can end up
# waiting on a head-defined receipt job is by keeping one, under either name,
# inside ci.yml itself.
#
# THE FOUR RULES (each independently falsifiable; B-for-"base-owned")
#
#   B1  exactly ONE workflow under .github/workflows/*.yml invokes
#       check_pr_review_arm4.sh WITHOUT --self-test on a non-comment `run:`
#       line — the "this PR's own receipt" invocation. Zero means the receipt
#       is judged nowhere (theater); two or more means two definitions to keep
#       in step, and today's defect (ci.yml carries it TWICE — once in
#       `pr-review-receipt`, a shadow job with no `needs:` weight, and again in
#       `pr-review-present`, which `gate` depends on) is exactly this shape.
#       `--self-test` invocations (the case-table run) do not count and may
#       live anywhere.
#
#   B2  that workflow's top-level `on:` declares BOTH `pull_request_target`
#       and `merge_group`, and does NOT declare `pull_request`. Parsed in its
#       two YAML spellings — a mapping under `on:` and a flow list
#       `on: [a, b]` — anything else is refused, never guessed.
#
#   B3  in .github/workflows/ci.yml, the job named `gate` needs: NO job which
#       (a) is itself named `pr-review-present`, or (b) invokes
#       check_pr_review_arm4.sh without --self-test. `needs:` cannot name a
#       job in another file, so this rule only ever fires against a job
#       ci.yml still carries internally — which is today's defect.
#
#   B4  the invoking job (identified by B1) carries a JOB-level `if:` (4-space
#       indent under the job key — the same convention check_pr_review_wiring.sh
#       uses for R3/R4) that evaluates TRUE on `pull_request_target` and
#       `merge_group`, and FALSE on `push` and `workflow_dispatch`. A missing
#       job-level `if:` is a FAIL; the evaluator understands only a
#       disjunction of `github.event_name == '<literal>'` and refuses anything
#       else rather than guess.
#
#   bash scripts/check_receipt_gate_base_owned.sh              # check
#   bash scripts/check_receipt_gate_base_owned.sh --self-test  # case table
#
# ENVIRONMENT
#   RECEIPT_GATE_WORKFLOWS_DIR   directory to scan for B1/B2/B4, and where
#                                ci.yml is expected for B3 (default:
#                                .github/workflows). Used by --self-test to
#                                point at fixture directories. There is no
#                                value of it that turns this check off.
#
# EXIT: 0 all four rules hold; 1 anything else, including a shape this guard
# cannot evaluate; 2 on a usage error.

set -uo pipefail

PROG=${0##*/}
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORKFLOWS_DIR="${RECEIPT_GATE_WORKFLOWS_DIR:-$REPO_ROOT/.github/workflows}"

GUARD_BASENAME='check_pr_review_arm4.sh'
# Same invocation shape check_pr_review_wiring.sh uses: a MENTION (a trailing
# `#` comment) is not an invocation, so comments are stripped from the first
# `#` before matching. `[.]` and not `\.` — a dynamic-regex `\.` warns on some
# awk builds, and a warning scrolling past a guard's own diagnostics is how a
# real one gets missed.
GUARD_RE="(^|[[:space:];&|(])((ba)?sh[[:space:]]+|[.]/)?[^[:space:]]*check_pr_review_arm4[.]sh([[:space:]]|\$|['\"])"

EVENTS_TRUE='pull_request_target merge_group'
EVENTS_FALSE='push workflow_dispatch'

# ---------------------------------------------------------------------------
# invocation_loci <dir> — one "<file>:<job>" line per (file, job) whose steps
# invoke check_pr_review_arm4.sh WITHOUT --self-test on a non-comment `run:`
# line, across every *.yml/*.yaml directly inside <dir>. Prints nothing when
# there are none.
# ---------------------------------------------------------------------------
invocation_loci() {
    local dir=$1 f jobs j
    find "$dir" -maxdepth 1 \( -name '*.yml' -o -name '*.yaml' \) 2>/dev/null | LC_ALL=C sort |
    while IFS= read -r f; do
        jobs=$(awk -v re="$GUARD_RE" '
            /^jobs:[[:space:]]*$/  { injobs = 1; next }
            !injobs                { next }
            /^[^[:space:]#]/       { injobs = 0; next }
            /^  [A-Za-z0-9_-]+:/   { job = $0; sub(/^  /, "", job); sub(/:.*$/, "", job); next }
            {
                line = $0; sub(/#.*$/, "", line)
                if (job != "" && line ~ re && line !~ /--self-test/) { print job }
            }
        ' "$f" | LC_ALL=C sort -u)
        [ -z "$jobs" ] && continue
        for j in $jobs; do
            printf '%s:%s\n' "$f" "$j"
        done
    done
}

# ---------------------------------------------------------------------------
# job_level_if <file> <job> — the job-level `if:` expression, verbatim.
# Indentation IS the definition: 4 spaces is the job's own key, 6+ belongs to
# a step. Prints nothing when the job has no job-level `if:`.
# ---------------------------------------------------------------------------
job_level_if() {
    awk -v want="$2" '
        /^jobs:[[:space:]]*$/  { injobs = 1; next }
        !injobs                { next }
        /^[^[:space:]#]/       { injobs = 0; next }
        /^  [A-Za-z0-9_-]+:/   { cur = $0; sub(/^  /, "", cur); sub(/:.*$/, "", cur); inj = (cur == want); next }
        inj && /^    if:[[:space:]]*[^[:space:]]/ {
            line = $0
            sub(/^    if:[[:space:]]*/, "", line)
            sub(/[[:space:]]+$/, "", line)
            print line
        }
    ' "$1"
}

# ---------------------------------------------------------------------------
# job_needs <file> <job> — one job name per line from the job's `needs:`,
# handling both `needs: [a, b]` (flow list, same line) and a block list
# (`needs:` then `      - a` items). Prints nothing when the job has none.
# ---------------------------------------------------------------------------
job_needs() {
    awk -v want="$2" '
        /^jobs:[[:space:]]*$/  { injobs = 1; next }
        !injobs                { next }
        /^[^[:space:]#]/       { injobs = 0; next }
        /^  [A-Za-z0-9_-]+:/   { cur = $0; sub(/^  /, "", cur); sub(/:.*$/, "", cur); inj = (cur == want); inlist = 0; next }
        inj && /^    needs:[[:space:]]*\[/ {
            line = $0
            sub(/^    needs:[[:space:]]*\[/, "", line)
            sub(/\][[:space:]]*$/, "", line)
            n = split(line, arr, ",")
            for (i = 1; i <= n; i++) {
                item = arr[i]
                gsub(/^[[:space:]]+/, "", item)
                gsub(/[[:space:]]+$/, "", item)
                if (item != "") print item
            }
            next
        }
        inj && /^    needs:[[:space:]]*$/ { inlist = 1; next }
        inj && inlist && /^      - / {
            item = $0
            sub(/^      - /, "", item)
            gsub(/[[:space:]]+$/, "", item)
            print item
            next
        }
        inj && inlist && !/^      - / { inlist = 0 }
    ' "$1"
}

# ---------------------------------------------------------------------------
# top_level_on_events <file> — one event name per line from the workflow's
# top-level `on:`. Returns 1 (nothing printed) when the shape is not one of
# the two this guard understands: a mapping under `on:`, or a same-line flow
# list `on: [a, b]`. REFUSES rather than guesses.
# ---------------------------------------------------------------------------
top_level_on_events() {
    local f=$1 flow inner tokens
    flow=$(grep -m1 -E '^on:[[:space:]]*\[.*\][[:space:]]*$' "$f" 2>/dev/null || true)
    if [ -n "$flow" ]; then
        inner=$(printf '%s\n' "$flow" | sed -E 's/^on:[[:space:]]*\[[[:space:]]*//; s/[[:space:]]*\][[:space:]]*$//')
        printf '%s\n' "$inner" | tr ',' '\n' | sed -E "s/^[[:space:]]*['\"]?//; s/['\"]?[[:space:]]*\$//"
        return 0
    fi
    # An `on: [...` whose bracket never closes on the same line, or any other
    # single-line scalar (`on: "pull_request_target"`), is unevaluable.
    if grep -qE '^on:[[:space:]]*\[' "$f" 2>/dev/null; then
        return 1
    fi

    if ! grep -qE '^on:[[:space:]]*$' "$f" 2>/dev/null; then
        return 1
    fi
    tokens=$(awk '
        /^on:[[:space:]]*$/     { ion = 1; next }
        ion && /^[^[:space:]#]/ { ion = 0 }
        ion {
            line = $0; sub(/#.*$/, "", line)
            if (line ~ /^[[:space:]]*$/)   { next }
            if (line ~ /^  [A-Za-z0-9_]+:/) { key = line; sub(/^  /, "", key); sub(/:.*$/, "", key); print key; next }
            if (line ~ /^    /)             { next }
            print "UNEVALUABLE"
        }
    ' "$f")
    case "$tokens" in
        *UNEVALUABLE*) return 1 ;;
        *) [ -n "$tokens" ] && printf '%s\n' "$tokens"; return 0 ;;
    esac
}

# ---------------------------------------------------------------------------
# eval_if <expr> <event_name>
#   0 -> the expression is TRUE for that event
#   1 -> the expression is FALSE for that event
#   2 -> this guard cannot evaluate the expression (a FAILURE, never a pass)
# Copied from check_pr_review_wiring.sh's evaluator (same narrow shape:
# a disjunction of github.event_name == '<literal>', refuse anything else).
# ---------------------------------------------------------------------------
eval_if() {
    local expr=$1 ev=$2 norm lits
    norm=$(printf '%s' "$expr" | sed 's/[[:space:]][[:space:]]*/ /g; s/^ //; s/ $//')
    if ! grep -Eq -- "^github\.event_name == '[a-z_]+'( \|\| github\.event_name == '[a-z_]+')*\$" <<<"$norm"; then
        return 2
    fi
    lits=$(grep -oE "'[a-z_]+'" <<<"$norm" | tr -d "'" | tr '\n' ' ')
    case " $lits " in
        *" $ev "*) return 0 ;;
        *)         return 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# check_b1 <dir> — sets B1_FILE / B1_JOB on success (exactly one locus).
# Prints diagnostics. 0 = holds.
# ---------------------------------------------------------------------------
B1_FILE=""
B1_JOB=""
check_b1() {
    local dir=$1 loci n
    B1_FILE=""
    B1_JOB=""
    loci=$(invocation_loci "$dir")
    if [ -z "$loci" ]; then
        printf 'FAIL B1: 0 real invocations of %s across %s/*.yml\n' "$GUARD_BASENAME" "$dir"
        printf '        (a --self-test-only run, or one named only in a comment, does not\n'
        printf '        count). The gate would be theater — nothing judges the receipt.\n'
        return 1
    fi
    n=$(printf '%s\n' "$loci" | grep -c .)
    if [ "$n" -ne 1 ]; then
        printf 'FAIL B1: %s invokes %s in %s places:\n' "$dir" "$GUARD_BASENAME" "$n"
        printf '%s\n' "$loci" | sed 's|^|          |'
        printf '        Two or more is two definitions to keep in step.\n'
        return 1
    fi
    B1_FILE=${loci%%:*}
    B1_JOB=${loci##*:}
    printf 'ok  B1  %s invoked once, by job `%s` in %s\n' "$GUARD_BASENAME" "$B1_JOB" "$(basename "$B1_FILE")"
    return 0
}

# ---------------------------------------------------------------------------
# check_b2 <file> — 0 = holds.
# ---------------------------------------------------------------------------
check_b2() {
    local f=$1 events
    if ! events=$(top_level_on_events "$f"); then
        printf 'FAIL B2: cannot evaluate the top-level `on:` of %s.\n' "$(basename "$f")"
        printf '        This guard understands only a mapping under `on:` or a same-line\n'
        printf '        flow list `on: [a, b]`. Extend it with a case-table row; do not\n'
        printf '        widen the pattern to make this pass.\n'
        return 1
    fi
    events=" $(printf '%s' "$events" | tr '\n' ' ') "
    if [[ "$events" != *" pull_request_target "* ]]; then
        printf 'FAIL B2: %s does not declare `pull_request_target` in its top-level `on:`.\n' "$(basename "$f")"
        return 1
    fi
    if [[ "$events" != *" merge_group "* ]]; then
        printf 'FAIL B2: %s does not declare `merge_group` in its top-level `on:` — the\n' "$(basename "$f")"
        printf '        merge queue would deadlock waiting on an event it never fires for.\n'
        return 1
    fi
    if [[ "$events" == *" pull_request "* ]]; then
        printf 'FAIL B2: %s ALSO declares `pull_request` — that reintroduces the\n' "$(basename "$f")"
        printf '        head-defined reading for the same job (PRQ-013).\n'
        return 1
    fi
    printf 'ok  B2  %s declares pull_request_target + merge_group, not pull_request\n' "$(basename "$f")"
    return 0
}

# ---------------------------------------------------------------------------
# check_b3 <ci_yml> — 0 = holds.
# ---------------------------------------------------------------------------
check_b3() {
    local f=$1 needs invoking bad j ih
    if [ ! -f "$f" ]; then
        printf 'FAIL B3: no workflow at %s\n' "$f"
        return 1
    fi
    needs=$(job_needs "$f" gate)
    if [ -z "$needs" ]; then
        printf 'FAIL B3: job `gate` in %s has no `needs:` this guard can parse.\n' "$(basename "$f")"
        return 1
    fi
    invoking=$(invocation_loci "$(dirname "$f")" | awk -F: -v want="$f" '$1==want{ $1=""; sub(/^:/,""); print }')
    bad=""
    for j in $needs; do
        if [ "$j" = "pr-review-present" ]; then
            bad=$j
            break
        fi
        for ih in $invoking; do
            if [ "$j" = "$ih" ]; then
                bad=$j
                break 2
            fi
        done
    done
    if [ -n "$bad" ]; then
        printf 'FAIL B3: job `gate` in %s needs `%s` — a head-defined receipt job\n' "$(basename "$f")" "$bad"
        printf '        (PRQ-013). `gate` must not wait on a job whose own definition the\n'
        printf '        PR under review controls.\n'
        return 1
    fi
    printf 'ok  B3  job `gate` in %s needs no head-defined receipt job\n' "$(basename "$f")"
    return 0
}

# ---------------------------------------------------------------------------
# check_b4 <file> <job> — 0 = holds.
# ---------------------------------------------------------------------------
check_b4() {
    local f=$1 job=$2 ifexpr ev rc=0
    ifexpr=$(job_level_if "$f" "$job")
    if [ -z "$ifexpr" ]; then
        printf 'FAIL B4: job `%s` in %s has no JOB-level `if:` (4-space indent under the\n' "$job" "$(basename "$f")"
        printf '        job key).\n'
        return 1
    fi
    for ev in $EVENTS_TRUE; do
        eval_if "$ifexpr" "$ev"
        case $? in
            0) printf 'ok  B4  %-20s -> runs\n' "$ev" ;;
            1) printf 'FAIL B4: %s -> SKIPPED, but the receipt must be judged there.\n' "$ev"; rc=1 ;;
            *) printf 'FAIL B4: this guard cannot evaluate `%s`.\n' "$ifexpr"
               printf '        It understands only a disjunction of\n'
               printf "        github.event_name == '<literal>'. Extend the evaluator and add a\n"
               printf '        case-table row; do not widen the pattern to make this pass.\n'
               return 1 ;;
        esac
    done
    for ev in $EVENTS_FALSE; do
        eval_if "$ifexpr" "$ev"
        case $? in
            1) printf 'ok  B4  %-20s -> skipped\n' "$ev" ;;
            0) printf 'FAIL B4: %s -> runs. There is no PR to judge on this event.\n' "$ev"; rc=1 ;;
            *) printf 'FAIL B4: this guard cannot evaluate `%s`.\n' "$ifexpr"; return 1 ;;
        esac
    done
    return "$rc"
}

# ---------------------------------------------------------------------------
# check_all <workflows_dir> — B1..B4. 0 = all hold.
# ---------------------------------------------------------------------------
check_all() {
    local dir=$1 rc=0
    if ! check_b1 "$dir"; then
        rc=1
        printf 'SKIP B2/B4: no single invoking workflow identified by B1.\n'
    else
        check_b2 "$B1_FILE" || rc=1
    fi
    check_b3 "$dir/ci.yml" || rc=1
    if [ -n "$B1_FILE" ]; then
        check_b4 "$B1_FILE" "$B1_JOB" || rc=1
    fi
    if [ "$rc" -eq 0 ]; then
        printf 'ok  B1..B4 hold: %s judges the PR'"'"'s own receipt from the base\n' "$(basename "$B1_FILE")"
    fi
    return "$rc"
}

# ---------------------------------------------------------------------------
# --self-test — must-hold / must-fail rows over synthesized workflow pairs.
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
    TD=$(mktemp -d) || exit 1
    trap 'rm -rf "${TD:?}"' EXIT
    fails=0
    row=0

    # emit_ci <file> <needs-list-flow> <extra-job-block>
    emit_ci() {
        {
            printf 'name: CI\n\non:\n  push:\n    branches: [main]\n  pull_request:\n    branches: [main]\n  merge_group:\n  workflow_dispatch:\n\njobs:\n'
            printf '  ci:\n    runs-on: [self-hosted]\n    steps:\n      - run: echo build\n'
            [ -n "${3:-}" ] && printf '%s\n' "$3"
            printf '  gate:\n    runs-on: [self-hosted]\n    needs: [%s]\n    if: always()\n    steps:\n      - run: echo gate\n' "$2"
        } > "$1"
    }

    # emit_quorum <file> <on-block> <job-name> <job-if-line> <invocation-lines>
    emit_quorum() {
        {
            printf 'name: PR Review Quorum\n\n'
            printf '%s\n' "$2"
            printf '\njobs:\n  %s:\n    runs-on: [self-hosted]\n' "$3"
            [ -n "${4:-}" ] && printf '%s\n' "$4"
            printf '    steps:\n'
            printf '%s\n' "$5"
        } > "$1"
    }

    ON_MAP='on:
  pull_request_target:
    branches: [main]
  merge_group:
  workflow_dispatch:'
    ON_FLOW='on: [pull_request_target, merge_group]'
    JOBIF="    if: github.event_name == 'pull_request_target' || github.event_name == 'merge_group'"
    SELFTEST_LINE='      - run: bash scripts/check_pr_review_arm4.sh --self-test'
    REAL_LINE='      - run: bash scripts/check_pr_review_arm4.sh'

    # assert <label> <PASS|FAIL> <dir> <ci_yml_name> [<expected-message-substring>]
    assert() {
        row=$((row + 1))
        local label=$1 want=$2 d=$3 ciyml=$4 msg=${5:-} out rc got
        out=$(RECEIPT_GATE_WORKFLOWS_DIR="$d" check_all "$d" 2>&1)
        rc=$?
        if [ "$rc" -eq 0 ]; then got=PASS; else got=FAIL; fi
        if [ "$got" != "$want" ]; then
            printf 'FAIL  row %-2s %s: wanted %s, got %s\n' "$row" "$label" "$want" "$got"
            printf '%s\n' "$out" | sed 's|^|             |'
            fails=1
            return
        fi
        if [ -n "$msg" ]; then
            case "$out" in
                *"$msg"*) ;;
                *)  printf 'FAIL  row %-2s %s: %s, but on the wrong branch.\n' "$row" "$label" "$want"
                    printf '             expected the diagnostic to contain: %s\n' "$msg"
                    printf '%s\n' "$out" | sed 's|^|             |'
                    fails=1
                    return ;;
            esac
        fi
        printf 'ok    row %-2s %s\n' "$row" "$label"
        : "${ciyml:-}"
    }

    # --- control: correctly wired, everything holds -------------------------
    d=$TD/control; mkdir -p "$d"
    emit_ci "$d/ci.yml" 'ci' ''
    emit_quorum "$d/pr-review-quorum.yml" "$ON_MAP" 'pr-review-present' "$JOBIF" "$SELFTEST_LINE
$REAL_LINE"
    assert 'a correctly wired pair' PASS "$d" ci.yml

    # --- B1 -------------------------------------------------------------------
    d=$TD/b1-zero; mkdir -p "$d"
    emit_ci "$d/ci.yml" 'ci' ''
    emit_quorum "$d/pr-review-quorum.yml" "$ON_MAP" 'pr-review-present' "$JOBIF" "$SELFTEST_LINE"
    assert 'B1 zero real invocations' FAIL "$d" ci.yml '0 real invocations'

    d=$TD/b1-two; mkdir -p "$d"
    emit_ci "$d/ci.yml" 'ci' ''
    emit_quorum "$d/pr-review-quorum.yml" "$ON_MAP" 'pr-review-present' "$JOBIF" "$SELFTEST_LINE
$REAL_LINE"
    emit_quorum "$d/pr-review-quorum-2.yml" "$ON_MAP" 'pr-review-present-2' "$JOBIF" "$REAL_LINE"
    assert 'B1 two workflows both invoke it' FAIL "$d" ci.yml 'places'

    d=$TD/b1-selftest-only; mkdir -p "$d"
    emit_ci "$d/ci.yml" 'ci' ''
    emit_quorum "$d/pr-review-quorum.yml" "$ON_MAP" 'pr-review-present' "$JOBIF" "$SELFTEST_LINE"
    assert 'B1 a --self-test-only invocation does not count' FAIL "$d" ci.yml '0 real invocations'

    d=$TD/b1-comment; mkdir -p "$d"
    emit_ci "$d/ci.yml" 'ci' ''
    emit_quorum "$d/pr-review-quorum.yml" "$ON_MAP" 'pr-review-present' "$JOBIF" \
        '      - run: echo skipped # bash scripts/check_pr_review_arm4.sh'
    assert 'B1 named only in a trailing comment' FAIL "$d" ci.yml '0 real invocations'

    # --- B2 -------------------------------------------------------------------
    d=$TD/b2-map; mkdir -p "$d"
    emit_ci "$d/ci.yml" 'ci' ''
    emit_quorum "$d/pr-review-quorum.yml" "$ON_MAP" 'pr-review-present' "$JOBIF" "$REAL_LINE"
    assert 'B2 mapping form' PASS "$d" ci.yml

    d=$TD/b2-flow; mkdir -p "$d"
    emit_ci "$d/ci.yml" 'ci' ''
    emit_quorum "$d/pr-review-quorum.yml" "$ON_FLOW" 'pr-review-present' "$JOBIF" "$REAL_LINE"
    assert 'B2 flow-list form' PASS "$d" ci.yml

    d=$TD/b2-pullreq; mkdir -p "$d"
    emit_ci "$d/ci.yml" 'ci' ''
    emit_quorum "$d/pr-review-quorum.yml" 'on:
  pull_request_target:
    branches: [main]
  pull_request:
    branches: [main]
  merge_group:
  workflow_dispatch:' 'pr-review-present' "$JOBIF" "$REAL_LINE"
    assert 'B2 pull_request also declared' FAIL "$d" ci.yml 'ALSO declares'

    d=$TD/b2-nomerge; mkdir -p "$d"
    emit_ci "$d/ci.yml" 'ci' ''
    emit_quorum "$d/pr-review-quorum.yml" 'on:
  pull_request_target:
    branches: [main]
  workflow_dispatch:' 'pr-review-present' "$JOBIF" "$REAL_LINE"
    assert 'B2 missing merge_group' FAIL "$d" ci.yml 'merge_group'

    d=$TD/b2-opaque; mkdir -p "$d"
    emit_ci "$d/ci.yml" 'ci' ''
    emit_quorum "$d/pr-review-quorum.yml" 'on: "pull_request_target"' 'pr-review-present' "$JOBIF" "$REAL_LINE"
    assert 'B2 an on: this guard cannot evaluate' FAIL "$d" ci.yml 'cannot evaluate'

    # --- B3 -------------------------------------------------------------------
    d=$TD/b3-named; mkdir -p "$d"
    emit_ci "$d/ci.yml" 'ci, pr-review-present' \
        '  pr-review-present:
    runs-on: [self-hosted]
    steps:
      - run: echo hi
'
    emit_quorum "$d/pr-review-quorum.yml" "$ON_MAP" 'pr-review-quorum-job' "$JOBIF" "$REAL_LINE"
    assert 'B3 gate needs literal pr-review-present' FAIL "$d" ci.yml 'needs `pr-review-present`'

    d=$TD/b3-renamed; mkdir -p "$d"
    emit_ci "$d/ci.yml" 'ci, custom-receipt' \
        '  custom-receipt:
    runs-on: [self-hosted]
    steps:
      - run: bash scripts/check_pr_review_arm4.sh
'
    emit_quorum "$d/pr-review-quorum.yml" "$ON_MAP" 'pr-review-quorum-job' "$JOBIF" "$REAL_LINE"
    assert 'B3 gate needs a differently-named job that invokes arm4' FAIL "$d" ci.yml 'needs `custom-receipt`'

    d=$TD/b3-clean; mkdir -p "$d"
    emit_ci "$d/ci.yml" 'ci, workspace-test' \
        '  workspace-test:
    runs-on: [self-hosted]
    steps:
      - run: echo hi
'
    emit_quorum "$d/pr-review-quorum.yml" "$ON_MAP" 'pr-review-present' "$JOBIF" "$REAL_LINE"
    assert 'B3 a clean gate' PASS "$d" ci.yml

    # --- B4 -------------------------------------------------------------------
    d=$TD/b4-noif; mkdir -p "$d"
    emit_ci "$d/ci.yml" 'ci' ''
    emit_quorum "$d/pr-review-quorum.yml" "$ON_MAP" 'pr-review-present' '' "$REAL_LINE"
    assert 'B4 job has no if:' FAIL "$d" ci.yml 'no JOB-level'

    d=$TD/b4-ok; mkdir -p "$d"
    emit_ci "$d/ci.yml" 'ci' ''
    emit_quorum "$d/pr-review-quorum.yml" "$ON_MAP" 'pr-review-present' "$JOBIF" "$REAL_LINE"
    assert 'B4 correct if: pull_request_target || merge_group' PASS "$d" ci.yml

    d=$TD/b4-push; mkdir -p "$d"
    emit_ci "$d/ci.yml" 'ci' ''
    emit_quorum "$d/pr-review-quorum.yml" "$ON_MAP" 'pr-review-present' \
        "    if: github.event_name == 'pull_request_target' || github.event_name == 'push'" "$REAL_LINE"
    assert 'B4 if: also true on push' FAIL "$d" ci.yml 'runs. There is no PR'

    d=$TD/b4-opaque; mkdir -p "$d"
    emit_ci "$d/ci.yml" 'ci' ''
    emit_quorum "$d/pr-review-quorum.yml" "$ON_MAP" 'pr-review-present' '    if: always()' "$REAL_LINE"
    assert 'B4 an if: this guard cannot evaluate' FAIL "$d" ci.yml 'cannot evaluate'

    [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED (%s/%s rows)\n' "$((row - fails))" "$row"; exit 1; }
    printf '\nSELF-TEST PASSED (%s/%s rows)\n' "$row" "$row"
    exit 0
fi

if [ "${1:-}" != "" ]; then
    printf 'usage: %s [--self-test]\n' "$PROG" >&2
    exit 2
fi

printf '=== the PR'"'"'s own receipt is judged from the base, not the head (%s) ===\n' "$PROG"
printf 'workflows dir: %s\n' "$WORKFLOWS_DIR"
if check_all "$WORKFLOWS_DIR"; then
    printf 'PASS\n'
    exit 0
fi
printf '\nPRQ-013: the job that invokes check_pr_review_arm4.sh (without --self-test)\n'
printf 'must live in a workflow triggered by pull_request_target + merge_group, never\n'
printf 'pull_request; `gate` must not needs: a head-defined receipt job.\n'
exit 1
