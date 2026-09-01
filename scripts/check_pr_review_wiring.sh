#!/usr/bin/env bash
# check_pr_review_wiring.sh — how the PR-review receipt guard is allowed to be
# wired into ci.yml, made mechanical.
#
# WHY THIS EXISTS
# ---------------
# PR-REVIEW-SKILL-002 v2 §9 row 6 states the rule PRREV-006 must not get wrong:
#
#   "wire into ci.yml beside existing guards | job-level `if:`, **not**
#    workflow-level `paths:` — a path-filtered required check never reports and
#    blocks branch protection forever"
#
# Both halves are scars, not style. A workflow-level `paths:` filter means the
# workflow does not run at all on a PR that misses the filter, so a check
# GitHub is told to require never produces a check run: the PR sits PENDING
# forever and nothing can merge — the phantom-required-check deadlock this
# repository has already hit once. A job-level `if:` is the opposite shape: the
# workflow runs, the job reports `skipped`, and `gate` can read that result and
# decide. `mutants` (ci.yml) is the in-repo precedent.
#
# Until now that rule lived in a COMMENT, and a comment is not a trigger. The
# same file already carries the receipt of what that costs: check_hardcoded_paths.sh
# was left unwired behind a comment saying to promote it "once the fleet carries
# pmat >= 3.32.0", nothing re-evaluated the condition, and 20 machine-specific
# paths landed through the gap. So the rule is checked here rather than written
# down here.
#
# THE FOUR RULES
#
#   R1  ci.yml INVOKES scripts/check_pr_review_receipt.sh on a non-comment line.
#       check_guards_are_wired.sh already asserts this generically, but through a
#       shrink-only BASELINE — and a baseline entry is exactly what someone
#       removing this wiring would reach for. Naming the file here means the
#       wiring cannot be traded away for a line in a text file.
#
#   R2  ci.yml declares NO workflow-level `paths:` / `paths-ignore:` filter.
#       ci.yml is where BOTH required checks live (`ci / gate` and
#       `workspace-test`), so a path filter here is the deadlock, not a
#       hypothetical. Scoped to ci.yml and named: check_workflow_path_filters.sh
#       governs the *other* workflows, where a filter is legal and the risk is
#       going dark instead of deadlocking.
#
#   R3  the job that invokes the receipt guard carries a JOB-level `if:`.
#       Job-level and step-level are distinguished by indentation, which is what
#       makes "job-level" mechanical: a job key sits at 2 spaces, its `if:` at 4,
#       and a step's `if:` at 8 or more. A step-level `if:` would leave the JOB
#       reporting success on an event where it checked nothing.
#
#   R4  that `if:` evaluates TRUE on `pull_request` and FALSE on `push`,
#       `merge_group` and `workflow_dispatch` — both polarities, because
#       "it has an `if:`" is satisfied by `if: false`, which is a gate that
#       never runs, and by `if: always()`, which is no gate at all.
#
# THE EVALUATOR IS DELIBERATELY NARROW, AND REFUSES RATHER THAN GUESSES.
# It understands exactly one expression shape — a disjunction of
# `github.event_name == '<literal>'` — and any other `if:` is a hard FAILURE
# saying so. That is the correct direction: a guard that silently widens its
# pattern to cover a form it cannot reason about is the defect this repository
# has shipped six times. Extending the evaluator is a code change with a new
# case-table row, not an accident.
#
#   bash scripts/check_pr_review_wiring.sh             # check
#   bash scripts/check_pr_review_wiring.sh --self-test # case table, both polarities
#
# ENVIRONMENT
#   PR_REVIEW_CI_YML   workflow to inspect (default: .github/workflows/ci.yml).
#                      Used by --self-test to drive fixtures; there is no value
#                      of it that turns a check off.
#
# EXIT: 0 all four rules hold; 1 anything else, including an `if:` this guard
# cannot evaluate.

set -uo pipefail

PROG=${0##*/}
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CI_YML="${PR_REVIEW_CI_YML:-$REPO_ROOT/.github/workflows/ci.yml}"

GUARD_BASENAME='check_pr_review_receipt.sh'
# Same invocation shape check_guards_are_wired.sh uses: a MENTION is not an
# invocation, and a trailing `#` comment defeated the first version of that
# pattern, so comments are stripped from the first `#` before matching.
# `[.]` and not `\.`: awk warns "escape sequence treated as plain ." on a
# dynamic regex, and a warning on stderr from a guard is how a real diagnostic
# gets scrolled past.
GUARD_RE="(^|[[:space:];&|(])((ba)?sh[[:space:]]+|[.]/)?[^[:space:]]*check_pr_review_receipt[.]sh([[:space:]]|$|['\"])"

# Events the workflow can be triggered by, and whether the receipt job must run.
# Driven as a table rather than asserted once: the FALSE rows are what stop
# `if: always()` and a step-level `if:` from reading as compliance.
EVENTS_TRUE='pull_request'
EVENTS_FALSE='push merge_group workflow_dispatch'

# ---------------------------------------------------------------------------
# invoking_job <ci.yml> — name of the job whose steps invoke the receipt guard.
# Prints nothing when no job does.
# ---------------------------------------------------------------------------
invoking_job() {
    awk -v re="$GUARD_RE" '
        /^jobs:[[:space:]]*$/            { injobs = 1; next }
        !injobs                          { next }
        /^[^[:space:]#]/                 { injobs = 0; next }
        /^  [A-Za-z0-9_-]+:/             { job = $0; sub(/^  /, "", job); sub(/:.*$/, "", job); next }
        {
            line = $0; sub(/#.*$/, "", line)
            if (job != "" && line ~ re) { print job }
        }
    ' "$1" | LC_ALL=C sort -u
}

# ---------------------------------------------------------------------------
# job_level_if <ci.yml> <job> — the job-level `if:` expression, verbatim.
# Indentation IS the definition: 4 spaces is the job's own key, 6+ belongs to a
# step. Prints nothing when the job has no job-level `if:`.
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
# workflow_path_filters <ci.yml> — every `paths:`/`paths-ignore:` key inside the
# top-level `on:` block, as "line: text". Prints nothing when there are none.
# ---------------------------------------------------------------------------
workflow_path_filters() {
    awk '
        /^on:/            { ino = 1; next }
        ino && /^[^[:space:]#]/ { ino = 0 }
        ino {
            line = $0; sub(/#.*$/, "", line)
            if (line ~ /^[[:space:]]+paths(-ignore)?:/) { printf "%d: %s\n", NR, line }
        }
    ' "$1"
}

# ---------------------------------------------------------------------------
# eval_if <expr> <event_name>
#   0 -> the expression is TRUE for that event
#   1 -> the expression is FALSE for that event
#   2 -> this guard cannot evaluate the expression (a FAILURE, never a pass)
# ---------------------------------------------------------------------------
eval_if() {
    local expr=$1 ev=$2 norm lits
    norm=$(printf '%s' "$expr" | sed 's/[[:space:]][[:space:]]*/ /g; s/^ //; s/ $//')
    # A herestring, never `printf ... | grep -q`: on a pipe grep can exit 141 on
    # SIGPIPE despite having MATCHED, and this repository has shipped that four
    # times in one day.
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
# check_file <ci.yml> — R1..R4. 0 = all hold. Diagnostics on stdout.
# ---------------------------------------------------------------------------
check_file() {
    local f=$1 job ifexpr ev filters rc

    if [ ! -f "$f" ]; then
        printf 'FAIL R0: no workflow at %s\n' "$f"
        return 1
    fi

    # R1
    job=$(invoking_job "$f")
    if [ -z "$job" ]; then
        printf 'FAIL R1: no job in %s invokes %s.\n' "$f" "$GUARD_BASENAME"
        printf '        A guard named only in a comment is not wired (PRREV-006, spec 9.6).\n'
        return 1
    fi
    if [ "$(printf '%s\n' "$job" | grep -c .)" -ne 1 ]; then
        printf 'FAIL R1: %s is invoked by more than one job:\n' "$GUARD_BASENAME"
        printf '%s\n' "$job" | sed 's|^|          |'
        printf '        Two jobs means two `if:` conditions to keep in step; R3/R4 would\n'
        printf '        then hold for one of them while the other went dark.\n'
        return 1
    fi
    printf 'ok  R1  %s is invoked by job `%s`\n' "$GUARD_BASENAME" "$job"

    # R2
    filters=$(workflow_path_filters "$f")
    if [ -n "$filters" ]; then
        printf 'FAIL R2: %s declares a workflow-level path filter:\n' "$f"
        printf '%s\n' "$filters" | sed 's|^|          |'
        printf '        This file carries BOTH required checks. A path-filtered required\n'
        printf '        check never reports: every PR that misses the filter sits PENDING\n'
        printf '        forever and nothing merges. Gate the JOB, never the workflow.\n'
        return 1
    fi
    printf 'ok  R2  no workflow-level paths:/paths-ignore: filter in %s\n' "$(basename "$f")"

    # R3
    ifexpr=$(job_level_if "$f" "$job")
    if [ -z "$ifexpr" ]; then
        printf 'FAIL R3: job `%s` has no JOB-level `if:` (4-space indent under the job key).\n' "$job"
        printf '        A step-level `if:` leaves the JOB reporting success on an event\n'
        printf '        where it checked nothing.\n'
        return 1
    fi
    if [ "$(printf '%s\n' "$ifexpr" | grep -c .)" -ne 1 ]; then
        printf 'FAIL R3: job `%s` has %s job-level `if:` keys; YAML keeps the last and\n' \
            "$job" "$(printf '%s\n' "$ifexpr" | grep -c .)"
        printf '        the earlier ones read as enforcement that is not there.\n'
        return 1
    fi
    printf 'ok  R3  job `%s` carries a job-level if: %s\n' "$job" "$ifexpr"

    # R4 — both polarities.
    rc=0
    for ev in $EVENTS_TRUE; do
        eval_if "$ifexpr" "$ev"
        case $? in
            0) printf 'ok  R4  %-18s -> runs\n' "$ev" ;;
            1) printf 'FAIL R4: %s -> SKIPPED, but the receipt is addressed to a PR.\n' "$ev"; rc=1 ;;
            *) printf 'FAIL R4: this guard cannot evaluate `%s`.\n' "$ifexpr"
               printf '        It understands only a disjunction of\n'
               printf "        github.event_name == '<literal>'. Extend the evaluator and add a\n"
               printf '        case-table row; do not widen the pattern to make this pass.\n'
               return 1 ;;
        esac
    done
    for ev in $EVENTS_FALSE; do
        eval_if "$ifexpr" "$ev"
        case $? in
            1) printf 'ok  R4  %-18s -> skipped\n' "$ev" ;;
            0) printf 'FAIL R4: %s -> runs. There is no PR number on this event, so the\n' "$ev"
               printf '        receipt path evidence/pr-review/<pr>/<sha>/ has no subject.\n'; rc=1 ;;
            *) printf 'FAIL R4: this guard cannot evaluate `%s`.\n' "$ifexpr"; return 1 ;;
        esac
    done
    return "$rc"
}

# ---------------------------------------------------------------------------
# --self-test — must-hold / must-fail rows over synthesized workflows, plus the
# evaluator's own truth table. Every rule gets a mutation that turns it RED and
# a control that must stay GREEN, because "refuse everything" reads green
# otherwise.
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
    TD=$(mktemp -d) || exit 1
    trap 'rm -rf "${TD:?}"' EXIT
    fails=0
    row=0

    emit_ci() {  # emit_ci <file> <on-extra> <job-if-line> <invocation>
        {
            printf 'name: CI\n\non:\n  push:\n    branches: [main]\n  pull_request:\n    branches: [main]\n'
            [ -n "$2" ] && printf '%s\n' "$2"
            printf '  merge_group:\n  workflow_dispatch:\n\njobs:\n  guard-runner-labels:\n    runs-on: [self-hosted]\n    steps:\n      - run: bash scripts/check_runner_labels.sh\n  pr-review-receipt:\n    runs-on: [self-hosted]\n'
            [ -n "$3" ] && printf '%s\n' "$3"
            printf '    steps:\n'
            [ -n "$4" ] && printf '%s\n' "$4"
        } > "$1"
    }

    INVOKE='      - run: bash scripts/check_pr_review_receipt.sh tests/fixtures/pr-review/row-14-complete-gpu-review'
    JOBIF="    if: github.event_name == 'pull_request'"

    # assert_file <label> <PASS|FAIL> <file> [<expected-message-substring>]
    #
    # THE MESSAGE IS ASSERTED, NOT ONLY THE VERDICT, and that is measured rather
    # than stylistic. R1 and R3 each have a zero branch and a more-than-one
    # branch, and the more-than-one branch also rejects when the count is ZERO
    # (0 != 1). So `if [ -z "$job" ]` -> `if false` left the guard still
    # rejecting, on the neighbouring branch, with a message about "more than one
    # job" for a file that had none: verdict-only, three mutants of this guard
    # SURVIVED. This is the same finding tests/pr-review.bats records for the
    # receipt guard's B1, one level up.
    assert_file() {
        row=$((row + 1))
        local label=$1 want=$2 f=$3 msg=${4:-} out rc got
        out=$(PR_REVIEW_CI_YML="$f" check_file "$f" 2>&1)
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
    }

    # The control FIRST. Without a row that must stay GREEN, every mutation
    # below passes against a guard that refuses everything.
    emit_ci "$TD/good.yml" '' "$JOBIF" "$INVOKE"
    assert_file 'a correctly wired ci.yml' PASS "$TD/good.yml"

    # R1: the invocation removed.
    emit_ci "$TD/r1-none.yml" '' "$JOBIF" ''
    assert_file 'R1 no invocation at all' FAIL "$TD/r1-none.yml" 'no job in'

    # R1: a MENTION, not an invocation. This is the exact defect
    # check_guards_are_wired.sh had: the name inside a comment read as wiring.
    emit_ci "$TD/r1-comment.yml" '' "$JOBIF" \
        '      - run: echo skipped # bash scripts/check_pr_review_receipt.sh'
    assert_file 'R1 named only in a trailing comment' FAIL "$TD/r1-comment.yml" 'no job in'

    # R1: two jobs invoking it. Two jobs is two `if:` conditions to keep in step,
    # and R3/R4 would then hold for whichever one this guard happened to pick.
    emit_ci "$TD/r1-two.yml" '' "$JOBIF" "$INVOKE"
    printf '  second-receipt-job:\n    runs-on: [self-hosted]\n%s\n    steps:\n%s\n' \
        "$JOBIF" "$INVOKE" >> "$TD/r1-two.yml"
    assert_file 'R1 two jobs invoke it' FAIL "$TD/r1-two.yml" 'more than one job'

    # R2 DISCRIMINATION: a `paths:` key inside a JOB is not a workflow filter and
    # must stay GREEN. Several actions take one. Without this row, widening the
    # `on:`-block terminator so the scan never leaves it is a SURVIVING mutant:
    # the guard would start flagging `paths:` anywhere in the file, and every
    # other fixture here happens to have none, so nothing would notice.
    emit_ci "$TD/r2-injob.yml" '' "$JOBIF" \
        "      - uses: some/action@v1
        with:
          paths: |
            evidence/pr-review/**
$INVOKE"
    assert_file 'R2 a paths: key inside a job is not a workflow filter' PASS "$TD/r2-injob.yml"

    # R2: a workflow-level paths filter — the deadlock the spec names.
    emit_ci "$TD/r2-paths.yml" '    paths:
      - "evidence/pr-review/**"' "$JOBIF" "$INVOKE"
    assert_file 'R2 workflow-level paths: filter' FAIL "$TD/r2-paths.yml"

    emit_ci "$TD/r2-ignore.yml" '    paths-ignore:
      - "docs/**"' "$JOBIF" "$INVOKE"
    assert_file 'R2 workflow-level paths-ignore: filter' FAIL "$TD/r2-ignore.yml"

    # R3: no job-level if: at all.
    emit_ci "$TD/r3-noif.yml" '' '' "$INVOKE"
    assert_file 'R3 job has no if:' FAIL "$TD/r3-noif.yml" 'has no JOB-level'

    # R3: two job-level `if:` keys. YAML keeps the last, so the earlier one reads
    # as enforcement that is not there.
    emit_ci "$TD/r3-twoif.yml" '' "$JOBIF
    if: github.event_name == 'push'" "$INVOKE"
    assert_file 'R3 two job-level if: keys' FAIL "$TD/r3-twoif.yml" 'job-level `if:` keys'

    # R3: a STEP-level if: is not a job-level if:. The job would report success
    # on an event where the step was skipped.
    emit_ci "$TD/r3-stepif.yml" '' '' \
        "      - name: validate the receipt
        if: github.event_name == 'pull_request'
        run: bash scripts/check_pr_review_receipt.sh evidence/pr-review/1/deadbeef"
    assert_file 'R3 step-level if: does not count' FAIL "$TD/r3-stepif.yml" 'has no JOB-level'

    # R4: an if: that is never true.
    emit_ci "$TD/r4-never.yml" '' "    if: github.event_name == 'release'" "$INVOKE"
    assert_file 'R4 if: that never runs on a PR' FAIL "$TD/r4-never.yml"

    # R4: an if: that is always true — "has an if:" is not the property.
    emit_ci "$TD/r4-always.yml" '' \
        "    if: github.event_name == 'pull_request' || github.event_name == 'push'" "$INVOKE"
    assert_file 'R4 if: that also runs where there is no PR' FAIL "$TD/r4-always.yml"

    # R4: a form the evaluator does not understand must FAIL, not pass.
    emit_ci "$TD/r4-opaque.yml" '' '    if: always()' "$INVOKE"
    assert_file 'R4 an if: this guard cannot evaluate' FAIL "$TD/r4-opaque.yml"

    emit_ci "$TD/r4-negated.yml" '' "    if: github.event_name != 'push'" "$INVOKE"
    assert_file 'R4 a negated if: is refused, not guessed' FAIL "$TD/r4-negated.yml"

    # The evaluator's own truth table, independent of any workflow.
    eval_row() {  # eval_row <expr> <event> <0|1|2>
        row=$((row + 1))
        eval_if "$1" "$2"; local got=$?
        if [ "$got" -eq "$3" ]; then
            printf 'ok    row %-2s eval %-12s on %-16s -> %s\n' "$row" "$3" "$2" "$1"
        else
            printf 'FAIL  row %-2s eval on %s of `%s`: wanted %s, got %s\n' "$row" "$2" "$1" "$3" "$got"
            fails=1
        fi
    }
    eval_row "github.event_name == 'pull_request'" pull_request      0
    eval_row "github.event_name == 'pull_request'" push              1
    eval_row "github.event_name == 'pull_request'" merge_group       1
    eval_row "github.event_name == 'pull_request'" workflow_dispatch 1
    eval_row "github.event_name == 'push' || github.event_name == 'pull_request'" push 0
    # A LITERAL IS A WHOLE WORD, NOT A SUBSTRING. `pull_request` is a prefix of
    # the real event names `pull_request_target` and `pull_request_review`, so
    # dropping the space delimiters around the membership test turns a job
    # gated on `pull_request_target` into one this guard reports as running on
    # `pull_request`. Measured: without this row that mutation SURVIVES.
    eval_row "github.event_name == 'pull_request_target'" pull_request 1
    eval_row "github.event_name == 'pull_request'" pull_request_target 1
    # Irregular spacing is still the same expression. Without the normaliser the
    # shape check misses and every such `if:` is reported unevaluable, which is a
    # hard FAIL — a guard that reds a correctly wired workflow. Also a surviving
    # mutant without this row.
    eval_row "github.event_name  ==   'pull_request'" pull_request      0
    eval_row "always()"                            pull_request      2
    eval_row "github.event_name != 'push'"         pull_request      2
    eval_row "\${{ github.event_name == 'pull_request' }}" pull_request 2

    [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED\n'; exit 1; }
    printf '\nSELF-TEST PASSED (%s/%s)\n' "$row" "$row"
    exit 0
fi

printf '=== the PR-review receipt guard is wired job-level, not path-filtered (%s) ===\n' "$PROG"
printf 'workflow: %s\n' "$CI_YML"
if check_file "$CI_YML"; then
    printf 'PASS\n'
    exit 0
fi
printf '\nPR-REVIEW-SKILL-002 v2 §9 row 6: job-level `if:`, never a workflow-level\n'
printf '`paths:` filter. A path-filtered required check never reports and blocks\n'
printf 'branch protection forever.\n'
exit 1
