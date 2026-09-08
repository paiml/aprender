#!/usr/bin/env bash
# check_pr_review_wiring.sh — the PR-review receipt guard family is reachable,
# and the 150-minute receipt JOB does not come back onto pull requests.
#
# WHAT CHANGED, AND WHY THIS FILE INVERTED (2026-09-08, BSE-15, BSE-001 §4 wave 5)
# -------------------------------------------------------------------------------
# Until today this guard asserted that ci.yml carried a `pr-review-receipt` job
# with a job-level `if:`, and R4 asserted that `if:`'s polarity per event. The
# job itself is now DELETED, so R1's old subject ("the job that invokes the
# receipt guard") no longer exists and R3/R4 have nothing to evaluate. Asserting
# the presence of a deleted job is not a weaker guard, it is a broken one.
#
# The deletion was gated on a query, recorded in ci.yml beside the removal:
# nothing `needs:` the job, it uploaded no artifact so no consumer could read
# one, and it is not a required context. `gate` had already stopped reading it
# (PP-066 C0-5, PRQ-013, #2982). It cost 150 minutes of two mutation sweeps on a
# clean-room runner on every push to every open PR.
#
# So the rules invert, and the FALSIFIER SURVIVES THE DELETION — that is the
# whole point of keeping this file rather than deleting it too:
#
#   R1  ci.yml declares NO job that invokes check_pr_review_receipt.sh.
#       This is the standing falsifier. Putting the job back — on
#       `pull_request` as it used to be, or on any other event — turns this
#       guard RED rather than passing quietly. The case table carries the old
#       wiring VERBATIM, asserted FAIL, so the row is about the thing that
#       actually happened and not about a shape someone invented.
#
#       Restoring the job deliberately is therefore a code change here plus a
#       new case-table row, which is the friction this repository asks for. It
#       is not a rule that can be traded away for a line in a baseline file.
#
#   R2  ci.yml declares NO workflow-level `paths:` / `paths-ignore:` filter.
#       UNCHANGED, and unrelated to the job: ci.yml is where BOTH required
#       checks live (`ci / gate`, `workspace-test`), so a path filter here is a
#       deadlock — every PR that misses the filter sits PENDING forever.
#       check_workflow_path_filters.sh governs the other workflows, where a
#       filter is legal and the risk is going dark instead of deadlocking.
#
#   R3  the receipt guard family is still REACHABLE. `check_pr_review_receipt.sh`
#       is a tracked `scripts/check_*.sh`, which is exactly guard_tree.sh's
#       derived universe (`git ls-files 'scripts/check_*.sh'`, BSE-001 PR-A), so
#       it runs in `guard-tree`, a job `gate` needs. Without this rule the
#       deletion could be followed by renaming or untracking the guard and
#       nothing would notice: R1 would still hold, vacuously, over a family that
#       no longer runs anywhere. R1 and R3 are the two halves of one claim —
#       the job is gone AND the guards it used to carry still run.
#
# WHAT THIS FILE DOES NOT CLAIM. The two mutation sweeps the job carried
# (`scripts/mutate-guard.sh`, `scripts/mutate_quorum_arm.sh`) and its 43-row bats
# fixture table are not `check_*.sh` and no workflow invokes them, so they run
# nowhere in CI today. That is stated in ci.yml at the deletion site and in the
# PR that removed them; it is deliberately NOT asserted here, because a rule
# nobody can satisfy is the mirror of one that cannot fail.
#
#   bash scripts/check_pr_review_wiring.sh             # check
#   bash scripts/check_pr_review_wiring.sh --self-test # case table, both polarities
#
# ENVIRONMENT
#   PR_REVIEW_CI_YML   workflow to inspect (default: .github/workflows/ci.yml).
#                      Used by --self-test to drive fixtures; there is no value
#                      of it that turns a check off.
#
# EXIT: 0 all three rules hold; 1 anything else.

set -uo pipefail

PROG=${0##*/}
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CI_YML="${PR_REVIEW_CI_YML:-$REPO_ROOT/.github/workflows/ci.yml}"

GUARD_BASENAME='check_pr_review_receipt.sh'
GUARD_RE="(^|[[:space:];&|(])((ba)?sh[[:space:]]+|[.]/)?[^[:space:]]*check_pr_review_receipt[.]sh([[:space:]]|$|['\"])"

# ---------------------------------------------------------------------------
# invoking_job <ci.yml> — name of each job whose steps invoke the receipt guard.
# Prints nothing when no job does. A MENTION is not an invocation: comments are
# stripped from the first `#` before matching, because a trailing `#` comment
# defeated the first version of this pattern.
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
# guard_is_tracked — R3. The receipt guard is a tracked scripts/check_*.sh, so
# guard_tree.sh's derived universe runs it. `git ls-files` and not a filesystem
# probe: guard_tree derives its universe the same way, and an untracked file is
# invisible to it however present it looks on disk.
# ---------------------------------------------------------------------------
guard_is_tracked() {
    git -C "$REPO_ROOT" ls-files --error-unmatch "scripts/$GUARD_BASENAME" >/dev/null 2>&1
}

# ---------------------------------------------------------------------------
# check_file <ci.yml> — R1..R3. 0 = all hold. Diagnostics on stdout.
# ---------------------------------------------------------------------------
check_file() {
    local f=$1 job filters n

    if [ ! -f "$f" ]; then
        printf 'FAIL R0: no workflow at %s\n' "$f"
        return 1
    fi

    # R1 — the standing falsifier: the receipt job must not be back.
    job=$(invoking_job "$f")
    if [ -n "$job" ]; then
        n=$(printf '%s\n' "$job" | grep -c .)
        printf 'FAIL R1: %s job(s) in %s invoke %s:\n' "$n" "$f" "$GUARD_BASENAME"
        printf '%s\n' "$job" | sed 's|^|          |'
        printf '        That job was DELETED on 2026-09-08 (BSE-15): 150 minutes of two\n'
        printf '        mutation sweeps on a clean-room runner, on every push to every open\n'
        printf '        PR, gating nothing — no `needs:`, no artifact, not a required\n'
        printf '        context, and `gate` had already stopped reading it (C0-5, #2982).\n'
        printf '        The guard itself still runs: it is a scripts/check_*.sh and so is in\n'
        printf '        guard_tree.sh'"'"'s derived universe (R3 below).\n'
        printf '        Bringing the job back is a deliberate change to this rule plus a new\n'
        printf '        case-table row — not an edit to ci.yml alone.\n'
        return 1
    fi
    printf 'ok  R1  no job in %s invokes %s (the receipt job stays deleted)\n' "$(basename "$f")" "$GUARD_BASENAME"

    # R2 — unchanged.
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

    # R3 — the other half of R1: deleted job, guards still reachable.
    if ! guard_is_tracked; then
        printf 'FAIL R3: scripts/%s is not tracked by git.\n' "$GUARD_BASENAME"
        printf '        guard_tree.sh derives its universe from\n'
        printf "        \`git ls-files 'scripts/check_*.sh'\`, so an untracked or renamed\n"
        printf '        guard runs NOWHERE — and R1 would keep holding over a family that\n'
        printf '        no longer exists. Track it under that name, or move its rules to a\n'
        printf '        guard that is tracked and say so here.\n'
        return 1
    fi
    printf 'ok  R3  scripts/%s is tracked, so guard_tree.sh runs it\n' "$GUARD_BASENAME"

    return 0
}

# ---------------------------------------------------------------------------
# --self-test — must-hold / must-fail rows over synthesized workflows. Every
# rule gets a mutation that turns it RED and a control that must stay GREEN,
# because "refuse everything" reads green otherwise.
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

    # assert_file <label> <PASS|FAIL> <file> [<expected-message-substring>]
    #
    # THE MESSAGE IS ASSERTED, NOT ONLY THE VERDICT. R1 has a zero branch and a
    # more-than-one branch and they reject on neighbouring conditions, so a
    # verdict-only table let three mutants of the old guard SURVIVE: the wrong
    # branch still said no.
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
    emit_ci "$TD/good.yml" '' '' ''
    assert_file 'ci.yml with no receipt job at all (the state this repo is in)' PASS "$TD/good.yml"

    # R1, THE STANDING FALSIFIER — the wiring this repository ran until
    # 2026-09-08, carried here verbatim. This row is why the job cannot come
    # back quietly, and it is the reason this file was kept rather than deleted
    # along with the job.
    emit_ci "$TD/r1-prback.yml" '' "    if: github.event_name == 'pull_request'" "$INVOKE"
    assert_file 'R1 the per-PR wiring this repository deleted on 2026-09-08' FAIL "$TD/r1-prback.yml" 'invoke check_pr_review_receipt.sh'

    # R1: dispatch-only is ALSO refused. The deletion was not "move it to
    # workflow_dispatch" — a dispatch-only job whose PR_NUMBER comes from a
    # pull_request context it no longer has is reachable and non-functional,
    # which is worse than absent. Restoring it needs this rule changed.
    emit_ci "$TD/r1-dispatch.yml" '' "    if: github.event_name == 'workflow_dispatch'" "$INVOKE"
    assert_file 'R1 a dispatch-only receipt job is refused too' FAIL "$TD/r1-dispatch.yml" 'invoke check_pr_review_receipt.sh'

    # R1: no `if:` at all — the property is the invocation, not the condition.
    emit_ci "$TD/r1-noif.yml" '' '' "$INVOKE"
    assert_file 'R1 an unconditional receipt job is refused' FAIL "$TD/r1-noif.yml" 'invoke check_pr_review_receipt.sh'

    # R1 must not fire on a MENTION. This is the exact defect
    # check_guards_are_wired.sh had, in the opposite direction: the name inside
    # a comment must NOT read as an invocation, or the guard reds a clean file.
    emit_ci "$TD/r1-comment.yml" '' '' \
        '      - run: echo skipped # bash scripts/check_pr_review_receipt.sh'
    assert_file 'R1 a name in a trailing comment is not an invocation' PASS "$TD/r1-comment.yml"

    # R2: a workflow-level path filter, both spellings.
    emit_ci "$TD/r2-paths.yml" '    paths: [src/**]' '' ''
    assert_file 'R2 workflow-level paths:' FAIL "$TD/r2-paths.yml" 'path filter'
    emit_ci "$TD/r2-ignore.yml" '    paths-ignore: [docs/**]' '' ''
    assert_file 'R2 workflow-level paths-ignore:' FAIL "$TD/r2-ignore.yml" 'path filter'

    # R0: a workflow that is not there is a failure, never a pass.
    row=$((row + 1))
    if out=$(check_file "$TD/nope.yml" 2>&1); then
        printf 'FAIL  row %-2s R0 a missing workflow must FAIL\n' "$row"; fails=1
    else
        case "$out" in *'no workflow at'*) printf 'ok    row %-2s R0 a missing workflow fails\n' "$row" ;;
                       *) printf 'FAIL  row %-2s R0 failed on the wrong branch: %s\n' "$row" "$out"; fails=1 ;; esac
    fi

    # R3 both polarities, against real repositories rather than a stub: the
    # rule is about `git ls-files`, so only a git tree can drive it.
    row=$((row + 1))
    R3REPO="$TD/r3repo"
    mkdir -p "$R3REPO/scripts" && git -C "$R3REPO" init -q 2>/dev/null
    : > "$R3REPO/scripts/$GUARD_BASENAME"
    git -C "$R3REPO" add "scripts/$GUARD_BASENAME" >/dev/null 2>&1
    if (REPO_ROOT="$R3REPO"; guard_is_tracked); then
        printf 'ok    row %-2s R3 a tracked guard satisfies the reachability rule\n' "$row"
    else
        printf 'FAIL  row %-2s R3 a tracked guard was reported untracked\n' "$row"; fails=1
    fi
    row=$((row + 1))
    git -C "$R3REPO" rm -q --cached "scripts/$GUARD_BASENAME" >/dev/null 2>&1
    if (REPO_ROOT="$R3REPO"; guard_is_tracked); then
        printf 'FAIL  row %-2s R3 an UNTRACKED guard passed — guard_tree.sh would never run it\n' "$row"; fails=1
    else
        printf 'ok    row %-2s R3 an untracked guard is refused (guard_tree derives from git ls-files)\n' "$row"
    fi

    [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED\n'; exit 1; }
    printf '\nSELF-TEST PASSED (%s/%s)\n' "$row" "$row"
    exit 0
fi

printf '=== the receipt JOB stays deleted and its guards stay reachable (%s) ===\n' "$PROG"
printf 'workflow: %s\n' "$CI_YML"
if check_file "$CI_YML"; then
    printf 'PASS\n'
    exit 0
fi
printf '\nBSE-15 (BSE-001 §4 wave 5): the pr-review-receipt job was deleted on\n'
printf '2026-09-08 after a recorded query showed nothing needed it and nothing read\n'
printf 'an artifact from it. Its guards still run through guard_tree.sh. Restoring\n'
printf 'the job is a change to R1 plus a case-table row, never an edit to ci.yml alone.\n'
exit 1
