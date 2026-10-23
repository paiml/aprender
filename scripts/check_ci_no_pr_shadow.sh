#!/usr/bin/env bash
# check_ci_no_pr_shadow.sh -- pr-review-shadow is not run on the PR or merge-queue path. #5996, C324.
#
# THE DEFECT. The pr-review-shadow section records a REFUSE verdict and exits 0 on a pull request: it can neither
# block nor pass, which is the third state C324 bans on a PR or release path (a check is a GATE that can block, or a
# LAB that never runs there). It was found by the quorum on #5994 and had been inside x86-main's section lists.
#
# RULE. No `fat_driver.py run --sections ...` list in the workflow names pr-review-shadow. The section stays defined in
# ci/sections.yml for a LAB lane; only the PR and merge_group lists are checked.
#
# Usage:  bash scripts/check_ci_no_pr_shadow.sh [--self-test] [WORKFLOW_FILE]
# Wired:  ci/explicit-test-commands.d/843-ci-no-pr-shadow.cmd (--self-test).
set -euo pipefail

check() { # FILE -> one FAIL line per --sections list that names the section; rc 1 when any
    local f="$1" bad=0 n=0 line
    while IFS= read -r line; do
        n=$((n + 1))
        case "$line" in
            *pr-review-shadow*) echo "FAIL  a section list names pr-review-shadow: $line"; bad=1 ;;
        esac
    done < <(awk '/--sections/ { print }' "$f")
    if [ "$n" -eq 0 ]; then echo "FAIL  no --sections list found in $f (the check would be vacuous)"; bad=1; fi
    return "$bad"
}

self_test() {
    local src="$1" td fails=0 got
    td="$(mktemp -d)"; trap 'rm -rf "${td:?}"' RETURN
    if got="$(check "$src")"; then echo "ok    green the workflow as committed"; else echo "FAIL  green: committed workflow is RED: $got"; fails=1; fi
    sed "s/pr-review-sign,determinism/pr-review-shadow,pr-review-sign,determinism/" "$src" > "$td/m1.yml"
    if check "$td/m1.yml" > /dev/null; then echo "FAIL  mutant: shadow put back in a list stayed green"; fails=1; else echo "ok    red   MUTANT: shadow put back in a --sections list"; fi
    sed "s/--sections/--other/" "$src" > "$td/m2.yml"
    if check "$td/m2.yml" > /dev/null; then echo "FAIL  mutant: a workflow with no --sections list stayed green"; fails=1; else echo "ok    red   MUTANT: no --sections list is vacuous, not green"; fi
    return "$fails"
}

case "${1:-}" in
    --self-test) self_test "${2:-.github/workflows/ci.yml}"; exit $? ;;
    *) check "${1:-.github/workflows/ci.yml}" && echo "ok    no section list names pr-review-shadow" ;;
esac
