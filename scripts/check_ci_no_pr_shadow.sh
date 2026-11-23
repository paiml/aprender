#!/usr/bin/env bash
# check_ci_no_pr_shadow.sh -- pr-review-shadow is not run on the PR or merge-queue path. #5996, C324.
#
# THE DEFECT. The pr-review-shadow section records a REFUSE verdict and exits 0 on a pull request: it can neither
# block nor pass, which is the third state C324 bans on a PR or release path (a check is a GATE that can block, or a
# LAB that never runs there). It was found by the quorum on #5994 and had been inside x86-main's section lists.
#
# RULE. No `fat_driver.py run --sections ...` list in the workflow resolves to pr-review-shadow (globs included; an indirect $VAR value is refused). The section stays defined in
# ci/sections.yml for a LAB lane; only the PR and merge_group lists are checked.
#
# Usage:  bash scripts/check_ci_no_pr_shadow.sh [--self-test] [WORKFLOW_FILE]
# Wired:  ci/explicit-test-commands.d/893-ci-no-pr-shadow.cmd (--self-test).
set -euo pipefail

# values(): every --sections value in the workflow, one per line. Comment lines are dropped BEFORE lines are joined, so a comment
# ending in a backslash cannot swallow code and a folded `run: >-` block whose value sits on the next line still reads as one string.
# ${{ ... }} expressions become @EXPR@; only the workspace-test shard names may carry one (it is a matrix index), any other part with one is refused.
values() {
    grep -v '^[[:space:]]*#' "$1" | tr '\n' ' ' | sed -e 's/\\ / /g' -e 's/\${{[^}]*}}/@EXPR@/g' \
        | grep -oE -- "--sec[a-z]*(=|[[:space:]]+)('[^']*'|\"[^\"]*\"|[^[:space:]]+)" | sed -E "s/^--sec[a-z]*(=|[[:space:]]+)//; s/^['\"]//; s/['\"]\$//"
}

check() { # FILE -> one FAIL line per --sections list that resolves to the section (globs included); rc 1 when any
    local f="$1" bad=0 n=0 v part
    while IFS= read -r v; do
        n=$((n + 1))
        case "$v" in *'$'*|*'`'*) echo "FAIL  indirect --sections value cannot be resolved: $v"; bad=1; continue ;; esac
        case "$v" in *\'*|*\"*) echo "FAIL  --sections value still holds a quote (splicing) and cannot be resolved: $v"; bad=1; continue ;; esac
        local IFS=,
        for part in $v; do
            part="${part#"${part%%[![:space:]]*}"}"; part="${part%"${part##*[![:space:]]}"}"   # fat_driver strips each part
            case "$part" in *@EXPR@*) case "$part" in workspace-test-shard?@EXPR@?@EXPR@?) part="${part//@EXPR@/?}" ;; *) echo "FAIL  --sections part holds an expression and cannot be resolved: '$part'"; bad=1; continue ;; esac ;; esac
            # shellcheck disable=SC2053  # the unquoted right side is the point: fat_driver matches a section name with fnmatch
            if [[ pr-review-shadow == $part ]]; then echo "FAIL  --sections '$v' resolves to pr-review-shadow (via '$part')"; bad=1; fi
        done
    done < <(values "$f")
    if [ "$n" -eq 0 ]; then echo "FAIL  no --sections list found in $f (the check would be vacuous)"; bad=1; fi
    return "$bad"
}

self_test() {
    local src="$1" td fails=0 got
    td="$(mktemp -d)"; trap 'rm -rf "${td:?}"' RETURN
    if got="$(check "$src")"; then echo "ok    green the workflow as committed"; else echo "FAIL  green: committed workflow is RED: $got"; fails=1; fi
    local anchor="--sections 'guard-tree,pr-review-sign"
    if ! grep -qF -- "$anchor" "$src"; then echo "FAIL  anchor missing from $src; self-test is vacuous"; return 1; fi
    mutant() { # LABEL SED-EXPR -> red expected
        sed -e "$2" "$src" > "$td/m.yml"
        if check "$td/m.yml" > /dev/null; then echo "FAIL  mutant stayed green: $1"; fails=1; else echo "ok    red   MUTANT: $1"; fi
    }
    mutant "shadow put back in a list" "s/--sections 'guard-tree,pr-review-sign/--sections 'guard-tree,pr-review-shadow,pr-review-sign/"
    mutant "glob pr-review-* resolves to shadow" "s/--sections 'guard-tree,pr-review-sign/--sections 'guard-tree,pr-review-*/"
    mutant "glob pr-review-s?a*" "s/--sections 'guard-tree,pr-review-sign/--sections 'guard-tree,pr-review-s?a*/"
    mutant "value on the next line (folded block)" "s/--sections 'guard-tree,pr-review-sign/--sections\n          'guard-tree,pr-review-shadow,pr-review-sign/"
    mutant "--sections=value form" "s/--sections 'guard-tree,pr-review-sign/--sections='guard-tree,pr-review-shadow,pr-review-sign/"
    mutant "indirect \$VAR value" "s/--sections 'guard-tree,pr-review-sign/--sections \"\$PR_SECS\" --x 'guard-tree,pr-review-sign/"
    mutant "no --sections list is vacuous, not green" "s/--sections/--other/"
    mutant "space after a comma" "s/--sections 'guard-tree,pr-review-sign/--sections 'guard-tree, pr-review-shadow,pr-review-sign/"
    mutant "abbreviated option --section" "s/--sections 'guard-tree,pr-review-sign/--section 'guard-tree,pr-review-shadow,pr-review-sign/"
    mutant "quote splicing" "s/--sections 'guard-tree,pr-review-sign/--sections 'guard-tree,'pr-review-shadow,pr-review-sign/"
    mutant "whole-value expression" "s/--sections 'guard-tree,pr-review-sign/--sections '\${{ vars.PR_SECS }}' --x 'guard-tree,pr-review-sign/"
    mutant "partial expression" "s/--sections 'guard-tree,pr-review-sign/--sections 'guard-tree,pr-review-\${{ env.S }},pr-review-sign/"
    mutant "adjacent quoting" "s/--sections 'guard-tree,pr-review-sign/--sections 'guard-tree,pr-review-''shadow,pr-review-sign/"
    # a comment ending in a backslash must not hide the code on the next line
    printf '  - run: |\n      # merge queue: sign only \\\n      python3 scripts/ci/fat_driver.py run --sections pr-review-shadow\n' > "$td/c.yml"
    if check "$td/c.yml" > /dev/null; then echo "FAIL  mutant: comment ending in a backslash hid the next line"; fails=1; else echo "ok    red   MUTANT: comment ending in a backslash does not swallow the next line"; fi
    return "$fails"
}

case "${1:-}" in
    --self-test) self_test "${2:-.github/workflows/ci.yml}"; exit $? ;;
    *) check "${1:-.github/workflows/ci.yml}" && echo "ok    no --sections list resolves to pr-review-shadow" ;;
esac
