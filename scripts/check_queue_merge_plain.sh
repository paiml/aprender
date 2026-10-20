#!/usr/bin/env bash
# check_queue_merge_plain.sh -- every arming call on a merge-queue repo is plain `gh pr merge <pr> [--repo R]`:
# no method flag of any spelling, no --auto, no variable-held or extra argument. #5794, from the #5789 loss of 2026-10-20.
#
# THE DEFECT. #5789's auto-merge was armed at 12:39Z with method MERGE on a queue repo whose queue method is SQUASH.
# GitHub accepted it and the PR sat un-queued until 14:14Z, when a plain `gh pr merge` queued it. A method flag on a
# queue repo is a second source of truth that can disagree with the queue; the queue owns the method.
#
# RULE (after joining backslash continuations): in each non-comment line, the words after `pr merge` up to the end of
# the command (`;` `|` `&&` `||` `)` or a redirection) may be only: ONE positional (the PR/branch), and `--repo R` /
# `-R R` / `--repo=R`. Anything else -- --squash/-s/--squash=1/--merge/-m/--rebase/-r/--auto/--delete-branch, a
# second positional such as a held `$M` -- is flagged.
#
# Scope: scripts/**/*.sh, .claude/skills/**/*.md, docs/specifications/*.md. CLAUDE.md is operator-owned and not scanned.
# Wired: ci/explicit-test-commands.d/814-queue-merge-plain-5794.cmd runs it on the merge path (--self-test, which also
# scans the real tree), so a restored method flag turns CI RED.
# Usage:  bash scripts/check_queue_merge_plain.sh [--self-test]
set -euo pipefail

# Lines that name a method flag on purpose: fixtures that stub gh and classify its calls, not arming paths.
ALLOW='scripts/check_release_host_receipts.sh scripts/release/rehearse.sh scripts/check_queue_merge_plain.sh'

# flagged FILE -> "file:text" per offending command
flagged() {
    [ "${QMP_MUTANT:-}" = broken ] && return 0 # test-only: a scanner that flags nothing
    sed -e ':a' -e '/\\$/{N;s/\\\n/ /;ta' -e '}' "$1" 2> /dev/null | awk -v f="$1" '
        /^[[:space:]]*#/ { next }
        {
            line = $0
            while (match(line, /(^|[^[:alnum:]_])pr merge/) > 0) {
                rest = substr(line, RSTART + RLENGTH); line = rest
                n = split(rest, w, /[[:space:]]+/); pos = 0; bad = 0; skip = 0
                for (k = 1; k <= n; k++) {
                    t = w[k]; if (t == "") continue
                    if (t ~ /^(2?>|&>|&&|\|\||;|\||\)|`)/) break
                    closing = (t ~ /`$/)
                    gsub(/[)"`]+$/, "", t)
                    if (skip) { skip = 0; continue }
                    if (t == "--repo" || t == "-R") { skip = 1; continue }
                    if (t ~ /^--repo=/) continue
                    if (t ~ /^-/) { bad = 1; break }
                    pos++; if (pos > 1) { bad = 1; break }
                    if (closing) break
                }
                if (bad) { print f ":" $0; break }
            }
        }'
}

scan() {
    local root="$1" f a skip
    while IFS= read -r f; do
        skip=0
        for a in $ALLOW; do [ "$f" = "$a" ] && skip=1; done
        [ "$skip" = 1 ] && continue
        (cd "$root" && flagged "$f")
    done < <(cd "$root" && { find scripts -name '*.sh' -type f; find .claude/skills -name '*.md' -type f; find docs/specifications -maxdepth 1 -name '*.md' -type f; } 2> /dev/null | sort)
}

self_test() {
    local td fails=0 got flag
    td="$(mktemp -d)"; trap 'rm -rf "${td:?}"' RETURN
    mkdir -p "$td/scripts"
    check() { # WANT(red|green) LABEL  (content on stdin)
        cat > "$td/scripts/x.sh"; got="$(scan "$td")"
        if { [ "$1" = red ] && [ -n "$got" ]; } || { [ "$1" = green ] && [ -z "$got" ]; }; then echo "ok    $1  $2"; else echo "FAIL  $1  $2 (got: ${got:-nothing})"; fails=1; fi
    }
    check green "plain with --repo and a redirect" < <(printf 'out=$(gh pr merge "$pr" --repo "$repo" 2>&1)\n')
    check green "plain then &&" < <(printf 'gh pr merge "$BR" >/dev/null 2>&1 && say armed\n')
    check green "-R form and a comment" < <(printf 'gh pr merge -R o/r 12\n# gh pr merge 1 --squash --auto\n')
    for flag in --squash -s --merge -m --rebase -r --auto --squash=1 --delete-branch; do
        check red "method/option flag $flag" < <(printf 'gh pr merge "$pr" %s\n' "$flag")
    done
    check red "mid-line (arm_pr_automerge shape)" < <(printf 'out=$(gh pr merge "$N" --repo r --squash --auto 2>&1)\n')
    check red "flags before the positional (session_docs_commit shape)" < <(printf 'gh pr merge --squash --auto "$BR" >/dev/null\n')
    check red "backslash continuation" < <(printf 'gh pr merge "$pr" \\\n   --squash --auto\n')
    check red "flag held in a variable" < <(printf 'M=--squash\ngh pr merge "$pr" $M\n')
    check red "array-held arguments" < <(printf 'gh pr merge "$pr" "${ARGS[@]}"\n')
    mkdir -p "$td/.claude/skills/s"
    printf 'Runs `gh pr merge <N> --squash --auto` on PERMIT\n' > "$td/.claude/skills/s/SKILL.md"
    got="$(scan "$td")"
    if [ -n "$got" ]; then echo "ok    red  skill markdown is scanned"; else echo "FAIL  red  markdown not scanned"; fails=1; fi
    rm -rf "${td:?}/.claude"
    if [ -z "${QMP_MUTANT:-}" ]; then # MUTANT: a scanner that flags nothing must turn --self-test RED
        if QMP_MUTANT=broken bash "$0" --self-test > /dev/null 2>&1; then echo "FAIL  mutant: --self-test stayed green on a broken scanner"; fails=1; else echo "ok    red  MUTANT: broken scanner turns --self-test RED"; fi
    fi
    got="$(scan "$(git rev-parse --show-toplevel)")"
    if [ -z "$got" ]; then echo "ok    green the real tree has no method flag on an arming path"; else echo "FAIL  real tree: $got"; fails=1; fi
    return "$fails"
}

case "${1:-}" in
    --self-test) self_test; exit $? ;;
    "")
        bad="$(scan "$(git rev-parse --show-toplevel)")"
        if [ -n "$bad" ]; then printf 'FAIL  an arming call carries a method flag or extra argument (#5794):\n%s\n' "$bad"; exit 1; fi
        echo "ok    every gh pr merge is plain" ;;
    *) echo "usage: $0 [--self-test]" >&2; exit 2 ;;
esac
