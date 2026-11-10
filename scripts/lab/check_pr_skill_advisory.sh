#!/usr/bin/env bash
# check_pr_skill_advisory.sh - the two advisory lines of the PR-review skill stay advisory (Refs #6600, #6601).
#
# THE LINES. pr-skill-spec.md S6: a `FINDINGS` verdict "proceeds" (never blocks the merge queue).
# pr-review/SKILL.md: the `cargo mutants` step on Rust source is "advisory". Both are rules about what must NOT
# block. The spec sweep (2026-10-31) found no check that holds them.
#
# THE CHECKS (each RED line names its rule):
#   S6   spec table row for FINDINGS still reads "**proceeds**"
#   S6W  no workflow that runs on pull_request / pull_request_target / merge_group both names FINDINGS and exits non-zero
#   M1   SKILL.md table row "Rust source" (cargo mutants) still ends in "advisory"
#   M2   no pull_request / pull_request_target / merge_group workflow runs `cargo mutants` (it lives in the nightly)
#
# TIER. LAB (C324): runs nightly on an idle machine, never in a PR, merge-queue or release job. A red opens or
# updates one ticket with one owner (scripts/lab_ticket.sh) and stays silent. Promotion to a gate needs the
# operator's yes, three green nights on main and one gate retired.
#
# Usage: scripts/lab/check_pr_skill_advisory.sh [repo-root] | --self-test      Exit: 0 green, 1 RED, 2 cannot measure
set -euo pipefail

SPEC=docs/specifications/pr-skill-spec.md
SKILL=.claude/skills/pr-review/SKILL.md
WF=.github/workflows

# Workflow files whose `on:` block names a PR-path trigger.
pr_path_workflows() {
    local f
    for f in "$1"/$WF/*.yml "$1"/$WF/*.yaml; do
        [ -f "$f" ] || continue
        awk '/^on:/ {on=1; next} on && /^[^[:space:]#]/ {on=0} on && /(pull_request|pull_request_target|merge_group)/ {hit=1} END{exit !hit}' "$f" && echo "$f"
    done
    return 0
}

scan() { # root -> RED lines on stdout
    local root=$1 f
    [ -f "$root/$SPEC" ] && [ -f "$root/$SKILL" ] || { echo "CANNOT-MEASURE: $SPEC or $SKILL missing under $root" >&2; return 2; }
    awk '/^\|[[:space:]]*`FINDINGS`/ && /\*\*proceeds\*\*/ {ok=1} END{exit !ok}' "$root/$SPEC" \
        || echo "RED S6 $SPEC: the FINDINGS row no longer says **proceeds**"
    awk '/^\|[[:space:]]*Rust source[[:space:]]*\|/ && /\|[[:space:]]*advisory[[:space:]]*\|[[:space:]]*$/ {ok=1} END{exit !ok}' "$root/$SKILL" \
        || echo "RED M1 $SKILL: the Rust source (cargo mutants) row no longer ends in advisory"
    while read -r f; do
        [ -n "$f" ] || continue
        grep -qE 'cargo[ -]mutants' "$f" && echo "RED M2 $f: a PR-path workflow runs cargo mutants"
        if grep -q 'FINDINGS' "$f" && grep -qE '(^|[^[:alnum:]_])exit[[:space:]]+[1-9]' "$f"; then
            echo "RED S6W $f: a PR-path workflow names FINDINGS and exits non-zero"
        fi
    done < <(pr_path_workflows "$root")
    return 0
}

self_test() {
    local t out rc
    t=$(mktemp -d) || exit 2
    trap 'rm -rf -- "${t:?}"' RETURN
    mkdir -p "$t/$WF" "$t/docs/specifications" "$t/.claude/skills/pr-review"
    good() {
        printf '| `FINDINGS` | **proceeds**, findings posted | x |\n' > "$t/$SPEC"
        printf '| Rust source | `cargo mutants --in-diff` | advisory |\n' > "$t/$SKILL"
        printf 'on:\n  pull_request:\njobs:\n  a:\n    steps:\n      - run: cargo test\n' > "$t/$WF/pr.yml"
        printf 'on:\n  schedule:\n    - cron: "0 1 * * *"\njobs:\n  m:\n    steps:\n      - run: cargo mutants\n' > "$t/$WF/nightly.yml"
    }
    expect() { # label want-substring
        out=$(scan "$t" 2>&1) && rc=0 || rc=$?
        if [ -z "$2" ]; then
            [ -z "$out" ] || { echo "SELF-TEST FAIL $1: expected green, got: $out"; exit 1; }
        else
            case $out in *"$2"*) ;; *) echo "SELF-TEST FAIL $1: expected '$2', got: ${out:-green}"; exit 1 ;; esac
        fi
    }
    good; expect "real-shape tree is green" ""
    good; printf '| `FINDINGS` | **BLOCKS** | x |\n' > "$t/$SPEC"; expect "S6 spec flips" "RED S6 "
    good; printf '| Rust source | `cargo mutants` | **yes** |\n' > "$t/$SKILL"; expect "M1 skill flips" "RED M1 "
    good; printf 'on:\n  merge_group:\njobs:\n  a:\n    steps:\n      - run: cargo mutants --in-diff d\n' > "$t/$WF/pr.yml"; expect "M2 mutants on PR path" "RED M2 "
    good; printf 'on:\n  pull_request:\njobs:\n  a:\n    steps:\n      - run: |\n          [ "$v" = FINDINGS ] && exit 1\n' > "$t/$WF/pr.yml"; expect "S6W FINDINGS exits" "RED S6W "
    good; rm "$t/$SKILL"; out=$(scan "$t" 2>&1) && rc=0 || rc=$?
    [ "$rc" -eq 2 ] || { echo "SELF-TEST FAIL: a missing skill file must be 'cannot measure' (rc 2), got rc $rc"; exit 1; }
    echo "self-test: 6 cases ok (green, S6, M1, M2, S6W, cannot-measure)"
}

case ${1-} in
    --self-test) self_test; exit $? ;;
esac
root=${1:-$(cd "$(dirname "$0")/../.." && pwd)}
out=$(scan "$root") || exit 2
if [ -n "$out" ]; then printf '%s\n' "$out"; exit 1; fi
echo "check_pr_skill_advisory: green (S6, S6W, M1, M2)"
