#!/usr/bin/env bash
# check_readme_derived.sh — the README's CONTRACT_COUNT block is DERIVED, and has one writer (GEN-001, #4526).
#
# THE DEFECT
# ----------
# `<!-- CONTRACT_COUNT_START -->N<!-- CONTRACT_COUNT_END -->` is written by scripts/readme_sync.sh
# (BSE-03, PMAT-1068) and a human writes neither the markers nor the number. But nothing REFUSED a
# hand edit, and check_readme_claims.sh judged the block by EQUALITY against the merge tree, so every
# PR that added a contract had to bump the number. Two such PRs conflicted on the same line, and the
# conflict was resolved by hand: six feature PRs in 14 days (#6833 "CONTRACT_COUNT 2010 -> 2011,
# main red", #6586, #6581, #6582, #6585, #6504). An output that behaved like an input.
#
# THE RULE (the same one contracts/census.json lives under, #3569, scripts/check_census_derived.sh)
# --------
#   1. A pull request may not change the text between the CONTRACT_COUNT markers. The release train
#      (T-0) is its only writer: a `release/X.Y.Z` branch, or a run with CENSUS_WRITER=train, where
#      `make census` rewrites the block. Any other diff against the comparand is refused, and the
#      refusal names the block.
#   2. check_readme_claims.sh therefore judges the block by the ratchet on a PR (it may lag, it may
#      never overstate; same polarity as crates/aprender-core/tests/readme_contract.rs) and by
#      EQUALITY only on the train, where the writer runs.
#
# The comparand is the one every ratchet here uses (scripts/lib_baseline_ratchet.sh). If it does not
# resolve, the check FAILS; it is never degraded to comparing the branch against itself.
#
#   bash scripts/check_readme_derived.sh               # the touch check
#   bash scripts/check_readme_derived.sh --self-test   # the case table; every RED row must go RED
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
README_REL="README.md"
START='<!-- CONTRACT_COUNT_START -->'
END='<!-- CONTRACT_COUNT_END -->'
# shellcheck source=scripts/lib_baseline_ratchet.sh
. "$REPO_ROOT/scripts/lib_baseline_ratchet.sh" || exit 2
# shellcheck source=scripts/lib_release_train.sh
. "$REPO_ROOT/scripts/lib_release_train.sh" || exit 2

block_texts() { # the text between each marker pair, one line per block, on stdin
    { grep -oE "${START}[^<]*${END}" || true; } | sed -E "s|${START}||; s|${END}||"
}

touch_check() { # touch_check ROOT BRANCH -> 0 not touched (or train); 1 touched or unmeasurable
    local root="$1" branch="$2" res mode ref base now
    res=$(baseline_ratchet_resolve "$root" "$BASELINE_RATCHET_BASE_REF" "$README_REL")
    mode=${res%%$'\t'*}; ref=${res##*$'\t'}
    case "$mode" in
        MERGEBASE | FIRSTPARENT | TIP) ;;
        *)
            printf 'FAIL  comparand %s resolved %s for %s: whether this branch edits the CONTRACT_COUNT block is UNMEASURED, and that is not "untouched".\n' "$ref" "$mode" "$README_REL"
            printf '      In CI: %s\n' "$(baseline_ratchet_fetch_hint "$ref")"
            return 1 ;;
    esac
    # The comparand's blocks vs the WORKING TREE's, so a local run sees an unstaged edit too.
    base=$(git -C "$root" show "$ref:$README_REL" 2>/dev/null | block_texts) || base=""
    now=$(block_texts < "$root/$README_REL") || now=""
    if [ "$base" = "$now" ]; then
        printf 'ok    the CONTRACT_COUNT block is untouched against %s (%s)\n' "${ref:0:12}" "$mode"
        return 0
    fi
    if is_train "$branch"; then
        printf 'ok    the CONTRACT_COUNT block changed on <%s>: the release train is its one writer\n' "$branch"
        return 0
    fi
    printf 'FAIL  this branch <%s> edits the CONTRACT_COUNT block of %s against %s (%s): %s -> %s.\n' \
        "$branch" "$README_REL" "${ref:0:12}" "$mode" "$(printf '%s' "$base" | tr '\n' ' ')" "$(printf '%s' "$now" | tr '\n' ' ')"
    printf '      The block is derived, and the release train (T-0) is its only writer (GEN-001 #4526).\n'
    printf '      A PR may leave it lagging; it may never overstate (check_readme_claims.sh). Restore the comparand text:\n'
    printf '          git checkout %s -- %s\n' "${ref:0:12}" "$README_REL"
    return 1
}

self_test() {
    local t pass=0 fail=0 rc r
    t=$(mktemp -d)
    case "$t" in /tmp/* | /var/tmp/* | "${TMPDIR:-/nonexistent}"/*) : ;; *) printf 'NO-GO: odd mktemp path %s\n' "$t" >&2; return 2 ;; esac
    row() { # row NAME GOT WANT
        if [ "$2" = "$3" ]; then pass=$((pass + 1)); printf '  ok    rc=%s  %s\n' "$2" "$1"
        else fail=$((fail + 1)); printf '  FAIL  rc=%s (wanted %s)  %s\n' "$2" "$3" "$1"; fi
    }
    printf 'check_readme_derived self-test\n'
    local GITHUB_EVENT_PATH='' GITHUB_EVENT_NAME='' CENSUS_WRITER=''
    readme() { printf '# apr\n\n| contracts | %s%s%s |\n\nprose about 7 contracts\n%s\n' "$START" "$1" "$END" "${2:-}" > "$r/$README_REL"; }
    r="$t/repo"; mkdir -p "$r"
    git -C "$r" init -q -b main
    git -C "$r" config user.email t@t; git -C "$r" config user.name t; git -C "$r" config commit.gpgsign false; git -C "$r" config core.hooksPath /dev/null
    readme 100; git -C "$r" add -A; git -C "$r" commit -qm base
    git -C "$r" update-ref refs/remotes/origin/main HEAD
    git -C "$r" checkout -qb feat
    rc=0; touch_check "$r" feat >/dev/null || rc=$?; row "a PR that does not touch the README is GREEN" "$rc" 0
    readme 100 "an unrelated prose line"; git -C "$r" commit -qam "prose"
    rc=0; touch_check "$r" feat >/dev/null || rc=$?; row "a PR that edits README prose outside the block is GREEN" "$rc" 0
    readme 101; git -C "$r" commit -qam "hand-edit the count"
    rc=0; touch_check "$r" feat >"$t/out" || rc=$?; row "RED: a PR that hand-edits the CONTRACT_COUNT block (planted, committed)" "$rc" 1
    grep -q 'edits the CONTRACT_COUNT block' "$t/out"; row "the refusal names the block" "$?" 0
    rc=0; touch_check "$r" release/0.70.0 >/dev/null || rc=$?; row "a release/X.Y.Z branch may rewrite the block" "$rc" 0
    rc=0; CENSUS_WRITER=train touch_check "$r" feat >/dev/null || rc=$?; row "CENSUS_WRITER=train may rewrite the block" "$rc" 0
    rc=0; touch_check "$r" release/next >/dev/null || rc=$?; row "RED: release/<not a version> is not the train" "$rc" 1
    rc=0; touch_check "$r" fix/release/0.70.0 >/dev/null || rc=$?; row "RED: a branch that merely CONTAINS release/X.Y.Z is not the train" "$rc" 1
    printf '{"pull_request":{"head":{"repo":{"full_name":"someone/aprender"}},"base":{"repo":{"full_name":"paiml/aprender"}}}}\n' > "$t/fork.json"
    rc=0; GITHUB_EVENT_PATH="$t/fork.json" touch_check "$r" release/0.70.0 >/dev/null || rc=$?; row "RED: a FORK PR named release/X.Y.Z is not the train" "$rc" 1
    git -C "$r" reset -q --hard HEAD~1; git -C "$r" reset -q --hard HEAD~1   # back to base
    readme 100; git -C "$r" status --short >/dev/null
    printf 'X\n' >> "$r/$README_REL"
    rc=0; touch_check "$r" feat >/dev/null || rc=$?; row "an unstaged edit outside the block is GREEN" "$rc" 0
    readme 102
    rc=0; touch_check "$r" feat >/dev/null || rc=$?; row "RED: an UNCOMMITTED hand-edit of the block" "$rc" 1
    readme 100; sed -i "s|${START}100${END}|100|" "$r/$README_REL"
    rc=0; touch_check "$r" feat >/dev/null || rc=$?; row "RED: deleting the markers changes the block set" "$rc" 1
    git -C "$r" checkout -q -- "$README_REL"
    git -C "$r" update-ref -d refs/remotes/origin/main
    rc=0; touch_check "$r" feat >/dev/null || rc=$?; row "RED: no comparand is UNMEASURED, not untouched" "$rc" 1
    printf 'self-test: %s passed, %s failed\n' "$pass" "$fail"
    [ -n "$t" ] && [ -d "$t" ] && rm -rf -- "${t:?}"
    [ "$fail" -eq 0 ] && [ "$pass" -gt 0 ]
}

main() {
    command -v jq >/dev/null 2>&1 || { printf 'NO-GO: jq is required\n' >&2; return 2; }
    case "${1:-}" in
        --self-test) self_test; return $? ;;
        '') ;;
        *) printf 'usage: %s [--self-test]\n' "$(basename "$0")" >&2; return 2 ;;
    esac
    printf '== the README CONTRACT_COUNT block is derived; the train is its one writer (GEN-001 #4526) ==\n'
    touch_check "$REPO_ROOT" "$(current_branch "$REPO_ROOT")" && printf 'PASS\n'
}

main "$@"
