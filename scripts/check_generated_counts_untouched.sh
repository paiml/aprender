#!/usr/bin/env bash
# check_generated_counts_untouched.sh — README counts come from ONE writer; PRs never hand-edit them.
#
# GEN-001 (#4526). scripts/readme_sync.sh generates two numbers into README.md:
#
#   * the frontmatter `contract_count:` (census n_files), and
#   * every <!-- CONTRACT_COUNT_START -->N<!-- CONTRACT_COUNT_END --> block.
#
# Both are derived, and the release train (T-0: `make census`, which runs
# readme_sync --write) is their one writer, the same writer as
# contracts/census.json (#3569, scripts/lib_train_writer.sh). While every
# contract-adding PR had to rewrite them, parallel PRs conflicted on the same
# line and each one hand-typed a number. So, against the comparand (the
# merge-base with origin/main, scripts/lib_baseline_ratchet.sh):
#
#   * the frontmatter contract_count changes only on the train;
#   * a CONTRACT_COUNT block changes only on the train, or goes DOWN: a PR that
#     removes contracts lowers the block so check_readme_claims.sh (which lets
#     the block lag, never overstate) stays true. A block that goes UP, appears,
#     disappears or splits into two numbers off the train is RED;
#   * an unresolvable comparand is UNMEASURED, and that is RED, not "untouched".
#
#   bash scripts/check_generated_counts_untouched.sh              # this checkout
#   bash scripts/check_generated_counts_untouched.sh --self-test  # the case table
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
README_REL="README.md"
BLOCK_START='<!-- CONTRACT_COUNT_START -->'
BLOCK_END='<!-- CONTRACT_COUNT_END -->'
. "$REPO_ROOT/scripts/lib_baseline_ratchet.sh" || exit 2
# shellcheck source=scripts/lib_train_writer.sh
. "$REPO_ROOT/scripts/lib_train_writer.sh" || exit 2

# ── the two generated numbers, read from README text on stdin ────────────────
frontmatter_count() { # the contract_count: of the leading --- frontmatter, or empty
    awk 'NR == 1 && $0 != "---" { exit } NR > 1 && $0 == "---" { exit }
         NR > 1 && /^contract_count:/ { sub(/^contract_count:[ \t]*/, ""); print; exit }'
}
block_values() { # each CONTRACT_COUNT block body, one per line, deduplicated
    { grep -oE "${BLOCK_START}[^<]*${BLOCK_END}" || true; } \
        | sed -E "s|${BLOCK_START}||; s|${BLOCK_END}||" | sort -u
}

# ── the touch check ──────────────────────────────────────────────────────────
touch_check() { # touch_check ROOT BRANCH -> 0 untouched, lowered, or train; 1 hand-edited or unmeasurable
    local root="$1" branch="$2" res mode ref base_text head_text bf hf bb hb rc=0
    res=$(baseline_ratchet_resolve "$root" "$BASELINE_RATCHET_BASE_REF" "$README_REL")
    mode=${res%%$'\t'*}; ref=${res##*$'\t'}
    case "$mode" in
        MERGEBASE | TIP) ;;
        *)
            printf 'FAIL  comparand %s resolved %s for %s: whether this branch hand-edits its generated counts is UNMEASURED, and that is not "untouched".\n' "$ref" "$mode" "$README_REL"
            printf '      In CI: git fetch --no-tags --depth=1 origin +refs/heads/main:refs/remotes/origin/main\n'
            return 1 ;;
    esac
    base_text=$(git -C "$root" show "$ref:$README_REL" 2>/dev/null) || {
        printf 'FAIL  cannot read %s at %s: UNMEASURED\n' "$README_REL" "${ref:0:12}"; return 1; }
    # The working tree, so a local run sees an unstaged edit too; in CI it is the merge tree.
    head_text=$(cat "$root/$README_REL" 2>/dev/null) || {
        printf 'FAIL  cannot read %s in the checkout: UNMEASURED\n' "$README_REL"; return 1; }
    bf=$(frontmatter_count <<<"$base_text"); hf=$(frontmatter_count <<<"$head_text")
    bb=$(block_values <<<"$base_text");      hb=$(block_values <<<"$head_text")
    if [ "$bf" = "$hf" ] && [ "$bb" = "$hb" ]; then
        printf 'ok    the generated counts in %s are untouched against %s (%s)\n' "$README_REL" "${ref:0:12}" "$mode"
        return 0
    fi
    if is_train "$branch"; then
        printf 'ok    the generated counts in %s changed on <%s>: the release train is their one writer\n' "$README_REL" "$branch"
        return 0
    fi
    if [ "$bf" != "$hf" ]; then
        printf 'FAIL  this branch <%s> changes the frontmatter contract_count of %s from %s to %s against %s (%s).\n' \
            "$branch" "$README_REL" "${bf:-<none>}" "${hf:-<none>}" "${ref:0:12}" "$mode"
        rc=1
    fi
    if [ "$bb" != "$hb" ] && ! block_lowered "$bb" "$hb"; then
        printf 'FAIL  this branch <%s> changes the CONTRACT_COUNT block(s) of %s from [%s] to [%s] against %s (%s); off the train a block may only go DOWN (a net contract removal).\n' \
            "$branch" "$README_REL" "$(printf '%s' "$bb" | tr '\n' ' ')" "$(printf '%s' "$hb" | tr '\n' ' ')" "${ref:0:12}" "$mode"
        rc=1
    fi
    if [ "$rc" = 0 ]; then
        printf 'ok    the CONTRACT_COUNT block of %s went DOWN from %s to %s against %s: a net contract removal, so the README may not overstate\n' \
            "$README_REL" "$bb" "$hb" "${ref:0:12}"
        return 0
    fi
    printf '      These numbers are generated, and the release train (T-0, make census) is their only writer (#4526).\n'
    printf '      check_readme_claims.sh lets the block lag the tree off the train. Drop the edit:\n'
    printf '          git checkout %s -- %s   (then re-apply your prose)\n' "${ref:0:12}" "$README_REL"
    return 1
}

block_lowered() { # block_lowered BASE HEAD -> 0 iff each side is ONE number and HEAD < BASE
    [[ "$1" =~ ^[0-9]+$ ]] && [[ "$2" =~ ^[0-9]+$ ]] && [ "$2" -lt "$1" ]
}

# ── the case table ───────────────────────────────────────────────────────────
self_test() {
    local t pass=0 fail=0 rc r
    t=$(mktemp -d)
    case "$t" in /tmp/* | /var/tmp/* | "${TMPDIR:-/nonexistent}"/*) : ;; *) printf 'NO-GO: odd mktemp path %s\n' "$t" >&2; return 2 ;; esac
    row() { # row NAME GOT_RC WANT_RC
        if [ "$2" = "$3" ]; then pass=$((pass + 1)); printf '  ok    rc=%s  %s\n' "$2" "$1"
        else fail=$((fail + 1)); printf '  FAIL  rc=%s (wanted %s)  %s\n' "$2" "$3" "$1"; fi
    }
    readme() { # readme FRONTMATTER_N BLOCK_N [SECOND_BLOCK_N] -> a README on stdout
        printf -- '---\nkind: library\ncontract_count: %s\n---\n\n# apr\n\n| contracts | %s%s%s |\n' "$1" "$BLOCK_START" "$2" "$BLOCK_END"
        printf 'Prose that a PR may edit freely.\n'
        [ -z "${3:-}" ] || printf '**%s%s%s** provable contracts\n' "$BLOCK_START" "$3" "$BLOCK_END"
    }
    printf 'check_generated_counts_untouched self-test\n'
    # The runner's own event payload must not decide a row: a fork PR would turn the train rows RED.
    local GITHUB_EVENT_PATH='' GITHUB_EVENT_NAME=''

    row "frontmatter_count reads the leading frontmatter" "$(readme 7 8 | frontmatter_count)" 7
    row "frontmatter_count ignores a contract_count: outside the frontmatter" "$(printf '# apr\ncontract_count: 9\n' | frontmatter_count)" ""
    row "block_values lists each distinct block once" "$(readme 7 8 8 | block_values | tr '\n' ,)" "8,"

    r="$t/repo"
    mkdir -p "$r"
    git -C "$r" init -q -b main
    git -C "$r" config user.email t@t; git -C "$r" config user.name t; git -C "$r" config commit.gpgsign false; git -C "$r" config core.hooksPath /dev/null
    readme 10 12 12 > "$r/$README_REL"
    git -C "$r" add -A; git -C "$r" commit -qm base
    git -C "$r" update-ref refs/remotes/origin/main HEAD
    git -C "$r" checkout -qb feat
    check() { # check NAME WANT [BRANCH] -- run against the current fixture README
        rc=0; CENSUS_WRITER='' touch_check "$r" "${3:-feat}" > "$t/out" || rc=$?; row "$1" "$rc" "$2"
    }
    printf 'A PR edits prose only.\n' >> "$r/$README_REL"
    check "a PR that edits prose and leaves the counts is GREEN" 0
    readme 10 13 13 > "$r/$README_REL"
    check "RED: an UNCOMMITTED hand-edit raising both blocks (planted)" 1
    rc=0; grep -q 'may only go DOWN' "$t/out" || rc=$?; row "the refusal says a block may only go down" "$rc" 0
    git -C "$r" commit -qam "bump the count by hand"
    check "RED: a committed hand-edit raising both blocks (planted)" 1
    readme 11 12 12 > "$r/$README_REL"
    check "RED: a hand-edit of the frontmatter contract_count (planted)" 1
    rc=0; grep -q 'frontmatter contract_count of README.md from 10 to 11' "$t/out" || rc=$?; row "the refusal names both frontmatter numbers" "$rc" 0
    readme 9 12 12 > "$r/$README_REL"
    check "RED: the frontmatter may not go down either (it is the census, train-only)" 1
    readme 10 13 12 > "$r/$README_REL"
    check "RED: one block raised, the other left: two numbers (planted)" 1
    readme 10 11 > "$r/$README_REL"
    check "a block LOWERED by a net contract removal is GREEN" 0
    readme 10 11 11 > "$r/$README_REL"
    check "both blocks lowered to one number is GREEN" 0
    readme 10 11 12 > "$r/$README_REL"
    check "RED: blocks lowered to two different numbers" 1
    printf -- '---\nkind: library\ncontract_count: 10\n---\n# apr, no block\n' > "$r/$README_REL"
    check "RED: deleting the block is not lowering it" 1
    readme 11 13 13 > "$r/$README_REL"
    check "the release/X.Y.Z train may write both" 0 release/0.72.0
    rc=0; CENSUS_WRITER=train touch_check "$r" feat >/dev/null || rc=$?; row "CENSUS_WRITER=train may write both" "$rc" 0
    check "RED: release/<not a version> is not the train" 1 release/next
    check "RED: a batch branch is not the train" 1 batch/0.72.0
    printf '{"pull_request":{"head":{"repo":{"full_name":"someone/aprender"}},"base":{"repo":{"full_name":"paiml/aprender"}}}}\n' > "$t/fork.json"
    rc=0; CENSUS_WRITER='' GITHUB_EVENT_PATH="$t/fork.json" touch_check "$r" release/0.72.0 >/dev/null || rc=$?; row "RED: a FORK PR named release/X.Y.Z is not the train" "$rc" 1
    rc=0; CENSUS_WRITER=train GITHUB_EVENT_PATH="$t/fork.json" touch_check "$r" feat >/dev/null || rc=$?; row "RED: a FORK PR cannot claim CENSUS_WRITER=train" "$rc" 1
    git -C "$r" update-ref -d refs/remotes/origin/main
    git -C "$r" checkout -q -- "$README_REL"
    check "RED: no comparand is UNMEASURED, not untouched" 1

    printf 'self-test: %s passed, %s failed\n' "$pass" "$fail"
    [ -n "$t" ] && [ -d "$t" ] && rm -rf -- "$t"
    [ "$fail" -eq 0 ] && [ "$pass" -gt 0 ]
}

main() {
    case "${1:-}" in
        --self-test) self_test; return $? ;;
        '') ;;
        *) printf 'usage: %s [--self-test]\n' "$(basename "$0")" >&2; return 2 ;;
    esac
    printf '== README counts are generated; the train is their one writer (#4526) ==\n'
    touch_check "$REPO_ROOT" "$(current_branch "$REPO_ROOT")" && printf 'PASS\n'
}

main "$@"
