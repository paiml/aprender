#!/usr/bin/env bash
# batch_fold.sh -- fold receipted PR branches into ONE integration batch branch
# (APR-RELEASE-001 §13; operator 2026-09-21: "we are arbitrarily going slow
# because each PR adds time, but in most cases 80% of PR can be batched").
#
# Run INSIDE the batch worktree (release/<train>-batch, cut from origin/main).
# Each branch is merged --no-ff, in the order given (cut-blockers first).
#
# THE GENERATED SET is the only thing this script ever resolves by itself:
#   docs/roadmaps/roadmap.yaml   aggregate of docs/roadmaps/entries/
#   contracts/census.json        pv census
#   contracts/contracts.nt       pv extract
#   contracts/shapes.ttl         pv extract (every `shape:` block)
#   README.md CONTRACT_COUNT     scripts/readme_sync.sh -- the COUNT BLOCKS only
# A conflict in those is taken and marked for regeneration. README.md is merged
# three-way with the counts normalised out, so a README conflict OUTSIDE a count
# block is a real conflict, not a generated one. Any real conflict aborts that
# branch's merge and reports SKIP with the paths -- never resolved by picking a
# side. The branch is left out of the batch and named.
#
# A merge that folds CLEANLY can still leave the generated set wrong: two PRs
# that each add a contract merge textually and the census count is then off by
# one. So every fold that touches a generated file marks the set STALE, and a
# stale set is never silent: --regen regenerates it ONCE, with the pv built from
# THIS tree (scripts/pv_bin.sh), asserts the three fixed points, and commits the
# regeneration; without --regen the run ends `REGEN REQUIRED` and exits 3.
#
#   bash scripts/batch_fold.sh [--regen] BRANCH...
#   bash scripts/batch_fold.sh --regen                 regenerate only
#
# Output, one line per branch:
#   folded <branch> generated=[<paths>]      merged; <paths> were taken for regeneration
#   SKIP <branch>: conflict in <paths>       merge aborted, branch left out
#   ERROR <branch>: <why>                    git could not merge it at all
# Exit: 0 every branch folded and the generated set is consistent; 1 any SKIP;
#       2 a usage, git or regeneration failure (including a failed fixed-point
#       check); 3 folded, but the generated set is stale (run --regen).
# The case table is scripts/check_batch_fold.sh (guard-tree runs it).
#
# Test seam: BATCH_FOLD_PV=<pv> skips building pv (scripts/pv_bin.sh) and uses it.
set -uo pipefail

GENERATED_WHOLE="docs/roadmaps/roadmap.yaml contracts/census.json contracts/contracts.nt contracts/shapes.ttl"
COUNT_RE='(<!-- CONTRACT_COUNT_START -->)[0-9]+(<!-- CONTRACT_COUNT_END -->)'

die() { printf 'ERROR %s\n' "$*" >&2; exit 2; }
rmtree() { case "${1:-}" in ''|/) return 0 ;; *) [ -d "$1" ] && rm -rf -- "$1" ;; esac; return 0; }

is_generated_whole() { # PATH -> 0 iff the whole file is derived
    local g
    for g in $GENERATED_WHOLE; do [ "$1" = "$g" ] && return 0; done
    return 1
}

touches_generated() { # REV_A REV_B -> prints the generated paths that differ
    local p
    while IFS= read -r p; do
        [ -n "$p" ] || continue
        if is_generated_whole "$p" || [ "$p" = README.md ]; then printf '%s\n' "$p"; fi
    done < <(git diff --no-renames --name-only "$1" "$2")
}

# readme_counts_only -> 0 iff README.md's conflict is ONLY in CONTRACT_COUNT blocks.
# On 0 the file is written merged (counts left for readme_sync.sh) and staged.
readme_counts_only() {
    local t rc=0
    t=$(mktemp -d "${TMPDIR:-/tmp}/batch-fold-readme.XXXXXX") || return 1
    if ! { git show :1:README.md > "$t/base" && git show :2:README.md > "$t/ours" && git show :3:README.md > "$t/theirs"; } 2>/dev/null; then
        rmtree "$t"; return 1   # an add/add or delete conflict is not a count conflict
    fi
    local n
    n=$(grep -m1 -oE "$COUNT_RE" "$t/ours" | sed -E 's/[^0-9]//g')
    sed -E -i "s/$COUNT_RE/\\1N\\2/g" "$t/base" "$t/ours" "$t/theirs"
    git merge-file -p "$t/ours" "$t/base" "$t/theirs" > "$t/merged" 2>/dev/null || rc=$?
    if [ "$rc" = 0 ]; then
        # put OUR count back so the file stays well-formed; --regen rewrites every block
        sed -E -i "s/(<!-- CONTRACT_COUNT_START -->)N(<!-- CONTRACT_COUNT_END -->)/\\1${n:-0}\\2/g" "$t/merged"
        cp -- "$t/merged" README.md && git add -- README.md || rc=1
    fi
    rmtree "$t"
    return "$rc"
}

fold_one() { # BRANCH -> one verdict line; 0 folded, 1 skipped, 2 git failure
    local br=$1 f real="" gen="" conf before
    before=$(git rev-parse HEAD) || return 2
    if git merge --no-ff --no-edit "$br" >/dev/null 2>&1; then
        gen=$(touches_generated "$before" HEAD | tr '\n' ' ')
        [ -z "$gen" ] || STALE=1
        printf 'folded %s generated=[%s]\n' "$br" "${gen% }"; return 0
    fi
    if ! git rev-parse -q --verify MERGE_HEAD >/dev/null 2>&1; then
        printf 'ERROR %s: git merge failed without a conflict state\n' "$br"; return 2
    fi
    conf=$(git diff --name-only --diff-filter=U)
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        if is_generated_whole "$f"; then
            gen="$gen $f"
        elif [ "$f" = README.md ] && readme_counts_only; then
            gen="$gen README.md"
        else
            real="$real $f"
        fi
    done <<< "$conf"
    if [ -n "$real" ]; then
        git merge --abort >/dev/null 2>&1
        printf 'SKIP %s: conflict in%s\n' "$br" "$real"; return 1
    fi
    for f in $gen; do
        [ "$f" = README.md ] && continue           # already merged and staged above
        { git checkout --theirs -- "$f" && git add -- "$f"; } >/dev/null 2>&1 \
            || { git merge --abort >/dev/null 2>&1; printf 'ERROR %s: cannot take %s\n' "$br" "$f"; return 2; }
    done
    git -c commit.gpgsign=false commit -q --no-edit >/dev/null 2>&1 \
        || { git merge --abort >/dev/null 2>&1; printf 'ERROR %s: commit failed\n' "$br"; return 2; }
    STALE=1
    printf 'folded %s generated=[%s]\n' "$br" "${gen# }"; return 0
}

regen() { # regenerate the generated set once, assert the fixed points, commit it
    local PV extra
    if [ -n "${BATCH_FOLD_PV:-}" ]; then
        PV=$BATCH_FOLD_PV
    else
        # shellcheck source=scripts/pv_bin.sh
        . scripts/pv_bin.sh || die "regen: scripts/pv_bin.sh could not build/resolve pv from this tree"
    fi
    make roadmap-aggregate >/dev/null 2>&1                              || die "regen: make roadmap-aggregate failed"
    "$PV" census contracts --format json > contracts/census.json 2>/dev/null || die "regen: pv census failed"
    "$PV" extract contracts >/dev/null 2>&1                             || die "regen: pv extract failed"
    bash scripts/readme_sync.sh --write >/dev/null 2>&1                 || die "regen: readme_sync.sh --write failed"
    # the fixed points: each regenerated artifact must equal a second regeneration
    make roadmap-aggregate-check >/dev/null 2>&1                        || die "regen: fixed point FAILED -- make roadmap-aggregate-check"
    "$PV" extract contracts --check >/dev/null 2>&1                     || die "regen: fixed point FAILED -- pv extract contracts --check"
    bash scripts/readme_sync.sh --check >/dev/null 2>&1                 || die "regen: fixed point FAILED -- readme_sync.sh --check"
    # only the generated set may have moved; anything else is a tool writing where it should not
    extra=$(git status --porcelain --untracked-files=no | awk '{print $2}' \
            | grep -vxE 'docs/roadmaps/roadmap\.yaml|contracts/census\.json|contracts/contracts\.nt|contracts/shapes\.ttl|README\.md' || true)
    [ -z "$extra" ] || die "regen: wrote outside the generated set: $(printf '%s ' $extra)"
    if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
        git add -- docs/roadmaps/roadmap.yaml contracts/census.json contracts/contracts.nt contracts/shapes.ttl README.md 2>/dev/null
        git -c commit.gpgsign=false commit -q -m "batch: regenerate the generated set once (batch_fold.sh --regen)

roadmap aggregate, pv census, pv extract (contracts.nt + shapes.ttl) and the
README CONTRACT_COUNT blocks, regenerated with the pv built from this tree;
fixed points asserted: make roadmap-aggregate-check, pv extract --check,
readme_sync.sh --check." >/dev/null 2>&1 || die "regen: commit failed"
        printf 'regenerated the generated set and committed it (fixed points asserted)\n'
    else
        printf 'regenerated the generated set: already consistent (fixed points asserted)\n'
    fi
}

case "${1:-}" in -h|--help) sed -n '2,42p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac

REGEN=0; branches=()
while [ $# -gt 0 ]; do
    case "$1" in
        --regen) REGEN=1; shift ;;
        -*) printf 'batch_fold.sh: unknown flag %s\n' "$1" >&2; exit 2 ;;
        *) branches+=("$1"); shift ;;
    esac
done
[ "${#branches[@]}" -gt 0 ] || [ "$REGEN" = 1 ] \
    || { printf 'usage: bash scripts/batch_fold.sh [--regen] BRANCH... | --regen\n' >&2; exit 2; }
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "not inside a git work tree"
[ -z "$(git status --porcelain --untracked-files=no)" ] \
    || die "the batch worktree has uncommitted changes; fold into a clean tree only"

STALE=0; status=0
for br in "${branches[@]}"; do
    fold_one "$br"; rc=$?
    [ "$rc" = 2 ] && exit 2
    [ "$rc" = 1 ] && status=1
done
if [ "$REGEN" = 1 ]; then
    regen
elif [ "$STALE" = 1 ]; then
    printf 'REGEN REQUIRED: a folded branch touched the generated set; run: bash scripts/batch_fold.sh --regen\n'
    [ "$status" = 0 ] && status=3
fi
exit "$status"
