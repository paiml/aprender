#!/usr/bin/env bash
#
# predict_merge.sh — predict whether merge(origin/main, HEAD) would pass the
# tree-property guards, BEFORE pushing (BSE-14, docs/specifications/
# build-system-enhancement.md §4 wave 3, infra repo, report O1).
#
# WHY THIS EXISTS
# ---------------
# `make gate` (BSE-16) only ever runs against the tree the developer already
# has checked out. It says nothing about the tree GitHub will actually build:
# origin/main plus this branch's commits. A branch that passes `make gate`
# locally can still fail CI the moment origin/main has moved underneath it —
# the guard universe (scripts/guard_tree.sh) or the touched-crate selection
# (scripts/gate_touched_crates.sh) can behave differently once merged with
# whatever landed on main in the meantime. This script builds that merge tree
# in a throwaway worktree, runs the same two BSE-16 gates against it, and
# records the verdict so a fast `--check` can refuse a stale prediction
# without redoing the (expensive) merge-and-run step every time.
#
# TWO MODES
# ---------
#   predict_merge.sh            build merge(origin/main, HEAD), run the
#                                cargo-free guard universe (guard_tree.sh
#                                --no-cargo) and the BSE-16 touched-crate
#                                selection (gate_touched_crates.sh --dry-run,
#                                plan only — no cargo test/check is ever run
#                                by this script) against it, record the
#                                verdict in .predict/last-<branch>.json.
#   predict_merge.sh --check    cheap freshness check: refuse (exit 3) if
#                                origin/main has moved, HEAD has moved, or the
#                                working tree is dirty since the last predict
#                                run recorded — otherwise replay that
#                                recorded verdict (§13 F-14).
#
# `git fetch origin main` runs FIRST in both modes: a network failure there
# is reported as exit 4, distinct from every other refusal (exit 3, "the
# record is stale") because a stale record is a fact about THIS branch and a
# fetch failure is a fact about the network — conflating the two would make a
# transient outage look like a rebase-worthy staleness.
#
# THE MERGE HAPPENS IN A THROWAWAY WORKTREE, NEVER THE WORKING TREE
# -------------------------------------------------------------------
# `git worktree add --detach <tmp>/merge <head-sha>` then
# `git merge --no-edit origin/main` INSIDE that worktree. This repo's own
# working tree, branch and HEAD are never touched — no `git merge` or
# `git rebase` ever runs against the branch this script is invoked from. The
# worktree path is a fresh `mktemp -d` per invocation (never a fixed name),
# so two workers running this at the same time never collide, and an EXIT
# trap always removes it (both `git worktree remove` to unregister it from
# .git/worktrees and `rm -rf` as a backstop), whether the script exits 0, a
# refusal code, or is killed mid-run.
#
# THE RECORD IS A TUPLE, NOT A BOOLEAN
# --------------------------------------
# (origin_main_sha, head_sha, tree_clean) in .predict/last-<branch>.json.
# `--check` recomputes the first two live and compares — a `tree_clean: true`
# recorded against a origin/main that has since moved, or a HEAD that has
# since moved, or a tree that has since gone dirty, is worthless and MUST be
# refused rather than replayed as if still true.
#
# Usage: scripts/predict_merge.sh [--check]
set -uo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)" || exit 1
cd "$REPO_ROOT" || exit 1

usage() {
    printf 'usage: %s [--check]\n' "$0" >&2
    exit 2
}

check_mode=0
for arg in "$@"; do
    case "$arg" in
        --check) check_mode=1 ;;
        *) usage ;;
    esac
done

branch="$(git rev-parse --abbrev-ref HEAD)" || exit 1
record_dir="$REPO_ROOT/.predict"
record_file="$record_dir/last-${branch}.json"

# git fetch origin main -- ALWAYS first, in both modes. A failure here is a
# network fact, not a staleness fact, so it gets its own exit code (4).
if ! git fetch origin main >/dev/null 2>&1; then
    echo "predict_merge: git fetch origin main failed (network unreachable?)" >&2
    exit 4
fi

origin_main_sha="$(git rev-parse origin/main)" || exit 1
head_sha="$(git rev-parse HEAD)" || exit 1

if [ "$check_mode" -eq 1 ]; then
    if [ ! -f "$record_file" ]; then
        echo "predict_merge --check: no record at ${record_file} -- run 'make predict' first" >&2
        exit 3
    fi

    rec_origin_main_sha="$(jq -r '.origin_main_sha' "$record_file")" || exit 1
    rec_head_sha="$(jq -r '.head_sha' "$record_file")" || exit 1
    rec_tree_clean="$(jq -r '.tree_clean' "$record_file")" || exit 1

    # BEGIN freshness-check-main-moved
    if [ "$origin_main_sha" != "$rec_origin_main_sha" ]; then
        echo "predict_merge --check: origin/main moved since predict (${rec_origin_main_sha} -> ${origin_main_sha})" >&2
        exit 3
    fi
    # END freshness-check-main-moved

    # BEGIN freshness-check-head-mismatch
    if [ "$head_sha" != "$rec_head_sha" ]; then
        echo "predict_merge --check: HEAD changed since predict (${rec_head_sha} -> ${head_sha})" >&2
        exit 3
    fi
    # END freshness-check-head-mismatch

    # BEGIN freshness-check-tree-dirty
    if [ -n "$(git status --porcelain)" ]; then
        echo "predict_merge --check: working tree is dirty since predict" >&2
        exit 3
    fi
    # END freshness-check-tree-dirty

    if [ "$rec_tree_clean" = "true" ]; then
        echo "predict_merge --check: fresh -- merge(origin/main, HEAD) predicted to PASS"
        exit 0
    fi
    echo "predict_merge --check: fresh, but merge(origin/main, HEAD) was predicted to FAIL the tree-property guards -- see .predict/last-${branch}.json" >&2
    exit 1
fi

# --- predict mode: build merge(origin/main, HEAD) in a throwaway worktree.
tmpbase="$(mktemp -d)" || exit 1
mergetree="$tmpbase/merge"

cleanup() {
    git worktree remove --force "$mergetree" >/dev/null 2>&1
    rm -rf "${tmpbase:?tmpbase must be set}"
}
trap cleanup EXIT

if ! git worktree add --detach --quiet "$mergetree" "$head_sha" >/dev/null 2>&1; then
    echo "predict_merge: failed to create merge worktree" >&2
    exit 1
fi

merge_ok=1
if ! git -C "$mergetree" merge --no-edit "$origin_main_sha" >/dev/null 2>&1; then
    merge_ok=0
    git -C "$mergetree" merge --abort >/dev/null 2>&1
fi

tree_clean=false
if [ "$merge_ok" -eq 1 ]; then
    guard_rc=0
    ( cd "$mergetree" && scripts/guard_tree.sh --no-cargo ) || guard_rc=$?
    gate_rc=0
    ( cd "$mergetree" && scripts/gate_touched_crates.sh --dry-run ) || gate_rc=$?
    if [ "$guard_rc" -eq 0 ] && [ "$gate_rc" -eq 0 ]; then
        tree_clean=true
    fi
else
    echo "predict_merge: merge(origin/main, HEAD) has conflicts -- cannot predict" >&2
fi

mkdir -p "$record_dir" || exit 1
printf '{"origin_main_sha":"%s","head_sha":"%s","tree_clean":%s}\n' \
    "$origin_main_sha" "$head_sha" "$tree_clean" > "$record_file" || exit 1

if [ "$tree_clean" = "true" ]; then
    echo "predict_merge: merge(origin/main, HEAD) predicted to PASS the tree-property guards"
    exit 0
fi
echo "predict_merge: merge(origin/main, HEAD) predicted to FAIL the tree-property guards" >&2
exit 1
