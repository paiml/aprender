#!/usr/bin/env bash
# dogfood.sh (user scope) — A SHIM. The protocol is NOT here.
#
# The canonical runner is scripts/dogfood.sh in the aprender repo. This file
# used to be a 1172-line SECOND COPY that had silently diverged in nine places
# (#2640, docs/audits/dogfood-divergence-2640.md). Canon is the repo copy: a
# canon at ~/.claude/skills/ is not in git, not reviewed, not CI-reachable and
# not diffable (#2361). scripts/check_dogfood_shim.sh caps this file, keeps it
# gateless and proves it fails CLOSED — from protected main, not from here.
#
# WHICH COPY RUNS (#2701). The runner at a pinned git REF (default origin/main),
# never the checkout's WORKING TREE: aprender on a feature branch without
# scripts/dogfood.sh made the fleet's release gate exit 2, and cost a release
# on 2026-08-23. The ref's scripts/ is extracted once per commit into a cache,
# and the commit is printed, so a receipt names the runner that produced it.
# DOGFOOD_CANON_REF=worktree runs the working tree — for editing the runner.
#
# REVISIT TRIGGER: dogfood invoked on a crate whose OWN repo is checked out but
# `aprender` is NOT. That is the signal to move the protocol into shared infra —
# not to copy the runner back here. It fails LOUDLY below, naming this trigger.
set -uo pipefail

CANON_ROOT="${DOGFOOD_CANON_ROOT:-$HOME/src/aprender}"
CANON_REF="${DOGFOOD_CANON_REF:-origin/main}"

die() {
    printf 'dogfood: %s\n  (override with DOGFOOD_CANON_ROOT=/path/to/aprender' "$1" >&2
    printf ' and DOGFOOD_CANON_REF=<ref>)\n\n' >&2
    printf '  REVISIT TRIGGER: if the repo you are releasing IS checked out but\n' >&2
    printf '  aprender is NOT, this protocol has outgrown the aprender repo and\n' >&2
    printf '  belongs in shared infra. Move it there. Do NOT restore a local copy\n' >&2
    printf '  of the runner: a second copy is the defect aprender#2640 removed.\n' >&2
    exit 2
}

git -C "$CANON_ROOT" rev-parse --git-dir >/dev/null 2>&1 \
    || die "the canonical runner is not reachable: $CANON_ROOT is not an aprender checkout."

if [ "$CANON_REF" = worktree ]; then
    CANON="$CANON_ROOT/scripts/dogfood.sh"
    [ -x "$CANON" ] || die "the canonical runner is not reachable: no $CANON in the working tree."
    printf 'dogfood: canon = working tree of %s\n' "$CANON_ROOT" >&2
    exec "$CANON" "$@"
fi

sha=$(git -C "$CANON_ROOT" rev-parse --verify --quiet "$CANON_REF^{commit}") \
    || die "the canonical runner is not reachable: ref $CANON_REF does not resolve in $CANON_ROOT."
cache="${XDG_CACHE_HOME:-$HOME/.cache}/dogfood-canon"
dest="$cache/$sha"
if [ ! -x "$dest/scripts/dogfood.sh" ]; then
    mkdir -p "$cache" && tmp=$(mktemp -d "$cache/.x.XXXXXX") \
        || die "cannot create the canon cache under $cache."
    if ! git -C "$CANON_ROOT" archive "$sha" scripts | tar -x -C "$tmp" \
        || [ ! -x "$tmp/scripts/dogfood.sh" ]; then
        rm -rf "${tmp:?}"
        die "the canonical runner is not reachable: $CANON_REF ($sha) has no scripts/dogfood.sh."
    fi
    mv -T "$tmp" "$dest" 2>/dev/null || rm -rf "${tmp:?}"   # lost a race: theirs is identical
fi
printf 'dogfood: canon = %s @ %s\n' "$CANON_REF" "$sha" >&2
exec "$dest/scripts/dogfood.sh" "$@"
