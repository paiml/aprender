#!/usr/bin/env bash
# dogfood.sh (user scope) — A SHIM. The protocol is NOT here.
#
# The canonical runner is scripts/dogfood.sh in the aprender repo. This file
# used to be a 1172-line SECOND COPY of it, and the two had silently diverged in
# nine places (#2640, docs/audits/dogfood-divergence-2640.md). Every hunk was a
# real hardening that the other copy did not have, so neither copy was safe to
# delete and neither was safe to trust.
#
# Canon is the repo copy because a canon at ~/.claude/skills/ is not in git, not
# PR-reviewed, not CI-reachable and not diffable — the four properties whose
# absence produced #2361, where hardening this directory edited a file that
# never ran.
#
# WHAT KEEPS THIS FILE A SHIM. Nothing here, deliberately: a rule enforced by
# the file it constrains is not enforced. scripts/check_dogfood_shim.sh in the
# repo asserts that this file stays under its line cap, invokes no gate of its
# own, and fails CLOSED. That guard lives on protected main, so this file cannot
# raise its own ceiling.
#
# REVISIT TRIGGER — the one signal that this arrangement has outgrown itself:
#
#     the first time dogfood is invoked on a crate whose OWN repo is checked out
#     but `aprender` is NOT.
#
# That means the protocol is being asked to serve the fleet from a repo that is
# not present, and the correct answer is to move it into shared infra — not to
# copy the runner back here. It fails LOUDLY below, naming this trigger, so the
# person who hits it knows what they are looking at instead of patching around
# it. A second copy is precisely the defect #2640 removed.
set -uo pipefail

CANON_ROOT="${DOGFOOD_CANON_ROOT:-$HOME/src/aprender}"
CANON="$CANON_ROOT/scripts/dogfood.sh"

if [ ! -x "$CANON" ]; then
    printf 'dogfood: the canonical runner is not reachable.\n' >&2
    printf '  looked for: %s\n' "$CANON" >&2
    printf '  (override the checkout with DOGFOOD_CANON_ROOT=/path/to/aprender)\n\n' >&2
    printf '  REVISIT TRIGGER: if the repo you are releasing IS checked out but\n' >&2
    printf '  aprender is NOT, that is the signal this protocol has outgrown the\n' >&2
    printf '  aprender repo and belongs in shared infra. Move it there. Do NOT\n' >&2
    printf '  restore a local copy of the runner: a second copy is the defect\n' >&2
    printf '  aprender#2640 removed, and it took a 97-line triage to establish\n' >&2
    printf '  which of the two copies was safe to delete (neither was).\n' >&2
    exit 2
fi

exec "$CANON" "$@"
