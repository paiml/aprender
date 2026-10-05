#!/usr/bin/env bash
# pretag_gate.sh -- the tag step's gate: a release tag is cut only on a commit that
# can already be published (#4805).
#
#   bash scripts/release/pretag_gate.sh [ROOT]     # ROOT: the release worktree (default: cwd)
#
# WHY. The publish gates (scripts/check_publish_preflight.sh and the clean-room gate)
# ran only AFTER the tag. v0.70.0 was tagged and then refused at publish (R4/R5/R7/R8),
# so the tag named a release that could not ship and a new patch number had to be cut.
# This script asks the same two gates the same question, on the same commit, BEFORE
# `git tag`:
#
#   1. check_publish_preflight.sh --pre-tag on HEAD: R1 R2 R4 R5 R6 R7 R8 exactly as at
#      publish. R3 (tag at HEAD) cannot hold yet; it is replaced by "no v<version>
#      names another commit" and is still judged at publish.
#   2. clean_room_gate_sha on HEAD (scripts/release/lib_clean_room_gate.sh, the code
#      cascade-publish.sh runs): a green `clean-room (aprender)` job in paiml/infra
#      clean-room.yml whose recorded tested commit is exactly HEAD.
#
# Nothing at publish is removed or loosened: cascade-publish.sh still runs both gates
# on the tag, which then resolves to this same HEAD, so a green pre-tag run is
# re-proved there rather than trusted.
#
# EXIT  0 both green on HEAD · 1 a gate refused · 2 not measured (no clean-room run
#       for HEAD yet, a preflight that cannot judge, no repository). 2 is not a pass.
#
# SEAM (the self-test only; production never sets it):
#   PRETAG_PREFLIGHT   the preflight script (default: ROOT/scripts/check_publish_preflight.sh)
#
# Self-test: scripts/release/check_pretag_gate.sh
set -uo pipefail

PROG=pretag_gate.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-$PWD}"

# shellcheck source=scripts/release/lib_clean_room_gate.sh
. "$HERE/lib_clean_room_gate.sh" || { echo "NOT_MEASURED $PROG: cannot load lib_clean_room_gate.sh"; exit 2; }

head="$(git -C "$ROOT" rev-parse --verify --quiet 'HEAD^{commit}' 2>/dev/null)" || head=""
[ -n "$head" ] || { echo "NOT_MEASURED $PROG: $ROOT is not a git repository with a HEAD"; exit 2; }
preflight="${PRETAG_PREFLIGHT:-"$ROOT/scripts/check_publish_preflight.sh"}"
echo "$PROG: judging HEAD $head in $ROOT before any tag"

# 1. the publish preflight, pre-tag mode, on HEAD
pf_rc=0
pf_out="$(PUBLISH_PREFLIGHT_ROOT="$ROOT" bash "$preflight" --pre-tag 2>&1)" || pf_rc=$?
printf '%s\n' "$pf_out" | sed 's/^/  preflight | /'
case "$pf_rc" in
    0) pf=PASS ;;
    1) pf=REFUSE ;;
    *) pf=NOT_MEASURED ;;
esac

# 2. the clean-room gate, on HEAD by sha
cr_rc=0
cr_out="$(clean_room_gate_sha "$ROOT" "$head" "pre-tag HEAD" 2>&1)" || cr_rc=$?
printf '%s\n' "$cr_out" | sed 's/^/  clean-room | /'
if [ "$cr_rc" -eq 0 ]; then
    cr=PASS
elif [[ "$cr_out" == *"-- found none"* ]]; then
    cr=NOT_MEASURED   # no clean-room run on HEAD at all: dispatch one, then ask again
else
    cr=REFUSE
fi

verdict="preflight=$pf clean-room=$cr sha=$head"
if [ "$pf" = PASS ] && [ "$cr" = PASS ]; then
    echo "PRETAG PASS $verdict"
    exit 0
fi
if [ "$pf" = REFUSE ] || [ "$cr" = REFUSE ]; then
    echo "PRETAG REFUSE $verdict -- no tag"
    exit 1
fi
echo "PRETAG NOT_MEASURED $verdict -- no tag"
exit 2
