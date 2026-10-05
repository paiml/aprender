#!/usr/bin/env bash
# release_gate.sh -- the gate in front of `gh release create` (#4805): the GitHub
# release (the public step that fires binary-release.yml and names the version) is
# made only for a tag that can already be published.
#
#   bash scripts/release/release_gate.sh <tag> [ROOT]   # ROOT: the release worktree (default: cwd)
#
# WHY. The publish gates ran only at publish. v0.70.0 was tagged and released, then
# refused at publish (R4/R5/R7/R8), so a new patch number had to be cut. This script
# asks the two publish gates the same question, on the same tag, BEFORE the release:
#
#   1. scripts/check_publish_preflight.sh, unchanged: every rule, R3 (tag at HEAD) included.
#   2. clean_room_gate ROOT TAG (scripts/release/lib_clean_room_gate.sh, the code
#      cascade-publish.sh runs): a green `clean-room (aprender)` job whose recorded
#      tested commit is exactly the commit the tag names.
#
# No check moves and none is loosened: these are the publish gates, run once more
# earlier. cascade-publish.sh still runs both again at publish.
#
# EXIT  0 both green · 1 a gate refused · 2 not measured (no clean-room run for the
#       tag yet, a preflight that cannot judge, no repository, tag not at HEAD of ROOT).
#       2 is not a pass.
#
# SEAM (the self-test only; production never sets it):
#   RELEASE_PREFLIGHT   the preflight script (default: ROOT/scripts/check_publish_preflight.sh)
#
# Self-test: scripts/release/check_release_gate.sh
set -uo pipefail

PROG=release_gate.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="${1:-}"
ROOT="${2:-$PWD}"
[ -n "$TAG" ] || { echo "usage: $PROG <tag> [ROOT]" >&2; exit 2; }

# shellcheck source=scripts/release/lib_clean_room_gate.sh
. "$HERE/lib_clean_room_gate.sh" || { echo "NOT_MEASURED $PROG: cannot load lib_clean_room_gate.sh"; exit 2; }

head="$(git -C "$ROOT" rev-parse --verify --quiet 'HEAD^{commit}' 2>/dev/null)" || head=""
[ -n "$head" ] || { echo "NOT_MEASURED $PROG: $ROOT is not a git repository with a HEAD"; exit 2; }
tagged="$(git -C "$ROOT" rev-parse --verify --quiet "refs/tags/${TAG}^{commit}" 2>/dev/null)" || tagged=""
[ "$tagged" = "$head" ] || { echo "NOT_MEASURED $PROG: tag $TAG names '${tagged:-nothing}', not HEAD $head of $ROOT"; exit 2; }
preflight="${RELEASE_PREFLIGHT:-"$ROOT/scripts/check_publish_preflight.sh"}"
echo "$PROG: judging $TAG = $head in $ROOT before the release"

# 1. the publish preflight, unchanged, on HEAD (= the tag)
pf_rc=0
pf_out="$(PUBLISH_PREFLIGHT_ROOT="$ROOT" bash "$preflight" 2>&1)" || pf_rc=$?
printf '%s\n' "$pf_out" | sed 's/^/  preflight | /'
case "$pf_rc" in
    0) pf=PASS ;;
    1) pf=REFUSE ;;
    *) pf=NOT_MEASURED ;;
esac

# 2. the clean-room gate on the tag, exactly as cascade-publish.sh calls it
cr_rc=0
cr_out="$(clean_room_gate "$ROOT" "$TAG" 2>&1)" || cr_rc=$?
printf '%s\n' "$cr_out" | sed 's/^/  clean-room | /'
if [ "$cr_rc" -eq 0 ]; then
    cr=PASS
elif [[ "$cr_out" == *"-- found none"* ]]; then
    cr=NOT_MEASURED   # no clean-room run on the tag at all: dispatch one, then ask again
else
    cr=REFUSE
fi

verdict="preflight=$pf clean-room=$cr tag=$TAG sha=$head"
if [ "$pf" = PASS ] && [ "$cr" = PASS ]; then
    echo "RELEASE-GATE PASS $verdict"
    exit 0
fi
if [ "$pf" = REFUSE ] || [ "$cr" = REFUSE ]; then
    echo "RELEASE-GATE REFUSE $verdict -- no release"
    exit 1
fi
echo "RELEASE-GATE NOT_MEASURED $verdict -- no release"
exit 2
