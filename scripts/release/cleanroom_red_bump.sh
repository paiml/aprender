#!/usr/bin/env bash
# cleanroom_red_bump.sh -- ANDON T1 (#6491): clean-room (aprender) went red on a tag, so open the next-patch bump
# PR at once instead of waiting for a human to notice the fix has merged (v0.72.1 -> v0.72.2 took >2 h).
#
#   cleanroom_red_bump.sh <failed-version> [<clean-room-run-id>]   prints NEXT <version> and what it did
#   cleanroom_red_bump.sh --next <version>                          prints the next patch version only
#   cleanroom_red_bump.sh --self-test                               the case table, no network
#
# The next patch is X.Y.(Z+1) of the failed tag. Idempotent: nothing is done when a tag or a `release: <next>`
# PR (open or merged) exists. The tag test is an EXACT ref match (refs/tags/v<next>), never a prefix: v0.72.2 must
# not be hidden by an existing v0.72.20. It drafts the bump with prepare_bump.sh, fills the one-paragraph placeholder with
# the andon cause, and ships it with prepare_bump.sh --ship.
#
# --SHIP SCOPE: --ship is prepare_bump.sh's own path and does all of this, with its own gates unchanged:
# judges the PR body by §6 R-2, requires the model-ladder receipts for <next>, runs the pre-push checks,
# commits, pushes the release-<next> branch, opens the `release: <next>` PR in milestone <next> and arms
# auto-merge. It never tags, publishes, yanks or merges past the required checks; autopilot.sh runs the rest
# of the train once the PR merges. No gate is waived here (T-2 GO receipt, R-2 body check, auto-merge arm). Any refusal is printed as `ANDON-BUMP-BLOCKED <why>`, rc 1.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
GH=${CRB_GH:-gh}; PREPARE=${CRB_PREPARE:-$HERE/prepare_bump.sh}; REPO=${CRB_REPO:-paiml/aprender}

crb_next_patch() { # crb_next_patch v0.72.1 -> 0.72.2 ; rc 1 for anything but vX.Y.Z / X.Y.Z
    local v=${1#v}
    [[ $v =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]] || return 1
    printf '%s.%s.%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "$((BASH_REMATCH[3] + 1))"
}

if [ "${1:-}" = --self-test ]; then
    fail=0
    chk() { if [ "$2" = "$3" ]; then echo "  ok   $1"; else echo "  FAIL $1: got '$2' want '$3'"; fail=1; fi; }
    chk "v0.72.1 -> 0.72.2" "$(crb_next_patch v0.72.1)" 0.72.2
    chk "0.72.9 -> 0.72.10" "$(crb_next_patch 0.72.9)" 0.72.10
    chk "rc candidate refused" "$(crb_next_patch 0.72.1-rc.1 || echo refused)" refused
    chk "empty refused" "$(crb_next_patch '' || echo refused)" refused
    d=$(mktemp -d) || exit 2
    printf '#!/bin/sh\ncase "$*" in *matching-refs*) echo 0;; *) echo 0;; esac\n' > "$d/gh"
    printf '#!/bin/sh\necho "$*" >> "%s/calls"\n' "$d" > "$d/prep"
    chmod +x "$d/gh" "$d/prep"
    out=$(CRB_GH="$d/gh" CRB_PREPARE="$d/prep" CRB_NO_EDIT=1 bash "$0" v0.72.1 99 2>&1); rc=$?
    chk "no tag, no PR -> bump drafted then shipped" "$(tr '\n' '|' < "$d/calls" 2>/dev/null)" "0.72.2|0.72.2 --ship|"
    chk "rc 0" "$rc" 0
    rm -f "$d/calls"
    printf '#!/bin/sh\ncase "$*" in *matching-refs*) echo 1;; *) echo 0;; esac\n' > "$d/gh"
    CRB_GH="$d/gh" CRB_PREPARE="$d/prep" CRB_NO_EDIT=1 bash "$0" v0.72.1 > /dev/null 2>&1
    chk "tag exists -> prepare_bump not called" "$([ -e "$d/calls" ] && echo called || echo none)" none
    printf '#!/bin/sh\ncase "$*" in *matching-refs*) echo 0;; *) echo 1;; esac\n' > "$d/gh"
    CRB_GH="$d/gh" CRB_PREPARE="$d/prep" CRB_NO_EDIT=1 bash "$0" v0.72.1 > /dev/null 2>&1
    chk "PR exists -> prepare_bump not called" "$([ -e "$d/calls" ] && echo called || echo none)" none
    printf '#!/bin/sh\nexit 1\n' > "$d/gh"
    out=$(CRB_GH="$d/gh" CRB_PREPARE="$d/prep" CRB_NO_EDIT=1 bash "$0" v0.72.1 2>&1); rc=$?
    chk "gh failure -> not_measured, blocked rc 1" "$rc:${out%% *}" "1:ANDON-BUMP-BLOCKED"
    # exact tag match: a stub that APPLIES the --jq filter to canned matching-refs JSON, so the filter itself is tested
    cat > "$d/gh" <<'STUB'
#!/bin/bash
expr=; while [ $# -gt 0 ]; do [ "$1" = --jq ] && expr=$2; shift; done
case "$expr" in *refs/tags*) printf '%s' "$CRB_REFS" | jq "$expr" ;; *) echo 0 ;; esac
STUB
    chmod +x "$d/gh"; rm -f "${d:?}/calls"
    CRB_REFS='[{"ref":"refs/tags/v0.72.20"}]' CRB_GH="$d/gh" CRB_PREPARE="$d/prep" CRB_NO_EDIT=1 bash "$0" v0.72.1 > /dev/null 2>&1
    chk "only v0.72.20 exists (prefix of v0.72.2) -> bump still drafted" "$(tr '\n' '|' < "$d/calls" 2>/dev/null)" "0.72.2|0.72.2 --ship|"
    rm -f "${d:?}/calls"
    CRB_REFS='[{"ref":"refs/tags/v0.72.2"},{"ref":"refs/tags/v0.72.20"}]' CRB_GH="$d/gh" CRB_PREPARE="$d/prep" CRB_NO_EDIT=1 bash "$0" v0.72.1 > /dev/null 2>&1
    chk "exact v0.72.2 exists -> prepare_bump not called" "$([ -e "$d/calls" ] && echo called || echo none)" none
    # wiring: autopilot.sh calls this script on a non-success clean-room, before its die
    w=$(awk '/cleanroom_red_bump.sh/ {c = NR} /die "clean-room \(aprender\) on/ {d = NR} END {print (c && d && c < d) ? "wired" : "unwired"}' "$HERE/autopilot.sh")
    chk "autopilot calls the bump before the clean-room die" "$w" wired
    rm -rf -- "${d:?}"
    if [ "$fail" -eq 0 ]; then echo "cleanroom_red_bump self-test: PASS"; else echo "cleanroom_red_bump self-test: FAIL"; fi
    exit "$fail"
fi
if [ "${1:-}" = --next ]; then crb_next_patch "${2:-}" || { echo "not a release version: ${2:-}" >&2; exit 2; }; exit 0; fi

failed=${1:-}; run=${2:-unknown}
next=$(crb_next_patch "$failed") || { echo "usage: cleanroom_red_bump.sh <failed-version> [<run-id>]" >&2; exit 2; }
blocked() { printf 'ANDON-BUMP-BLOCKED %s\n' "$*"; exit 1; }

tags=$($GH api "repos/$REPO/git/matching-refs/tags/v$next" --jq "[.[] | select(.ref == \"refs/tags/v$next\")] | length") || blocked "cannot read tags for v$next (not_measured)"
[ "${tags:-0}" = 0 ] || { echo "NEXT $next: tag v$next exists; nothing to open"; exit 0; }
prs=$($GH pr list --repo "$REPO" --state all --search "\"release: $next\" in:title" --json title --jq "[.[] | select(.title == \"release: $next\")] | length") \
    || blocked "cannot list PRs for release: $next (not_measured)"
[ "${prs:-0}" = 0 ] || { echo "NEXT $next: a release: $next PR exists; nothing to open"; exit 0; }

bash "$PREPARE" "$next" || blocked "prepare_bump.sh $next (draft) failed"
if [ "${CRB_NO_EDIT:-}" != 1 ]; then
    REPO_ROOT="$(cd "$HERE/../.." && pwd)" || blocked "cannot resolve the repo root"
    # shellcheck source=scripts/release/lib_release_params.sh
    . "$REPO_ROOT/scripts/release/lib_release_params.sh" || blocked "cannot source lib_release_params.sh"
    release_params "$next" "$REPO_ROOT" || blocked "release_params $next failed"
    cl="$AP/bump/CHANGELOG.md"
    mark='<!-- one-paragraph summary of the train: EDIT BEFORE --ship -->'
    msg="Patch for v${failed#v}: clean-room (aprender) went red on the tag (infra run $run); this bump was opened automatically by the ANDON T1 branch (#6491)."
    [ -f "$cl" ] || blocked "no drafted CHANGELOG at $cl"
    n=$(grep -cF "$mark" "$cl"); [ "$n" = 1 ] || blocked "$cl carries $n placeholder lines, expected 1"
    sed -i "s|$mark|$msg|" "$cl" || blocked "cannot fill the CHANGELOG placeholder"
fi
bash "$PREPARE" "$next" --ship || blocked "prepare_bump.sh $next --ship refused"
echo "NEXT $next: bump PR opened for the red clean-room on v${failed#v}"
