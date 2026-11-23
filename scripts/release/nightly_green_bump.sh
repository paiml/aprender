#!/usr/bin/env bash
# nightly_green_bump.sh -- C4 (#6675): a CI bot opens the version-bump PR from the green nightly, so no patch
# release needs an agent edit. Extends BLD-002 (cleanroom_red_bump.sh, #6491); it forks nothing: the draft, the
# CHANGELOG fill and --ship are cleanroom_red_bump.sh -> prepare_bump.sh, with every one of their gates unchanged.
#
#   nightly_green_bump.sh                run: latest tag -> next patch, only when the newest scheduled Nightly Train on main is green
#   nightly_green_bump.sh --self-test    the case table, no network
#
# Fail closed: an unreadable tag list or nightly run is not_measured and opens nothing (rc 1, NIGHTLY-BUMP-BLOCKED).
# A red, running or absent nightly opens nothing (rc 0). Idempotent through cleanroom_red_bump.sh (tag / PR exists).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
GH=${NGB_GH:-gh}; CRB=${NGB_CRB:-$HERE/cleanroom_red_bump.sh}; REPO=${NGB_REPO:-paiml/aprender}

if [ "${1:-}" = --self-test ]; then
    fail=0
    chk() { if [ "$2" = "$3" ]; then echo "  ok   $1"; else echo "  FAIL $1: got '$2' want '$3'"; fail=1; fi; }
    d=$(mktemp -d "${TMPDIR:-/tmp}/ngb.XXXXXX") || exit 2
    trap 'rm -rf -- "${d:?}"' EXIT
    printf '#!/bin/sh\necho "$CRB_REASON|$*" >> "%s/calls"\n' "$d" > "$d/crb"; chmod +x "$d/crb"
    cat > "$d/gh" <<'STUB'
#!/bin/bash
case "$*" in
  *git/matching-refs/tags/v*) [ "$NGB_TAGS_FAIL" = 1 ] && exit 1; printf '%s' "$NGB_TAG" ;;
  */actions/workflows/*/runs*) [ "$NGB_RUN_FAIL" = 1 ] && exit 1; a="$*"; w=${a#*actions/workflows/}; echo "${w%%/*}" >> "$NGB_D/wf"; printf '%s' "$NGB_RUN" ;;
  *) exit 1 ;;
esac
STUB
    chmod +x "$d/gh"
    printf '#!/bin/sh\necho "${CRB_NEXT:-}|$*" >> "%s/next"\n' "$d" > "$d/crbn"; chmod +x "$d/crbn"
    export NGB_D="$d"
    run() { rm -f "${d:?}/calls" "${d:?}/wf"; NGB_GH="$d/gh" NGB_CRB="$d/crb" bash "$0" "$@" 2>&1; echo "rc=$?"; }
    export NGB_TAG=v0.73.0
    out=$(NGB_RUN=success run)
    chk "green nightly -> bump drafted from the latest tag" "$(cut -d'|' -f2 "$d/calls" | cut -d' ' -f1)" "v0.73.0"
    chk "reason names the nightly, not the clean-room" "$(grep -c 'Nightly Train on main is green' "$d/calls")" 1
    chk "rc 0" "${out##*rc=}" 0
    out=$(NGB_RUN=failure run); chk "red nightly -> nothing opened" "$([ -e "$d/calls" ] && echo called || echo none)" none
    chk "red nightly rc 0" "${out##*rc=}" 0
    out=$(NGB_RUN=in_progress run); chk "unfinished nightly -> nothing opened" "$([ -e "$d/calls" ] && echo called || echo none)" none
    out=$(NGB_RUN= run); chk "no nightly run at all -> nothing opened" "$([ -e "$d/calls" ] && echo called || echo none)" none
    out=$(NGB_RUN_FAIL=1 NGB_RUN=success run)
    chk "run read fails -> blocked rc 1" "${out%% *}:${out##*rc=}" "NIGHTLY-BUMP-BLOCKED:1"
    chk "run read fails -> nothing opened" "$([ -e "$d/calls" ] && echo called || echo none)" none
    out=$(NGB_TAGS_FAIL=1 NGB_RUN=success run)
    chk "tag read fails -> blocked rc 1" "${out%% *}:${out##*rc=}" "NIGHTLY-BUMP-BLOCKED:1"
    out=$(NGB_TAG= NGB_RUN=success run)
    chk "no tag at all -> blocked, not guessed" "${out%% *}" "NIGHTLY-BUMP-BLOCKED"
    # --workflow: which nightly decides green (default unchanged: nightly-train.yml)
    out=$(NGB_RUN=success run); chk "default nightly is nightly-train.yml" "$(cat "$d/wf")" nightly-train.yml
    out=$(NGB_RUN=success run --workflow nightly.yml); chk "--workflow nightly.yml reads that workflow" "$(cat "$d/wf")" nightly.yml
    out=$(NGB_RUN=success run --workflow '../x'); chk "--workflow outside [a-z0-9-].yml -> usage rc 2" "${out##*rc=}" 2
    chk "bad --workflow opens nothing" "$([ -e "$d/calls" ] && echo called || echo none)" none
    # --to: an explicit target (a minor train, e.g. 0.73.0) handed to cleanroom_red_bump.sh as CRB_NEXT
    export NGB_TAG=v0.72.4
    rm -f "$d/next"; out=$(NGB_RUN=success NGB_GH="$d/gh" NGB_CRB="$d/crbn" bash "$0" --to 0.73.0 2>&1; echo "rc=$?")
    chk "--to 0.73.0 -> CRB_NEXT=0.73.0 from v0.72.4" "$(cat "$d/next" 2>/dev/null)" "0.73.0|v0.72.4 nightly"
    chk "--to rc 0" "${out##*rc=}" 0
    out=$(NGB_RUN=success run --to 0.72.4); chk "--to not above the latest tag -> blocked rc 1" "${out%% *}:${out##*rc=}" "NIGHTLY-BUMP-BLOCKED:1"
    chk "--to not above the tag opens nothing" "$([ -e "$d/calls" ] && echo called || echo none)" none
    out=$(NGB_RUN=success run --to 0.72.10); chk "--to compares numerically (0.72.10 > 0.72.4)" "${out##*rc=}" 0
    out=$(NGB_RUN=success run --to 0.73); chk "--to not X.Y.Z -> usage rc 2" "${out##*rc=}" 2
    out=$(NGB_RUN=failure run --to 0.73.0); chk "--to with a red nightly opens nothing" "$([ -e "$d/calls" ] && echo called || echo none)" none
    out=$(NGB_RUN=success run --bogus); chk "unknown argument -> usage rc 2" "${out##*rc=}" 2
    if [ "$fail" -eq 0 ]; then echo "nightly_green_bump self-test: PASS"; else echo "nightly_green_bump self-test: FAIL"; fi
    exit "$fail"
fi

blocked() { printf 'NIGHTLY-BUMP-BLOCKED %s\n' "$*"; exit 1; }
usage() { printf 'usage: nightly_green_bump.sh [--to X.Y.Z] [--workflow <file>.yml] | --self-test (%s)\n' "$*" >&2; exit 2; }
# ngb_gt A B -> rc 0 when X.Y.Z A is strictly above X.Y.Z B (numeric per field: 0.72.10 > 0.72.4)
ngb_gt() {
    local a1 a2 a3 b1 b2 b3
    IFS=. read -r a1 a2 a3 <<< "${1#v}"; IFS=. read -r b1 b2 b3 <<< "${2#v}"
    [ "$a1" -ne "$b1" ] && { [ "$a1" -gt "$b1" ]; return; }
    [ "$a2" -ne "$b2" ] && { [ "$a2" -gt "$b2" ]; return; }
    [ "$a3" -gt "$b3" ]
}
to="" wf=nightly-train.yml
while [ $# -gt 0 ]; do
    case "$1" in
        --to) [ $# -ge 2 ] || usage "--to needs a value"; to=$2; shift 2 ;;
        --workflow) [ $# -ge 2 ] || usage "--workflow needs a value"; wf=$2; shift 2 ;;
        *) usage "unknown argument: $1" ;;
    esac
done
[[ $wf =~ ^[a-z0-9-]+\.ya?ml$ ]] || usage "--workflow must be a workflow file name, got '$wf'"
[ -z "$to" ] || [[ $to =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || usage "--to must be X.Y.Z, got '$to'"
# newest tag by version order; a read failure is not_measured
tag=$($GH api "repos/$REPO/git/matching-refs/tags/v" --jq '[.[].ref | ltrimstr("refs/tags/") | select(test("^v[0-9]+\\.[0-9]+\\.[0-9]+$"))] | sort_by(ltrimstr("v") | split(".") | map(tonumber)) | last // empty') \
    || blocked "cannot read tags (not_measured)"
[ -n "$tag" ] || blocked "no vX.Y.Z tag found; will not guess a base version"
if [ -n "$to" ]; then ngb_gt "$to" "$tag" || blocked "--to $to is not above the latest tag $tag; will not open a bump that goes down or sideways"; fi
if [ "$wf" = nightly-train.yml ]; then wname="Nightly Train"; else wname="nightly $wf"; fi
concl=$($GH api "repos/$REPO/actions/workflows/$wf/runs?branch=main&event=schedule&per_page=1" --jq '.workflow_runs[0] | if . == null then "" elif .status != "completed" then "in_progress" else .conclusion end') \
    || blocked "cannot read the $wname run (not_measured)"
[ "$concl" = success ] || { echo "NIGHTLY-BUMP: newest scheduled $wname on main is '${concl:-absent}', not success; nothing opened"; exit 0; }
if [ -n "$to" ]; then
    CRB_NEXT="$to" CRB_REASON="Release $to after $tag: the $wname on main is green; this bump was opened automatically by the green nightly bot (#6675)." bash "$CRB" "$tag" nightly
else
    CRB_REASON="Patch for $tag: the $wname on main is green; this bump was opened automatically by the green nightly bot (#6675)." bash "$CRB" "$tag" nightly
fi
