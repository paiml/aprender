#!/usr/bin/env bash
# carry_milestone.sh -- on release, open issues/PRs in the released milestone move to the next one
#
#   bash scripts/release/carry_milestone.sh X.Y.Z [--repo O/R] [--execute]
#
# Operator, 2026-11-03 06:30Z: "Milestones are labels, NOT gates. ... On each release: open issues in that
# milestone auto-move to the next one." Called by promote_nightly.sh after the tag is pushed.
# DEFAULT IS A DRY RUN (prints the moves). With --execute it PATCHes each item's milestone (REST write,
# GH-2) and checks the returned object: a null/mismatched milestone is a failure, never a silent pass.
#
# NEXT milestone = the open milestone with the smallest X.Y.Z title strictly greater than the released one
# (non-semver titles such as "backlog" are never a target). None: refuse, move nothing.
# Writes are paced (CARRY_PACE seconds, default 3; the content-creation limit is shared fleet-wide).
# gh is $GH (default gh) so the contract test can substitute a fake.
#
# exit: 0 done (or nothing to move) · 1 a move failed / no next milestone · 2 usage
set -uo pipefail
PROG=carry_milestone
GH=${GH:-gh}
PACE=${CARRY_PACE:-3}
die() { printf '%s: REFUSE -- %s\n' "$PROG" "$*" >&2; exit 1; }
usage() { printf '%s: %s\nusage: carry_milestone.sh X.Y.Z [--repo O/R] [--execute]\n' "$PROG" "$*" >&2; exit 2; }

ver="" repo="paiml/aprender" execute=0
while [ $# -gt 0 ]; do
    case "$1" in
        --repo) [ $# -ge 2 ] || usage "--repo needs a value"; repo=$2; shift 2 ;;
        --execute) execute=1; shift ;;
        -*) usage "unknown argument: $1" ;;
        *) [ -z "$ver" ] || usage "one version only"; ver=$1; shift ;;
    esac
done
ver=${ver#v}
[[ $ver =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || usage "version must be X.Y.Z"

ms=$($GH api "repos/$repo/milestones?state=open&per_page=100" --paginate 2> /dev/null) || die "cannot list milestones of $repo (not_measured)"
# one JSON array per page -> flatten
rows=$(printf '%s' "$ms" | jq -s -r 'add // [] | .[] | [.number, .title] | @tsv') || die "milestone list is not JSON"
from_n="" next_n="" next_t="" next_key=""
vkey() { printf '%s' "$1" | awk -F. '{printf "%06d%06d%06d", $1, $2, $3}'; }
vk=$(vkey "$ver")
while IFS=$'\t' read -r n t; do
    [ -n "$n" ] || continue
    [ "$t" = "$ver" ] && from_n=$n
    [[ $t =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || continue
    k=$(vkey "$t")
    if [[ $k > $vk ]] && { [ -z "$next_key" ] || [[ $k < $next_key ]]; }; then next_key=$k next_n=$n next_t=$t; fi
done <<< "$rows"
if [ -z "$from_n" ]; then printf '%s: milestone %s is not open in %s: nothing to carry\n' "$PROG" "$ver" "$repo"; exit 0; fi
[ -n "$next_n" ] || die "no open milestone newer than $ver in $repo to carry into"

nums=$( { $GH issue list -R "$repo" --milestone "$ver" --state open --limit 500 --json number --jq '.[].number' \
        && $GH pr list -R "$repo" --milestone "$ver" --state open --limit 500 --json number --jq '.[].number'; } 2> /dev/null) \
    || die "cannot list open items of milestone $ver (not_measured)"
fail=0 moved=0
for n in $nums; do
    if [ "$execute" != 1 ]; then printf '%s: dry-run: #%s %s -> %s\n' "$PROG" "$n" "$ver" "$next_t"; continue; fi
    got=$(printf '{"milestone":%s}' "$next_n" | $GH api -X PATCH "repos/$repo/issues/$n" --input - --jq '.milestone.number' 2> /dev/null)
    if [ "$got" = "$next_n" ]; then moved=$((moved + 1)); printf '%s: #%s %s -> %s\n' "$PROG" "$n" "$ver" "$next_t"
    else fail=$((fail + 1)); printf '%s: FAILED #%s (returned milestone "%s", wanted %s)\n' "$PROG" "$n" "$got" "$next_n" >&2; fi
    sleep "$PACE"
done
printf '%s: %s -> %s: moved=%s failed=%s%s\n' "$PROG" "$ver" "$next_t" "$moved" "$fail" "$([ "$execute" = 1 ] || echo ' (dry-run)')"
[ "$fail" = 0 ] || exit 1
exit 0
