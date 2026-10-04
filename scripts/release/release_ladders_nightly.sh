#!/usr/bin/env bash
# release_ladders_nightly.sh -- RQ-2 (ruling A, aprender-a7 2026-10-04): the 02:00Z nightly that runs
# the release ladders (deep, dogfood, models) at main HEAD once a version bump is on main and not yet
# tagged, so release day cuts AT that commit and reads the result (scripts/release/check_nightly_cut.sh)
# instead of running them (HANDOFF.md §2, §6, §7).
#
# Each ladder is autopilot.sh's OWN step body, via `autopilot.sh --ladders` -- one copy, one set of
# guards. Runs on the train host (lambda), from a dedicated checkout: autopilot builds into that
# checkout's target/. Triggered by a forjar-managed systemd timer (paiml/infra), never by CI.
#
# Leaves, under ROOT (RELEASE_LADDERS_ROOT, default /mnt/nvme-raid0/release-ladders):
#   index.tsv                         <sha>\t<version>\t<deep rc>\t<dogfood rc>\t<models rc>\t<finished epoch>
#   <sha>/STATUS, <sha>/*.log         autopilot's own state dir for the run (RELEASE_AP)
#   <sha>/dogfood/receipt.json        the dogfood receipt whose `commit` is <sha>, selected by commit
#   <sha>/models-t1/<host>.json       models_t1.sh's receipts, where autopilot already writes them
#
# A step's rc in the index: 0 GO · 1 RED (a finding: release day STOPs) · 2 not measured (a decline,
# an ENV/usage exit, or apr-gpu.lock busy (75), read from autopilot's "NO-GO rc=N" STOP line;
# release day falls back to running it). An unparseable STOP line is 1: fail closed toward STOP.
#
# EXIT 0 a row was written with every step GO, or there is nothing to measure
#      1 a row was written with a step not GO · 2 cannot run (fetch, lock, ROOT)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" || exit 2
ROOT="${RELEASE_LADDERS_ROOT:-/mnt/nvme-raid0/release-ladders}"
AUTOPILOT="${RELEASE_LADDERS_AUTOPILOT:-$REPO_ROOT/scripts/release/autopilot.sh}"   # the table's seam
REF="${RELEASE_LADDERS_REF:-origin/main}"
STEP_TIMEOUT="${RELEASE_LADDERS_STEP_TIMEOUT:-21600}"   # 6 h per step; a hung ladder must not hold the lock forever
say() { printf '%s release-ladders %s\n' "$(date -u +%FT%TZ)" "$*"; }

mkdir -p "$ROOT" || { say "cannot create $ROOT"; exit 2; }
exec 9> "$ROOT/.lock"
flock -n 9 || { say "another nightly holds $ROOT/.lock -- not starting a second"; exit 2; }

if [ "$REF" = origin/main ]; then
    git -C "$REPO_ROOT" fetch -q origin main --tags || { say "fetch failed: not measured"; exit 2; }
fi
S=$(git -C "$REPO_ROOT" rev-parse --verify "$REF^{commit}") || { say "cannot resolve $REF"; exit 2; }
V=$(git -C "$REPO_ROOT" show "$S:Cargo.toml" | sed -n '/^\[workspace\.package\]/,/^\[/s/^version *= *"\([0-9.]*\)".*/\1/p' | head -n 1)
[[ $V =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { say "cannot read [workspace.package] version at ${S:0:9}"; exit 2; }

# F4: only a bumped, untagged version is a release to measure for. Any other night writes no row.
if git -C "$REPO_ROOT" rev-parse -q --verify "refs/tags/v$V" > /dev/null; then
    say "v$V is already tagged: no pending release at ${S:0:9}, nothing to measure"; exit 0
fi
# A sha is measured again only while its newest row holds a "not measured" (2): a busy GPU lock one
# night must not cost the release its nightly (quorum r2 F3). A 0 or a 1 is final.
if [ -f "$ROOT/index.tsv" ] && awk -F"\t" -v s="$S" '$1 == s { last = $3 $4 $5 } END { exit !(last != "" && last !~ /2/) }' "$ROOT/index.tsv"; then
    say "${S:0:9} already has a final row: not measuring it twice"; exit 0
fi

AP="$ROOT/$S"
mkdir -p "$AP" || exit 2
say "measuring $V at $S into $AP"

# step_rc <step> -> the index rc for one autopilot --ladders run
step_rc() {
    local rc n from
    from=$(( $(wc -l 2> /dev/null < "$AP/STATUS" || echo 0) + 1 ))   # only THIS step's STOP line (quorum F3/F4)
    # 9>&-: no child (a model server, an ssh) inherits the nightly lock and wedges every later night.
    RELEASE_AP="$AP" timeout --kill-after=60 "$STEP_TIMEOUT" bash "$AUTOPILOT" --ladders "$V" "$S" "$1" > "$AP/nightly-$1.out" 2>&1 9>&-; rc=$?
    [ "$rc" = 0 ] && { echo 0; return; }
    [ "$rc" = 124 ] || [ "$rc" = 137 ] && { echo 2; return; }   # timed out: not measured, never a pass
    n=$(tail -n +"$from" "$AP/STATUS" 2> /dev/null | grep -E " STOP " | tail -n 1 | sed -n 's/.*NO-GO rc=\([0-9][0-9]*\).*/\1/p')
    case "$rc:$n" in
        2:*) echo 2 ;;            # autopilot's own usage/ENV exit: nothing was judged
        *:2 | *:75) echo 2 ;;     # the step declined, or the GPU lock was busy
        *) echo 1 ;;
    esac
}
DRC=$(step_rc deep); say "deep rc $DRC"
FRC=$(step_rc dogfood); say "dogfood rc $FRC"
MRC=$(step_rc models); say "models rc $MRC"

# The dogfood receipt by `commit`, never by newest mtime (HANDOFF §6 F2): a stale one is not copied.
mkdir -p "$AP/dogfood"
rm -f -- "$AP/dogfood/receipt.json"
for r in "$AP"/wt/.dogfood/receipt-*.json; do
    [ -f "$r" ] || continue
    if [ "$(jq -r '.commit // empty' -- "$r" 2> /dev/null)" = "$S" ]; then cp -- "$r" "$AP/dogfood/receipt.json"; fi
done

printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$S" "$V" "$DRC" "$FRC" "$MRC" "$(date -u +%s)" >> "$ROOT/index.tsv"
say "row: $S $V deep=$DRC dogfood=$FRC models=$MRC"
[ "$DRC$FRC$MRC" = 000 ]
