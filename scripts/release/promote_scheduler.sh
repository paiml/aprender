#!/usr/bin/env bash
# promote_scheduler.sh -- the unattended driver of promote_nightly.sh (N26: "the nightly is the release", PMAT-5306)
#
#   bash scripts/release/promote_scheduler.sh --repo DIR --state DIR [--not-before ISO8601Z] [--dispatch-cleanroom] [--execute]
#
# Run by a systemd user timer on the build host (scripts/release/systemd/). Dry-run unless --execute. Each run:
#   1. before --not-before: do nothing (exit 0).
#   2. fetch origin main + nightly-evidence. Candidates = models-crux/<sha>/ dirs whose verdict is state=green, sha on
#      origin/main, newest commit first (at most 5).
#   3. per candidate, newest first: find a paiml/infra clean-room run whose 'clean-room (aprender)' job succeeded and whose
#      log says tested-sha == exactly that sha. None: with --dispatch-cleanroom and none already dispatched for it, dispatch
#      one on the NEWEST candidate only (gh workflow run clean-room.yml -f repos=aprender -f ref=<sha>), exit 3 (pending;
#      the timer's next tick judges it). No fallback to an older sha while the newest waits.
#   4. run rc_publish_gate.sh --verify in a fresh worktree of the sha; build receipts (receipts_from_night.sh);
#      run promote_nightly.sh DRY-RUN; with --execute and the dry-run exit 0, run it again with --execute.
# It never waives, never edits a receipt, never tags itself, never deletes a tag, never force-pushes: the one write to git
# is promote_nightly.sh --execute, which refuses an existing tag and goes through pre-push-tags.sh --arm-release.
# Env overrides (the falsifier uses them): GH, PROMOTE, RECEIPTS, DRYRUN_GATE, EVIDENCE_DIR.
# exit: 0 promoted (or dry-run would promote, or too early) - 1 no candidate passed - 3 pending clean-room - 2 usage
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
GH=${GH:-gh}; PROMOTE=${PROMOTE:-$HERE/promote_nightly.sh}; RECEIPTS=${RECEIPTS:-$HERE/receipts_from_night.sh}
INFRA=paiml/infra
repo="" state="" nb="" disp=0 exe=0
usage() { printf 'promote_scheduler: %s\n' "$*" >&2; exit 2; }
say() { printf 'promote_scheduler: %s\n' "$*"; }
while [ $# -gt 0 ]; do
    case "$1" in
        --repo) [ $# -ge 2 ] || usage "--repo needs a value"; repo=$2; shift 2 ;;
        --state) [ $# -ge 2 ] || usage "--state needs a value"; state=$2; shift 2 ;;
        --not-before) [ $# -ge 2 ] || usage "--not-before needs a value"; nb=$2; shift 2 ;;
        --dispatch-cleanroom) disp=1; shift ;;
        --execute) exe=1; shift ;;
        *) usage "unknown argument: $1" ;;
    esac
done
[ -d "${repo:-/nonexistent}" ] && [ -n "$state" ] || usage "--repo DIR and --state DIR are required"
mkdir -p -- "$state" || usage "cannot create $state"
if [ -n "$nb" ]; then
    nbs=$(date -u -d "$nb" +%s 2> /dev/null) || usage "bad --not-before '$nb'"
    [ "$(date -u +%s)" -ge "$nbs" ] || { say "too early (before $nb): nothing done"; exit 0; }
fi
G() { git -C "$repo" "$@"; }
G fetch -q origin main nightly-evidence 2> /dev/null || { say "cannot fetch origin main/nightly-evidence: nothing measured"; exit 1; }

ev="${EVIDENCE_DIR:-}" own_ev=0 evroot=""
if [ -z "$ev" ]; then
    evroot=$(mktemp -d) && G worktree add -q --detach "$evroot/ev" refs/remotes/origin/nightly-evidence > /dev/null 2>&1 || { say "cannot read nightly-evidence"; exit 1; }
    ev="$evroot/ev"; own_ev=1
fi
cleanup() { if [ "$own_ev" = 1 ]; then G worktree remove --force "$ev" > /dev/null 2>&1; rm -rf -- "${evroot:?}"; fi; return 0; }
trap cleanup EXIT

cands=()
while read -r _ s; do cands+=("$s"); done < <(
    for d in "$ev"/models-crux/*/; do
        s=$(basename "$d"); [[ $s =~ ^[0-9a-f]{40}$ ]] || continue
        grep -qx 'state=green' "$d/verdict" 2> /dev/null || continue
        G merge-base --is-ancestor "$s" refs/remotes/origin/main 2> /dev/null || continue
        printf '%s %s\n' "$(G show -s --format=%ct "$s" 2> /dev/null || echo 0)" "$s"
    done | sort -rn | head -n 5)
[ "${#cands[@]}" -gt 0 ] || { say "no green models-crux sha on origin/main: nothing to promote"; exit 1; }

# clean-room summary for sha -> file; 0 found success, 1 none
cleanroom_for() {
    local sha=$1 out=$2 id jid jc shas
    while read -r id; do
        [ -n "$id" ] || continue
        read -r jc jid < <("$GH" run view "$id" --repo "$INFRA" --json jobs --jq '.jobs[] | select(.name=="clean-room (aprender)") | select(.status=="completed") | "\(.conclusion) \(.databaseId)"' 2> /dev/null | head -1)
        [ "${jc:-}" = success ] || continue
        shas=$("$GH" run view --job "$jid" --repo "$INFRA" --log 2> /dev/null | grep -oE 'tested-sha: [0-9a-f]{40}' | sort -u)
        [ "$shas" = "tested-sha: $sha" ] || continue
        jq -n --arg r "$id" --arg s "$sha" '{run_id:$r,job:"clean-room (aprender)",conclusion:"success",tested_shas:[$s]}' > "$out"
        return 0
    done < <("$GH" run list --repo "$INFRA" --workflow clean-room.yml --event workflow_dispatch --limit 20 --json databaseId --jq '.[].databaseId' 2> /dev/null)
    return 1
}

first=1
for sha in "${cands[@]}"; do
    d="$state/$sha"; mkdir -p -- "$d"
    if ! cleanroom_for "$sha" "$d/cleanroom.json"; then
        if [ "$first" = 1 ] && [ "$disp" = 1 ] && [ ! -e "$d/cleanroom-dispatched" ]; then
            if "$GH" workflow run clean-room.yml --repo "$INFRA" -f repos=aprender -f ref="$sha" > "$d/dispatch.log" 2>&1; then
                date -u +%FT%TZ > "$d/cleanroom-dispatched"; say "clean-room dispatched on ${sha:0:9}; pending"; exit 3
            fi
            say "clean-room dispatch on ${sha:0:9} failed ($(tail -n 1 "$d/dispatch.log"))"; exit 1
        fi
        if [ "$first" = 1 ] && [ -e "$d/cleanroom-dispatched" ]; then say "clean-room for ${sha:0:9} dispatched, not green yet; pending"; exit 3; fi
        say "${sha:0:9}: no clean-room success tested on this exact sha; skip"; first=0; continue
    fi
    first=0
    w=$(mktemp -d) && G worktree add -q --detach "$w/wt" "$sha" > /dev/null 2>&1 || { say "cannot make a worktree of $sha"; continue; }
    bash "${DRYRUN_GATE:-$HERE/rc_publish_gate.sh}" --verify "$w/wt" > "$d/publish-dryrun.log" 2>&1; drc=$?
    G worktree remove --force "$w/wt" > /dev/null 2>&1; rm -rf -- "${w:?}"
    rm -rf -- "${d:?}/receipts"
    bash "$RECEIPTS" --sha "$sha" --out "$d/receipts" --models-dir "$ev/models-crux/$sha" --cleanroom-json "$d/cleanroom.json" \
        --dryrun-rc "$drc" --dryrun-commit "$sha" > "$d/receipts.log" 2>&1 || { say "${sha:0:9}: receipts incomplete ($(tr '\n' ' ' < "$d/receipts.log"))"; continue; }
    if ! bash "$PROMOTE" --sha "$sha" --receipts "$d/receipts" --repo "$repo" > "$d/promote-dry.log" 2>&1; then
        say "${sha:0:9}: promote dry-run REFUSED: $(grep -m1 REFUSE "$d/promote-dry.log")"; continue
    fi
    if [ "$exe" != 1 ]; then say "${sha:0:9}: dry-run: would promote (re-run with --execute)"; exit 0; fi
    bash "$PROMOTE" --sha "$sha" --receipts "$d/receipts" --repo "$repo" --execute > "$d/promote-exec.log" 2>&1; rc=$?
    tail -n 6 "$d/promote-exec.log"; exit "$rc"
done
say "no candidate passed every gate"; exit 1
