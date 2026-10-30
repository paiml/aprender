#!/usr/bin/env bash
# promote_scheduler.sh -- the unattended driver of promote_nightly.sh (N26: "the nightly is the release", PMAT-5306)
#
#   bash scripts/release/promote_scheduler.sh --repo DIR --state DIR [--not-before ISO8601Z] [--dispatch-cleanroom] [--execute] [--inbox FILE]
#
# Run by a systemd user timer on the build host (scripts/release/systemd/). Dry-run unless --execute.
# SHA-SELECTION RULE (full text: handoff/aprender-cop/scheduler-sha-rule.md):
#   candidates = the newest 5 models-crux/<sha>/ bundles on nightly-evidence, on origin/main, commit age <= 3 days.
#   A candidate must pass, on that EXACT sha: (a) models_nightly.sh --relay --measure crux (the real verifier: SHA256SUMS,
#   commit, measure, version, C's own models_t1.sh blob, replayed verdict) = green; (b) the release-day nightly lanes
#   (deep-doctests deep-nodefault deep-examples deep-bins-build cleanroom-gpu preflight milestone) each succeeded on it;
#   (c) a paiml/infra clean-room run whose log says tested-sha == exactly that sha; (d) rc_publish_gate.sh --verify.
#   The newest candidate that passes (a)+(b) is the one a clean-room is dispatched for. If the newest nightly is red the
#   rule is "last candidate that passes all of the above"; none -> SKIP and a line to the cop inbox. Never silent.
# GATES RUN FROM A CLEAN WORKTREE OF origin/main (never the checkout's own files); the checkout must be clean and AT
# origin/main or the run refuses. No environment variable changes behaviour: GH PROMOTE RECEIPTS DRYRUN_GATE EVIDENCE_DIR
# MODELS_NIGHTLY are unset at start; the test seams are read only from a file named .promote-test-seams NEXT TO THIS
# SCRIPT, which exists only in the falsifier's sandbox copy.
# It never waives, never edits a receipt, never tags itself, never deletes a tag, never force-pushes: the one write to git
# is promote_nightly.sh --execute, which refuses an existing tag and goes through pre-push-tags.sh --arm-release.
# exit: 0 promoted / would promote (dry-run) / too early - 1 nothing promoted (a line is in the cop inbox) - 3 clean-room running - 2 usage
set -uo pipefail
unset GH PROMOTE RECEIPTS DRYRUN_GATE EVIDENCE_DIR MODELS_NIGHTLY PUSHTAGS
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
GH=gh; INFRA=paiml/infra; APR=paiml/aprender; MAX_AGE_DAYS=3; NCAND=5; EVIDENCE_DIR=""; DISPATCH_WAIT=5
repo="" state="" nb="" disp=0 exe=0 inbox=/mnt/nvme-raid0/cop-inbox/inbox.md
usage() { printf 'promote_scheduler: %s\n' "$*" >&2; exit 2; }
say() { printf 'promote_scheduler: %s\n' "$*"; }
# inbox_line MSG: one line, <= 300 chars, to the cop inbox. A skipped or failed promotion is never silent.
inbox_line() {
    local t; t=$(date -u +%H:%MZ) # bashrs disable-line=DET002
    printf '%s | promote-scheduler | PMAT-5306 | %s\n' "$t" "$1" | cut -c1-300 >> "$inbox" 2> /dev/null || true
}
fail() { say "$*"; inbox_line "RED promote: $* (state $state)"; exit 1; }
while [ $# -gt 0 ]; do
    case "$1" in
        --repo) [ $# -ge 2 ] || usage "--repo needs a value"; repo=$2; shift 2 ;;
        --state) [ $# -ge 2 ] || usage "--state needs a value"; state=$2; shift 2 ;;
        --not-before) [ $# -ge 2 ] || usage "--not-before needs a value"; nb=$2; shift 2 ;;
        --inbox) [ $# -ge 2 ] || usage "--inbox needs a value"; inbox=$2; shift 2 ;;
        --dispatch-cleanroom) disp=1; shift ;;
        --execute) exe=1; shift ;;
        *) usage "unknown argument: $1" ;;
    esac
done
[ -d "${repo:-/nonexistent}" ] && [ -n "$state" ] || usage "--repo DIR and --state DIR are required"
if [ -n "$nb" ]; then
    nbs=$(date -u -d "$nb" +%s 2> /dev/null) || usage "bad --not-before '$nb'"
    [ "$(date -u +%s)" -ge "$nbs" ] || { say "too early (before $nb): nothing done"; exit 0; }
fi
install -d -- "$state" || usage "cannot create $state"
# a seams file inside the repo checkout (tracked or not) would stub the gates in production: refused
if [ -e "$HERE/.promote-test-seams" ]; then
    case "$(realpath -e -- "$HERE")/" in "$(realpath -e -- "$repo")"/*) fail "test seams file inside the repo checkout $repo: refused" ;; esac
fi
exec 9> "${state:?}/.lock"
flock -n 9 || { say "another promote_scheduler holds the lock: nothing done"; inbox_line "SKIP lock held by another promote_scheduler run"; exit 0; }
G() { git -C "$repo" "$@"; }
G fetch -q origin main nightly-evidence 2> /dev/null || fail "cannot fetch origin main/nightly-evidence: nothing measured"
[ -z "$(G status --porcelain --untracked-files=no 2> /dev/null)" ] || fail "checkout $repo is dirty: refused"
# liveness: a clean checkout that is behind origin/main fast-forwards (never reset/force); diverged stays refused below
[ "$(G rev-parse HEAD 2> /dev/null)" = "$(G rev-parse refs/remotes/origin/main 2> /dev/null)" ] || G merge -q --ff-only refs/remotes/origin/main > /dev/null 2>&1
[ "$(G rev-parse HEAD 2> /dev/null)" = "$(G rev-parse refs/remotes/origin/main 2> /dev/null)" ] || fail "checkout $repo is not at origin/main: refused"

work=$(mktemp -d) || usage "no temp dir"
cleanup() { G worktree remove --force "$work/gate" > /dev/null 2>&1; G worktree remove --force "$work/ev" > /dev/null 2>&1; G worktree prune > /dev/null 2>&1; rm -rf -- "${work:?}"; return 0; }
trap cleanup EXIT
G worktree add -q --detach "$work/gate" refs/remotes/origin/main > /dev/null 2>&1 || fail "cannot make a gate worktree of origin/main"
S="$work/gate/scripts/release"
PROMOTE="$S/promote_nightly.sh"; RECEIPTS="$S/receipts_from_night.sh"; DRYRUN_GATE="$S/rc_publish_gate.sh"
MODELS_NIGHTLY="$S/models_nightly.sh"; TRAIN="$S/nightly_train.sh"; PUSHTAGS="$work/gate/scripts/hooks/pre-push-tags.sh"
# test seams: only a file beside this script, i.e. only in the falsifier's sandbox copy
# shellcheck disable=SC1091
[ ! -f "$HERE/.promote-test-seams" ] || . "$HERE/.promote-test-seams"
if [ -z "$EVIDENCE_DIR" ]; then
    G worktree add -q --detach "$work/ev" refs/remotes/origin/nightly-evidence > /dev/null 2>&1 || fail "cannot read nightly-evidence"
    EVIDENCE_DIR="$work/ev"
fi
ev=$EVIDENCE_DIR
# GH-1: below the REST floor only the release path may call; unreadable = not_measured. Skip, never silent.
rem=$("$GH" api rate_limit --jq .resources.core.remaining 2> /dev/null)
if ! [[ $rem =~ ^[0-9]+$ ]] || [ "$rem" -lt 1000 ]; then say "GitHub REST allowance '${rem:-unreadable}' < 1000: nothing done"; inbox_line "SKIP GitHub REST remaining '${rem:-unreadable}' < 1000 (GH-1); will retry next tick"; exit 0; fi

# candidates: the newest NCAND bundles (any state), on origin/main, no older than MAX_AGE_DAYS
cutoff=$(( $(date -u +%s) - MAX_AGE_DAYS * 86400 ))
cands=()
while read -r _ s; do cands+=("$s"); done < <(
    for d in "$ev"/models-crux/*/; do
        s=$(basename "$d"); [[ $s =~ ^[0-9a-f]{40}$ ]] || continue
        G merge-base --is-ancestor "$s" refs/remotes/origin/main 2> /dev/null || continue
        ct=$(G show -s --format=%ct "$s" 2> /dev/null) || continue
        [ "$ct" -ge "$cutoff" ] || continue
        printf '%s %s\n' "$ct" "$s"
    done | sort -rn | head -n "$NCAND")
[ "${#cands[@]}" -gt 0 ] || fail "no models-crux bundle on origin/main newer than $MAX_AGE_DAYS days: SKIP"

# models_ok SHA DIR: the real verifier. green only.
models_ok() {
    local sha=$1 d=$2 go="$2/relay.out"
    : > "$go"
    (cd "$work/gate" && GITHUB_OUTPUT="$go" bash "$MODELS_NIGHTLY" --relay --commit "$sha" --out "$d/relay" --evidence "$ev/models-crux/$sha" --measure crux) > "$d/relay.log" 2>&1 || return 1
    [ "$(sed -n 's/^state=//p' "$go" | head -n 1)" = green ]
}
# lanes_ok SHA DIR: the train's own judgement (ci-main, deep-*, models, milestone, cleanroom-gpu, preflight ... incl. FLOW-003 G4
# attempt>=2 = red and newest-completed-run-only), read for exactly SHA: nightly_train.sh --commit SHA --exit-verdict from the gate
# worktree. A missing train script, a failed read or NOT RELEASABLE = fail closed (prints why).
lanes_ok() {
    local sha=$1 d=$2
    [ -f "$TRAIN" ] || { printf 'nightly_train.sh missing in the gate worktree (not_measured)'; return 1; }
    rm -rf -- "${d:?}/train"
    bash "$TRAIN" --out "$d/train" --commit "$sha" --exit-verdict > "$d/train.log" 2>&1 || { printf 'train: %s' "$(grep -m1 -E 'RELEASABLE|trap|read=' "$d/train.log" | cut -c1-120)"; return 1; }
}
# cleanroom_for SHA OUT: a successful clean-room (aprender) job whose log says tested-sha == exactly SHA
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
# the dispatched clean-room run: pending (3), red (1, never forever-pending), or fall through when green
cleanroom_marker() { # SHA DIR FIRST
    local sha=$1 d=$2 id st
    if [ -s "$d/cleanroom-run-id" ]; then
        id=$(cat "$d/cleanroom-run-id")
        st=$("$GH" run view "$id" --repo "$INFRA" --json status,conclusion --jq '"\(.status) \(.conclusion)"' 2> /dev/null) || st=""
        case $st in
            "completed success") return 0 ;;
            completed*) fail "clean-room run $id for ${sha:0:9} ended '${st#completed }': nothing promoted" ;;
            "") fail "clean-room run $id for ${sha:0:9} cannot be read: not_measured" ;;
            *) say "clean-room run $id for ${sha:0:9} still running; pending"; inbox_line "PENDING clean-room run $id on ${sha:0:9} still running"; exit 3 ;;
        esac
    fi
    [ "$3" = 1 ] && [ "$disp" = 1 ] || return 0
    t0=$(date -u +%Y-%m-%dT%H:%M:%SZ) # bashrs disable-line=DET002
    "$GH" workflow run clean-room.yml --repo "$INFRA" -f repos=aprender -f ref="$sha" > "$d/dispatch.log" 2>&1 || fail "clean-room dispatch on ${sha:0:9} failed ($(tail -n 1 "$d/dispatch.log"))"
    id=""
    for _ in 1 2 3; do
        id=$("$GH" run list --repo "$INFRA" --workflow clean-room.yml --event workflow_dispatch --limit 10 --json databaseId,createdAt --jq "[.[] | select(.createdAt >= \"$t0\")] | .[0].databaseId" 2> /dev/null)
        [[ $id =~ ^[1-9][0-9]*$ ]] && break
        sleep "$DISPATCH_WAIT"
    done
    [[ $id =~ ^[1-9][0-9]*$ ]] || fail "clean-room dispatched on ${sha:0:9} but its run id cannot be read"
    printf '%s\n' "$id" > "$d/cleanroom-run-id"
    say "clean-room run $id dispatched on ${sha:0:9}; pending"; inbox_line "PENDING clean-room run $id dispatched on ${sha:0:9}"; exit 3
}

first=1 newest=1 notes=""
for sha in "${cands[@]}"; do
    d="$state/$sha"; install -d -- "$d"
    if ! models_ok "$sha" "$d"; then
        notes+="${sha:0:9} models not green ($(tail -n 1 "$d/relay.log" | cut -c1-80)); "
        [ "$newest" = 1 ] && inbox_line "RED newest nightly crux ${sha:0:9} is not a verified green: falling back to the last verified green"
        newest=0; continue
    fi
    newest=0
    if miss=$(lanes_ok "$sha" "$d"); then :; else notes+="${sha:0:9} the train does not judge it releasable: $miss; "; continue; fi
    if ! cleanroom_for "$sha" "$d/cleanroom.json"; then
        cleanroom_marker "$sha" "$d" "$first"
        if ! cleanroom_for "$sha" "$d/cleanroom.json"; then
            if [ "$first" = 1 ]; then fail "no clean-room success tested on exactly ${sha:0:9} (pass --dispatch-cleanroom to start one)"; fi
            notes+="${sha:0:9} no clean-room on this exact sha; "; continue
        fi
    fi
    first=0
    w=$(mktemp -d) && G worktree add -q --detach "$w/wt" "$sha" > /dev/null 2>&1 || { notes+="${sha:0:9} no worktree; "; continue; }
    wsha=$(git -C "$w/wt" rev-parse HEAD)
    bash "$DRYRUN_GATE" --verify "$w/wt" > "$d/publish-dryrun.log" 2>&1; drc=$?
    G worktree remove --force "$w/wt" > /dev/null 2>&1; rm -rf -- "${w:?}"
    rcpt="$work/receipts-$sha"; rm -rf -- "${rcpt:?}"
    bash "$RECEIPTS" --sha "$sha" --out "$rcpt" --models-dir "$ev/models-crux/$sha" --cleanroom-json "$d/cleanroom.json" \
        --dryrun-rc "$drc" --dryrun-commit "$wsha" > "$d/receipts.log" 2>&1 || { notes+="${sha:0:9} receipts incomplete ($(tr '\n' ' ' < "$d/receipts.log" | cut -c1-80)); "; continue; }
    if ! bash "$PROMOTE" --sha "$sha" --receipts "$rcpt" --repo "$repo" > "$d/promote-dry.log" 2>&1; then
        notes+="${sha:0:9} promote dry-run refused: $(grep -m1 REFUSE "$d/promote-dry.log" | cut -c1-100); "; continue
    fi
    if [ "$exe" != 1 ]; then say "${sha:0:9}: dry-run: would promote (re-run with --execute)"; exit 0; fi
    (cd "$repo" && { bash "$PUSHTAGS" --verify || { bash "$PUSHTAGS" --install && bash "$PUSHTAGS" --verify; }; }) > "$d/pushtags.log" 2>&1 \
        || fail "pre-push-tags guard is not installed/verified in $repo: not executing ($d/pushtags.log)"
    bash "$PROMOTE" --sha "$sha" --receipts "$rcpt" --repo "$repo" --execute > "$d/promote-exec.log" 2>&1; rc=$?
    tail -n 6 "$d/promote-exec.log"
    [ "$rc" = 0 ] && inbox_line "DONE promoted ${sha:0:9}: $(grep -m1 TAGGED "$d/promote-exec.log" | cut -c1-120)" || inbox_line "RED promote --execute failed rc=$rc at ${sha:0:9} ($d/promote-exec.log)"
    exit "$rc"
done
fail "no candidate passed every gate: ${notes:-none}"
