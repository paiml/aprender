#!/usr/bin/env bash
# rc_cut.sh -- cut vX.Y.Z-rc.N when CI goes green on the tip of release/X.Y.Z (#4285,
# RC-DOGFOOD-001 §4.1, paiml/infra#1030), or on the tip of main (#4434).
#
# MAIN MODE (#4434, APR-RELEASE-001 rule 17: "rc.N = tag on a queue-green main sha,
# cut within 5 min of green. No RC branches and no RC CI run.")
# ---------------------------------------------------------------------------------
# A merge-queue batch lands on main as a push, and that push re-runs CI on the merged
# sha. rc-cut.yml also fires on those runs. For main the decider requires, beyond the
# release/** checks:
#   - the run's event is `push`. A merge_group run's head branch is
#     gh-readonly-queue/..., never main, and a pull_request run is not main's sha.
#   - X.Y.Z is [workspace.package].version at head_sha. A release branch names its
#     version; main does not, so the version is read, not trusted from a name.
#   - no final vX.Y.Z tag exists. After the final, main still carries X.Y.Z until the
#     next bump, and a green push must not mint an rc of a shipped version.
#   - no vX.Y.Z-rc.* exists yet, unless --next. So every green push on main does not
#     mint a new rc: the automatic cut is rc.1 only. A promotion failure is answered by
#     rc.N+1 (rule 17), which a person asks for with --next on the chosen green run.
#
# WHY THIS EXISTS
# ---------------
# Before this, every rc was cut by hand. v0.69.3-rc.1, the only -rc tag in the repo's
# history, was tagged at 10:25:19Z on 2026-09-24. Its prerelease was published by
# hand 23 minutes later (10:48:32Z), and binary-release.yml then attached 16 assets
# in 8 minutes. This script removes the 23 manual minutes. CI cuts the tag and the
# prerelease. It never publishes a crate (RP-001).
#
# WHERE "GREEN" COMES FROM
# ------------------------
# ci.yml does not run on a push to release/**. A release branch is gated by CI on
# its release PR (release/X.Y.Z -> main, e.g. #4224). Every push to the branch
# re-runs that PR's CI with head_sha = the new branch tip. rc-cut.yml fires on the
# completion of any CI run whose head branch is release/**, and this script decides:
#   - the run's head repository is THIS repository. A fork can name its branch
#     release/9.9.9, so this check is required.
#   - the branch is exactly release/X.Y.Z.
#   - both required checks, `ci / gate` and `workspace-test`, concluded success on
#     THIS run. The run's overall conclusion is not read: it also covers jobs that
#     branch protection does not require.
#   - head_sha is still the branch tip. A run for a superseded push cuts nothing,
#     because the newer push has its own run.
#   - no vX.Y.Z-rc.* tag already points at head_sha. A re-run cuts nothing.
# Then N = 1 + the highest existing rc number for X.Y.Z. The spec says "1 + the
# count". The two agree while the rc numbers are contiguous. Max+1 also stays
# collision-free if an rc tag is ever deleted, where count+1 would reuse a live name.
#
# A release created with GITHUB_TOKEN does NOT fire `release: published` in another
# workflow (GitHub suppresses recursive triggers). So the script dispatches
# binary-release.yml itself with inputs.tag, the path that workflow already takes
# for backfills. Without that dispatch the prerelease would carry zero assets.
#
#   rc_cut.sh --run-id <CI run id>      decide, then cut (needs GH_TOKEN, GITHUB_REPOSITORY)
#   rc_cut.sh --run-id <id> --dry-run   decide and print; write nothing
#   rc_cut.sh --run-id <id> --next      main only: allow rc.N+1 (a promotion failure)
#   rc_cut.sh --self-test               case table for the decision, no network
#
# EXIT CODES
#   0  cut, or a deliberate skip. The result is written to $GITHUB_OUTPUT as
#      result=cut|skip|dry-run. A red CI run cutting nothing is correct, not an error.
#   1  a write failed (the tag already exists from a race, the release could not be
#      created, or the dispatch was refused).
#   2  ENV/usage: the run or the refs could not be read. This is never a skip.
set -uo pipefail
PROG=rc_cut
REQUIRED_CHECKS=("ci / gate" "workspace-test")
PER_PAGE=100   # the jobs API's maximum page size

# rc_version_of_branch BRANCH -> prints X.Y.Z; returns 1 if BRANCH is not release/X.Y.Z.
rc_version_of_branch() {
    [[ "$1" =~ ^release/([0-9]+\.[0-9]+\.[0-9]+)$ ]] || return 1
    printf '%s\n' "${BASH_REMATCH[1]}"
}

# rc_version_of_run BRANCH EVENT WS_VERSION -> prints X.Y.Z for a candidate run.
# release/X.Y.Z names its version; main carries it in [workspace.package].version.
# Returns 1 for anything else. The push-only rule for main is rc_decide's, and run_cut
# reads WS_VERSION only for a push, so EVENT is not re-checked here.
rc_version_of_run() {
    if [ "$1" = main ]; then
        [[ "$3" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || return 1
        printf '%s\n' "$3"; return 0
    fi
    rc_version_of_branch "$1"
}

# rc_decide -- the whole decision, pure. Inputs are named variables so the case table
# can vary exactly one at a time:
#   D_REPO D_HEAD_REPO D_BRANCH D_HEAD_SHA D_TIP_SHA
#   D_EVENT D_WS_VERSION D_NEXT   main mode only: the run's event, the workspace
#           version at head_sha, and 1 when a person asked for rc.N+1 (--next)
#   D_FINAL "1" when the final tag v<X.Y.Z> exists (main mode refuses it)
#   D_JOBS  lines "<job name>\t<conclusion>"
#   D_TAGS  lines "<tag name>\t<commit sha>" for existing tags matching v<X.Y.Z>-rc.*
# Prints "cut <tag>" or "skip <reason>". Returns 0 for both.
rc_decide() {
    local v name c n max=0 have
    if [ "${D_HEAD_REPO:-}" != "${D_REPO:-}" ]; then
        printf 'skip head repository %s is not %s\n' "${D_HEAD_REPO:-?}" "${D_REPO:-?}"; return 0
    fi
    if [ "${D_BRANCH:-}" = main ]; then
        if [ "${D_EVENT:-}" != push ]; then printf 'skip a %s run on main is not a push of a queue-merged sha\n' "${D_EVENT:-?}"; return 0; fi
        if ! v=$(rc_version_of_run main push "${D_WS_VERSION:-}"); then
            printf 'skip workspace version %s on main is not X.Y.Z\n' "${D_WS_VERSION:-?}"; return 0
        fi
    elif ! v=$(rc_version_of_branch "${D_BRANCH:-}"); then
        printf 'skip branch %s is not release/X.Y.Z or main\n' "${D_BRANCH:-?}"; return 0
    fi
    for name in "${REQUIRED_CHECKS[@]}"; do
        # Every job with this name must be success, and at least one must exist:
        # a missing required check is not a green one.
        have=$(printf '%s\n' "${D_JOBS:-}" | awk -F'\t' -v n="$name" '$1==n {print $2}')
        if [ -z "$have" ]; then printf 'skip required check "%s" is absent from the run\n' "$name"; return 0; fi
        if [ -n "$(printf '%s\n' "$have" | grep -v '^success$')" ]; then
            printf 'skip required check "%s" concluded %s\n' "$name" "$(printf '%s' "$have" | paste -sd, -)"; return 0
        fi
    done
    if [ -z "${D_HEAD_SHA:-}" ] || [ "${D_HEAD_SHA:-}" != "${D_TIP_SHA:-}" ]; then
        printf 'skip %s is no longer the tip of %s (tip %s)\n' "${D_HEAD_SHA:-?}" "$D_BRANCH" "${D_TIP_SHA:-?}"; return 0
    fi
    while IFS=$'\t' read -r name c; do
        [ -n "$name" ] || continue
        [[ "$name" =~ ^v${v//./\\.}-rc\.([0-9]+)$ ]] || continue
        n=$((10#${BASH_REMATCH[1]}))
        if [ "$c" = "$D_HEAD_SHA" ]; then printf 'skip %s already points at %s\n' "$name" "$D_HEAD_SHA"; return 0; fi
        [ "$n" -gt "$max" ] && max=$n
    done <<< "${D_TAGS:-}"
    if [ "$D_BRANCH" = main ]; then
        if [ "${D_FINAL:-}" = 1 ]; then printf 'skip v%s is already final; main cuts no rc of a shipped version\n' "$v"; return 0; fi
        if [ "$max" -gt 0 ] && [ "${D_NEXT:-}" != 1 ]; then
            printf 'skip v%s-rc.%d exists; rc.%d on main is cut only with --next (a promotion failure)\n' "$v" "$max" "$((max + 1))"; return 0
        fi
    fi
    printf 'cut v%s-rc.%d\n' "$v" "$((max + 1))"
}

self_test() {
    local fail=0 got mut loop_head
    local J_OK=$'ci / gate\tsuccess\nworkspace-test\tsuccess\nlint\tfailure'
    local A=aaaaaaaa B=bbbbbbbb
    expect() {  # expect <want> <label> -- rc_decide under the current D_* vs the full line
        got=$(rc_decide)
        if [ "$got" = "$1" ]; then printf '  ok   %s\n' "$2"; else printf '  FAIL %s\n       want: %s\n       got:  %s\n' "$2" "$1" "$got"; fail=1; fi
    }
    base() { D_REPO=paiml/aprender D_HEAD_REPO=paiml/aprender D_BRANCH=release/0.70.0 D_HEAD_SHA=$A D_TIP_SHA=$A D_JOBS=$J_OK D_TAGS=''; }
    echo "$PROG self-test: rc_decide case table"
    base; expect 'cut v0.70.0-rc.1' 'first green run on a fresh release branch -> rc.1 (a red non-required job does not block)'
    base; D_TAGS=$'v0.70.0-rc.1\t'"$B"; expect 'cut v0.70.0-rc.2' 'rc.1 exists on an older commit -> rc.2'
    base; D_TAGS=$'v0.70.0-rc.1\tc1\nv0.70.0-rc.3\tc3'; expect 'cut v0.70.0-rc.4' 'gap in numbering -> max+1, never reuse a name'
    base; D_TAGS=$'v0.70.0-rc.1\tc1\nv0.70.0-rc.10\tc10\nv0.70.0-rc.9\tc9'; expect 'cut v0.70.0-rc.11' 'numeric max over the API'"'"'s LEXICAL order (rc.10 sorts before rc.9)'
    base; D_TAGS=$'v0.69.3-rc.7\tc7\nv0.70.0\tcf\nv0.70.00-rc.5\tcx\nv0x70x0-rc.6\tcy'; expect 'cut v0.70.0-rc.1' 'other versions, the final tag and near-miss names (dots are literal) do not count'
    base; D_TAGS=$'v0.70.0-rc.2\t'$A; expect "skip v0.70.0-rc.2 already points at $A" 're-run on an already-cut commit cuts nothing'
    base; D_BRANCH=release/0.69.1-batch-2; expect 'skip branch release/0.69.1-batch-2 is not release/X.Y.Z or main' 'suffixed release branch refused'
    base; D_BRANCH=release/0.70; expect 'skip branch release/0.70 is not release/X.Y.Z or main' 'two-part version refused'
    base; D_BRANCH=gh-readonly-queue/main/pr-1-abc; expect 'skip branch gh-readonly-queue/main/pr-1-abc is not release/X.Y.Z or main' 'a merge_group ref is not main: the cut waits for the push'
    # main mode (#4434): rc = tag on a queue-green main sha
    main_base() { base; D_BRANCH=main D_EVENT=push D_WS_VERSION=0.70.0 D_FINAL='' D_NEXT=''; }
    main_base; expect 'cut v0.70.0-rc.1' 'main: first green push at a new version -> rc.1'
    main_base; D_EVENT=pull_request; expect 'skip a pull_request run on main is not a push of a queue-merged sha' 'main: a pull_request run cuts nothing'
    main_base; D_EVENT=merge_group; expect 'skip a merge_group run on main is not a push of a queue-merged sha' 'main: a merge_group run cuts nothing'
    main_base; D_EVENT=''; expect 'skip a ? run on main is not a push of a queue-merged sha' 'main: an unread event cuts nothing'
    main_base; D_WS_VERSION=''; expect 'skip workspace version ? on main is not X.Y.Z' 'main: an unread version cuts nothing'
    main_base; D_WS_VERSION=0.70.0-dev; expect 'skip workspace version 0.70.0-dev on main is not X.Y.Z' 'main: a pre-release workspace version cuts nothing'
    main_base; D_FINAL=1; expect 'skip v0.70.0 is already final; main cuts no rc of a shipped version' 'main: the final exists -> no rc of a shipped version'
    main_base; D_TAGS=$'v0.70.0-rc.1\t'"$B"; expect 'skip v0.70.0-rc.1 exists; rc.2 on main is cut only with --next (a promotion failure)' 'main: every green push does not mint an rc'
    main_base; D_TAGS=$'v0.70.0-rc.1\t'"$B"; D_NEXT=1; expect 'cut v0.70.0-rc.2' 'main: --next after a promotion failure -> rc.N+1'
    main_base; D_TAGS=$'v0.70.0-rc.1\t'"$A"; D_NEXT=1; expect "skip v0.70.0-rc.1 already points at $A" 'main: --next on an already-cut sha cuts nothing'
    main_base; D_TIP_SHA="$B"; expect "skip $A is no longer the tip of main (tip $B)" 'main: a superseded push cuts nothing'
    main_base; D_JOBS=$'ci / gate\tsuccess\nworkspace-test\tfailure'; expect 'skip required check "workspace-test" concluded failure' 'main: red workspace-test cuts nothing'
    base; D_NEXT=1; D_TAGS=$'v0.70.0-rc.1\t'"$B"; expect 'cut v0.70.0-rc.2' 'release/**: rc.N+1 needs no --next (unchanged)'
    base; D_HEAD_REPO=someone/aprender; expect 'skip head repository someone/aprender is not paiml/aprender' 'fork PR with a release/X.Y.Z head refused'
    base; D_JOBS=$'ci / gate\tfailure\nworkspace-test\tsuccess'; expect 'skip required check "ci / gate" concluded failure' 'red ci / gate cuts nothing'
    base; D_JOBS=$'ci / gate\tsuccess\nworkspace-test\tcancelled'; expect 'skip required check "workspace-test" concluded cancelled' 'cancelled workspace-test cuts nothing'
    base; D_JOBS=$'ci / gate\tsuccess'; expect 'skip required check "workspace-test" is absent from the run' 'a missing required check is not green'
    base; D_JOBS=$'ci / gate\tsuccess\nci / gate\tfailure\nworkspace-test\tsuccess'; expect 'skip required check "ci / gate" concluded success,failure' 'every job of the name must pass'
    base; D_JOBS=$'ci / gate\tskipped\nworkspace-test\tsuccess'; expect 'skip required check "ci / gate" concluded skipped' 'skipped is not success'
    base; D_TIP_SHA=$B; expect "skip $A is no longer the tip of release/0.70.0 (tip $B)" 'superseded push cuts nothing'
    base; D_HEAD_SHA=''; D_TIP_SHA=''; expect 'skip ? is no longer the tip of release/0.70.0 (tip ?)' 'empty head sha never cuts'
    # MUTANT: a decider with the required-check loop deleted must turn the red-gate row
    # the other way. Build it from THIS file, so the mutant tracks the shipped code.
    mut=$(mktemp) || return 2
    sed '/for name in "\${REQUIRED_CHECKS\[@\]}"; do/,/^    done$/d' "${BASH_SOURCE[0]}" > "$mut"
    # the mutant must hold FEWER uses of REQUIRED_CHECKS than this file (the loop is gone)
    loop_head='REQUIRED_CHECKS[@]'
    if [ "$(grep -cF "$loop_head" "$mut")" -ge "$(grep -cF "$loop_head" "${BASH_SOURCE[0]}")" ]; then echo "  FAIL mutant was not built (the check loop is still present)"; fail=1
    else
        got=$(D_REPO=paiml/aprender D_HEAD_REPO=paiml/aprender D_BRANCH=release/0.70.0 D_HEAD_SHA=$A D_TIP_SHA=$A \
              D_JOBS=$'ci / gate\tfailure\nworkspace-test\tsuccess' D_TAGS='' bash -c ". '$mut' --source-only; rc_decide")
        if [ "$got" = 'cut v0.70.0-rc.1' ]; then echo "  ok   mutant (check loop deleted) cuts on a red gate: the red-gate row can see that defect"
        else printf '  FAIL mutant did not behave as a mutant: %s\n' "$got"; fail=1; fi
    fi
    rm -f -- "$mut"
    if [ "$fail" = 0 ]; then echo "$PROG self-test: PASS"; else echo "$PROG self-test: FAIL"; fi
    return "$fail"
}

# ---- I/O: GitHub REST via curl + python3 (the fleet boxes carry no gh) ------------
api_get() {  # api_get PATH -> body on stdout; returns 2 on any failure
    curl -sSf -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json" \
        "https://api.github.com/repos/$GITHUB_REPOSITORY${1:+/$1}" || return 2
}
api_post() {  # api_post PATH JSON -> line 1 is the HTTP code, the rest is the body
    local out code
    out=$(mktemp) || return 2
    code=$(curl -sS -o "$out" -w '%{http_code}' -X POST -H "Authorization: Bearer $GH_TOKEN" \
        -H "Accept: application/vnd.github+json" "https://api.github.com/repos/$GITHUB_REPOSITORY/$1" -d "$2") || { rm -f -- "$out"; return 2; }
    printf '%s\n' "$code"; cat "$out"; rm -f -- "$out"
}
json() { python3 -c "import json,sys; d=json.load(sys.stdin); $1"; }
emit() { if [ -n "${GITHUB_OUTPUT:-}" ]; then printf '%s=%s\n' "$1" "$2" >> "$GITHUB_OUTPUT"; fi; }
summary() { if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then printf '%s\n' "$@" >> "$GITHUB_STEP_SUMMARY"; fi; printf '%s\n' "$@"; }
post_ok() {  # post_ok <want code> <what> <api_post output>
    local code; code=$(printf '%s\n' "$3" | head -1)
    [ "$code" = "$1" ] && return 0
    printf '%s: %s returned HTTP %s: %s\n' "$PROG" "$2" "$code" "$(printf '%s\n' "$3" | tail -n +2 | head -c 400)" >&2
    return 1
}

# ws_version_at SHA -> [workspace.package].version of Cargo.toml at SHA, by section
# (the fleet boxes run python 3.10, which has no tomllib).
ws_version_at() {
    local toml
    toml=$(api_get "contents/Cargo.toml?ref=$1" | json 'import base64; sys.stdout.write(base64.b64decode(d["content"]).decode())') || return 2
    printf '%s\n' "$toml" | awk -F'"' '/^\[/ { s = $0 } s == "[workspace.package]" && /^version *=/ && !f { print $2; f = 1 }'
}

run_cut() {
    local run_id="$1" dry="$2" next="${3:-0}" run v page body n decision tag out merged_at notes ref rc
    : "${GH_TOKEN:?GH_TOKEN is required}" "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
    run=$(api_get "actions/runs/$run_id") || { echo "$PROG: cannot read run $run_id" >&2; return 2; }
    D_REPO=$GITHUB_REPOSITORY
    D_HEAD_REPO=$(printf '%s' "$run" | json 'print((d.get("head_repository") or {}).get("full_name",""))') || return 2
    D_BRANCH=$(printf '%s' "$run" | json 'print(d.get("head_branch") or "")') || return 2
    D_HEAD_SHA=$(printf '%s' "$run" | json 'print(d.get("head_sha") or "")') || return 2
    D_EVENT=$(printf '%s' "$run" | json 'print(d.get("event") or "")') || return 2
    D_JOBS='' D_TAGS='' D_TIP_SHA='' D_WS_VERSION='' D_FINAL='' D_NEXT="$next"
    if [ "$D_BRANCH" = main ] && [ "$D_EVENT" = push ] && [ "$D_HEAD_REPO" = "$D_REPO" ]; then
        D_WS_VERSION=$(ws_version_at "$D_HEAD_SHA") || { echo "$PROG: cannot read [workspace.package].version at $D_HEAD_SHA" >&2; return 2; }
    fi
    # Read the jobs, tip and tags only for a candidate; rc_decide still judges every field.
    if v=$(rc_version_of_run "$D_BRANCH" "$D_EVENT" "$D_WS_VERSION") && [ "$D_HEAD_REPO" = "$D_REPO" ]; then
        page=1
        while :; do  # CI can exceed one page of jobs once shards and matrices are counted: paginate
            body=$(api_get "actions/runs/$run_id/jobs?filter=latest&per_page=$PER_PAGE&page=$page") || { echo "$PROG: cannot read the jobs of run $run_id" >&2; return 2; }
            D_JOBS+=$(printf '%s' "$body" | json 'print("\n".join(j["name"]+"\t"+str(j.get("conclusion")) for j in d["jobs"]))')$'\n' || return 2
            n=$(printf '%s' "$body" | json 'print(len(d["jobs"]))') || return 2
            [ "$n" -lt "$PER_PAGE" ] && break
            page=$((page + 1))
        done
        body=$(api_get "git/ref/heads/$D_BRANCH") || { echo "$PROG: cannot read the tip of $D_BRANCH" >&2; return 2; }
        D_TIP_SHA=$(printf '%s' "$body" | json 'print(d["object"]["sha"])') || return 2
        # matching-refs returns [] when nothing matches; an annotated tag is dereferenced to its commit.
        body=$(api_get "git/matching-refs/tags/v$v-rc.") || { echo "$PROG: cannot list the v$v-rc.* tags" >&2; return 2; }
        D_TAGS=$(printf '%s' "$body" | python3 -c '
import json, os, sys, urllib.request
def get(url):
    r = urllib.request.Request(url, headers={"Authorization": "Bearer " + os.environ["GH_TOKEN"], "Accept": "application/vnd.github+json"})
    return json.load(urllib.request.urlopen(r))
for ref in json.load(sys.stdin):
    o = ref["object"]
    sha = o["sha"] if o["type"] == "commit" else get(o["url"])["object"]["sha"]
    print(ref["ref"][len("refs/tags/"):] + "\t" + sha)
') || { echo "$PROG: cannot resolve the v$v-rc.* tags" >&2; return 2; }
        # The final tag: matching-refs is a PREFIX match, so v0.70.0 also lists v0.70.0-rc.*
        # and v0.70.01; only the exact ref counts.
        body=$(api_get "git/matching-refs/tags/v$v") || { echo "$PROG: cannot list the v$v tags" >&2; return 2; }
        D_FINAL=$(printf '%s' "$body" | V="v$v" json 'import os; print(1 if any(r["ref"] == "refs/tags/" + os.environ["V"] for r in d) else "")') || return 2
    fi
    decision=$(rc_decide)
    summary "### rc-cut: CI run $run_id on ${D_BRANCH:-?} @ ${D_HEAD_SHA:0:9}" "" "$decision"
    case "$decision" in
        skip*) emit result skip; return 0 ;;
        "cut "*) tag=${decision#cut } ;;
        *) echo "$PROG: rc_decide printed '$decision'" >&2; return 2 ;;
    esac
    emit tag "$tag"
    emit sha "$D_HEAD_SHA"   # rc-cut.yml gates exactly this commit before the real cut (#4287)

    # 0. The fleet (#4328 C4): no rc is cut while a fleet cell is RED, unless a dated waiver
    #    in scripts/release/fleet-waivers.tsv names it. The cells are infra-64's andon output
    #    (C3) on the fleet-state branch; missing or stale cells refuse too.
    local cells gate
    cells=$(mktemp) || return 2
    api_get "contents/fleet/cells.tsv?ref=fleet-state" 2>/dev/null \
        | json 'import base64; sys.stdout.write(base64.b64decode(d["content"]).decode())' > "$cells" 2>/dev/null || : > "$cells"
    gate=$(bash "$(dirname -- "${BASH_SOURCE[0]}")/fleet_cells_gate.sh" --cells "$cells" \
        --waivers "$(dirname -- "${BASH_SOURCE[0]}")/fleet-waivers.tsv"); rc=$?
    rm -f -- "$cells"
    summary "fleet cells: $gate"
    [ "$rc" = 0 ] || { echo "$PROG: REFUSED to cut $tag -- $gate (#4328 C4)" >&2; emit result fleet-red; return 1; }
    if [ "$dry" = 1 ]; then summary "(dry run: no tag, release or dispatch written)"; emit result dry-run; return 0; fi

    # 1. The tag. Creating the ref first makes the name the lock: a concurrent cut of
    #    the same N gets 422 here and writes nothing else.
    out=$(api_post git/refs "{\"ref\":\"refs/tags/$tag\",\"sha\":\"$D_HEAD_SHA\"}") || return 1
    post_ok 201 "creating tag $tag" "$out" || return 1

    # 2. The prerelease, never latest, created as a DRAFT (#4327). Operator, 2026-09-24:
    #    "THE INSTANT we do a release candidate it needs to be on all hardware either before
    #    or same time". A draft is invisible to install.sh and the fleet poller; it becomes
    #    visible only when rc_fleet_stage.sh has installed and verified it on every
    #    reachable fleet host. The notes carry the provenance the fleet pins by (commit and
    #    gating run) and the merge time the <=45 min target is measured from.
    merged_at=$(api_get "commits/$D_HEAD_SHA" | json 'print(d["commit"]["committer"]["date"])') || merged_at=unknown
    cut_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)  # bashrs disable-line=DET002
    notes="Release candidate of ${v}, cut by CI (#4285). Not on crates.io.

Commit \`$D_HEAD_SHA\` on \`$D_BRANCH\`, merged $merged_at.
Gated by CI run https://github.com/$GITHUB_REPOSITORY/actions/runs/$run_id (\`ci / gate\` and \`workspace-test\` green).
Cut at $cut_at. binary-release.yml attaches the apr and pv assets to this DRAFT;
scripts/release/rc_fleet_stage.sh publishes it only after every reachable fleet host runs it (#4327).

Install: \`install.sh --version $tag\`, or \`install.sh --channel rc\` for the newest rc."
    body=$(TAG=$tag NOTES=$notes python3 -c 'import json,os; print(json.dumps({"tag_name":os.environ["TAG"],"name":os.environ["TAG"],"body":os.environ["NOTES"],"prerelease":True,"draft":True,"make_latest":"false"}))') || return 1
    out=$(api_post releases "$body") || return 1
    post_ok 201 "creating prerelease $tag (the tag exists; create the release and dispatch binary-release.yml by hand)" "$out" || return 1

    # 3. The assets. A GITHUB_TOKEN-created release does not fire `release: published`.
    ref=$(api_get "" | json 'print(d["default_branch"])') || ref=main
    out=$(api_post actions/workflows/binary-release.yml/dispatches "{\"ref\":\"$ref\",\"inputs\":{\"tag\":\"$tag\"}}") || return 1
    post_ok 204 "dispatching binary-release.yml for $tag" "$out" || return 1
    summary "Cut $tag at ${D_HEAD_SHA:0:9} as a DRAFT prerelease and dispatched binary-release.yml on $ref. Next: rc_fleet_stage.sh $tag --publish (#4327)."
    emit result cut
}

main() {
    local run_id='' dry=0 next=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --self-test) self_test; return $? ;;
            --source-only) return 0 ;;
            --run-id) run_id=${2:-}; shift 2 || return 2 ;;
            --dry-run) dry=1; shift ;;
            --next) next=1; shift ;;   # main only: allow rc.N+1 after a promotion failure
            -h|--help) sed -n '2,63p' "${BASH_SOURCE[0]}"; return 0 ;;
            *) echo "$PROG: unknown argument $1" >&2; return 2 ;;
        esac
    done
    [[ "$run_id" =~ ^[0-9]+$ ]] || { echo "$PROG: --run-id <numeric CI run id> is required" >&2; return 2; }
    run_cut "$run_id" "$dry" "$next"
}

# Sourced with --source-only (the self-test's mutant): define the functions, run nothing.
if [ "${1:-}" = --source-only ]; then return 0 2>/dev/null || exit 0; fi
main "$@"
