#!/usr/bin/env bash
# promote_nightly.sh -- receipt-gated promote: verify a green nightly SHA, and with --execute tag it (N26, FLOW-008, PMAT-5306)
#
#   bash scripts/release/promote_nightly.sh --sha <40-hex> --receipts DIR [--tag vX.Y.Z] [--repo DIR] [--execute]
#
# DEFAULT IS A DRY RUN: every check runs, the decision and the exact tag commands are printed, nothing
# is tagged, armed or pushed. With --execute it does exactly what autopilot.sh cut_tag() does today
# (git tag -a, pre-push-tags.sh --arm-release 120, git push origin <tag>) and no more. It has no tagging
# logic of its own: the tag guard stays the one that already exists.
#
# THE CHECKS (each one marked `# CHK:<name>`; falsify_promote_nightly.sh --mutants drops each in turn
# and requires a case-table row to go red):
#   head         --repo's HEAD is exactly --sha and its tracked tree is clean, so every gate script
#                below is the SHA's own, as in autopilot's release worktree
#   onmain       --sha is an ancestor of origin/main (fetched now)
#   version      [workspace.package] version in Cargo.toml at --sha is a final X.Y.Z; the tag is v<that>;
#                --tag, if given, must equal it
#   tag_local / tag_origin
#                that tag exists neither locally nor on origin (never re-tag); an unreachable origin refuses
#   policy       the standing release policy covers that version (scripts/lib/release_policy.sh, the
#                same judgement as autopilot's ap_policy_applies). Not covered or unjudgeable: refuse.
#                The readiness path (policy 0) is not supported here.
#   cert         evidence/crux/<ver>/prompt-certification.json exists at --sha and parses
#   cleanroom, models, dryrun
#                DIR/<lane>.json is {"lane":"<lane>","sha":"<--sha>","status":"green"} for the lanes
#                clean-room, models (CRUX smoke GO), publish-dry-run. Missing, unreadable, red,
#                not_measured, another lane or another sha: refuse. Absent is never green.
#   coverage     scripts/release/tag_coverage_gate.sh --resolve <sha> (the existing coverage path) exits 0
#
# NEVER here: cargo build/test/publish, gh workflow/release, tag delete, force push. The crates.io
# publish stays the manual `make publish` of RP-001; the Binary Release dispatch stays a printed step.
#
# exit: 0 all checks passed (dry-run: would promote; --execute: tagged and pushed) · 1 refused · 2 usage
set -uo pipefail

PROG=promote_nightly
die() { printf '%s: REFUSE -- %s\n' "$PROG" "$*" >&2; exit 1; }
usage() { printf '%s: %s\nusage: promote_nightly.sh --sha <40-hex> --receipts DIR [--tag vX.Y.Z] [--repo DIR] [--execute]\n' "$PROG" "$*" >&2; exit 2; }

sha="" rdir="" want_tag="" repo="." execute=0
while [ $# -gt 0 ]; do
    case "$1" in
        --sha) [ $# -ge 2 ] || usage "--sha needs a value"; sha=$2; shift 2 ;;
        --receipts) [ $# -ge 2 ] || usage "--receipts needs a value"; rdir=$2; shift 2 ;;
        --tag) [ $# -ge 2 ] || usage "--tag needs a value"; want_tag=$2; shift 2 ;;
        --repo) [ $# -ge 2 ] || usage "--repo needs a value"; repo=$2; shift 2 ;;
        --execute) execute=1; shift ;;
        *) usage "unknown argument: $1" ;;
    esac
done
[ -n "$sha" ] || usage "--sha is required"
[ -n "$rdir" ] || usage "--receipts is required"
[[ $sha =~ ^[0-9a-f]{40}$ ]] || usage "--sha must be a full 40-character lowercase hex sha"
command -v jq > /dev/null 2>&1 || die "jq is not installed: no receipt can be read"
[ -d "$repo" ] || die "--repo $repo is not a directory"
G() { git -C "$repo" "$@"; }

# head: the checkout is the SHA
check_head() {
    local h
    h=$(G rev-parse HEAD 2> /dev/null) || die "$repo is not a git checkout"
    [ "$h" = "$sha" ] || die "the checkout HEAD ($h) is not --sha ($sha): check out the candidate first"
    G diff --quiet HEAD -- 2> /dev/null || die "the checkout is not clean (tracked files differ from HEAD)"
}
check_head # CHK:head

# onmain: --sha is on origin/main
check_onmain() {
    G fetch -q origin main 2> /dev/null || die "cannot fetch origin main: nothing is measured"
    G merge-base --is-ancestor "$sha" refs/remotes/origin/main 2> /dev/null || die "$sha is not on origin/main"
}
check_onmain # CHK:onmain

# version: the workspace version at the SHA, and the tag it names
ver=$(G show "$sha:Cargo.toml" 2> /dev/null | awk '
    /^\[/ { s = ($0 ~ /^\[workspace\.package\][[:space:]]*$/) }
    s && /^version[[:space:]]*=/ { v = $0; sub(/^[^"]*"/, "", v); sub(/".*$/, "", v); print v; exit }')
[ -n "$ver" ] || die "no [workspace.package] version in Cargo.toml at $sha"
tag="v$ver"
check_version() {
    [[ $ver =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "workspace version '$ver' is not a final X.Y.Z (pre-releases are not promoted here)"
    [ -z "$want_tag" ] || [ "$want_tag" = "$tag" ] || die "--tag $want_tag does not match the workspace version tag $tag"
}
check_version # CHK:version

# never re-tag: absent locally and on origin
check_tag_local() {
    G rev-parse -q --verify "refs/tags/$tag" > /dev/null 2>&1 && die "tag $tag already exists locally"
    return 0
}
check_tag_local # CHK:tag_local
check_tag_origin() {
    local out
    out=$(G ls-remote origin "refs/tags/$tag" "refs/tags/$tag^{}" 2> /dev/null) || die "cannot read tags on origin: nothing is measured"
    [ -z "$out" ] || die "tag $tag already exists on origin"
}
check_tag_origin # CHK:tag_origin

# policy: the standing release policy covers this version (autopilot ap_policy_applies, from this commit's own library)
check_policy() {
    local pol
    pol=$(cd "$repo" && . scripts/lib/release_policy.sh || exit 2
        t=$(mktemp) || exit 2
        release_policy_ladder contracts/model-capability-ladder-v1.yaml "$ver" > "$t"; r=$?
        l=$(cat "$t"); rm -f "${t:?}"
        [ -z "$l" ] || [ "$l" = contracts/model-capability-ladder-v1.yaml ] || rm -f "${l:?}"
        [ "$r" = 0 ] || { printf '%s\n' "$RP_WHY" >&2; exit "$r"; }
        printf '%s\n' "$RP_APPLIES") || die "the standing release policy cannot be judged for $ver: nothing is measured"
    [ "$pol" = 1 ] || die "the standing release policy does not cover $ver (policy=$pol): the readiness path is not supported by this adapter"
}
check_policy # CHK:policy

# cert: the CRUX prompt certification is in the commit
check_cert() {
    local p="evidence/crux/$ver/prompt-certification.json"
    G cat-file -e "$sha:$p" 2> /dev/null || die "no $p at $sha"
    G show "$sha:$p" 2> /dev/null | jq -e . > /dev/null 2>&1 || die "$p at $sha is not valid JSON"
}
check_cert # CHK:cert

# receipts: {"lane","sha","status":"green"} for exactly this sha
[ -d "$rdir" ] || die "receipts directory '$rdir' does not exist"
receipt() {
    local lane=$1 f="$rdir/$1.json"
    [ -s "$f" ] || die "$lane receipt missing: $f"
    jq -e --arg l "$lane" --arg s "$sha" '.lane == $l and .sha == $s and .status == "green"' "$f" > /dev/null 2>&1 \
        || die "$lane receipt $f is unreadable, not for lane $lane, not bound to $sha, or not green (status: $(jq -r '.status // "absent"' "$f" 2> /dev/null || echo unreadable))"
}
receipt clean-room # CHK:cleanroom
receipt models # CHK:models
receipt publish-dry-run # CHK:dryrun

# coverage: the existing path (coverage-nightly's receipt, judged by tag_coverage_gate.sh) before the tag
check_coverage() {
    local cov="$repo/scripts/release/tag_coverage_gate.sh" o
    [ -f "$cov" ] || die "coverage: $cov is missing at $sha"
    o=$(cd "$repo" && bash scripts/release/tag_coverage_gate.sh --resolve "$sha" 2>&1) || die "coverage gate refused: $(printf '%s' "$o" | tail -n 2 | tr '\n' ' ')"
}
check_coverage # CHK:coverage

arm="bash scripts/hooks/pre-push-tags.sh --arm-release $tag 120"
printf '%s: DECISION: PROMOTE %s at %s (%s)\n' "$PROG" "$tag" "$sha" "$([ "$execute" = 1 ] && echo execute || echo dry-run)"
printf '  git tag -a %s -m "aprender %s" %s\n  %s\n  git push origin %s\n' "$tag" "$tag" "$sha" "$arm" "$tag"

if [ "$execute" != 1 ]; then
    printf '%s: dry-run: nothing tagged, armed or pushed. Re-run with --execute to perform the above.\n' "$PROG"
    exit 0
fi

# exactly cut_tag's tail: tag, arm the pre-push tag guard for 120 s, push
G tag -a "$tag" -m "aprender $tag" "$sha" || die "git tag failed"
(cd "$repo" && $arm) || die "arming $tag for the pre-push tag guard failed -- tag NOT pushed (the local tag $tag stays; a human decides)"
G push origin "$tag" || die "tag push failed ($tag exists locally, not on origin)"

printf '%s: TAGGED and pushed %s at %s. Next, by hand:\n' "$PROG" "$tag" "$sha"
printf '  1. gh release create %s --verify-tag --draft ...   (then Binary Release)\n' "$tag"
printf '  2. gh workflow run binary-release.yml --ref %s -f tag=%s   (Binary Release dispatch)\n' "$tag" "$tag"
printf '  3. crates.io publish stays manual: `make publish` per RP-001 (ask before the cascade)\n'
exit 0
