#!/usr/bin/env bash
# ci_train_cutover.sh -- the cut-over step of #7001 (heavy tier once per train), as a tool with a case table.
#
# Why it exists (non-author review of #7006, blocker 2, "stale greens"): branch protection reads the check runs on a PR's
# HEAD SHA. An open PR whose `ci / gate`, `gate` and `workspace-test` went green under the OLD ci.yml keeps those greens on
# that sha after the train-only ci.yml lands, so it stays mergeable alone until its head moves. A re-run does not help: it
# replays the old run's merge commit and therefore the old ci.yml. The only invalidation is a NEW head commit that
# descends from the cut-over commit: every run on such a head evaluates a merge ref that contains the train-only ci.yml.
#
# What it does, per open PR (drafts included: `gh pr ready` would expose a draft's old greens the same way):
#   (a) DESCENDS: GET compare/<cut-over>...<head>; status `ahead` or `identical` means the head contains the cut-over.
#       Anything else (behind, diverged, an unreadable compare) is STALE -- a failed read is not measured, never a pass.
#   (b) NO GREEN OFF A TRAIN REF: on a head that descends, a non-`train/aprender-*` PR must not carry a `success` on any of
#       the three required names (the train-only ci.yml refuses them; a green there means the PR edited ci.yml, which is
#       the rollback path of §7 and needs a human eye, so it is reported, not judged).
# With --apply, every STALE PR gets `gh pr update-branch` (a merge of main into the branch; never a force-push). The heads
# then move, so --apply always exits 1: run it again without --apply after the new runs exist, and that exit 0 is the
# cut-over receipt. Exit 0 = every open PR passes (a) and (b); 1 = at least one does not; 2 = usage / not measured.
#
# GH-1/GH-2: one GraphQL list read, one REST compare (+1 check-runs read) per open PR, one REST write per stale PR, after a
# rate-limit check (below 1000 REST calls left it refuses: not measured).
set -euo pipefail

REQUIRED_NAMES=("ci / gate" "gate" "workspace-test")
GH="${GH:-gh}"

usage() {
    printf 'usage: %s --cutover SHA [--repo OWNER/NAME] [--apply] | --self-test\n' "$(basename "$0")" >&2
    exit 2
}

descends() { # descends REPO CUTOVER HEAD -> prints the compare status; 0 when HEAD contains CUTOVER
    local st
    st="$("$GH" api "repos/$1/compare/$2...$3" --jq '.status' 2>/dev/null)" || { printf 'unread'; return 1; }
    printf '%s' "${st:-unread}"
    case "$st" in ahead|identical) return 0 ;; *) return 1 ;; esac
}

greens_on() { # greens_on REPO SHA -> the required names that carry a success check run on SHA, one per line
    local names
    names="$("$GH" api "repos/$1/commits/$2/check-runs?per_page=100" --paginate \
        --jq '.check_runs[] | select(.conclusion == "success") | .name' 2>/dev/null)" || { printf 'UNREAD\n'; return 0; }
    local n r
    for r in "${REQUIRED_NAMES[@]}"; do
        while IFS= read -r n; do [ "$n" = "$r" ] && { printf '%s\n' "$r"; break; }; done <<<"$names"
    done
    return 0
}

cutover() { # cutover REPO CUTOVER APPLY -> 0 all clean, 1 something stale or green, 2 not measured
    local repo="$1" cut="$2" apply="$3" left prs n head ref st g bad=0 total=0
    left="$("$GH" api rate_limit --jq '.resources.core.remaining' 2>/dev/null)" || { printf 'NOT MEASURED: rate_limit unreadable\n'; return 2; }
    if ! [[ "$left" =~ ^[0-9]+$ ]] || [ "$left" -lt 1000 ]; then
        printf 'NOT MEASURED: %s REST calls left (< 1000, fleet rule GH-1); wait for the reset\n' "$left"; return 2
    fi
    prs="$("$GH" pr list -R "$repo" --state open --limit 500 --json number,headRefOid,headRefName \
        --jq '.[] | "\(.number) \(.headRefOid) \(.headRefName)"' 2>/dev/null)" || { printf 'NOT MEASURED: open PR list unreadable\n'; return 2; }
    while read -r n head ref; do
        [ -n "${n:-}" ] || continue
        total=$((total+1))
        if st="$(descends "$repo" "$cut" "$head")"; then
            g="$(greens_on "$repo" "$head")"
            case "$ref" in
                train/aprender-*) printf 'ok    #%s %s descends (%s); train ref\n' "$n" "${head:0:9}" "$st" ;;
                *)
                    if [ -n "$g" ]; then
                        printf 'GREEN #%s %s descends but carries success on: %s (does it edit ci.yml? §7 rollback path)\n' \
                            "$n" "${head:0:9}" "$(printf '%s' "$g" | paste -sd, -)"; bad=1
                    else
                        printf 'ok    #%s %s descends (%s); no required green\n' "$n" "${head:0:9}" "$st"
                    fi ;;
            esac
            continue
        fi
        bad=1
        printf 'STALE #%s %s does not contain %s (compare: %s)\n' "$n" "${head:0:9}" "${cut:0:9}" "$st"
        if [ "$apply" = 1 ]; then
            if "$GH" pr update-branch "$n" -R "$repo" >/dev/null 2>&1; then printf '      update-branch #%s: requested\n' "$n"
            else printf '      update-branch #%s: FAILED (conflict or refused); its owner merges main (STALE PR RULE)\n' "$n"; fi
        fi
    done <<<"$prs"
    if [ "$total" -eq 0 ]; then printf 'ok    no open PR\n'; fi
    [ "$apply" = 1 ] && [ "$bad" = 1 ] && printf 'applied: heads moved; re-run without --apply after the new runs exist\n'
    return "$bad"
}

self_test() {
    local pass=0 fail=0 rc
    # A fake gh as a shell FUNCTION (GH=fake_gh), so the table needs no temp files; update-branch writes the PR number to fd 3. FX_* select the fixture per row.
    fake_gh() {
        case "$1 $2" in
            "api rate_limit") [ "${FX_RATE:-}" = unread ] && return 1; printf '%s\n' "${FX_RATE:-5000}" ;;
            "pr list") [ "${FX_LIST:-}" = unread ] && return 1; printf '%b' "${FX_LIST:-}" ;;
            "pr update-branch") printf "%s," "$3" >&3; [ "${FX_UPD_FAIL:-}" != "$3" ] ;;
            api\ repos/*/compare/*)
                local h="${2##*...}"; local v; v="$(printf '%b' "${FX_CMP:-}" | awk -v h="$h" '$1 == h { print $2 }')"
                [ "$v" = unread ] && return 1; printf '%s\n' "${v:-diverged}" ;;
            api\ repos/*/commits/*)
                [ "${FX_GREEN:-}" = unread ] && return 1
                local s="${2#*/commits/}"; s="${s%%/*}"; printf '%b' "${FX_GREEN:-}" | awk -v s="$s" -F'\t' '$1 == s { print $2 }' ;;
            *) printf 'fake gh: unexpected: %s\n' "$*" >&2; return 1 ;;
        esac
    }
    GH=fake_gh
    row() { # row NAME WANT_RC APPLY WANT_UPDATED  (fixtures via FX_* set by the caller)
        local name="$1" want="$2" apply="$3" wantu="$4" got_u
        # errexit stays ON inside the substitution (it is not an if/&&/|| operand), so a stray non-zero status in cutover
        # aborts the row exactly as it aborts a real run; a `set +e` around the call would hide it (it did, once).
        set +e; got_u="$( set -e; cutover paiml/aprender CUT "$apply" 3>&1 >/dev/null 2>&1 )"; rc=$?; set -e
        got_u="${got_u%,}"
        if [ "$rc" -eq "$want" ] && [ "$got_u" = "$wantu" ]; then pass=$((pass+1)); printf '  ok    %-52s rc=%s updated=[%s]\n' "$name" "$rc" "$got_u"
        else fail=$((fail+1)); printf '  FAIL  %-52s want rc=%s updated=[%s] got rc=%s updated=[%s]\n' "$name" "$want" "$wantu" "$rc" "$got_u"; fi
    }
    local A='1 h1 feat/a\n' B='2 h2 feat/b\n' T='3 h3 train/aprender-x\n'
    FX_RATE=5000 FX_LIST="$A$B" FX_CMP='h1 ahead\nh2 identical\n' FX_GREEN='' row "all descend, no green" 0 0 ""
    FX_RATE=5000 FX_LIST="$A$B" FX_CMP='h1 ahead\nh2 diverged\n' FX_GREEN='' row "one diverged: stale" 1 0 ""
    FX_RATE=5000 FX_LIST="$A$B" FX_CMP='h1 behind\nh2 diverged\n' FX_GREEN='' row "apply: both updated, still rc 1" 1 1 "1,2"
    FX_RATE=5000 FX_LIST="$A$B" FX_CMP='h1 ahead\nh2 diverged\n' FX_GREEN='' row "apply: only the stale one updated" 1 1 "2"
    FX_RATE=5000 FX_LIST="$A$B" FX_CMP='h1 ahead\nh2 diverged\n' FX_GREEN='' FX_UPD_FAIL=2 row "apply: update conflict reported, rc 1" 1 1 "2"
    FX_RATE=5000 FX_LIST="$A" FX_CMP='h1 unread\n' FX_GREEN='' row "compare unreadable is stale, not ok" 1 0 ""
    FX_RATE=5000 FX_LIST="$A" FX_CMP='' FX_GREEN='' row "compare missing (empty) is stale" 1 0 ""
    FX_RATE=5000 FX_LIST="$A" FX_CMP='h1 ahead\n' FX_GREEN='h1\tgate\n' row "descends but gate green off train" 1 0 ""
    FX_RATE=5000 FX_LIST="$A" FX_CMP='h1 ahead\n' FX_GREEN='h1\tci / gate\n' row "descends but ci / gate green off train" 1 0 ""
    FX_RATE=5000 FX_LIST="$A" FX_CMP='h1 ahead\n' FX_GREEN='h1\tworkspace-test\n' row "descends but workspace-test green" 1 0 ""
    FX_RATE=5000 FX_LIST="$A" FX_CMP='h1 ahead\n' FX_GREEN=unread row "check-runs unreadable is flagged, not ok" 1 0 ""
    FX_RATE=5000 FX_LIST="$A" FX_CMP='h1 ahead\n' FX_GREEN='h1\tpr-light\n' row "pr-light green is fine" 0 0 ""
    FX_RATE=5000 FX_LIST="$A" FX_CMP='h1 ahead\n' FX_GREEN='h1\tgate-extra\n' row "similar name is not a required one" 0 0 ""
    FX_RATE=5000 FX_LIST="$T" FX_CMP='h3 ahead\n' FX_GREEN='h3\tgate\n' row "train ref green is legit" 0 0 ""
    FX_RATE=5000 FX_LIST="$T" FX_CMP='h3 diverged\n' FX_GREEN='' row "a stale train ref is stale too" 1 0 ""
    FX_RATE=5000 FX_LIST='' FX_CMP='' FX_GREEN='' row "no open PR" 0 0 ""
    FX_RATE=999 FX_LIST="$A" FX_CMP='h1 ahead\n' FX_GREEN='' row "rate < 1000: not measured" 2 0 ""
    FX_RATE=unread FX_LIST="$A" FX_CMP='h1 ahead\n' FX_GREEN='' row "rate unreadable: not measured" 2 0 ""
    FX_RATE=5000 FX_LIST=unread FX_CMP='' FX_GREEN='' row "PR list unreadable: not measured" 2 0 ""
    printf 'self-test: %s passed, %s failed\n' "$pass" "$fail"
    [ "$fail" -eq 0 ]
}

main() {
    local repo="paiml/aprender" cut="" apply=0
    [ $# -gt 0 ] || usage
    if [ "$1" = "--self-test" ]; then self_test; return $?; fi
    while [ $# -gt 0 ]; do
        case "$1" in
            --cutover) cut="${2:-}"; shift 2 ;;
            --repo)    repo="${2:-}"; shift 2 ;;
            --apply)   apply=1; shift ;;
            *) usage ;;
        esac
    done
    [[ "$cut" =~ ^[0-9a-f]{7,40}$ ]] || usage
    cutover "$repo" "$cut" "$apply"
}

main "$@"
