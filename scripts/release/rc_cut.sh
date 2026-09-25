#!/usr/bin/env bash
# rc_cut.sh -- cut vX.Y.Z-rc.N when CI goes green on the tip of release/X.Y.Z (#4285,
# RC-DOGFOOD-001 §4.1, paiml/infra#1030).
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

# rc_decide -- the whole decision, pure. Inputs are named variables so the case table
# can vary exactly one at a time:
#   D_REPO D_HEAD_REPO D_BRANCH D_HEAD_SHA D_TIP_SHA
#   D_JOBS  lines "<job name>\t<conclusion>"
#   D_TAGS  lines "<tag name>\t<commit sha>" for existing tags matching v<X.Y.Z>-rc.*
# Prints "cut <tag>" or "skip <reason>". Returns 0 for both.
rc_decide() {
    local v name c n max=0 have
    if [ "${D_HEAD_REPO:-}" != "${D_REPO:-}" ]; then
        printf 'skip head repository %s is not %s\n' "${D_HEAD_REPO:-?}" "${D_REPO:-?}"; return 0
    fi
    if ! v=$(rc_version_of_branch "${D_BRANCH:-}"); then
        printf 'skip branch %s is not release/X.Y.Z\n' "${D_BRANCH:-?}"; return 0
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
    printf 'cut v%s-rc.%d\n' "$v" "$((max + 1))"
}

# rcc_io_rows -- the jq reads against fixture answers, with a fake curl for the tag
# dereference. Prints one line per row; returns 1 if any row is wrong.
rcc_io_rows() {
    local bad=0 got d
    row() {  # row <want> <label> <got>
        if [ "$3" = "$1" ]; then printf '  ok   %s\n' "$2"; else printf '  FAIL %s\n       want: %q\n       got:  %q\n' "$2" "$1" "$3"; bad=1; fi
    }
    local run='{"head_repository":null,"head_branch":"release/0.70.0","head_sha":null}'
    row '' 'a null head_repository reads as "" (a fork check that cannot pass)' "$(printf '%s' "$run" | rcc_head_repo)"
    row 'release/0.70.0' 'head_branch' "$(printf '%s' "$run" | rcc_or_empty head_branch)"
    row '' 'a null head_sha reads as "", never "None"' "$(printf '%s' "$run" | rcc_or_empty head_sha)"
    got=$(printf '%s' '{"jobs":[{"name":"ci / gate","conclusion":"success"},{"name":"workspace-test","conclusion":null}]}' | rcc_jobs_tsv)
    row $'ci / gate\tsuccess\nworkspace-test\tNone' 'an unfinished job reads as None, which is not success' "$got"
    printf '%s' '{"total_count":0}' | rcc_jobs_tsv > /dev/null 2>&1; row 5 'an answer with no jobs key is an error, not zero jobs' "$?"
    row 2 'the page count' "$(printf '%s' '{"jobs":[{},{}]}' | rcc_jobs_count)"
    row None 'print(d["object"]["sha"]) of a null sha' "$(printf '%s' '{"object":{"sha":null}}' | rcc_path 'key("object") | key("sha")')"
    got=$(printf '%s' "{\"content\":\"$(printf 'lambda\tapr\tGREEN\nyoga\tpv\tRED\n' | base64 -w 16 | awk '{printf "%s\\n", $0}')\"}" | rcc_b64_content)
    row $'lambda\tapr\tGREEN\nyoga\tpv\tRED' 'the contents API base64, wrapped, decodes to the cells' "$got"
    got=$(printf '%s' '{"content":"bGFtYmRhC"}' | rcc_b64_content 2>/dev/null; echo "rc=$?")
    row 'rc=5' 'truncated base64 writes nothing and fails (the gate then refuses empty cells)' "$got"
    got=$(rcc_release_body 'v0.70.0-rc.1' $'notes "q"\n' | jq -c '[.tag_name, .name, .body, .prerelease, .draft, .make_latest]')
    row '["v0.70.0-rc.1","v0.70.0-rc.1","notes \"q\"\n",true,true,"false"]' 'the release body: a draft prerelease, never latest' "$got"
    d=$(mktemp -d) || return 1
    printf '%s\n' '#!/usr/bin/env bash' 'for a; do u=$a; done' \
        'case $u in *tag-ok) echo "{\"object\":{\"sha\":\"c0ffee\",\"type\":\"commit\"}}" ;; *) exit 22 ;; esac' > "$d/curl"
    chmod +x "$d/curl"
    got=$(printf '%s' '[{"ref":"refs/tags/v0.70.0-rc.1","object":{"type":"commit","sha":"aaa"}},{"ref":"refs/tags/v0.70.0-rc.2","object":{"type":"tag","sha":"ttt","url":"https://x/tag-ok"}}]' \
        | GH_TOKEN=t PATH="$d:$PATH" rcc_tags_tsv)
    row $'v0.70.0-rc.1\taaa\nv0.70.0-rc.2\tc0ffee' 'an annotated tag is dereferenced to its commit' "$got"
    got=$(printf '%s' '[{"ref":"refs/tags/v0.70.0-rc.2","object":{"type":"tag","sha":"ttt","url":"https://x/gone"}}]' \
        | GH_TOKEN=t PATH="$d:$PATH" rcc_tags_tsv 2>/dev/null; echo "rc=$?")
    row 'rc=2' 'a failed dereference is ENV, never a missing tag' "$got"
    row '' 'no matching refs -> no tags' "$(printf '%s' '[]' | rcc_tags_tsv)"
    rm -f -- "$d/curl"; rmdir -- "$d"
    return "$bad"
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
    base; D_TAGS=$'v0.70.0-rc.1\t'$B; expect 'cut v0.70.0-rc.2' 'rc.1 exists on an older commit -> rc.2'
    base; D_TAGS=$'v0.70.0-rc.1\tc1\nv0.70.0-rc.3\tc3'; expect 'cut v0.70.0-rc.4' 'gap in numbering -> max+1, never reuse a name'
    base; D_TAGS=$'v0.70.0-rc.1\tc1\nv0.70.0-rc.10\tc10\nv0.70.0-rc.9\tc9'; expect 'cut v0.70.0-rc.11' 'numeric max over the API'"'"'s LEXICAL order (rc.10 sorts before rc.9)'
    base; D_TAGS=$'v0.69.3-rc.7\tc7\nv0.70.0\tcf\nv0.70.00-rc.5\tcx\nv0x70x0-rc.6\tcy'; expect 'cut v0.70.0-rc.1' 'other versions, the final tag and near-miss names (dots are literal) do not count'
    base; D_TAGS=$'v0.70.0-rc.2\t'$A; expect "skip v0.70.0-rc.2 already points at $A" 're-run on an already-cut commit cuts nothing'
    base; D_BRANCH=main; expect 'skip branch main is not release/X.Y.Z' 'main is not a release branch'
    base; D_BRANCH=release/0.69.1-batch-2; expect 'skip branch release/0.69.1-batch-2 is not release/X.Y.Z' 'suffixed release branch refused'
    base; D_BRANCH=release/0.70; expect 'skip branch release/0.70 is not release/X.Y.Z' 'two-part version refused'
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
    echo "$PROG self-test: the jq reads of the GitHub answers (#4352)"
    rcc_io_rows || fail=1
    # MUTANTS of the reads: each must turn at least one row red. Built from THIS file.
    local m a b why src head tail marker=$'\n# ---- I/O: GitHub REST'
    src=$(cat -- "${BASH_SOURCE[0]}")
    for m in 1 2 3; do
        case $m in
            1) a='if key("type") == "commit" then'; b='if true then'; why='annotated tags not dereferenced' ;;
            2) a='| gsub("[^A-Za-z0-9+/=]"; "") |'; b='|'; why='the base64 line breaks kept' ;;
            3) a='prerelease: true, draft: true'; b='prerelease: true, draft: false'; why='the rc published at once, not as a draft' ;;
        esac
        mut=$(mktemp) || return 2
        # mutate only the I/O section: the case lines above hold the same anchors
        head=${src%%"$marker"*}; tail=${src#"$head"}
        printf '%s\n' "$head${tail/"$a"/"$b"}" > "$mut"
        if [ "$head" = "$src" ] || [ "${tail/"$a"/}" = "$tail" ]; then
            echo "  FAIL mutant $m ($why) was not built: its anchor is not in the helpers"; fail=1
        elif bash -c ". '$mut' --source-only; rcc_io_rows" > /dev/null 2>&1; then
            echo "  FAIL mutant $m ($why) survived the rows"; fail=1
        else echo "  ok   mutant $m ($why) is killed"; fi
        rm -f -- "$mut"
    done
    if [ "$fail" = 0 ]; then echo "$PROG self-test: PASS"; else echo "$PROG self-test: FAIL"; fi
    return "$fail"
}

# ---- I/O: GitHub REST via curl + jq (the fleet boxes carry no gh) ------------
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
# jq, not python3 (#4352). The helpers keep the python reads' semantics, which the parity
# table pinned: a missing key or a non-object is an error (the caller returns 2), a JSON
# null prints "None" as str(None) did, and `x or ""` treats every python-falsy value as "".
# iter: python iterates an empty dict or str as zero items; any other non-list is an error.
RCC_JQ='def obj: if type == "object" then . else error("not an object") end;
def key($k): obj | if has($k) then .[$k] else error("missing key \($k)") end;
def pystr: if . == null then "None" elif . == true then "True" elif . == false then "False"
    elif type == "string" then . else tojson end;
def truthy: . != null and . != false and . != 0 and . != "" and . != [] and . != {};
def str: if type == "string" then . else error("not a string") end;
def arr: if type == "array" then . else error("not an array") end;
def iter: if . == {} or . == "" then [] else arr end;'
json() { jq -r "$RCC_JQ $1"; }   # json FILTER: stdin JSON -> the filter's raw output
rcc_head_repo() { json 'obj | (.head_repository | if truthy then . else {} end) | obj | if has("full_name") then .full_name | pystr else "" end'; }
rcc_or_empty() { json "obj | .$1 | if truthy then pystr else \"\" end"; }   # print(d.get(K) or "")
rcc_jobs_tsv() { json '[key("jobs") | iter | .[] | obj | (key("name") | str) + "\t" + (.conclusion | pystr)] | join("\n")'; }
rcc_jobs_count() { json 'key("jobs") | if type == "array" or type == "object" or type == "string" then length else error("no len") end'; }
rcc_path() { json "$1 | pystr"; }   # rcc_path 'key("a") | key("b")' -> print(d["a"]["b"])
rcc_b64_content() {  # the contents API's base64 (wrapped at 60) -> the file; nothing on bad input
    jq -j "$RCC_JQ"' key("content") | str | if explode | any(. > 127) then error("non-ascii") else . end
        | gsub("[^A-Za-z0-9+/=]"; "") | (gsub("="; "") | length % 4) as $r | (capture("(?<p>=*)$").p | length) as $p
        | if $r == 1 or ($r == 2 and $p < 2) or ($r == 3 and $p < 1) then error("bad padding") else . end
        | gsub("="; "") | . + ["", "", "==", "="][$r] | @base64d
        | if index("\ufffd") then error("not utf-8") else . end'   # python .decode() refused bad UTF-8
}
# rcc_tags_tsv: matching-refs JSON on stdin -> "<tag>\t<commit sha>"; an annotated tag's
# object is fetched and dereferenced. All or nothing: any bad ref or fetch returns 2.
rcc_tags_tsv() {
    local lines tag sha url out=''
    lines=$(json 'iter | .[] | (key("ref") | str | .[10:]) as $t | key("object")
        | if key("type") == "commit" then "\($t)\u001f\(key("sha") | str)" else "\($t)\u001f\u001f\(key("url") | str)" end') || return 2
    while IFS=$'\x1f' read -r tag sha url; do
        [ -n "$tag$sha$url" ] || continue
        if [ -z "$sha" ]; then
            sha=$(curl -sSfL -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json" "$url" \
                | json 'key("object") | key("sha") | str') || return 2
        fi
        out+="$tag"$'\t'"$sha"$'\n'
    done <<< "$lines"
    printf '%s' "$out"
}
rcc_release_body() {  # rcc_release_body TAG NOTES -> the POST /releases payload
    jq -nc --arg tag "$1" --arg notes "$2" \
        '{tag_name: $tag, name: $tag, body: $notes, prerelease: true, draft: true, make_latest: "false"}'
}
emit() { if [ -n "${GITHUB_OUTPUT:-}" ]; then printf '%s=%s\n' "$1" "$2" >> "$GITHUB_OUTPUT"; fi; }
summary() { if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then printf '%s\n' "$@" >> "$GITHUB_STEP_SUMMARY"; fi; printf '%s\n' "$@"; }
post_ok() {  # post_ok <want code> <what> <api_post output>
    local code; code=$(printf '%s\n' "$3" | head -1)
    [ "$code" = "$1" ] && return 0
    printf '%s: %s returned HTTP %s: %s\n' "$PROG" "$2" "$code" "$(printf '%s\n' "$3" | tail -n +2 | head -c 400)" >&2
    return 1
}

run_cut() {
    local run_id=$1 dry=$2 run v page body n decision tag out merged_at notes ref rc
    : "${GH_TOKEN:?GH_TOKEN is required}" "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
    run=$(api_get "actions/runs/$run_id") || { echo "$PROG: cannot read run $run_id" >&2; return 2; }
    D_REPO=$GITHUB_REPOSITORY
    D_HEAD_REPO=$(printf '%s' "$run" | rcc_head_repo) || return 2
    D_BRANCH=$(printf '%s' "$run" | rcc_or_empty head_branch) || return 2
    D_HEAD_SHA=$(printf '%s' "$run" | rcc_or_empty head_sha) || return 2
    D_JOBS='' D_TAGS='' D_TIP_SHA=''
    # Read the jobs, tip and tags only for a candidate; rc_decide still judges every field.
    if v=$(rc_version_of_branch "$D_BRANCH") && [ "$D_HEAD_REPO" = "$D_REPO" ]; then
        page=1
        while :; do  # CI can exceed one page of jobs once shards and matrices are counted: paginate
            body=$(api_get "actions/runs/$run_id/jobs?filter=latest&per_page=$PER_PAGE&page=$page") || { echo "$PROG: cannot read the jobs of run $run_id" >&2; return 2; }
            D_JOBS+=$(printf '%s' "$body" | rcc_jobs_tsv)$'\n' || return 2
            n=$(printf '%s' "$body" | rcc_jobs_count) || return 2
            [ "$n" -lt "$PER_PAGE" ] && break
            page=$((page + 1))
        done
        body=$(api_get "git/ref/heads/release/$v") || { echo "$PROG: cannot read the tip of release/$v" >&2; return 2; }
        D_TIP_SHA=$(printf '%s' "$body" | rcc_path 'key("object") | key("sha")') || return 2
        # matching-refs returns [] when nothing matches; an annotated tag is dereferenced to its commit.
        body=$(api_get "git/matching-refs/tags/v$v-rc.") || { echo "$PROG: cannot list the v$v-rc.* tags" >&2; return 2; }
        D_TAGS=$(printf '%s' "$body" | rcc_tags_tsv) || { echo "$PROG: cannot resolve the v$v-rc.* tags" >&2; return 2; }
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
        | rcc_b64_content > "$cells" 2>/dev/null || : > "$cells"
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
    merged_at=$(api_get "commits/$D_HEAD_SHA" | rcc_path 'key("commit") | key("committer") | key("date")') || merged_at=unknown
    notes="Release candidate of ${v}, cut by CI (#4285). Not on crates.io.

Commit \`$D_HEAD_SHA\` on \`release/$v\`, merged $merged_at.
Gated by CI run https://github.com/$GITHUB_REPOSITORY/actions/runs/$run_id (\`ci / gate\` and \`workspace-test\` green).
Cut at $(date -u +%Y-%m-%dT%H:%M:%SZ). binary-release.yml attaches the apr and pv assets to this DRAFT;
scripts/release/rc_fleet_stage.sh publishes it only after every reachable fleet host runs it (#4327).

Install: \`install.sh --version $tag\`, or \`install.sh --channel rc\` for the newest rc."
    body=$(rcc_release_body "$tag" "$notes") || return 1
    out=$(api_post releases "$body") || return 1
    post_ok 201 "creating prerelease $tag (the tag exists; create the release and dispatch binary-release.yml by hand)" "$out" || return 1

    # 3. The assets. A GITHUB_TOKEN-created release does not fire `release: published`.
    ref=$(api_get "" | rcc_path 'key("default_branch")') || ref=main
    out=$(api_post actions/workflows/binary-release.yml/dispatches "{\"ref\":\"$ref\",\"inputs\":{\"tag\":\"$tag\"}}") || return 1
    post_ok 204 "dispatching binary-release.yml for $tag" "$out" || return 1
    summary "Cut $tag at ${D_HEAD_SHA:0:9} as a DRAFT prerelease and dispatched binary-release.yml on $ref. Next: rc_fleet_stage.sh $tag --publish (#4327)."
    emit result cut
}

main() {
    local run_id='' dry=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --self-test) self_test; return $? ;;
            --source-only) return 0 ;;
            --run-id) run_id=${2:-}; shift 2 || return 2 ;;
            --dry-run) dry=1; shift ;;
            -h|--help) sed -n '2,48p' "${BASH_SOURCE[0]}"; return 0 ;;
            *) echo "$PROG: unknown argument $1" >&2; return 2 ;;
        esac
    done
    [[ "$run_id" =~ ^[0-9]+$ ]] || { echo "$PROG: --run-id <numeric CI run id> is required" >&2; return 2; }
    run_cut "$run_id" "$dry"
}

# Sourced with --source-only (the self-test's mutant): define the functions, run nothing.
if [ "${1:-}" = --source-only ]; then return 0 2>/dev/null || exit 0; fi
main "$@"
