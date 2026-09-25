#!/usr/bin/env bash
# rc_tag_main.sh -- the rc tagger: tag vX.Y.Z-rc.N on a queue-green main commit (#4327, #4328 C4)
#
#   rc_tag_main.sh --tag vX.Y.Z-rc.N --sha <40-hex> [--dry-run]   judge, then cut
#   rc_tag_main.sh --self-test                                     case table + fake API + mutant
#
# Operator ruling 2026-09-25 16:58: an rc is a tag on a queue-green main, cut within 5 min.
# This script replaces rc_cut.sh (#4285, cut on the tip of release/X.Y.Z), which that ruling
# superseded. In order:
#   1. judge  -- the commit is on main (it is main's head or an ancestor of it), `ci / gate`
#                completed with success on it, its Cargo.toml reads X.Y.Z, and no tag of that
#                name exists yet. Any miss refuses and writes nothing.
#   2. fleet  -- scripts/release/fleet_cells_gate.sh on infra-64's cells (fleet/cells.tsv on
#                the fleet-state branch) with scripts/release/fleet-waivers.tsv. A RED cell
#                without a dated waiver, or missing or stale cells, refuses and writes nothing.
#   3. tag    -- POST git/refs. The name is the lock: a concurrent cut gets 422 here.
#   4. draft  -- POST releases as a DRAFT prerelease, never latest. A draft is invisible to
#                install.sh and to the fleet poller (#4327).
#   5. dispatch binary-release.yml -f tag=<tag>: creating a draft emits `created`, not
#                `published`, so nothing else starts the asset build.
# Next, by hand or by the caller: scripts/release/rc_fleet_stage.sh <tag> --publish.
#
# GitHub is read and written with curl + jq. GH_TOKEN is taken from `gh auth token` when unset.
# EXIT 0 cut (or --dry-run would cut) · 1 refused or a write failed · 2 usage / cannot read.
set -uo pipefail
PROG=rc_tag_main
HERE=${RC_TAG_HERE:-"$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"}   # a mutant copy runs from a tmp dir
REPO=${RC_TAG_REPO:-paiml/aprender}
# The input overrides are for the self-test's fake only. On a live run, a green cells file, a moved clock or a
# mutant gate would each write a real tag past the fleet gate, so a live run refuses to start when any is set.
if [ "${1:-}" != --self-test ] && [ -z "${RC_TAG_FAKE:-}" ] && [ -n "${RC_TAG_HERE:-}${RC_TAG_CELLS:-}${RC_TAG_NOW:-}" ]; then
    echo "$PROG: RC_TAG_HERE/RC_TAG_CELLS/RC_TAG_NOW are self-test only (RC_TAG_FAKE unset): refusing a live run" >&2; exit 2
fi

# rc_tag_decide -- pure. Reads D_TAG D_SHA D_ON_MAIN D_GATE D_VERSION D_TAG_AT.
# Prints `cut <tag> <sha>` or `refuse <why>`. Returns 0 for both.
rc_tag_decide() {
    local x
    [[ "${D_TAG:-}" =~ ^v([0-9]+\.[0-9]+\.[0-9]+)-rc\.[1-9][0-9]*$ ]] \
        || { printf 'refuse tag %s is not vX.Y.Z-rc.N\n' "${D_TAG:-?}"; return 0; }
    x=${BASH_REMATCH[1]}
    [[ "${D_SHA:-}" =~ ^[0-9a-f]{40}$ ]] || { printf 'refuse sha %s is not a full 40-hex commit\n' "${D_SHA:-?}"; return 0; }
    case "${D_ON_MAIN:-}" in
        identical | ahead) ;;   # compare <sha>...main: main is the commit, or is ahead of it
        *) printf 'refuse %s is not on main (compare: %s)\n' "${D_SHA:0:9}" "${D_ON_MAIN:-unread}"; return 0 ;;
    esac
    [ "${D_GATE:-}" = "completed success" ] \
        || { printf 'refuse ci / gate on %s is %s, not completed success\n' "${D_SHA:0:9}" "${D_GATE:-absent}"; return 0; }
    [ "${D_VERSION:-}" = "$x" ] \
        || { printf 'refuse Cargo.toml@%s reads %s, the tag wants %s\n' "${D_SHA:0:9}" "${D_VERSION:-?}" "$x"; return 0; }
    [ "${D_TAG_AT:-}" != '?unread' ] || { printf 'refuse cannot read whether %s exists\n' "$D_TAG"; return 0; }
    [ -z "${D_TAG_AT:-}" ] || { printf 'refuse %s already exists at %s\n' "$D_TAG" "${D_TAG_AT:0:9}"; return 0; }
    printf 'cut %s %s\n' "$D_TAG" "$D_SHA"
}

# ---- I/O: GitHub REST via curl + jq. RC_TAG_FAKE=<dir> swaps in the self-test's fake. ----
fkey() { printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_'; }
api_get() {  # api_get PATH -> body on stdout; returns 2 on any failure
    if [ -n "${RC_TAG_FAKE:-}" ]; then cat -- "$RC_TAG_FAKE/get/$(fkey "$1")" 2>/dev/null || return 2; return 0; fi
    curl -sSf -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json" \
        "https://api.github.com/repos/$REPO${1:+/$1}" || return 2
}
api_post() {  # api_post PATH JSON -> line 1 is the HTTP code, the rest is the body
    local out code
    if [ -n "${RC_TAG_FAKE:-}" ]; then
        printf 'POST %s %s\n' "$1" "$2" >> "$RC_TAG_FAKE/posts"
        cat -- "$RC_TAG_FAKE/code/$(fkey "$1")" 2>/dev/null || echo 500; return 0
    fi
    out=$(mktemp) || return 2
    code=$(curl -sS -o "$out" -w '%{http_code}' -X POST -H "Authorization: Bearer $GH_TOKEN" \
        -H "Accept: application/vnd.github+json" "https://api.github.com/repos/$REPO/$1" -d "$2") || { rm -f -- "${out:?}"; return 2; }
    printf '%s\n' "$code"; cat -- "$out"; rm -f -- "${out:?}"
}
post_ok() {  # post_ok <want code> <what> <api_post output>
    local code; code=$(printf '%s\n' "$3" | head -1)
    [ "$code" = "$1" ] && return 0
    printf '%s: %s returned HTTP %s: %s\n' "$PROG" "$2" "$code" "$(printf '%s\n' "$3" | tail -n +2 | head -c 400)" >&2
    return 1
}
b64file() { jq -r '.content' | tr -d '\n' | base64 -d 2>/dev/null; }   # contents API -> the file

run_tag() {
    local tag=$1 sha=$2 dry=$3 decision cells gate rc out notes cargo
    D_TAG=$tag D_SHA=$sha
    D_ON_MAIN=$(api_get "compare/$sha...main" | jq -r '.status // empty' 2>/dev/null) || D_ON_MAIN=''
    D_GATE=$(api_get "commits/$sha/check-runs?check_name=ci%20%2F%20gate&per_page=100" \
        | jq -r '[.check_runs[]? | select(.name == "ci / gate")] | sort_by(.started_at) | last
                 | if . == null then "" else .status + " " + (.conclusion // "-") end' 2>/dev/null) || D_GATE=''
    # read the whole file first: an early-exiting awk SIGPIPEs the decoder, and under pipefail the
    # `||` fallback would then wipe the version it had already printed
    cargo=$(api_get "contents/Cargo.toml?ref=$sha" | b64file) || cargo=''
    D_VERSION=$(printf '%s\n' "$cargo" | awk -F'"' '/^version *=/{v=$2; exit} END{print v}')
    D_TAG_AT=$(api_get "git/matching-refs/tags/$tag" \
        | jq -r --arg r "refs/tags/$tag" '.[]? | select(.ref == $r) | .object.sha' 2>/dev/null) || D_TAG_AT='?unread'   # a failed read must refuse, not read as "new"
    decision=$(rc_tag_decide)
    echo "$PROG: $decision"
    case "$decision" in cut\ *) ;; *) return 1 ;; esac

    # The fleet (#4328 C4): no rc is cut while a fleet cell is RED, unless a dated waiver in
    # scripts/release/fleet-waivers.tsv names it. Missing or stale cells refuse too.
    cells=$(mktemp) || return 2
    if [ -n "${RC_TAG_CELLS:-}" ]; then cat -- "$RC_TAG_CELLS" > "$cells"
    else api_get "contents/fleet/cells.tsv?ref=fleet-state" 2>/dev/null | b64file > "$cells" || : > "$cells"; fi
    gate=$(bash "$HERE/fleet_cells_gate.sh" --cells "$cells" --waivers "$HERE/fleet-waivers.tsv" ${RC_TAG_NOW:+--now "$RC_TAG_NOW"}); rc=$?
    rm -f -- "${cells:?}"
    echo "$PROG: fleet cells: $gate"
    [ "$rc" = 0 ] || { echo "$PROG: REFUSED to cut $tag -- the fleet cells gate refused (#4328 C4)" >&2; return 1; }
    if [ "$dry" = 1 ]; then echo "$PROG: dry run: no tag, draft or dispatch written"; return 0; fi

    out=$(api_post git/refs "{\"ref\":\"refs/tags/$tag\",\"sha\":\"$sha\"}") || return 1
    post_ok 201 "creating tag $tag" "$out" || return 1

    notes="Release candidate of ${tag#v}: a tag on queue-green main (operator ruling 2026-09-25 16:58). Not on crates.io.

Commit \`$sha\`; \`ci / gate\` completed success on it. Cut at $(date -u +%Y-%m-%dT%H:%M:%SZ) by scripts/release/rc_tag_main.sh.
binary-release.yml attaches the apr and pv assets to this DRAFT; scripts/release/rc_fleet_stage.sh publishes it
only after every reachable fleet host runs it (#4327).

Install: \`install.sh --version $tag\`, or \`install.sh --channel rc\` for the newest rc."
    out=$(api_post releases "$(jq -nc --arg t "$tag" --arg n "$notes" \
        '{tag_name: $t, name: "\($t) (release candidate)", body: $n, draft: true, prerelease: true, make_latest: "false"}')") || return 1
    post_ok 201 "creating the draft $tag (the tag exists: create the draft and dispatch binary-release.yml by hand)" "$out" || return 1

    out=$(api_post actions/workflows/binary-release.yml/dispatches "{\"ref\":\"main\",\"inputs\":{\"tag\":\"$tag\"}}") || return 1
    post_ok 204 "dispatching binary-release.yml for $tag (tag + draft exist: dispatch by hand)" "$out" || return 1
    echo "$PROG: cut $tag at ${sha:0:9}: tag, DRAFT prerelease, binary-release.yml dispatched. Next: rc_fleet_stage.sh $tag --publish (#4327)"
}

self_test() {
    local fail=0 d A B got rc
    A=$(printf 'a%.0s' {1..40}) B=$(printf 'b%.0s' {1..40})
    expect() {  # expect <want> <label>
        got=$(rc_tag_decide)
        if [ "$got" = "$1" ]; then echo "  ok   $2"; else echo "  FAIL $2: got '$got', want '$1'"; fail=1; fi
    }
    base() { D_TAG=v0.70.0-rc.1 D_SHA=$A D_ON_MAIN=identical D_GATE='completed success' D_VERSION=0.70.0 D_TAG_AT=''; }
    echo "$PROG self-test: decision table"
    base; expect "cut v0.70.0-rc.1 $A" 'main head, gate green, version matches, tag free'
    base; D_ON_MAIN=ahead; expect "cut v0.70.0-rc.1 $A" 'an ancestor of main cuts'
    base; D_ON_MAIN=behind; expect 'refuse aaaaaaaaa is not on main (compare: behind)' 'a commit ahead of main (a PR head) refused'
    base; D_ON_MAIN=diverged; expect 'refuse aaaaaaaaa is not on main (compare: diverged)' 'a side-branch commit refused'
    base; D_ON_MAIN=''; expect 'refuse aaaaaaaaa is not on main (compare: unread)' 'an unread compare never cuts'
    base; D_GATE='completed failure'; expect 'refuse ci / gate on aaaaaaaaa is completed failure, not completed success' 'red gate refused'
    base; D_GATE='in_progress -'; expect 'refuse ci / gate on aaaaaaaaa is in_progress -, not completed success' 'running gate refused'
    base; D_GATE=''; expect 'refuse ci / gate on aaaaaaaaa is absent, not completed success' 'no gate run refused'
    base; D_VERSION=0.69.3; expect 'refuse Cargo.toml@aaaaaaaaa reads 0.69.3, the tag wants 0.70.0' 'unbumped workspace refused'
    base; D_TAG_AT=$B; expect 'refuse v0.70.0-rc.1 already exists at bbbbbbbbb' 'an existing tag refused'
    base; D_TAG_AT='?unread'; expect 'refuse cannot read whether v0.70.0-rc.1 exists' 'an unreadable tag list refused'
    base; D_TAG=v0.70.0; expect 'refuse tag v0.70.0 is not vX.Y.Z-rc.N' 'a final tag is not this script'"'"'s'
    base; D_TAG=v0.70.0-rc.0; expect 'refuse tag v0.70.0-rc.0 is not vX.Y.Z-rc.N' 'rc.0 refused'
    base; D_SHA=aaaaaaaaa; expect 'refuse sha aaaaaaaaa is not a full 40-hex commit' 'a short sha refused'

    echo "$PROG self-test: fake API, real fleet_cells_gate.sh"
    d=$(mktemp -d) || return 2
    fake() {  # fake <cells state for lambda/apr>: a fresh fake GitHub where every judge input is green
        rm -rf -- "${d:?}/api"; mkdir -p "$d/api/get" "$d/api/code"
        printf '{"status":"identical"}' > "$d/api/get/$(fkey "compare/$A...main")"
        printf '{"check_runs":[{"name":"ci / gate","status":"completed","conclusion":"success","started_at":"2026-09-25T20:00:00Z"}]}' \
            > "$d/api/get/$(fkey "commits/$A/check-runs?check_name=ci%20%2F%20gate&per_page=100")"
        # large, as the real one is: a small file hides a SIGPIPE in the read
        printf '{"content":"%s"}' "$({ printf '[workspace.package]\nversion = "0.70.0"\n'; seq -f 'k%g = 1' 1 40000; } | base64 -w0)" > "$d/api/get/$(fkey "contents/Cargo.toml?ref=$A")"
        printf '[]' > "$d/api/get/$(fkey "git/matching-refs/tags/v0.70.0-rc.1")"
        echo 201 > "$d/api/code/$(fkey git/refs)"; echo 201 > "$d/api/code/releases"
        echo 204 > "$d/api/code/$(fkey actions/workflows/binary-release.yml/dispatches)"
        printf '# measured 2026-09-25T20:00:00Z\nlambda-labs\tapr\t%s\tprobe\n' "$1" > "$d/cells.tsv"
    }
    cut_with() {  # cut_with <script>: run a real cut against the fake; posts land in $d/api/posts
        RC_TAG_FAKE=$d/api RC_TAG_CELLS=$d/cells.tsv RC_TAG_NOW=$(date -u -d 2026-09-25T21:00:00Z +%s) bash "$1" --tag v0.70.0-rc.1 --sha "$A" > "$d/out" 2>&1
    }
    local me; me="$HERE/$(basename -- "${BASH_SOURCE[0]}")"
    fake GREEN; cut_with "$me"; rc=$?
    got=$(cut -d' ' -f2 "$d/api/posts" 2>/dev/null | tr '\n' ' ')
    if [ "$rc" = 0 ] && [ "$got" = "git/refs releases actions/workflows/binary-release.yml/dispatches " ] \
        && grep -q '"draft":true' "$d/api/posts"; then echo "  ok   green fleet: tag, then DRAFT, then dispatch"
    else echo "  FAIL green fleet: rc $rc, posts '$got': $(tail -n 2 "$d/out")"; fail=1; fi
    fake RED; cut_with "$me"; rc=$?
    if [ "$rc" = 1 ] && [ ! -s "$d/api/posts" ] && grep -q 'REFUSED' "$d/out"; then echo "  ok   RED fleet cell: refused, nothing written"
    else echo "  FAIL RED fleet cell: rc $rc, posts: $(cat "$d/api/posts" 2>/dev/null | cut -c1-80)"; fail=1; fi
    fake GREEN; printf '[{"ref":"refs/tags/v0.70.0-rc.1","object":{"sha":"%s"}}]' "$B" > "$d/api/get/$(fkey "git/matching-refs/tags/v0.70.0-rc.1")"
    cut_with "$me"; rc=$?
    if [ "$rc" = 1 ] && [ ! -s "$d/api/posts" ]; then echo "  ok   existing tag: refused before the fleet read, nothing written"
    else echo "  FAIL existing tag: rc $rc"; fail=1; fi
    fake GREEN; echo 422 > "$d/api/code/$(fkey git/refs)"; cut_with "$me"; rc=$?
    got=$(cut -d' ' -f2 "$d/api/posts" 2>/dev/null | tr '\n' ' ')
    if [ "$rc" = 1 ] && [ "$got" = "git/refs " ]; then echo "  ok   tag POST 422 (a concurrent cut): no draft, no dispatch"
    else echo "  FAIL tag POST 422: rc $rc, posts '$got'"; fail=1; fi
    fake GREEN; mv -- "$d/api/get/$(fkey git/matching-refs/tags/v0.70.0-rc.1)" "$d/api/unread"; cut_with "$me"; rc=$?
    if [ "$rc" = 1 ] && [ ! -s "$d/api/posts" ]; then echo "  ok   unreadable tag list: refused, nothing written"
    else echo "  FAIL unreadable tag list: rc $rc"; fail=1; fi
    for v in RC_TAG_HERE RC_TAG_CELLS RC_TAG_NOW; do   # live run (no RC_TAG_FAKE): exits before any API call
        env -u RC_TAG_FAKE "$v=$d" GH_TOKEN=unused bash "$me" --tag v0.70.0-rc.1 --sha "$A" --dry-run > /dev/null 2>&1; rc=$?
        if [ "$rc" = 2 ]; then echo "  ok   live run with $v set: refused (rc 2)"
        else echo "  FAIL live run with $v set: rc $rc"; fail=1; fi
    done

    echo "$PROG self-test: mutant"
    # the gate call deleted: the RED cell must now be tagged, or the refusal above was not the gate's
    mkdir -p "$d/m"; cp -- "$HERE/fleet_cells_gate.sh" "$HERE/fleet-waivers.tsv" "$d/m/"
    sed 's|^    gate=$(bash "$HERE/fleet_cells_gate.sh" --cells.*|    gate=mutant; rc=0|' "$me" > "$d/m/rc_tag_main.sh"
    fake RED; RC_TAG_HERE=$d/m cut_with "$d/m/rc_tag_main.sh"; rc=$?
    if grep -q '^POST git/refs' "$d/api/posts" 2>/dev/null; then echo "  ok   mutant without the gate call tags past a RED cell: the gate is load-bearing"
    else echo "  FAIL mutant still refuses (rc $rc): the RED refusal came from somewhere else"; fail=1; fi
    rm -rf -- "${d:?}"
    if [ "$fail" = 0 ]; then echo "$PROG self-test: PASS"; else echo "$PROG self-test: FAIL"; fi
    return "$fail"
}

main() {
    local tag='' sha='' dry=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --self-test) self_test; return $? ;;
            --tag) tag=${2:-}; shift 2 || return 2 ;;
            --sha) sha=${2:-}; shift 2 || return 2 ;;
            --dry-run) dry=1; shift ;;
            -h | --help) sed -n '2,25p' "${BASH_SOURCE[0]}"; return 0 ;;
            *) echo "$PROG: unknown argument $1" >&2; return 2 ;;
        esac
    done
    [ -n "$tag" ] && [ -n "$sha" ] || { echo "$PROG: --tag vX.Y.Z-rc.N and --sha <40-hex> are required" >&2; return 2; }
    command -v jq >/dev/null || { echo "$PROG: jq is required" >&2; return 2; }
    if [ -z "${RC_TAG_FAKE:-}" ] && [ -z "${GH_TOKEN:-}" ]; then
        GH_TOKEN=$(gh auth token 2>/dev/null) || { echo "$PROG: GH_TOKEN unset and gh auth token failed" >&2; return 2; }
    fi
    run_tag "$tag" "$sha" "$dry"
}

main "$@"
