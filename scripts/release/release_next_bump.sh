#!/usr/bin/env bash
# release_next_bump.sh -- #7017 child 1 (#7099): the NEXT version, taken from the open GitHub milestones, failing closed.
#
#   release_next_bump.sh [--current X.Y.Z] [--rate-floor N]   print the next version; rc 0
#   release_next_bump.sh --self-test                          the case table, no network
#
# THE NEXT VERSION, derived, never typed: the LOWEST open milestone titled X.Y.Z (canonical SemVer, version order:
# 0.74.2 before 0.74.10) strictly above the current version. The current version is --current, else main's
# [workspace.package] version read from the checkout's Cargo.toml.
# FAILS CLOSED: any read that is not whole (gh fails, GraphQL errors, null repository or milestones, a null title,
# more than 100 open milestones, allowance below the floor, unreadable current version) or no milestone above the
# current version prints `NEXT-BUMP-BLOCKED <why>` on stdout and exits 1. A guess is never printed.
# stdout on success: ONE line, the version. rc 2 = usage. It opens, tags and edits nothing.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
GH=${RNB_GH:-gh}; REPO=${RNB_REPO:-paiml/aprender}; CARGO=${RNB_CARGO:-$HERE/../../Cargo.toml}
SEMVER='^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$'

blocked() { printf 'NEXT-BUMP-BLOCKED %s\n' "$*"; exit 1; }
usage() { printf 'usage: release_next_bump.sh [--current X.Y.Z] [--rate-floor N] | --self-test (%s)\n' "$*" >&2; exit 2; }

if [ "${1:-}" = --self-test ]; then
    fail=0
    chk() { if [ "$2" = "$3" ]; then echo "  ok   $1"; else echo "  FAIL $1: got '$2' want '$3'"; fail=1; fi; }
    d=$(mktemp -d "${TMPDIR:-/tmp}/rnb.XXXXXX") || exit 2
    trap 'rm -rf -- "${d:?}"' EXIT
    # stub gh: RNB_T_MS (space list of milestone titles), RNB_T_REM, RNB_T_MODE (fail|null|msnull|mstitle|more|errors|remnull)
    cat > "$d/gh" << 'STUB'
#!/bin/bash
[ "${RNB_T_MODE:-}" = fail ] && exit 1
[ "${RNB_T_MODE:-}" = null ] && { printf '{"data":{"rateLimit":{"remaining":4000},"repository":null}}'; exit 0; }
ms=$(for m in ${RNB_T_MS:-}; do printf '{"title":"%s"},' "$m"; done); ms="[${ms%,}]"
[ "${RNB_T_MODE:-}" = mstitle ] && ms='[{"title":null},{"title":"0.75.0"}]'
more=false; [ "${RNB_T_MODE:-}" = more ] && more=true
errs='[]'; [ "${RNB_T_MODE:-}" = errors ] && errs='[{"type":"FORBIDDEN"}]'
rem=${RNB_T_REM:-4000}; [ "${RNB_T_MODE:-}" = remnull ] && rem=null
jq -n --argjson rem "$rem" --argjson ms "$ms" --argjson more "$more" --argjson errs "$errs" \
   '{errors:$errs,data:{rateLimit:{remaining:$rem},repository:{milestones:(if env.RNB_T_MODE == "msnull" then null else {pageInfo:{hasNextPage:$more},nodes:$ms} end)}}}'
STUB
    chmod +x "$d/gh"
    printf '[package]\nname = "aprender"\nversion = "9.9.9"\n\n[workspace.package]\nversion = "0.74.0"\n' > "$d/Cargo.toml"
    run() { out=$(RNB_GH="$d/gh" RNB_CARGO="$d/Cargo.toml" bash "$0" "$@" 2> /dev/null); rc=$?; }
    export RNB_T_MS="backlog 0.74.1 0.74.2 0.75.0 build-system 0.80.0"

    run;                       chk "lowest open milestone above Cargo.toml's [workspace.package] version" "$rc:$out" "0:0.74.1"
    run --current 0.74.1;      chk "--current replaces the Cargo.toml version" "$rc:$out" "0:0.74.2"
    RNB_T_MS="0.74.10 0.74.2 0.74.9" run --current 0.74.1
    chk "version order: 0.74.2 before 0.74.10" "$rc:$out" "0:0.74.2"
    RNB_T_MS="0.74.01 0.74 v0.75.0 0.75.0-rc.1 0.74.3" run --current 0.74.0
    chk "non-canonical titles are ignored" "$rc:$out" "0:0.74.3"
    run --current 0.80.0;      chk "no milestone above -> blocked, nothing printed but the verdict" "$rc:${out%% *}" "1:NEXT-BUMP-BLOCKED"
    RNB_T_MS="" run;           chk "no milestones at all -> blocked" "$rc:${out%% *}" "1:NEXT-BUMP-BLOCKED"
    RNB_T_MODE=fail run;       chk "gh fails -> blocked at the read, not later" "$rc:$(grep -c 'cannot read paiml' <<< "$out")" "1:1"
    RNB_T_MODE=remnull run;    chk "rateLimit.remaining null -> blocked as unreadable" "$rc:$(grep -c 'is unreadable' <<< "$out")" "1:1"
    RNB_T_MS="0.74.1 0.74.1 0.75.0" run; chk "two milestones with the same version -> that version, once" "$rc:$out" "0:0.74.1"
    RNB_T_MODE=null run;       chk "repository null with exit 0 -> blocked" "$rc:$(grep -c 'no repository' <<< "$out")" "1:1"
    RNB_T_MODE=msnull run;     chk "milestones null with exit 0 -> blocked" "$rc:${out%% *}" "1:NEXT-BUMP-BLOCKED"
    RNB_T_MODE=mstitle run;    chk "a null milestone title -> blocked, never skipped" "$rc:$(grep -c 'unreadable' <<< "$out")" "1:1"
    RNB_T_MODE=more run;       chk "more than 100 open milestones -> blocked" "$rc:${out%% *}" "1:NEXT-BUMP-BLOCKED"
    RNB_T_MODE=errors run;     chk "GraphQL errors with exit 0 -> blocked" "$rc:${out%% *}" "1:NEXT-BUMP-BLOCKED"
    RNB_T_REM=999 run;         chk "999 points left, floor 1000 -> blocked (GH-1)" "$rc:$(grep -c 'below the floor' <<< "$out")" "1:1"
    RNB_T_REM=1000 run;        chk "allowance exactly at the floor passes" "$rc:$out" "0:0.74.1"
    RNB_T_REM=999 run --rate-floor 500; chk "--rate-floor 500 lets 999 through" "$rc:$out" "0:0.74.1"
    printf '[workspace.package]\nname = "x"\n' > "$d/Cargo.toml"
    run;                       chk "no workspace version -> blocked" "$rc:${out%% *}" "1:NEXT-BUMP-BLOCKED"
    printf '[workspace.package]\nversion = "0.75.0-rc.1"\n' > "$d/Cargo.toml"
    run;                       chk "pre-release current version -> blocked" "$rc:$(grep -c 'not a final' <<< "$out")" "1:1"
    RNB_CARGO="$d/none.toml" run; chk "Cargo.toml missing -> blocked" "$rc:${out%% *}" "1:NEXT-BUMP-BLOCKED"
    run --current 0.75;        chk "--current not X.Y.Z -> usage rc 2" "$rc" 2
    run --bogus;               chk "unknown argument -> usage rc 2" "$rc" 2
    run --rate-floor x;        chk "--rate-floor not a number -> usage rc 2" "$rc" 2
    if [ "$fail" -eq 0 ]; then echo "release_next_bump self-test: PASS"; else echo "release_next_bump self-test: FAIL"; fi
    exit "$fail"
fi

cur="" floor=1000
while [ $# -gt 0 ]; do
    case "$1" in
        --current) [ $# -ge 2 ] || usage "--current needs a value"; cur=$2; shift 2 ;;
        --rate-floor) [ $# -ge 2 ] || usage "--rate-floor needs a value"; floor=$2; shift 2 ;;
        *) usage "unknown argument: $1" ;;
    esac
done
[ -z "$cur" ] || [[ $cur =~ $SEMVER ]] || usage "--current must be X.Y.Z, got '$cur'"
[[ $floor =~ ^[0-9]+$ ]] || usage "--rate-floor must be a number, got '$floor'"
command -v jq > /dev/null 2>&1 || blocked "jq is not installed: nothing can be read"

if [ -z "$cur" ]; then
    [ -r "$CARGO" ] || blocked "cannot read $CARGO for the current version"
    cur=$(awk '/^\[/ { s = ($0 ~ /^\[workspace\.package\][[:space:]]*$/) }
               s && /^version[[:space:]]*=/ { v = $0; sub(/^[^"]*"/, "", v); sub(/".*$/, "", v); print v; exit }' "$CARGO")
    [ -n "$cur" ] || blocked "no [workspace.package] version in $CARGO"
    [[ $cur =~ $SEMVER ]] || blocked "current version '$cur' is not a final X.Y.Z"
fi

w=$(mktemp -d "${TMPDIR:-/tmp}/rnb.XXXXXX") || blocked "no temp dir"
trap 'rm -rf -- "${w:?}"' EXIT
# shellcheck disable=SC2016
"$GH" api graphql -f owner="${REPO%%/*}" -f name="${REPO##*/}" -f query='
    query($owner: String!, $name: String!) {
      rateLimit { remaining }
      repository(owner: $owner, name: $name) { milestones(first: 100, states: OPEN) { pageInfo { hasNextPage } nodes { title } } }
    }' > "$w/r.json" 2> /dev/null || blocked "cannot read $REPO milestones (not_measured)"
jq -e '.data.repository | type == "object"' "$w/r.json" > /dev/null 2>&1 || blocked "the read returned no repository (not_measured)"
jq -e '(.errors // []) | length == 0' "$w/r.json" > /dev/null 2>&1 || blocked "the read carried GraphQL errors (not_measured)"
rem=$(jq -r '.data.rateLimit.remaining // empty' "$w/r.json" 2> /dev/null)
[[ $rem =~ ^[0-9]+$ ]] || blocked "rateLimit.remaining is unreadable (not_measured)"
[ "$rem" -ge "$floor" ] || blocked "GraphQL allowance $rem is below the floor $floor (GH-1): nothing judged"
jq -e '.data.repository.milestones | type == "object" and (.pageInfo.hasNextPage == false) and (.nodes | all(.title | type == "string"))' "$w/r.json" > /dev/null 2>&1 \
    || blocked "the open milestones are unreadable or past the first 100 (not_measured)"
next=$(jq -r --arg v "$cur" 'def key: split(".") | map(tonumber);
    [.data.repository.milestones.nodes[].title | select(test("^(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$")) | select(key > ($v | key))]
    | sort_by(key) | first // empty' "$w/r.json" 2> /dev/null) || blocked "the open milestones are unreadable (not_measured)"
[ -n "$next" ] || blocked "no open milestone titled X.Y.Z above $cur: will not guess a version"
printf '%s\n' "$next"
