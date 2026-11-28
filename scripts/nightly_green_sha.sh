#!/usr/bin/env bash
# nightly_green_sha.sh — the commit the nightly builds: HEAD when its checks are green, else the newest green
# first-parent main commit within a bounded window (aprender #7462, child 2 of #7017).
#
# Usage:
#   nightly_green_sha.sh <head-sha> [--repo O/R] [--window N]   prints one verdict on stdout:
#                                                                  <sha>        the commit to build
#                                                                  ci-pending   HEAD is still running: retry later
#                                                                  red-ci       no green commit in the window
#   nightly_green_sha.sh --self-test                            five falsifier cases, offline
#
# Rules (design receipt handoff/release-cop/quorum-7017-c2-1659, GO):
#   - HEAD pending -> ci-pending (nightly_gate_retry.sh retries; no fallback to an older commit).
#   - Walk main's first-parent history, newest first, at most WINDOW commits (default 20).
#   - A green commit is selected. A failed commit is skipped. A pending or unknown older commit is skipped too:
#     an older commit is shipped only when it is verified green.
#   - No green commit in the window -> red-ci.
# State: GitHub GraphQL statusCheckRollup, one read per commit. The self-test reads a fixture instead.
set -euo pipefail

WINDOW=20

usage() { echo "usage: nightly_green_sha.sh <head-sha> [--repo O/R] [--window N] | --self-test" >&2; exit 2; }

# GraphQL statusCheckRollup state -> success | failure | pending | unknown
map_state() {
  case "$1" in
    SUCCESS) echo success ;;
    FAILURE|ERROR) echo failure ;;
    PENDING|EXPECTED) echo pending ;;
    *) echo unknown ;;
  esac
}

# pick <state-fn> <sha>... (newest first, already windowed). Prints the verdict.
pick() {
  local state_fn="$1"; shift
  local i=0 sha st
  for sha in "$@"; do
    st="$("$state_fn" "$sha")"
    if [ "$i" -eq 0 ] && [ "$st" = pending ]; then echo ci-pending; return 0; fi
    if [ "$st" = success ]; then echo "$sha"; return 0; fi
    i=$((i + 1))
  done
  echo red-ci
}

# real state: one GraphQL read per commit
live_state() {
  local sha="$1" raw
  raw="$(gh api graphql -F owner="${REPO%%/*}" -F name="${REPO#*/}" -F oid="$sha" -f query='
    query($owner:String!,$name:String!,$oid:GitObjectID!){
      repository(owner:$owner,name:$name){ object(oid:$oid){ ... on Commit { statusCheckRollup { state } } } } }' \
    --jq '.data.repository.object.statusCheckRollup.state // "NONE"')"
  map_state "$raw"
}

# fixture state: the self-test writes "<sha> <state>" lines; a missing sha is unknown
fixture_state() {
  local s
  s="$(awk -v k="$1" '$1==k {print $2; exit}' "$NIGHTLY_FIXTURE")"
  echo "${s:-unknown}"
}

self_test() {
  local dir cases=0 fails=0 got
  dir="$(mktemp -d)"
  trap 'rm -rf "$dir"' RETURN
  NIGHTLY_FIXTURE="$dir/fix"; export NIGHTLY_FIXTURE

  # run <name> <want> <fixture lines, newest first>; the window is applied as in the real path
  run() {
    local name="$1" want="$2" line
    shift 2
    : > "$NIGHTLY_FIXTURE"
    local commits=()
    for line in "$@"; do printf '%s\n' "$line" >> "$NIGHTLY_FIXTURE"; commits+=("${line%% *}"); done
    commits=("${commits[@]:0:$WINDOW}")
    got="$(pick fixture_state "${commits[@]}")"
    cases=$((cases + 1))
    if [ "$got" = "$want" ]; then echo "PASS $name -> $got"; else echo "FAIL $name: want $want got $got"; fails=$((fails + 1)); fi
  }

  run "1 HEAD green builds HEAD"               "h1"      "h1 success" "p2 success"
  run "2 HEAD pending is ci-pending"           "ci-pending" "h1 pending" "p2 success"
  run "3 red HEAD falls back to newest green"  "p2"      "h1 failure" "p2 success" "p3 success"
  run "4 all red in window is red-ci"          "red-ci"  "h1 failure" "p2 failure" "p3 failure"
  local old=() i
  for i in $(seq 1 20); do old+=("c$i failure"); done
  run "5 green only past the window is red-ci" "red-ci"  "${old[@]}" "old success"

  echo "nightly_green_sha self-test: $cases case(s), $fails failure(s)"
  [ "$fails" -eq 0 ]
}

[ "$#" -ge 1 ] || usage
if [ "$1" = "--self-test" ]; then self_test; exit $?; fi

HEAD_SHA="$1"; shift
REPO="${GITHUB_REPOSITORY:-paiml/aprender}"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --repo) REPO="${2:?--repo needs O/R}"; shift 2 ;;
    --window) WINDOW="${2:?--window needs N}"; shift 2 ;;
    *) usage ;;
  esac
done

mapfile -t COMMITS < <(git rev-list --first-parent -n "$WINDOW" "$HEAD_SHA")
[ "${#COMMITS[@]}" -gt 0 ] || { echo "::error::no commits reachable from $HEAD_SHA" >&2; exit 2; }
pick live_state "${COMMITS[@]}"
