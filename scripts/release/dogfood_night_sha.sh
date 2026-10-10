#!/usr/bin/env bash
# dogfood_night_sha.sh -- the commit dogfood-nightly measures: the head of the night's SCHEDULED nightly.yml run (#5079).
#
# The train judges one commit per night: the head of the newest scheduled nightly.yml run on main (B1, the train's
# night pick). A producer that measured main's tip at its own start time measured some other commit whenever anything
# merged in between, and the train then read its lane as "no run on C". So dogfood-nightly checks out the commit
# this script prints, chosen the way the train chooses it:
#   the newest run with event schedule on main, created at or before NOW, with a 40-hex head_sha.
# One rule is tighter than the train's 24 h: the run must be less than 12 h old. Schedule events on this account
# land hours late, so if dogfood starts before tonight's nightly.yml run exists the newest run is LAST night's,
# and measuring it would be a confident answer about the wrong night. That refuses (rc 1) instead.
# The run's status does not matter here: its head is fixed when it is created.
#
#   dogfood_night_sha.sh --now EPOCH     print the sha (one gh api call; GH_TOKEN, GITHUB_REPOSITORY). The caller
#                                        passes the read time: the script reads no clock, so the same input decides the same.
#   dogfood_night_sha.sh --pick F NOW    pick from a runs JSON file F at epoch NOW (no network)
#   dogfood_night_sha.sh --self-test     case table, then mutants of the pick
# rc 0 = the sha on stdout; 1 = no night commit (reason on stderr); 2 = could not read.
set -uo pipefail

NIGHT_WF="nightly.yml"
NIGHT_MAX_AGE=43200

night_pick() { # night_pick RUNS.json NOW -> "sha" or "none<TAB>reason"
  jq -r --argjson t "$2" --argjson max "$NIGHT_MAX_AGE" '
    [(.workflow_runs // [])[] | select(.event == "schedule" and .head_branch == "main" and ((.created_at | fromdateiso8601) <= $t))]
    | (sort_by(.created_at) | last) as $r
    | if $r == null then "none\tno scheduled run on main at or before the read"
      elif ($t - ($r.created_at | fromdateiso8601)) >= $max then "none\tthe newest scheduled run \($r.id) (\($r.created_at)) is 12 h or more old: tonight'"'"'s has not started"
      elif (($r.head_sha // "") | test("^[0-9a-f]{40}$") | not) then "none\tthe scheduled run \($r.id) has no head sha"
      else $r.head_sha end' "$1" 2> /dev/null
}

emit() { # emit <pick output> -> rc
  case "$1" in
    "") echo "dogfood_night_sha: the runs list is unreadable" >&2; return 2 ;;
    none*) echo "dogfood_night_sha: ${1#none	}" >&2; return 1 ;;
    *) printf '%s\n' "$1" ;;
  esac
}

if [ "${1:-}" = --pick ]; then
  emit "$(night_pick "$2" "$3")"; exit $?
fi

if [ "${1:-}" = --self-test ]; then
  SELF="${BASH_SOURCE[0]}"
  d=$(mktemp -d) || exit 2
  trap 'rm -rf -- "${d:?}"' EXIT
  A=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; B=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
  NOW=1759802400   # 2025-10-07T02:00:00Z
  r() { printf '{"id":%s,"event":"%s","head_branch":"%s","created_at":"%s","status":"%s","head_sha":"%s"}' "$@"; }
  fx() { local n="$1"; shift; local IFS=,; printf '{"workflow_runs":[%s]}' "$*" > "$d/$n.json"; }
  fx tonight     "$(r 2 schedule main 2025-10-07T01:10:00Z in_progress $B)" "$(r 1 schedule main 2025-10-06T03:00:00Z completed $A)"
  fx completed   "$(r 2 schedule main 2025-10-07T00:30:00Z completed $B)"
  fx not-started "$(r 1 schedule main 2025-10-06T03:00:00Z completed $A)"
  fx future      "$(r 3 schedule main 2025-10-07T03:00:00Z queued $A)" "$(r 2 schedule main 2025-10-07T01:10:00Z in_progress $B)"
  fx dispatched  "$(r 3 workflow_dispatch main 2025-10-07T01:50:00Z completed $A)" "$(r 2 schedule main 2025-10-07T01:10:00Z completed $B)"
  fx other-branch "$(r 3 schedule release/0.71 2025-10-07T01:50:00Z completed $A)" "$(r 2 schedule main 2025-10-07T01:10:00Z completed $B)"
  fx no-sha      "$(r 2 schedule main 2025-10-07T01:10:00Z queued "")"
  fx empty       ""
  printf 'not json\n' > "$d/garbage.json"
  table() { # table <script> -> 0 all rows as expected
    local s="$1" bad=0 name want got rc
    while IFS='|' read -r name want; do
      [ -n "$name" ] || continue
      got=$(bash "$s" --pick "$d/$name.json" "$NOW" 2> /dev/null); rc=$?
      [ "$rc" = 0 ] || got="rc$rc"
      if [ "$got" = "$want" ]; then [ -n "${QUIET:-}" ] || printf '  ok    %-14s %s\n' "$name" "${got:0:12}"
      else [ -n "${QUIET:-}" ] || printf '  FAIL  %-14s got %s, want %s\n' "$name" "${got:0:12}" "${want:0:12}"; bad=1; fi
    done <<ROWS
tonight|$B
completed|$B
not-started|rc1
future|$B
dispatched|$B
other-branch|$B
no-sha|rc1
empty|rc1
garbage|rc2
ROWS
    return $bad
  }
  table "$SELF" || { echo "SELF-TEST FAILED: the shipped pick is wrong"; exit 1; }
  echo "self-test: 9/9 rows"
  # The pick decides nothing unless the producer checks out what it prints.
  WF="$(dirname "$SELF")/../../.github/workflows/dogfood-nightly.yml"
  if [ -f "$WF" ]; then
    for want in '          ref: ${{ needs.night.outputs.sha }}' '            sha=$(bash scripts/release/dogfood_night_sha.sh --now "$(date -u +%s)"); rc=$?' '    needs: night'; do
      grep -qxF -- "$want" "$WF" || { echo "SELF-TEST FAILED: dogfood-nightly.yml no longer has: $want"; exit 1; }
    done
    echo "self-test: dogfood-nightly.yml checks out the pick"
  fi
  bad=0
  mut() { # mut <label> <sed>
    local m="$d/m.sh"; sed "$2" "$SELF" > "$m"
    if cmp -s "$SELF" "$m"; then echo "SELF-TEST INCONCLUSIVE: mutant $1 changed nothing"; bad=1; return; fi
    if QUIET=1 table "$m"; then echo "SELF-TEST FAILED: mutant $1 SURVIVED"; bad=1; else echo "self-test: mutant $1 -- RED (expected)"; fi
  }
  mut any-event      's/select(.event == "schedule" and /select(/'
  mut any-branch     's/ and .head_branch == "main"//'
  mut future-allowed 's/ and ((.created_at | fromdateiso8601) <= \$t))\]/)]/'
  mut no-age-guard   's/^NIGHT_MAX_AGE=43200$/NIGHT_MAX_AGE=86400/'
  mut oldest-picked  's/(sort_by(.created_at) | last)/(sort_by(.created_at) | first)/'
  mut no-sha-ok      's/test("^\[0-9a-f\]{40}\$") | not/false/'
  [ "$bad" = 0 ] && echo "self-test: PASS -- red when the pick takes a dispatched, other-branch, future, oldest or sha-less run, or last night's"
  exit "$bad"
fi

[ "${1:-}" = --now ] && [[ "${2:-}" =~ ^[0-9]+$ ]] || { echo "usage: dogfood_night_sha.sh --now EPOCH | --pick F NOW | --self-test" >&2; exit 2; }
NOW="$2"
: "${GITHUB_REPOSITORY:?}"
f=$(mktemp) || exit 2
trap 'rm -f -- "${f:?}"' EXIT
gh api "repos/$GITHUB_REPOSITORY/actions/workflows/$NIGHT_WF/runs?event=schedule&branch=main&per_page=10" > "$f" 2> /dev/null \
  || { echo "dogfood_night_sha: listing the scheduled $NIGHT_WF runs failed" >&2; exit 2; }
emit "$(night_pick "$f" "$NOW")"
