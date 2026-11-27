#!/usr/bin/env bash
# runner_capacity.sh - read-only probe: are enough self-hosted X64 runners online?
# Usage: runner_capacity.sh [--min N] [--label L] [--json FILE] | --self-test
# Exit 0: >= N online runners carrying X64 and label L (default N=3, L=clean-room).
# Exit 1: fewer. Exit 2: could not measure (not_measured; make no decision on it).
set -euo pipefail

REPO="${RUNNER_CAPACITY_REPO:-paiml/aprender}"

# count_online <json-file> <label>: online runners with X64 and <label>.
count_online() {
  jq --arg l "$2" '(.runners | if type == "array" then . else error("runners is not a list") end)
    | [.[]
    | select(.status == "online")
    | select([.labels[].name] | (index("X64") != null and index($l) != null))] | length' "$1"
}

report() { # <json> <label> <min>
  local n rows
  { n=$(count_online "$1" "$2") && [[ $n =~ ^[0-9]+$ ]] &&
    rows=$(jq -r --arg l "$2" '.runners[] | "  \(.name) \(.status) has_label=\([.labels[].name]|index($l) != null)"' "$1"); } || { echo "runner_capacity: not_measured (unreadable runner list)" >&2; return 2; }
  [ -z "$rows" ] || printf '%s\n' "$rows"
  echo "runner_capacity: ${n} online X64 '$2' runners, need >= $3"
  [ "$n" -ge "$3" ]
}

self_test() {
  local d rc i self="${BASH_SOURCE[0]}"
  d=$(mktemp -d)
  trap 'rm -rf "${d:?}"' RETURN
  mk() { # <file> <online-count of 5>
    local i s lab='[{"name":"X64"},{"name":"clean-room"}]' out='' sep=''
    for i in 1 2 3 4 5; do
      s=offline; [ "$i" -le "$2" ] && s=online
      out="${out}${sep}{\"name\":\"r$i\",\"status\":\"$s\",\"labels\":$lab}"; sep=,
    done
    printf '{"runners":[%s]}' "$out" > "$1"
  }
  mk "$d/one.json" 1; mk "$d/three.json" 3
  printf '{"runners":[{"name":"k","status":"online","labels":[{"name":"X64"},{"name":"pr-x64"}]}]}' > "$d/nolabel.json"
  rc=0; report "$d/one.json" clean-room 3 >/dev/null || rc=$?
  [ "$rc" -eq 1 ] || { echo "FAIL: 1 online must exit 1, got $rc"; return 1; }
  rc=0; report "$d/three.json" clean-room 3 >/dev/null || rc=$?
  [ "$rc" -eq 0 ] || { echo "FAIL: 3 online must exit 0, got $rc"; return 1; }
  rc=0; report "$d/nolabel.json" clean-room 1 >/dev/null || rc=$?
  [ "$rc" -eq 1 ] || { echo "FAIL: unlabelled runner must not count, got $rc"; return 1; }
  printf '{"runners":[{"name":"a","status":"online","labels":[{"name":"clean-room"}]}]}' > "$d/nox64.json"
  rc=0; report "$d/nox64.json" clean-room 1 >/dev/null || rc=$?
  [ "$rc" -eq 1 ] || { echo "FAIL: a runner without X64 must not count, got $rc"; return 1; }
  # cli <name> <want-exit> <stderr text> [args]: run this script with the stub gh first on PATH
  cli() {
    local name=$1 want=$2 text=$3 got=0; shift 3
    PATH="$d:$PATH" bash "$self" "$@" >/dev/null 2>"$d/err" || got=$?
    { [ "$got" -eq "$want" ] && grep -qF -- "$text" "$d/err"; } || { echo "FAIL: $name: want exit $want and '$text' on stderr, got exit $got"; return 1; }
  }
  # a gh that prints a full list and then fails: unreadable is not_measured, never a capacity verdict
  printf '#!/bin/sh\ncat "%s"\nexit 1\n' "$d/three.json" > "$d/gh"; chmod +x "$d/gh"
  cli 'gh fails' 2 'not_measured (gh api failed)' || return 1
  TMPDIR="$d/absent" cli 'mktemp fails' 2 'not_measured (mktemp failed)' || return 1
  for i in --min --label --json; do
    cli "$i with no value" 2 'usage:' "$i" || return 1
    cli "$i with an empty value" 2 'usage:' "$i" '' || return 1
  done
  cli '--min not a number' 2 'usage:' --min abc || return 1
  printf '{"runners":{}}' > "$d/notalist.json"
  printf '{"runners":[{"name":"r","status":"offline","labels":null}]}' > "$d/nulllabels.json"
  : > "$d/empty.json"
  for i in notalist nulllabels empty; do
    cli "$i list" 2 'not_measured (unreadable runner list)' --json "$d/$i.json" || return 1
  done
  echo "runner_capacity self-test: PASS (16 cases)"
}

usage() { echo "usage: $0 [--min N] [--label L] [--json FILE] | --self-test" >&2; exit 2; }

min=3; label=clean-room; file=''
while [ $# -gt 0 ]; do
  case "$1" in
    --self-test) self_test; exit $? ;;
    --min) case "${2:-}" in ''|*[!0-9]*) usage ;; esac; min="$2"; shift 2 ;;
    --label) [ -n "${2:-}" ] || usage; label="$2"; shift 2 ;;
    --json) [ -n "${2:-}" ] || usage; file="$2"; shift 2 ;;
    *) usage ;;
  esac
done

if [ -z "$file" ]; then
  file=$(mktemp) || { echo "runner_capacity: not_measured (mktemp failed)" >&2; exit 2; }
  trap 'rm -f "${file:?}"' EXIT
  gh api "repos/${REPO}/actions/runners?per_page=100" > "$file" || { echo "runner_capacity: not_measured (gh api failed)" >&2; exit 2; }
fi
report "$file" "$label" "$min"
