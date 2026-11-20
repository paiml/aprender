#!/usr/bin/env bash
# runner_capacity.sh - read-only probe: are enough self-hosted X64 runners online?
# Usage: runner_capacity.sh [--min N] [--label L] [--json FILE] | --self-test
# Exit 0: >= N online runners carrying X64 and label L (default N=3, L=clean-room).
# Exit 1: fewer. Exit 2: could not measure (not_measured; make no decision on it).
set -euo pipefail

REPO="${RUNNER_CAPACITY_REPO:-paiml/aprender}"

# count_online <json-file> <label>: online runners with X64 and <label>.
count_online() {
  jq --arg l "$2" '[.runners[]
    | select(.status == "online")
    | select([.labels[].name] | (index("X64") != null and index($l) != null))] | length' "$1"
}

report() { # <json> <label> <min>
  local n
  n=$(count_online "$1" "$2") || return 2
  jq -r --arg l "$2" '.runners[] | "  \(.name) \(.status) has_label=\([.labels[].name]|index($l) != null)"' "$1"
  echo "runner_capacity: ${n} online X64 '$2' runners, need >= $3"
  [ "$n" -ge "$3" ]
}

self_test() {
  local d rc
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
  printf 'not json' > "$d/bad.json"
  rc=0; report "$d/bad.json" clean-room 3 >/dev/null 2>&1 || rc=$?
  [ "$rc" -eq 2 ] || { echo "FAIL: unreadable list must exit 2, got $rc"; return 1; }
  rc=0; report "$d/missing.json" clean-room 3 >/dev/null 2>&1 || rc=$?
  [ "$rc" -eq 2 ] || { echo "FAIL: missing list must exit 2, got $rc"; return 1; }
  echo "runner_capacity self-test: PASS (5 cases)"
}

min=3; label=clean-room; file=''
while [ $# -gt 0 ]; do
  case "$1" in
    --self-test) self_test; exit $? ;;
    --min) min="$2"; shift 2 ;;
    --label) label="$2"; shift 2 ;;
    --json) file="$2"; shift 2 ;;
    *) echo "usage: $0 [--min N] [--label L] [--json FILE] | --self-test" >&2; exit 2 ;;
  esac
done

if [ -z "$file" ]; then
  file=$(mktemp)
  trap 'rm -f "${file:?}"' EXIT
  gh api "repos/${REPO}/actions/runners?per_page=100" > "$file" || { echo "runner_capacity: not_measured (gh api failed)" >&2; exit 2; }
fi
report "$file" "$label" "$min"
