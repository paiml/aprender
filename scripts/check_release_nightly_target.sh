#!/usr/bin/env bash
# check_release_nightly_target.sh — the case table for release_nightly_target in
# scripts/release/lib_release_params.sh (#4672): the version the nightly dogfood measures.
# Every row runs the lib in a child shell under `set -eo pipefail` (as a workflow step does) with
# `curl` and `gh` stubbed on PATH: FX_INDEX is the crates.io index body (or "down"), FX_MS the
# open milestones as "title number" lines (or "down"). Then each mutant is applied to a copy of
# the lib and must turn at least one row WRONG; a table that only ever saw the shipped lib is
# indistinguishable from one that reads nothing.
# Exit 0 = every row as expected and every mutant killed · 1 = a row or a mutant landed wrong.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LIB="$HERE/release/lib_release_params.sh"
T=$(mktemp -d) || exit 1
trap 'rm -rf "${T:?}"' EXIT
mkdir -p "$T/bin"
cat > "$T/bin/curl" <<'STUB'
#!/usr/bin/env bash
[ "$FX_INDEX" = down ] && { echo "curl: (6) Could not resolve host" >&2; exit 6; }
printf '%b' "$FX_INDEX"
STUB
cat > "$T/bin/gh" <<'STUB'
#!/usr/bin/env bash
[ "$FX_MS" = down ] && { echo "HTTP 502" >&2; exit 1; }
printf '%b' "$FX_MS"
STUB
chmod +x "$T/bin/curl" "$T/bin/gh"

v() { printf '{"name":"aprender","vers":"%s"}\\n' "$@"; }
IDX=$(v 0.69.4 0.70.0 0.70.1)          # crates.io max stable: 0.70.1
# the 09:47Z shape: thirteen open milestones, five at or below the max
MS13='0.69.3 30\n0.69.4 31\n0.69.5 32\n0.70.0 33\n0.70.1 34\n0.70.2 35\n0.70.3 36\n0.71.0 37\n0.72.0 38\n0.73.0 39\n0.74.0 40\n0.75.0 41\n0.76.0 42\n'

# run <lib> <index> <milestones> -> "rc=N out" from a set -eo pipefail child
run() {
  local out rc
  out=$(PATH="$T/bin:$PATH" FX_INDEX="$2" FX_MS="$3" bash -c 'set -eo pipefail; . "$1"; release_nightly_target' _ "$1" 2>/dev/null)
  rc=$?
  printf 'rc=%s %s' "$rc" "$out"
}
CASES="thirteen_open_picks_the_lowest_above|$IDX|$MS13|rc=0 0.70.2 35 0.70.1
order_is_semver_not_lexical|$IDX|0.70.10 50\n0.70.9 51\n|rc=0 0.70.9 51 0.70.1
the_published_version_is_not_a_target|$IDX|0.70.1 34\n|rc=2 
nothing_above_is_not_measured|$IDX|0.69.5 32\n0.70.0 33\n|rc=2 
no_open_milestone_is_not_measured|$IDX||rc=2 
non_version_titles_are_ignored|$IDX|backlog 9\n0.70.2-rc 10\nv0.70.2 11\n0.70.3 36\n|rc=0 0.70.3 36 0.70.1
a_prerelease_on_crates_io_is_not_the_max|$(v 0.70.1 0.71.0-rc.1)|0.70.2 35\n0.71.0 37\n|rc=0 0.70.2 35 0.70.1
an_index_with_no_stable_version_is_not_measured|$(v 0.1.0-alpha)|0.70.2 35\n|rc=2 
an_html_index_is_not_measured|<html>captive portal</html>\n|0.70.2 35\n|rc=2 
index_down_is_not_measured|down|0.70.2 35\n|rc=2 
milestones_down_is_not_measured|$IDX|down|rc=2 "

table() { # table <lib> -> number of WRONG rows; prints each row
  local lib="$1" wrong=0 name idx ms want got
  while IFS='|' read -r name idx ms want; do
    [ -n "$name" ] || continue
    got=$(run "$lib" "$idx" "$ms")
    if [ "$got" = "$want" ]; then printf '  ok    %-48s %s\n' "$name" "$got"
    else printf '  WRONG %-48s got [%s] want [%s]\n' "$name" "$got" "$want"; wrong=$((wrong + 1)); fi
  done <<< "$CASES"
  return "$wrong"
}

bad=0
table "$LIB" || bad=1
rows=$(grep -c '|' <<< "$CASES")
[ "$rows" -ge 11 ] || { printf 'VACUOUS %s row(s), fewer than the 11 declared\n' "$rows"; bad=1; }

# mutant <name> <sed expression>: applied to a copy of the lib, the table must go WRONG
mutant() {
  local m="$T/m-$1.sh"
  sed "$2" "$LIB" > "$m"
  if cmp -s "$LIB" "$m"; then printf '  INCONCLUSIVE mutant %s changed nothing\n' "$1"; bad=1; return; fi
  bash -n "$m" || { printf "  BROKEN   mutant %s does not parse\n" "$1"; bad=1; return; }
  if table "$m" > /dev/null; then printf '  SURVIVED mutant %s\n' "$1"; bad=1
  else printf "  killed   mutant %s (%s)\n" "$1" "$(table "$m" | grep -m1 -o "WRONG [a-z_]*")"; fi
}
mutant equal-is-above    's/^        \[ "\$t" != "\$max" \] && \[/        [/'
mutant lowest-lexical    's/"\$best" "\$t" | sort -V | head -n 1/"$best" "$t" | sort | head -n 1/'
mutant highest-not-lowest 's/"\$best" "\$t" | sort -V | head -n 1/"$best" "$t" | sort -V | tail -n 1/'
mutant any-title         '/^        \[\[ \$t /s/.*/        :/'
mutant prerelease-max    's/grep -oE .\"vers\":\"\[0-9\]+\\\.\[0-9\]+\\\.\[0-9\]+\". <<< "\$idx"/grep -oE "\\"vers\\":\\"[^\\"]*\\"" <<< "$idx"/'
mutant empty-is-a-target 's/^    \[ -n "\$best" \] || {/    true || {/'
mutant curl-trusted      's/ -sSf -A "aprender-release/ -sS -A "aprender-release/; s/\[ -n "\$max" \] || {/true || {/'

[ "$bad" = 0 ] && printf 'PASS  %s row(s) and every mutant killed: the nightly target is the lowest open X.Y.Z milestone above crates.io, else NOT_MEASURED (#4672)\n' "$rows"
exit "$bad"
