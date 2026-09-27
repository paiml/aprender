#!/usr/bin/env bash
# check_readme_numeric_claims.sh — #3867: every numeric claim in README.md is either
# GATED or EXPLICITLY EXEMPTED WITH A REASON. No claim may be neither.
#
# WHY THIS EXISTS. README.md is not ungated. `check_readme_claims.sh` enforces six
# claims and goes RED when they drift. The defect is SCOPE: the gate covers a REGION,
# and the boundary is invisible in the rendered Markdown. Measured at 766326360, the
# boundary runs THROUGH the `At HEAD` table, between adjacent rows —
#   L45  **111** CLI commands        gated
#   L46  **112** Book CLI chapters   NOT gated, and drifted (measured 113)
#   L47  **71**  Book lib chapters   NOT gated, and drifted (measured 72)
# Nothing distinguishes row 45 from row 46 to a reader, or to whoever adds row 48.
# #3769's cookbook count (341 vs 1825, wrong by 5.4x) escaped the same way.
#
# THE ANTI-VACUITY RULE, and it is why the `--self-test` plant below exists.
# An enumerator that parses ZERO claims and reports "all 0 claims dispositioned" is
# `check_package_includes.sh` reborn — CLAUDE.md records that one as printing
# "OK: All 0 include!() files are included" and being unable to fail. So: finding no
# claims is a FAILURE OF THE INSTRUMENT, never a pass. That check is the first thing
# this script does after parsing, and the first row of its case table.
#
# WHAT COUNTS AS A CLAIM. A number followed by a countable noun from NOUNS, inside or
# outside bold. Bold-only would miss `... (82 crates total)`, which is precisely the
# ungated claim that motivated the ticket.
#
# FENCED BLOCKS ARE SCANNED, NOT SKIPPED — and this was a defect in this script's own
# first draft. Stripping fences is the obvious rule ("sample output is not a claim"),
# and it silently hid BOTH remaining ungated claims: `... (82 crates total)` and
# `(111 subcommands)` live inside README.md's directory-tree diagram, which is fenced.
# A fence-skipping rule is an INVISIBLE COVERAGE BOUNDARY — the exact defect this
# ticket exists to fix, reproduced inside the fix. So fenced claims are enumerated
# like any other and must be dispositioned; genuine sample output gets an
# `exempt_because` that SAYS it is sample output. The boundary becomes a written
# exemption instead of an unwritten rule.
#
# THE KEY IS THE LINE, NOT THE VALUE. A claim normalises to its whole line with every
# number replaced by N, so the inventory does not churn when a count changes. Keying
# on the phrase alone was the first draft's other defect: `**112** chapters` and
# `**71** chapters` (Book CLI vs Book lib, adjacent rows of one table) collapsed to one
# key, so dispositioning either would have dispositioned both — a collision that would
# have let a drifted claim ride on its neighbour's gate.
#
# Usage:
#   bash scripts/check_readme_numeric_claims.sh              # enforce
#   bash scripts/check_readme_numeric_claims.sh --list       # print claims + disposition
#   bash scripts/check_readme_numeric_claims.sh --self-test  # the case table
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
README="${README_PATH:-$REPO_ROOT/README.md}"
INVENTORY="${INVENTORY_PATH:-$REPO_ROOT/contracts/readme-claims-v1.yaml}"

# The countable nouns a claim can be about. Deliberately a closed list: an open one
# ("any word after a number") turns every version string and byte count into a claim
# and the gate becomes noise that gets switched off.
NOUNS='workspace crates|crates|directories|directory|provable contracts|contracts|CLI commands|commands|chapters|worked examples|examples|recipes|rungs|model rungs|subcommands'

# Emit "line<TAB>normalised whole line" for every line stating a claim. Fenced lines
# are INCLUDED (see the header): a skip rule here is an unwritten boundary.
enumerate_claims() {
  grep -nE "(\*\*)?[0-9][0-9,]*(\*\*)? (${NOUNS})\b" "$README" \
  | sed -E 's/^([0-9]+):/\1\t/' \
  | while IFS=$'\t' read -r ln text; do
      norm=$(printf '%s\n' "$text" \
             | sed -E 's/\*\*[0-9][0-9,]*\*\*/**N**/g; s/([^*0-9])[0-9][0-9,]*/\1N/g' \
             | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//' \
             | cut -c1-100)
      printf '%s\t%s\n' "$ln" "$norm"
    done
}

# Disposition for a normalised phrase, read from the inventory:
#   "gated:<claim>"  |  "exempt:<reason>"  |  "" (absent)
disposition_for() {
  local norm="$1"
  python3 - "$INVENTORY" "$norm" <<'PY'
import sys, yaml
inv_path, norm = sys.argv[1], sys.argv[2]
try:
    doc = yaml.safe_load(open(inv_path)) or {}
except Exception as e:
    print("ERR:%s" % e); sys.exit(0)
inv = ((doc.get("equations") or {}).get("numeric_claims_inventory") or {}).get("claims") or {}
ent = inv.get(norm)
if not isinstance(ent, dict):
    print("")
elif ent.get("gated_by"):
    print("gated:%s" % ent["gated_by"])
elif ent.get("exempt_because"):
    print("exempt:%s" % ent["exempt_because"])
else:
    print("MALFORMED")
PY
}

mode=enforce
for a in "${@:-}"; do
  case "$a" in
    --list) mode=list ;;
    --self-test) mode=selftest ;;
    "") ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

if [[ "$mode" = selftest ]]; then
  TD=$(mktemp -d)
  # SEC011: the delete is guarded — an empty or non-temp TD is left alone. Same idiom
  # as scripts/dogfood.sh::_rm_worklog; an unguarded `rm -rf "$VAR"` in a trap is one
  # typo away from deleting the tree it was meant to clean up after.
  _rm_td() {
    local v="${TD:-}"
    case "$v" in /tmp/?*|/var/folders/?*) ;; *) return 0 ;; esac
    [ -n "$v" ] && [ "$v" != "/" ] && rm -rf -- "$v" || :
  }
  trap _rm_td EXIT
  n=0; red=0
  row() { # row <want rc> <label> ; README fixture already written to $TD/README.md
    local want=$1 label=$2 rc=0
    n=$((n + 1))
    env README_PATH="$TD/README.md" INVENTORY_PATH="${FIX_INV:-$INVENTORY}" \
      bash "$0" >"$TD/out.$n" 2>&1 || rc=$?
    if [ "$rc" = "$want" ]; then printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
    else printf 'FAIL  row %-2s rc=%s (wanted %s)  %s\n' "$n" "$rc" "$want" "$label"; sed 's/^/        /' "$TD/out.$n"; red=1; fi
  }

  # ROW 1 IS THE ANTI-VACUITY PLANT. It is first because it is the one that decides
  # whether any other row means anything: an enumerator that finds nothing must FAIL,
  # not report "all 0 claims dispositioned".
  printf '# apr\n\nNo numbers here at all.\n' > "$TD/README.md"
  row 1 "README with ZERO numeric claims: RED (an enumerator that finds nothing is broken)"
  grep -q 'parsed 0' "$TD/out.1" || { printf 'FAIL  row 1  did not say it parsed zero\n'; red=1; }

  # A claim nobody dispositioned.
  printf '# apr\n\nThis repo has 42 recipes.\n' > "$TD/README.md"
  row 1 "an UNDISPOSITIONED claim: RED"

  # THE FENCE ROW. This asserts the property that this script's first draft got
  # wrong: a number inside a fenced block IS enumerated, so an undispositioned one
  # is RED. Skipping fences is the obvious rule and it silently hid both remaining
  # ungated claims (the directory-tree diagram's crate count and subcommand count).
  # If someone reinstates a fence-skip, this row goes green and catches them.
  printf '# apr\n\n```\n999 recipes\n```\n' > "$TD/README.md"
  row 1 "an undispositioned claim INSIDE a fence: RED (fences are scanned, not skipped)"

  # The real README must be fully dispositioned.
  cp "$REPO_ROOT/README.md" "$TD/README.md"
  row 0 "the committed README: every claim gated or exempted"

  printf '%s/%s rows\n' "$((n - red))" "$n"
  [ "$red" = 0 ] || exit 1
  exit 0
fi

claims=$(enumerate_claims || true)
count=$(printf '%s' "$claims" | grep -c . || true)

# ANTI-VACUITY. Before any verdict about dispositions.
if [[ "$count" -eq 0 ]]; then
  echo "FAIL FALSIFY-README-008 numeric_claims: parsed 0 numeric claims from $README — an enumerator that finds nothing has not verified anything. Either the README genuinely states no counts (then this gate is pointless and should be removed deliberately), or the pattern broke." >&2
  exit 1
fi

fail=0
while IFS=$'\t' read -r ln norm; do
  [ -n "$norm" ] || continue
  d=$(disposition_for "$norm")
  case "$d" in
    gated:*)  [[ "$mode" = list ]] && printf '  L%-5s %-34s GATED by %s\n' "$ln" "$norm" "${d#gated:}" ;;
    exempt:*) [[ "$mode" = list ]] && printf '  L%-5s %-34s exempt: %s\n' "$ln" "$norm" "${d#exempt:}" ;;
    MALFORMED)
      echo "FAIL FALSIFY-README-008 numeric_claims: L$ln '$norm' has an inventory entry with neither gated_by nor exempt_because — an exemption with no reason is the seam #3769 refused" >&2
      fail=1 ;;
    ERR:*)
      echo "FAIL FALSIFY-README-008 numeric_claims: cannot read $INVENTORY (${d#ERR:})" >&2
      exit 1 ;;
    *)
      echo "FAIL FALSIFY-README-008 numeric_claims: L$ln states '$norm' and the inventory does not disposition it. Add it to equations.numeric_claims_inventory.claims with either gated_by: <claim> or exempt_because: <reason>." >&2
      fail=1 ;;
  esac
done <<< "$claims"

if [[ "$mode" = list ]]; then exit 0; fi
if [[ "$fail" = 0 ]]; then
  echo "PASS FALSIFY-README-008 numeric_claims: $count claim(s), all gated or exempted with a reason"
fi
exit "$fail"
