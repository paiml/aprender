#!/usr/bin/env bash
# coverage_receipt.sh -- the coverage receipt coverage-nightly writes, keyed by the commit it measured
# (#4672, RQ-1's producer half). scripts/release/tag_coverage_gate.sh reads it for the release commit
# (#4734): "commit H, floor F, pct P over N passed tests", or "not measured".
#
# Usage: coverage_receipt.sh <coverage log> <sha> <floor> <out dir> <since>
#   <coverage log>  the `make coverage` output; its LAST "TOTAL: LH/LF lines covered" line is the number,
#                   and the sum of its libtest "test result: ... N passed;" lines is the test count (#4734)
#   <sha>           the 40-hex commit the run measured (git rev-parse HEAD in the job, never a ref name)
#   <floor>         COV_FLOOR as `make` resolves it, or anything else for "unknown"
#   <out dir>       the receipt lands at <out dir>/coverage-receipt-<sha>.json
#   <since>         a file touched before the coverage step: a log NOT newer than it is a previous run's
#
# A receipt is ALWAYS written for a valid sha. When the number cannot be trusted it says
# "status":"not_measured" with a reason and "pct":null -- never a stale or partial number (L25).
# Exit 0 receipt written (measured or not_measured) · 2 usage error, nothing written.
set -uo pipefail

[ "$#" -eq 5 ] || { echo "usage: coverage_receipt.sh <log> <sha> <floor> <out dir> <since>" >&2; exit 2; }
log=$1 sha=$2 floor=$3 out=$4 since=$5
[[ "$sha" =~ ^[0-9a-f]{40}$ ]] || { echo "coverage_receipt: sha '$sha' is not a 40-hex commit" >&2; exit 2; }
mkdir -p "$out" || exit 2

status=measured reason="" covered=null total=null pct=null passed=null
[[ "$floor" =~ ^(0|[1-9][0-9]*)(\.[0-9]+)?$ ]] || floor=null   # a JSON number: no leading zero

not_measured() { status=not_measured; reason=$1; covered=null total=null pct=null passed=null; }

if [ ! -s "$log" ]; then
  not_measured "no coverage log"
elif [ ! -e "$since" ] || ! [ "$log" -nt "$since" ]; then
  not_measured "coverage log is not from this run"
elif grep -q 'DID NOT MEASURE' "$log"; then
  not_measured "make coverage did not measure"
else
  # #4734: v0.70.1's tag coverage section went green having run 0 tests. Count what ran.
  passed=$(sed -nE 's/^test result: (ok|FAILED)\. ([0-9]+) passed;.*/\2/p' "$log" | awk '{ s += $1 } END { print s + 0 }')
  line=$(grep -E '^TOTAL: [0-9]+/[0-9]+ lines covered' "$log" | tail -n 1)
  if [[ "$line" =~ ^TOTAL:\ ([0-9]+)/([0-9]+)\  ]]; then
    covered=$((10#${BASH_REMATCH[1]})) total=$((10#${BASH_REMATCH[2]}))
    if [ "$total" -eq 0 ]; then
      not_measured "0 instrumented lines"
    elif [ "$passed" -eq 0 ]; then
      not_measured "0 tests ran"
    elif [ "$covered" -gt "$total" ]; then
      not_measured "covered exceeds total"
    else
      # Two decimals, truncated (never rounded up past the floor): 786448/885829 -> 88.78.
      bp=$((covered * 10000 / total))
      pct=$(printf '%d.%02d' $((bp / 100)) $((bp % 100)))
    fi
  else
    not_measured "no TOTAL line in the coverage log"
  fi
fi

f="$out/coverage-receipt-$sha.json"
if [ "$status" = measured ]; then rs=null; else rs="\"$reason\""; fi
printf '{"schema":"coverage-receipt/v1","sha":"%s","floor":%s,"pct":%s,"covered":%s,"total":%s,"status":"%s","reason":%s,"passed":%s}\n' \
  "$sha" "$floor" "$pct" "$covered" "$total" "$status" "$rs" "$passed" > "$f" || exit 2
cat "$f"
