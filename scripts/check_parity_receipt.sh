#!/usr/bin/env bash
# check_parity_receipt.sh — the parity validator must DISCRIMINATE before any
# release reads its verdict.
#
# The rule this repo keeps relearning: a validator that has only ever seen
# valid input is indistinguishable from `exit 0`. Every case below is a receipt
# shape that ALREADY HAPPENED here or is one edit away from happening, and the
# annotation on each fixture says which.
#
# The one that matters most is 04/05/06/07 — #2696. The published apr takes the
# CPU path even when handed --gpu, so measuring it against a CUDA llama.cpp
# yields 0.099x. That number is not a kernel defect and must not be reportable
# as one: 04 is the honest way to write it, and 05/06/07 are the three ways to
# write it dishonestly.
set -euo pipefail

VALIDATOR="scripts/lib/bench_receipt.py"
CASES="scripts/lib/parity_receipt_cases"
MIN_CASES=17

rc=0
printf -- '--- parity receipt validator --------------------------------------\n'

[ -f "$VALIDATOR" ] || { printf 'FAIL  %s is missing\n' "$VALIDATOR"; exit 2; }
[ -d "$CASES" ]     || { printf 'FAIL  %s is missing\n' "$CASES"; exit 2; }

n=0
for f in "$CASES"/*.json; do
    [ -e "$f" ] || break
    n=$((n + 1))
    expect=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['_expect']['result'])" "$f")
    why=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['_expect']['why'])" "$f")
    if python3 "$VALIDATOR" --parity "$f" >/dev/null 2>&1; then got=valid; else got=invalid; fi
    if [ "$expect" = "$got" ]; then
        printf 'ok    %-48s %-8s %s\n' "$(basename "$f" .json)" "$got" "$why"
    else
        printf 'FAIL  %-48s expected %s, got %s\n' "$(basename "$f" .json)" "$expect" "$got"
        rc=1
    fi
done

# VACUITY. A table that shrinks sweeps clean, and the count is asserted rather
# than trusted — the same reason check_bench_receipt.sh carries a floor.
if [ "$n" -lt "$MIN_CASES" ]; then
    printf 'FAIL  the case table has %s case(s); at least %s are required.\n' "$n" "$MIN_CASES"
    printf '      A shrinking table is a validator with less to discriminate.\n'
    rc=1
fi

# BOTH DIRECTIONS. A table of only-invalid cases is passed by a validator that
# rejects everything, which is as useless as one that accepts everything.
valid_n=$(grep -l '"result": "valid"' "$CASES"/*.json 2>/dev/null | wc -l | tr -d ' ')
invalid_n=$((n - valid_n))
if [ "$valid_n" -lt 3 ] || [ "$invalid_n" -lt 8 ]; then
    printf 'FAIL  the table is one-sided: %s valid / %s invalid. A validator that\n' "$valid_n" "$invalid_n"
    printf '      rejects everything passes an all-invalid table.\n'
    rc=1
fi

printf '\n'
if [ "$rc" -eq 0 ]; then
    printf 'PASS  %s cases (%s valid / %s invalid): the validator separates a\n' "$n" "$valid_n" "$invalid_n"
    printf '      real parity receipt from every fabricated shape in the table.\n'
else
    printf 'FAIL  see rows above (#2696).\n'
fi
exit "$rc"
