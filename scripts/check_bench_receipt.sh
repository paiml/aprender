#!/usr/bin/env bash
#
# check_bench_receipt.sh — the bench receipt validator, proven against its own
# case table before it is believed (PARITY-001 / PARITY-002, aprender#2668/#2669).
#
# WHY A CASE TABLE AND NOT A REVIEW. The verifier-pinning walker in this repo
# was wrong SIXTEEN times; every one was caught by a must-flag/must-not-flag
# table and none by reading the pattern. A validator whose own discrimination
# is unproven is a validator that can go all-GREEN or all-RED without anyone
# noticing, and either failure looks exactly like health.
#
# THE TWO RULES THIS EXISTS TO ENFORCE:
#
#   1. `compute_class` is REQUIRED and describes the path TAKEN, not the
#      hardware present. A receipt proving WHICH BINARY ran but not WHICH
#      PATH it took catches the wrong-binary class -- five in-tree
#      rediscoveries -- and misses the wrong-compute-class one entirely.
#      That miss is how a CPU-only apr side against a CUDA comparator
#      validates cleanly and reports the fabricated 14x regression already
#      documented at crates/apr-cli/src/dispatch.rs:165.
#
#   2. A cross-class run CANNOT carry a threshold. Born-disarmed made
#      unwriteable rather than discouraged.
#
# Exit: 0 = the table discriminates AND every real receipt is valid
#       1 = a case behaved wrongly, or a real receipt is invalid
#       2 = setup error
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

VALIDATOR="scripts/lib/bench_receipt.py"
CASES="scripts/lib/bench_receipt_cases"
rc=0

[ -f "$VALIDATOR" ] || { printf 'FAIL  %s is missing\n' "$VALIDATOR"; exit 2; }
[ -d "$CASES" ]     || { printf 'FAIL  %s is missing\n' "$CASES"; exit 2; }

printf -- '--- bench receipt validator -------------------------------------\n'
printf 'case table (the validator must be right before its verdict means anything)\n'

n=0; bad=0
for f in "$CASES"/*.json; do
    [ -e "$f" ] || continue
    n=$((n + 1))
    want=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('_expect','?'))" "$f" 2>/dev/null)
    python3 "$VALIDATOR" "$f" >/dev/null 2>&1
    got=$([ $? -eq 0 ] && echo GREEN || echo RED)
    if [ "$got" != "$want" ]; then
        printf 'FAIL  %-28s expected %s, got %s\n' "$(basename "$f" .json)" "$want" "$got"
        bad=$((bad + 1))
    fi
done

# VACUITY: a table that matched nothing would sweep clean. The floor is the
# committed case count; it may grow and may never shrink silently.
if [ "$n" -lt 14 ]; then
    printf 'FAIL  case table has %s case(s); at least 14 are required. A shrinking\n' "$n"
    printf '      table silently narrows what "the validator discriminates" means.\n'
    rc=1
elif [ "$bad" -eq 0 ]; then
    printf 'ok    all %s cases behaved as declared (RED on each defect, GREEN on each control)\n' "$n"
else
    rc=1
fi

# Every real receipt in the tree must validate.
printf '\nreal receipts\n'
found=0
while IFS= read -r r; do
    found=$((found + 1))
    if python3 "$VALIDATOR" "$r" >/dev/null 2>&1; then
        printf 'ok    %s\n' "$r"
    else
        printf 'FAIL  %s\n' "$r"
        python3 "$VALIDATOR" "$r" 2>&1 | sed 's/^/      /'
        rc=1
    fi
done < <(find evidence/dogfood -name 'bench-*.json' -type f 2>/dev/null | sort)
# SCOPED to evidence/dogfood/ deliberately. A bare `find evidence -name
# bench-*.json` also matches evidence/p2c-*/bench-epoch-NNN.json, which are
# TRAINING-EPOCH artifacts from a different subsystem with a different schema
# -- validating them here would report a defect in a file this validator has
# no claim over. Found by running this guard before believing it.
[ "$found" -eq 0 ] && printf 'none yet — the first arrives with PARITY-003 (aprender#2670)\n'

printf '\n'
if [ "$rc" -eq 0 ]; then
    printf 'PASS  the validator discriminates, and every receipt present is valid.\n'
else
    printf 'FAIL  see rows above. A receipt that cannot be trusted is not evidence.\n'
fi
exit "$rc"
