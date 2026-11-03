#!/usr/bin/env bash
# #6721-class falsifier: the guards that refuse a mktemp -d outside /tmp must also accept
# "${TMPDIR:-/tmp}" children (RS-5 per-runner temp). --mutants proves each file's check can fail.
set -euo pipefail
cd "$(dirname "$0")/../.."
FILES=(scripts/check_cascade_clean_room_gate.sh scripts/check_comparator_one_client.sh
    scripts/check_ladder_write_errors.sh scripts/check_perf_receipt_fields_have_producers.sh
    scripts/perf_gate.sh scripts/perf_receipt_sign.sh scripts/spec_conformance.sh
    scripts/release/carry_milestone_items.sh scripts/check_thresholds_in_matrix.sh
    scripts/check_perf_matrix_schema.sh)

# ok <file> : a case arm lists "${TMPDIR:-/tmp}", and the /tmp arm is still there
ok() { grep -qE '\|"\$\{TMPDIR:-/tmp\}"/(\?)?\*\)' "$1" && grep -qE '/tmp/(\?)?\*' "$1"; }

# fast behaviour: the cascade gate completes with a scratch dir outside /tmp
behave() {
    local d; d=$(mktemp -d); trap 'rm -rf "${d:?}"' RETURN
    mkdir "$d/scratch"
    TMPDIR=$d/scratch bash scripts/check_cascade_clean_room_gate.sh >/dev/null 2>&1
}

check() {
    local f bad=0
    for f in "${FILES[@]}"; do ok "$f" || { echo "FAIL: $f has no \${TMPDIR:-/tmp} arm"; bad=1; }; done
    return "$bad"
}

if [ "${1:-}" = "--mutants" ]; then
    check || exit 1
    d=$(mktemp -d); trap 'rm -rf "${d:?}"' EXIT
    for f in "${FILES[@]}"; do
        sed 's#|"\${TMPDIR:-/tmp}"/#|#' "$f" > "$d/m.sh"
        cmp -s "$f" "$d/m.sh" && { echo "FAIL: mutant of $f did not change it"; exit 1; }
        if ok "$d/m.sh"; then echo "FAIL: mutant of $f survived"; exit 1; fi
        echo "mutant $f: killed"
    done
    exit 0
fi
check && behave && echo "PASS: 10 guards accept \${TMPDIR:-/tmp}"
