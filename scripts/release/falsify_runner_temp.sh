#!/usr/bin/env bash
# RS-5 falsifier: nightly workflows and models_t1.sh must not use fixed /tmp names
# a concurrent job can sweep; they live under $RUNNER_TEMP. --mutants proves the check can fail.
set -euo pipefail
cd "$(dirname "$0")/../.."
FILES=(.github/workflows/cuda-nightly.yml .github/workflows/coverage-nightly.yml .github/workflows/nightly-bench.yml)
BAD='/tmp/(cov\.log|gaps\.txt|bench\.log|pp26-prev-witness)'

# scan <file...> : rc 1 and a line per hit when a fixed /tmp name is used outside a comment
scan() {
    local f hits=0
    for f in "$@"; do
        if grep -nE "^[^#]*${BAD}" "$f"; then hits=1; fi
    done
    return "$hits"
}

check() {
    scan "${FILES[@]}" || { echo "FAIL: fixed /tmp name in nightly workflow" >&2; return 1; }
    if ! grep -qE 'mktemp -d -p "\$\{RUNNER_TEMP:-' scripts/release/models_t1.sh; then
        echo "FAIL: models_t1.sh mt_self_test mktemp not under RUNNER_TEMP" >&2; return 1
    fi
    if grep -nE '^[[:space:]]*d=\$\(mktemp -d\)' scripts/release/models_t1.sh; then
        echo "FAIL: bare mktemp -d in models_t1.sh" >&2; return 1
    fi
    echo "PASS: no fixed /tmp names"
}

mutants() {
    local d f rc=0
    d=$(mktemp -d -p "${RUNNER_TEMP:-${TMPDIR:-/tmp}}") || exit 2
    for f in "${FILES[@]}"; do
        sed -E 's#"\$RUNNER_TEMP"/#/tmp/#' "$f" > "$d/m.yml"
        if scan "$d/m.yml" >/dev/null; then echo "MUTANT SURVIVED: $f"; rc=1; fi
    done
    echo "mutants: $([ "$rc" = 0 ] && echo killed || echo SURVIVED)"
    return "$rc"
}

case "${1:-}" in
    --mutants) mutants ;;
    *) check ;;
esac
