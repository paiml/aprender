#!/usr/bin/env bash
# RS-5 falsifier: nightly workflows, their guard and models_t1.sh must not use fixed /tmp names
# a concurrent job can sweep (C61-7 adds book, deploy-apr-install, mutants-nightly, qwen-story-daily); they live under $RUNNER_TEMP. --mutants proves each check can fail.
set -euo pipefail
cd "$(dirname "$0")/../.."
FILES=(.github/workflows/cuda-nightly.yml .github/workflows/coverage-nightly.yml .github/workflows/nightly-bench.yml scripts/check_coverage_receipt.sh
       .github/workflows/book.yml .github/workflows/deploy-apr-install.yml .github/workflows/mutants-nightly.yml .github/workflows/qwen-story-daily.yml)
M=scripts/release/models_t1.sh
BAD='/tmp/(cov\.log|gaps\.txt|bench\.log|pp26-prev-witness|bashrs\.log|deployed-install\.sh|mi\.log|story\.log|qwen-story-artifacts|prev/|issue-body\.md)'

# scan <file...> : rc 1 and a line per hit when a fixed /tmp name is used outside a comment
scan() {
    local f hits=0
    for f in "$@"; do
        if grep -nE "^[^#]*${BAD}" "$f"; then hits=1; fi
    done
    return "$hits"
}

# models_ok <file> : the self-test fixture dir is under RUNNER_TEMP and no bare mktemp -d remains
models_ok() {
    grep -qE 'mktemp -d -p "\$\{RUNNER_TEMP:-' "$1" || return 1
    if grep -qE '(^|[[:space:]])d=\$\(mktemp -d\)' "$1"; then return 1; fi
}

check() {
    scan "${FILES[@]}" || { echo "FAIL: fixed /tmp name" >&2; return 1; }
    models_ok "$M" || { echo "FAIL: models_t1.sh mktemp not under RUNNER_TEMP" >&2; return 1; }
    echo "PASS: no fixed /tmp names"
}

mutants() {
    local d f rc=0
    d=$(mktemp -d -p "${RUNNER_TEMP:-${TMPDIR:-/tmp}}") || exit 2
    trap 'rm -rf -- "${d:?}"' RETURN
    for f in "${FILES[@]}"; do
        sed -E 's#"\$RUNNER_TEMP"/#/tmp/#g; s#\$\{\{ runner\.temp \}\}/#/tmp/#g' "$f" > "$d/m"
        if scan "$d/m" >/dev/null; then echo "MUTANT SURVIVED: $f"; rc=1; fi
    done
    sed -E 's# -p "\$\{RUNNER_TEMP:-\$\{TMPDIR:-/tmp\}\}"##' "$M" > "$d/m"
    if models_ok "$d/m"; then echo "MUTANT SURVIVED: models_t1 drop -p"; rc=1; fi
    sed -E 's#mktemp -d -p "[^"]*"#mktemp -d#' "$M" > "$d/m"
    if models_ok "$d/m"; then echo "MUTANT SURVIVED: models_t1 bare mktemp"; rc=1; fi
    echo "mutants: $([ "$rc" = 0 ] && echo killed || echo SURVIVED)"
    return "$rc"
}

case "${1:-}" in
    --mutants) mutants ;;
    *) check ;;
esac
