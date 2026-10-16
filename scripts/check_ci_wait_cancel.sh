#!/usr/bin/env bash
# check_ci_wait_cancel.sh -- a cancel must stop the fat job's wait step, and the
# teardown must still run (13:46Z 2026-10-16, infra-6a).
#
# THE DEFECT. x86-main's step "Sections (wait for every one to finish)" was
# `if: always()`. always() is true on a cancelled run, so a cancel could not stop
# the `fat_driver.py wait` poll; orphan merge-group runs held queue runners until a
# force-cancel. `${{ !cancelled() }}` runs the wait after a failed step (its
# results are still published) and skips it on a cancel. The teardown step keeps
# `if: always()`: a cancelled run never reaches a section's own container removal,
# and fat_teardown.sh reaps the detached containers (#4948).
#
# Usage
#   bash scripts/check_ci_wait_cancel.sh [FILE]   check .github/workflows/ci.yml
#   bash scripts/check_ci_wait_cancel.sh --self-test
# Exit 0 ok, 1 a rule broke, 2 the file cannot be modelled.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# if_of <file> <run-substring>: the `if:` line of the step whose `run:` holds the substring.
if_of() {
    awk -v pat="$2" '
        /^      - / { cond = ""; if (index($0, "name:")) {} }
        /^        if: / { cond = $0; sub(/^        if: /, "", cond) }
        index($0, "run:") && index($0, pat) { print cond; found = 1 }
        index($0, "fat_driver.py wait") && pat == "fat_driver.py wait" && !found { print cond; found = 1 }
    ' "$1" | head -n 1
}

check() { # <file>
    local f=$1 w t rc=0
    w=$(if_of "$f" 'fat_driver.py wait') || true
    t=$(if_of "$f" 'fat_teardown.sh') || true
    if [ -z "$w" ] && ! grep -q 'fat_driver.py wait' "$f"; then echo "ENV: no fat_driver.py wait step in $f" >&2; return 2; fi
    case "$w" in
        *always\(\)*) echo "FAIL wait step is always(): a cancel cannot stop it ($w)"; rc=1 ;;
        *'!cancelled()'*) echo "ok   wait step is cancel-aware ($w)" ;;
        *) echo "FAIL wait step has no !cancelled() guard (if: '${w:-none}')"; rc=1 ;;
    esac
    case "$t" in
        *always\(\)*) echo "ok   teardown keeps always() ($t)" ;;
        *) echo "FAIL teardown must be always() so a cancel still reaps the containers (if: '${t:-none}')"; rc=1 ;;
    esac
    return "$rc"
}

self_test() {
    local td rc=0 out r
    td=$(mktemp -d "${TMPDIR:-/tmp}/ci-wait-cancel.XXXXXX")
    mk() { # <wait-if> <teardown-if> <file>
        printf '      - name: Sections\n        id: wait\n        if: %s\n        run: python3 scripts/ci/fat_driver.py wait --results x\n      - name: Remove\n        if: %s\n        run: if [ -f scripts/ci/fat_teardown.sh ]; then bash scripts/ci/fat_teardown.sh a b; fi\n' "$1" "$2" > "$3"
    }
    row() { # <want-rc> <label> <wait-if> <teardown-if>
        mk "$3" "$4" "$td/f.yml"; r=0; out=$(check "$td/f.yml" 2>&1) || r=$?
        if [ "$r" = "$1" ]; then echo "ok   self-test: $2"; else echo "FAIL self-test: $2 (rc $r, wanted $1): $out"; rc=1; fi
    }
    row 0 "cancel-aware wait + always() teardown passes" '${{ !cancelled() }}' 'always()'
    row 1 "MUTANT: wait back to always() is caught" 'always()' 'always()'
    row 1 "MUTANT: wait with no condition is caught" '${{ success() }}' 'always()'
    row 1 "MUTANT: teardown loses always() is caught" '${{ !cancelled() }}' '${{ !cancelled() }}'
    rm -f "${td:?}/f.yml"; rmdir "${td:?}"
    return "$rc"
}

case "${1:-}" in
    --self-test) self_test ;;
    *) check "${1:-$ROOT/.github/workflows/ci.yml}" ;;
esac
