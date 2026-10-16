#!/usr/bin/env bash
# check_ci_wait_cancel.sh -- a cancel must stop every fat job's wait step, and the
# teardown must still run (13:46Z 2026-10-16, infra-6a).
#
# THE DEFECT. x86-main's step "Sections (wait for every one to finish)" was
# `if: always()`. always() is true on a cancelled run, so a cancel could not stop
# the `fat_driver.py wait` poll; orphan merge-group runs held queue runners until a
# force-cancel. `${{ !cancelled() }}` runs the wait after a failed step (its
# results are still published) and skips it on a cancel. Every teardown step keeps
# `if: always()`: a cancelled run never reaches a section's own container removal,
# and fat_teardown.sh reaps the detached containers (#4948).
#
# EVERY `fat_driver.py wait` step (x86-main and determinism) must be `!cancelled()`;
# EVERY `fat_teardown.sh` step must be `always()`. A step with no `if:` at all fails.
#
# Usage
#   bash scripts/check_ci_wait_cancel.sh [FILE]   check .github/workflows/ci.yml
#   bash scripts/check_ci_wait_cancel.sh --self-test
# Exit 0 ok, 1 a rule broke, 2 the file cannot be modelled.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# ifs_of <file> <run-substring>: one line per step whose `run:` holds the substring,
# "<line>|<if or NONE>". The step's `if:` is the one between its `- ` and its `run:`.
ifs_of() {
    awk -v pat="$2" '
        /^      - / { cond = "NONE" }
        /^        if: / { cond = $0; sub(/^        if: /, "", cond) }
        /^        run: / && index($0, pat) { print NR "|" cond }
    ' "$1"
}

check() { # <file>
    local f=$1 rc=0 n=0 row ln cond
    for kind in 'fat_driver.py wait:!cancelled():wait' 'fat_teardown.sh:always():teardown'; do
        pat=${kind%%:*}; rest=${kind#*:}; want=${rest%%:*}; label=${rest#*:}
        while IFS= read -r row; do
            [ -n "$row" ] || continue
            n=$((n + 1)); ln=${row%%|*}; cond=${row#*|}
            case "$want|$cond" in
                '!cancelled()|'*always\(\)*) echo "FAIL $label step at line $ln is always(): a cancel cannot stop it ($cond)"; rc=1 ;;
                '!cancelled()|'*'!cancelled()'*) echo "ok   $label step at line $ln is cancel-aware" ;;
                'always()|'*always\(\)*) echo "ok   $label step at line $ln keeps always()" ;;
                *) echo "FAIL $label step at line $ln has if: '$cond', wanted $want"; rc=1 ;;
            esac
        done < <(ifs_of "$f" "$pat")
    done
    if [ "$n" -eq 0 ]; then echo "ENV: no wait or teardown step in $f" >&2; return 2; fi
    return "$rc"
}

self_test() {
    local td rc=0 out r
    td=$(mktemp -d "${TMPDIR:-/tmp}/ci-wait-cancel.XXXXXX")
    step() { # <name> <if-or-NONE> <run>
        printf '      - name: %s\n' "$1"
        [ "$2" = NONE ] || printf '        if: %s\n' "$2"
        printf '        run: %s\n' "$3"
    }
    mk() { # <wait1-if> <teardown1-if> <wait2-if> <teardown2-if>
        {
            step Sections "$1" 'python3 scripts/ci/fat_driver.py wait --results x'
            step Remove "$2" 'bash scripts/ci/fat_teardown.sh a b'
            step Compare "$3" 'python3 scripts/ci/fat_driver.py wait --results y'
            step Remove2 "$4" 'bash scripts/ci/fat_teardown.sh a b'
        } > "$td/f.yml"
    }
    row() { # <want-rc> <label> <w1> <t1> <w2> <t2>
        mk "$3" "$4" "$5" "$6"; r=0; out=$(check "$td/f.yml" 2>&1) || r=$?
        if [ "$r" = "$1" ]; then echo "ok   self-test: $2"; else echo "FAIL self-test: $2 (rc $r, wanted $1): $out"; rc=1; fi
    }
    local C='${{ !cancelled() }}' A='always()'
    row 0 "both waits cancel-aware, both teardowns always() passes" "$C" "$A" "$C" "$A"
    row 1 "MUTANT: first wait back to always() is caught" "$A" "$A" "$C" "$A"
    row 1 "MUTANT: the SECOND wait (determinism) always() is caught" "$C" "$A" "$A" "$A"
    row 1 "MUTANT: a wait with a non-cancel condition is caught" '${{ success() }}' "$A" "$C" "$A"
    row 1 "MUTANT: a wait with NO if: is caught" NONE "$A" "$C" "$A"
    row 1 "MUTANT: first teardown loses always() is caught" "$C" "$C" "$C" "$A"
    row 1 "MUTANT: the SECOND teardown loses always() is caught" "$C" "$A" "$C" "$C"
    row 1 "MUTANT: a teardown with NO if: is caught" "$C" NONE "$C" "$A"
    rm -f "${td:?}/f.yml"; rmdir "${td:?}"
    return "$rc"
}

case "${1:-}" in
    --self-test) self_test ;;
    *) check "${1:-$ROOT/.github/workflows/ci.yml}" ;;
esac
