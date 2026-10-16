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
# THE RULES. Steps are read as records (a `- ` at step indent starts one; comments are
# ignored; `- name:`, `- run:` and `- if:` layouts and `run: |` blocks all parse).
#   - EXACTLY 2 steps run `fat_driver.py wait` (x86-main, determinism) and each `if:` is
#     exactly `${{ !cancelled() }}`;
#   - EXACTLY 3 steps run `fat_teardown.sh` (x86-main, determinism, ARM64 determinism)
#     and each `if:` is exactly `always()`;
#   - a kind with no step at all is exit 2 (the file cannot be modelled).
# Override the counts with CI_WAIT_STEPS / CI_TEARDOWN_STEPS when a job is added.
#
# Usage
#   bash scripts/check_ci_wait_cancel.sh [FILE]   check .github/workflows/ci.yml
#   bash scripts/check_ci_wait_cancel.sh --self-test
# Exit 0 ok, 1 a rule broke, 2 the file cannot be modelled.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# steps_of <file> <substring>: "<line>|<if or NONE>" per step whose non-comment text holds it.
steps_of() {
    awk -v pat="$2" '
        function flush() {
            if (rec != "" && index(rec, pat)) print first "|" cond
            rec = ""; cond = "NONE"
        }
        function ifval(s) { sub(/^[ \t-]*if:[ \t]*/, "", s); sub(/[ \t]+$/, "", s); return s }
        /^[ \t]*#/ { next }
        /^      - / { flush(); first = NR; rec = $0 "\n"; if ($0 ~ /^      - if:/) cond = ifval($0); next }
        /^        if:/ { cond = ifval($0) }
        { rec = rec $0 "\n" }
        END { flush() }
    ' "$1"
}

check() { # <file>
    local f=$1 rc=0 spec pat want label want_n k row ln cond
    for spec in 'fat_driver.py wait|!cancelled()|wait|WAITS' 'fat_teardown.sh|always()|teardown|TEARDOWNS'; do
        IFS='|' read -r pat want label want_n <<< "$spec"
        case "$want_n" in WAITS) want_n=${CI_WAIT_STEPS:-2} ;; *) want_n=${CI_TEARDOWN_STEPS:-3} ;; esac
        k=0
        while IFS= read -r row; do
            [ -n "$row" ] || continue
            k=$((k + 1)); ln=${row%%|*}; cond=${row#*|}
            if [ "$want" = '!cancelled()' ]; then
                if [ "$cond" = '${{ !cancelled() }}' ]; then echo "ok   $label step at line $ln is cancel-aware"
                else echo "FAIL $label step at line $ln has if: '$cond', wanted exactly \${{ !cancelled() }}"; rc=1; fi
            else
                if [ "$cond" = 'always()' ]; then echo "ok   $label step at line $ln keeps always()"
                else echo "FAIL $label step at line $ln has if: '$cond', wanted exactly always()"; rc=1; fi
            fi
        done < <(steps_of "$f" "$pat")
        if [ "$k" -eq 0 ]; then echo "ENV: no $label step in $f -- the file cannot be modelled" >&2; return 2; fi
        if [ "$k" -ne "$want_n" ]; then echo "FAIL $k $label step(s) found, $want_n expected: a deleted or renamed step escapes the check (update CI_WAIT_STEPS/CI_TEARDOWN_STEPS with the step)"; rc=1; fi
    done
    return "$rc"
}

self_test() {
    local td rc=0 out r
    td=$(mktemp -d "${TMPDIR:-/tmp}/ci-wait-cancel.XXXXXX")
    local C='${{ !cancelled() }}' A='always()'
    step() { # <layout> <if-or-NONE> <run>   layouts: name | runfirst | block | iffirst
        case "$1" in
            name)     printf '      - name: S\n'; [ "$2" = NONE ] || printf '        if: %s\n' "$2"; printf '        run: %s\n' "$3" ;;
            runfirst) printf '      - run: %s\n' "$3"; [ "$2" = NONE ] || printf '        if: %s\n' "$2" ;;
            block)    printf '      - name: S\n'; [ "$2" = NONE ] || printf '        if: %s\n' "$2"; printf '        run: |\n          set -e\n          %s\n' "$3" ;;
            iffirst)  if [ "$2" = NONE ]; then printf '      - run: %s\n' "$3"; else printf '      - if: %s\n        run: %s\n' "$2" "$3"; fi ;;
        esac
    }
    W='python3 scripts/ci/fat_driver.py wait --results x'; T='bash scripts/ci/fat_teardown.sh a b'
    row() { # <want-rc> <label> <waits> <teardowns> <w1-layout> <w1-if> <w2-if> <t1-if> <t2-if> [<n-w> <n-t>]
        local wl=$5
        { [ "${10:-2}" -lt 1 ] || step "$wl" "$6" "$W"; [ "${10:-2}" -lt 2 ] || step name "$7" "$W"
          [ "${11:-2}" -lt 1 ] || step name "$8" "$T"; [ "${11:-2}" -lt 2 ] || step name "$9" "$T"; echo "      # a comment naming fat_teardown.sh and fat_driver.py wait"; } > "$td/f.yml"
        r=0; out=$(CI_WAIT_STEPS=2 CI_TEARDOWN_STEPS=2 check "$td/f.yml" 2>&1) || r=$?
        if [ "$r" = "$1" ]; then echo "ok   self-test: $2"; else echo "FAIL self-test: $2 (rc $r, wanted $1): $out"; rc=1; fi
    }
    row 0 "both waits cancel-aware, both teardowns always() passes" x x name "$C" "$C" "$A" "$A"
    row 1 "MUTANT: first wait back to always() is caught" x x name "$A" "$C" "$A" "$A"
    row 1 "MUTANT: the SECOND wait (determinism) always() is caught" x x name "$C" "$A" "$A" "$A"
    row 1 "MUTANT: a wait with a non-cancel condition is caught" x x name '${{ success() }}' "$C" "$A" "$A"
    row 1 "MUTANT: a wait with NO if: is caught" x x name NONE "$C" "$A" "$A"
    row 1 "MUTANT: BARE !cancelled() (no \${{ }}) is caught: exact match" x x name '!cancelled()' "$C" "$A" "$A"
    row 1 "MUTANT: a compound condition is caught: exact match" x x name '${{ !cancelled() && true }}' "$C" "$A" "$A"
    row 1 "MUTANT: first teardown loses always() is caught" x x name "$C" "$C" "$C" "$A"
    row 1 "MUTANT: the SECOND teardown loses always() is caught" x x name "$C" "$C" "$A" "$C"
    row 1 "MUTANT: a teardown with NO if: is caught" x x name "$C" "$C" NONE "$A"
    row 1 "MUTANT: a deleted wait step is caught (1 of 2)" x x name "$C" "$C" "$A" "$A" 1 2
    row 1 "MUTANT: a deleted teardown step is caught (1 of 2)" x x name "$C" "$C" "$A" "$A" 2 1
    row 2 "no wait step at all is exit 2" x x name "$C" "$C" "$A" "$A" 0 2
    row 2 "no teardown step at all is exit 2" x x name "$C" "$C" "$A" "$A" 2 0
    row 0 "a '- run:' first-key layout parses and passes" x x runfirst "$C" "$C" "$A" "$A"
    row 1 "a '- run:' first-key layout with always() is caught" x x runfirst "$A" "$C" "$A" "$A"
    row 0 "a '- if:' first-key layout parses and passes" x x iffirst "$C" "$C" "$A" "$A"
    row 1 "a '- if:' first-key layout with always() is caught" x x iffirst "$A" "$C" "$A" "$A"
    row 0 "a 'run: |' block layout parses and passes" x x block "$C" "$C" "$A" "$A"
    row 1 "a 'run: |' block layout with always() is caught" x x block "$A" "$C" "$A" "$A"
    rm -f "${td:?}/f.yml"; rmdir "${td:?}"
    return "$rc"
}

case "${1:-}" in
    --self-test) self_test ;;
    *) check "${1:-$ROOT/.github/workflows/ci.yml}" ;;
esac
