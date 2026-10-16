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

# steps_of <file> <substring>: "<line>|<if or NONE>|<flags>" per step whose non-comment text holds it.
# flags is "-", or "coe" (a continue-on-error that is not false) and/or "mask" (`|| true`, `|| :`,
# `exit 0`, `set +e` in the step), joined by "+". A record ends at the first line indented less than
# a step, so the next job's keys are never read as this step's.
steps_of() {
    awk -v pat="$2" '
        function flush(   fl) {
            if (rec != "" && index(rec, pat)) {
                fl = ""
                if (rec ~ /continue-on-error:[ \t]*[^ \t\n]/ && rec !~ /continue-on-error:[ \t]*false/) fl = "coe"
                if (rec ~ /\|\|[ \t]*(true|:)|exit[ \t]+0|set[ \t]+\+e/) fl = fl (fl == "" ? "" : "+") "mask"
                print first "|" cond "|" (fl == "" ? "-" : fl)
            }
            rec = ""; cond = "NONE"
        }
        function ifval(s) { sub(/^[ \t-]*if:[ \t]*/, "", s); sub(/[ \t]+$/, "", s); return s }
        /^[ \t]*#/ { next }
        /^ {0,5}[^ ]/ { flush(); live = 0; next }
        /^      - / { flush(); live = 1; first = NR; rec = $0 "\n"; if ($0 ~ /^      - if:/) cond = ifval($0); next }
        /^        if:/ { if (live) cond = ifval($0) }
        live { rec = rec $0 "\n" }
        END { flush() }
    ' "$1"
}

# The gate job's "Check required sections" script (the one whose env reads X86/DET), dedented.
gate_script() { # <file>
    awk '
        /X86: \$\{\{ needs\.x86-main\.outputs\.results \}\}/ { seen = 1 }
        seen && !run && /^        run: \|[ ]*$/ { run = 1; next }
        run { if ($0 ~ /^          / || $0 ~ /^[ ]*$/) { sub(/^          /, ""); print } else exit }
    ' "$1"
}

# r3 item 2: a CANCELLED fat job must turn the gate RED, never skipped or green. A cancelled
# x86-main/determinism publishes no `results` output, so the gate reads every section as missing.
# Runs the gate's own script: control (all green) exits 0; each cancelled fat need exits 1.
gate_probe() { # <file> -> prints ok/FAIL lines, returns 1 on a defect, 2 if the gate cannot be modelled
    local s g rc=0 r G='{"guard-tree":{"result":"success"},"guard-cargo":{"result":"success"},"sov.gate":{"result":"success"}}' D='{"determinism-compare":{"result":"success"}}'
    s=$(mktemp "${TMPDIR:-/tmp}/ci-gate.XXXXXX")
    gate_script "$1" > "$s"
    if [ ! -s "$s" ]; then echo "ENV: no gate 'Check required sections' script in $1" >&2; rm -f "${s:?}"; return 2; fi
    g() { # <label> <want-rc> <X86> <DET>
        r=0; EVT=pull_request MG=success MG_X86=0 MG_DET=0 MG_CITE= ADV=skipped WT=success MST=success X86="$3" DET="$4" bash "$s" > /dev/null 2>&1 || r=$?
        if [ "$r" = "$2" ]; then echo "ok   gate $1 -> exit $r"; else echo "FAIL gate $1: exit $r, wanted $2"; rc=1; fi
    }
    g "all fat needs green (control)" 0 "$G" "$D"
    g "x86-main CANCELLED (no results output) is RED" 1 "" "$D"
    g "determinism CANCELLED (no results output) is RED" 1 "$G" ""
    g "both fat needs cancelled is RED" 1 "" ""
    rm -f "${s:?}"
    return "$rc"
}

# r3 item 3: `if: !cancelled()` only stops a wait step that has not started. A step already polling is
# stopped by the runner's SIGTERM/SIGINT, so the driver must install a handler that sets CANCELLED and
# SIGTERMs its section process groups (its poll loops test CANCELLED). Static proof on the driver file.
sig_probe() { # <fat_driver.py> -> ok/FAIL lines, returns 1 on a defect
    local d=$1 rc=0 pat
    for pat in 'signal.signal(signal.SIGTERM, on_signal)' 'signal.signal(signal.SIGINT, on_signal)' 'CANCELLED.set()' 'os.killpg(p.pid, signal.SIGTERM)' 'CANCELLED.is_set() or time.time() > deadline'; do
        if grep -qF -- "$pat" "$d"; then echo "ok   driver cancel path: $pat"; else echo "FAIL driver cancel path missing in $d: $pat"; rc=1; fi
    done
    return "$rc"
}

check() { # <file>
    local f=$1 rc=0 spec pat want label want_n k row ln cond flg
    for spec in 'fat_driver.py wait|!cancelled()|wait|WAITS' 'fat_teardown.sh|always()|teardown|TEARDOWNS'; do
        IFS='|' read -r pat want label want_n <<< "$spec"
        case "$want_n" in WAITS) want_n=${CI_WAIT_STEPS:-2} ;; *) want_n=${CI_TEARDOWN_STEPS:-3} ;; esac
        case "$want_n" in ''|*[!0-9]*|0) echo "ENV: the expected $label step count '$want_n' is not a positive integer (CI_WAIT_STEPS/CI_TEARDOWN_STEPS): refusing to run the check without one" >&2; return 2 ;; esac
        k=0
        while IFS= read -r row; do
            [ -n "$row" ] || continue
            k=$((k + 1)); ln=${row%%|*}; cond=${row#*|}; flg=${cond##*|}; cond=${cond%|*}
            if [ "$flg" != - ]; then echo "FAIL $label step at line $ln is neutered ($flg): continue-on-error or a masked exit status lets it pass when it should fail"; rc=1; fi
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
    # sign-off r2 fixes: neutered steps, the real 3-teardown count, the job boundary
    fx() { # <want-rc> <label> <wait-extra-line> <teardown-extra-line> <third-teardown-if|NONE-STEP> [<env CI_TEARDOWN_STEPS unset>]
        { printf '    steps:\n'
          printf '      - name: W1\n        if: %s\n        run: %s\n%s' "$C" "$W" "$3"
          printf '      - name: W2\n        if: %s\n        run: %s\n' "$C" "$W"
          printf '      - name: T1\n        if: %s\n        run: %s\n%s' "$A" "$T" "$4"
          printf '      - name: T2\n        if: %s\n        run: %s\n' "$A" "$T"
          [ "$5" = NONE-STEP ] || printf '      - name: T3\n        if: %s\n        run: %s\n' "$5" "$T"; } > "$td/f.yml"
        r=0; out=$(env -u CI_TEARDOWN_STEPS -u CI_WAIT_STEPS GATE_PROBE=0 bash "$0" "$td/f.yml" 2>&1) || r=$?
        if [ "$r" = "$1" ]; then echo "ok   self-test: $2"; else echo "FAIL self-test: $2 (rc $r, wanted $1): $out"; rc=1; fi
    }
    fx 0 "the real shape: 2 waits + 3 teardowns under the DEFAULT counts passes" "" "" "$A"
    fx 1 "MUTANT: only 2 teardowns under the default count of 3 is caught" "" "" NONE-STEP
    fx 1 "MUTANT: the THIRD teardown (ARM64 determinism) loses always() is caught" "" "" "$C"
    fx 1 "MUTANT: continue-on-error: true on a wait is caught" "        continue-on-error: true
" "" "$A"
    fx 1 "MUTANT: continue-on-error: \${{ true }} on a teardown is caught" "" "        continue-on-error: \${{ true }}
" "$A"
    fx 0 "continue-on-error: false is not a defect" "        continue-on-error: false
" "" "$A"
    fx 1 "MUTANT: '|| true' on a wait is caught" "        shell: bash || true
" "" "$A"
    fx 1 "MUTANT: 'exit 0' on a teardown is caught" "" "        # not a comment-proof: below is real text
        env: { X: 'exit 0' }
" "$A"
    fx 1 "MUTANT: 'set +e' on a wait is caught" "        run2: set +e
" "" "$A"
    { printf '    steps:\n      - name: W1\n        run: %s\n  other:\n    env:\n        if: %s\n' "$W" "$C"
      printf '      - name: W2\n        if: %s\n        run: %s\n' "$C" "$W"
      printf '      - name: T1\n        if: %s\n        run: %s\n      - name: T2\n        if: %s\n        run: %s\n' "$A" "$T" "$A" "$T"
      printf '      - name: T3\n        if: %s\n        run: %s\n' "$A" "$T"; } > "$td/f.yml"
    r=0; out=$(env -u CI_TEARDOWN_STEPS -u CI_WAIT_STEPS GATE_PROBE=0 bash "$0" "$td/f.yml" 2>&1) || r=$?
    if [ "$r" = 1 ]; then echo "ok   self-test: MUTANT: a step record ends at the job boundary (next job's if: is not its own)"; else echo "FAIL self-test: job boundary (rc $r, wanted 1): $out"; rc=1; fi
    # r3 item 1: an invalid expected count fails closed (exit 2), never silently skips the count check
    for v in abc 0 "-1" "2x" " 2"; do
        r=0; out=$(GATE_PROBE=0 CI_WAIT_STEPS=$v bash "$0" "$td/f.yml" 2>&1) || r=$?
        if [ "$r" = 2 ] && grep -q "not a positive integer" <<< "$out"; then echo "ok   self-test: CI_WAIT_STEPS='$v' is refused (exit 2)"; else echo "FAIL self-test: CI_WAIT_STEPS='$v' (rc $r, wanted 2): $out"; rc=1; fi
    done
    r=0; out=$(GATE_PROBE=0 CI_TEARDOWN_STEPS=many bash "$0" "$td/f.yml" 2>&1) || r=$?
    if [ "$r" = 2 ] && grep -q "not a positive integer" <<< "$out"; then echo "ok   self-test: CI_TEARDOWN_STEPS='many' is refused (exit 2)"; else echo "FAIL self-test: CI_TEARDOWN_STEPS=many (rc $r, wanted 2): $out"; rc=1; fi
    # r3 item 2: the gate itself, from the real ci.yml, is RED on a cancelled fat need; a mutant gate that skips the check is caught
    out=$(gate_probe "$ROOT/.github/workflows/ci.yml" 2>&1) || r=$?
    echo "$out"; case "$out" in *FAIL*) rc=1 ;; esac
    sed 's/\[ "\$r" != success \]/[ "\$r" != success ] && false/' "$ROOT/.github/workflows/ci.yml" > "$td/mut-ci.yml"
    r=0; out=$(gate_probe "$td/mut-ci.yml" 2>&1) || r=$?
    if [ "$r" = 1 ]; then echo "ok   self-test: MUTANT gate that no longer fails a missing section is caught"; else echo "FAIL self-test: mutant gate not caught (rc $r): $out"; rc=1; fi
    rm -f "${td:?}/mut-ci.yml"
    out=$(sig_probe "$ROOT/scripts/ci/fat_driver.py" 2>&1) || r=$?; echo "$out"; case "$out" in *FAIL*) rc=1 ;; esac
    sed '/signal.signal(signal.SIGTERM, on_signal)/d' "$ROOT/scripts/ci/fat_driver.py" > "$td/mut-fd.py"
    r=0; out=$(sig_probe "$td/mut-fd.py" 2>&1) || r=$?
    if [ "$r" = 1 ]; then echo "ok   self-test: MUTANT driver without a SIGTERM handler is caught"; else echo "FAIL self-test: mutant driver not caught (rc $r)"; rc=1; fi
    rm -f "${td:?}/mut-fd.py"
    rm -f "${td:?}/f.yml"; rmdir "${td:?}"
    return "$rc"
}

case "${1:-}" in
    --self-test) self_test ;;
    *) wf=${1:-$ROOT/.github/workflows/ci.yml}; rc=0; check "$wf" || rc=$?; [ "${GATE_PROBE:-1}" = 0 ] || { gate_probe "$wf" || { r=$?; [ "$rc" -ge "$r" ] || rc=$r; }; sig_probe "$ROOT/scripts/ci/fat_driver.py" || { [ "$rc" -ge 1 ] || rc=1; }; }; exit "$rc" ;;
esac
