#!/usr/bin/env bash
# check_leanchecker_scoped.sh — no script or CI job runs leanchecker outside a memory/CPU cap (#4348).
#
# WHY THIS EXISTS
# ---------------
# leanchecker replays one full environment (Mathlib included) per concurrent module task, so its memory is
# threads x environment. Uncapped on lambda it took 51 threads and 58-67 GB and left the host 15 GB; the cop had to
# SIGTERM it. Operator, verbatim: "remember this host must be able to do other work, so never let it get
# overloaded". Where `pv discharge check --leanchecker` / `pv discharge run` exist they cap it by default
# (<= 8 threads inside `systemd-run --user --scope --slice=agent.slice -p MemoryMax=24G -p CPUQuota=800%`). This
# guard keeps every OTHER door shut: a tracked script, Makefile, workflow, CI-section or forjar line that runs
# leanchecker directly is RED unless the same line carries `systemd-run` and `MemoryMax=`, and
# `--leanchecker-unscoped` (pv's opt-out) is RED anywhere.
#
#   bash scripts/check_leanchecker_scoped.sh               # scan the tracked tree
#   bash scripts/check_leanchecker_scoped.sh --self-test   # the must-match / must-not-match case table
#
# Surfaces: tracked .github/workflows/*.yml|*.yaml, ci/ (sections.yml + the explicit-test .cmd fragments the fat
# driver runs -- the CI decision surface since #4441), forjar*.yaml, Makefile, *.mk, and *.sh anywhere.
# Comment lines and YAML `name:` labels (a step title, never executed) are skipped.
# A leanchecker word in an echo'd string in command position is a false RED: quote it ("... leanchecker ...").
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROG=check_leanchecker_scoped

# leanchecker in COMMAND position: after start/; & | ( ` or a space, behind any of exec/env/nice/`lake env`/
# `timeout [-opts] N`, optionally path-qualified. `--leanchecker` (pv's flag) and a quoted string are not.
CMD_RE='(^|[;&|(`]|[[:space:]])((exec|env|nice|lake[[:space:]]+env|timeout([[:space:]]+-[^[:space:]]+)*[[:space:]]+[0-9]+[smhd]?)[[:space:]]+)*("?[^[:space:]"]*/)?leanchecker("|[[:space:]]|$|[;)&|])'
OPTOUT_RE='--leanchecker-unscoped'

# judge_line <text> -> 0 clean / 1 RED (prints the reason)
judge_line() {
    local l=$1
    [[ "$l" =~ ^[[:space:]]*# ]] && return 0
    [[ "$l" =~ ^[[:space:]]*(-[[:space:]]+)?name: ]] && return 0
    if [[ "$l" =~ $OPTOUT_RE ]]; then
        printf 'pv discharge with --leanchecker-unscoped drops the 24G/800%% scope'; return 1
    fi
    if [[ "$l" =~ $CMD_RE ]] && ! { [[ "$l" == *systemd-run* ]] && [[ "$l" == *MemoryMax=* ]]; }; then
        printf 'leanchecker run outside systemd-run -p MemoryMax=...: use pv discharge check --leanchecker'; return 1
    fi
    return 0
}

scan() {
    local f n l why red=0 files=0
    while IFS= read -r f; do
        # This file's own patterns and case table are the RED lines by design; judging them is self-flagging.
        [ "$f" = scripts/check_leanchecker_scoped.sh ] && continue
        files=$((files + 1)); n=0
        while IFS= read -r l || [ -n "$l" ]; do
            n=$((n + 1))
            [[ "$l" == *leanchecker* ]] || continue
            if ! why=$(judge_line "$l"); then
                printf 'FAIL  %s: %s:%d: %s\n        %s\n' "$PROG" "$f" "$n" "$why" "$l"; red=1
            fi
        done < "$ROOT/$f"
    done < <(git -C "$ROOT" ls-files -- '.github/workflows/*.yml' '.github/workflows/*.yaml' \
        'ci/*.yml' 'ci/*.yaml' 'ci/*.cmd' 'ci/**/*.cmd' '*forjar*.yaml' 'Makefile' '*.mk' '*.sh')
    [ "$files" -gt 0 ] || { printf '%s: ENV - git ls-files named no file: UNMEASURED, not a pass\n' "$PROG" >&2; return 2; }
    [ "$red" = 0 ] && printf 'ok    %s: no unscoped leanchecker in %d tracked script/CI file(s)\n' "$PROG" "$files"
    return "$red"
}

self_test() {
    local n=0 red=0 want got l
    row() { # row <want 0|1> <line>
        n=$((n + 1)); want=$1; l=$2; got=0
        judge_line "$l" >/dev/null || got=1
        if [ "$got" = "$want" ]; then printf 'ok    row %-2s %s  %s\n' "$n" "$want" "$l"
        else printf 'FAIL  row %-2s got %s wanted %s  %s\n' "$n" "$got" "$want" "$l"; red=1; fi
    }
    # must be RED
    row 1 'lake env leanchecker ProvableContracts'
    row 1 'leanchecker ProvableContracts'
    row 1 '    exec timeout -k 30 3600 lake env leanchecker ProvableContracts'
    row 1 'timeout 3600 lake env leanchecker ProvableContracts'
    row 1 'cd lean && lake env leanchecker ProvableContracts'
    row 1 '"$SYSROOT/bin/leanchecker" ProvableContracts'
    row 1 '$SYSROOT/bin/leanchecker --fresh ProvableContracts'
    row 1 '      run: lake env leanchecker ProvableContracts'
    row 1 '	env LEAN_NUM_THREADS=8 lake env leanchecker ProvableContracts'
    row 1 'x=$(lake env leanchecker ProvableContracts)'
    row 1 'systemd-run --user --scope lake env leanchecker ProvableContracts'
    row 1 'nice lake env leanchecker ProvableContracts'
    row 1 'pv discharge check --leanchecker --leanchecker-unscoped lean'
    row 1 'pv discharge run lean --leanchecker-unscoped'
    # must NOT be RED
    row 0 'systemd-run --user --scope --slice=agent.slice -p MemoryMax=24G -p CPUQuota=800% lake env leanchecker ProvableContracts'
    row 0 'pv discharge check --leanchecker crates/aprender-contracts-staging/lean'
    row 0 'pv discharge check --leanchecker --leanchecker-threads 4 lean'
    row 0 'pv discharge run crates/aprender-contracts-staging/lean'
    row 0 '# lake env leanchecker ProvableContracts is what pv runs'
    row 0 '    # leanchecker ProvableContracts'
    row 0 'echo "leanchecker not in toolchain"'
    row 0 'grep -q leanchecker_exit discharge.json'
    row 0 'ls "$SYSROOT/bin/leanchecker.exe"'
    row 0 '      - name: pv discharge run (build.sh, then check --strict + comparator + leanchecker)'
    row 0 '        name: leanchecker ProvableContracts'
    row 1 '        run: lake env leanchecker ProvableContracts  # name: x'
    printf '%s/%s rows\n' "$((n - red))" "$n"
    [ "$red" = 0 ]
}

case "${1:-}" in
    --self-test) self_test; exit $? ;;
    -h|--help) printf 'usage: %s [--self-test]\n' "$PROG"; exit 0 ;;
    "") scan; exit $? ;;
    *) printf 'usage: %s [--self-test]\n' "$PROG" >&2; exit 2 ;;
esac
