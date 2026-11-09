#!/usr/bin/env bash
# Falsifier for three red nightlies whose cause was the host, not the code (2026-11-09 five-whys):
#  deep:    every deep-nightly job points TMPDIR at RUNNER_TEMP before its disk guard. framework16's /tmp
#           is a 28 GiB tmpfs under the 38 GiB floor, while the checkout disk had 209 GiB (run 42537290407).
#  conleche: the controls run lean4export under `elan run "$TC"`. Run bare, its `lean` child resolves the
#           elan proxy outside any lean-toolchain, and a fresh ELAN_HOME has no default (run 42543242566).
#  ladder:  an unwritable lean cache DECLINES (exit 2, names the host), never a bare `set -e` exit 1
#           (yoga-build, run 42536048000: "mkdir: cannot create directory ... Permission denied").
# usage: falsify_red_nightlies.sh [--mutants]
set -euo pipefail
cd "$(dirname "$0")/../.."
DEEP=.github/workflows/deep-nightly.yml
CL=crates/aprender-contracts-staging/lean/conleche.sh
SEC=ci/sections.yml

# deep_ok <file>: each job's TMPDIR export comes before its disk-guard step, one per job
deep_ok() {
    awk '
        /^  [a-z][a-z0-9-]*:$/ { job = $1; next }
        /TMPDIR=\$RUNNER_TEMP/ && /GITHUB_ENV/ && !/^[[:space:]]*#/ { seen[job] = 1 }
        /deep_disk_guard\.sh/ && !/^[[:space:]]*#/ && job != "" { if (seen[job]) ok[job] = 1; else bad = 1; guards++ }
        END { exit (bad || guards < 3) }
    ' "$1"
}

# conleche_ok <file>: no control export runs the exporter outside elan run "$TC"
conleche_ok() {
    grep -qE 'elan run "\$TC" "\$EXPORTER" "\$m"' "$1" || return 1
    # one awk, no pipe into grep -q (check_no_pipe_into_grep_q.sh)
    if awk '/^[^#]*"\$EXPORTER" "\$m"/ && !/elan run "\$TC"/ { bad = 1 } END { exit !bad }' "$1"; then return 1; fi
}

# ladder_ok <file>: the cache mkdir declines on failure and the cache must be writable
ladder_ok() {
    grep -qE '^[[:space:]]*mkdir -p "\$cache" 2>/dev/null && \[ -w "\$cache" \] \|\| decline ' "$1" || return 1
    if grep -qE '^[[:space:]]*mkdir -p "\$cache"$' "$1"; then return 1; fi
}

check() {
    local rc=0
    deep_ok "$DEEP" || { echo "FAIL deep: a deep-nightly job runs its disk guard without TMPDIR=\$RUNNER_TEMP first"; rc=1; }
    conleche_ok "$CL" || { echo "FAIL conleche: a control exports outside elan run \"\$TC\""; rc=1; }
    ladder_ok "$SEC" || { echo "FAIL ladder: an unwritable lean cache does not decline"; rc=1; }
    [ "$rc" = 0 ] && echo "PASS: deep TMPDIR, conleche elan run, ladder cache decline"
    return "$rc"
}

mutants() {
    local d rc=0
    d=$(mktemp -d -p "${RUNNER_TEMP:-${TMPDIR:-/tmp}}") || exit 2
    trap 'rm -rf -- "${d:?}"' RETURN
    # deep: drop each job's TMPDIR export in turn
    for n in $(grep -nE 'TMPDIR=\$RUNNER_TEMP' "$DEEP" | cut -d: -f1); do
        sed "${n}d" "$DEEP" > "$d/m"
        if deep_ok "$d/m"; then echo "MUTANT SURVIVED: deep drop line $n"; rc=1; fi
    done
    sed -E 's#elan run "\$TC" "\$EXPORTER"#"$EXPORTER"#' "$CL" > "$d/m"
    if conleche_ok "$d/m"; then echo "MUTANT SURVIVED: conleche bare exporter"; rc=1; fi
    { cat "$CL"; printf '%s\n' '"$EXPORTER" "$m" > "$d/$m.ndjson"'; } > "$d/m"
    if conleche_ok "$d/m"; then echo "MUTANT SURVIVED: conleche one bare exporter beside elan run"; rc=1; fi
    sed -E 's#mkdir -p "\$cache" 2>/dev/null && \[ -w "\$cache" \] \|\| decline .*#mkdir -p "$cache"#' "$SEC" > "$d/m"
    if ladder_ok "$d/m"; then echo "MUTANT SURVIVED: ladder bare mkdir"; rc=1; fi
    echo "mutants: $([ "$rc" = 0 ] && echo killed || echo SURVIVED)"
    return "$rc"
}

case "${1:-}" in
    --mutants) mutants ;;
    *) check ;;
esac
