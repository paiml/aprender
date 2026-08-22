#!/usr/bin/env bash
# STATIC counterpart to check_mcp_never_path_resolves_apr.sh.
#
# No MCP tool may pass the bare string "apr" to a process-spawning call. A bare
# "apr" is resolved through $PATH, so the server runs whatever binary the user
# happens to have -- during the 0.63.0 dogfood that was a 26-day-old 0.60.0 while
# the server reported itself as 0.63.0 (aprender#2563).
#
# WHY A STATIC GUARD EXISTS ALONGSIDE THE BEHAVIOURAL ONE. The behavioural guard
# is stronger -- it proves the property rather than the syntax -- but it needs a
# BUILT apr (a ~1 GB debug binary) and so costs minutes in CI. This one is a
# grep, runs in milliseconds, and catches the regression that actually happens:
# someone types "apr" instead of calling apr_binary(). Wire the cheap one; run
# the strong one when a binary already exists.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

SRC=crates/aprender-mcp/src
[ -d "$SRC" ] || { printf 'FAIL  %s does not exist -- the guard is looking at nothing.\n' "$SRC"; exit 1; }

# The spawn-shaped calls. Anchored on the FUNCTION so an unrelated "apr" in a
# doc comment or a test fixture cannot trip it.
PAT='(spawn_and_confirm|stream_with_sink|run_apr_streaming|Command::new)\([[:space:]]*"apr"'

# Strip comment lines BEFORE matching: apr_bin.rs and subprocess.rs both describe
# this very defect in prose, and a guard that trips on its own documentation is a
# guard people delete. `//` and `//!` only -- a block comment containing a spawn
# call would be pathological.
hits=$(grep -rnE --include='*.rs' "$PAT" "$SRC" 2>/dev/null \
       | grep -v '_tests\.rs' \
       | grep -vE '^[^:]+:[0-9]+:[[:space:]]*//' || true)

if [ -n "$hits" ]; then
    printf 'FAIL  an MCP tool passes a bare "apr" to a spawn call:\n'
    printf '%s\n' "$hits" | sed 's/^/        /'
    printf '      A bare "apr" is resolved through $PATH. Use apr_binary() --\n'
    printf '      crate::apr_bin::apr_binary() -- as every other tool does.\n'
    exit 1
fi

n=$(grep -rcE --include='*.rs' 'apr_binary\(\)' "$SRC" 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')
if [ "$n" -eq 0 ]; then
    # Anti-vacuity: if nothing calls apr_binary() at all, either the resolver was
    # removed or this guard is pointed at the wrong tree. Either way its PASS
    # above would mean nothing.
    printf 'FAIL  no call to apr_binary() found anywhere under %s.\n' "$SRC"
    printf '      A clean grep for bare "apr" is not evidence when the resolver\n'
    printf '      itself is absent -- the guard would pass on an empty directory.\n'
    exit 1
fi

printf 'ok    no MCP tool passes a bare "apr" to a spawn call (%s apr_binary() call sites)\n' "$n"
exit 0
