#!/usr/bin/env bash
# Falsifier: when `cargo update --workspace --offline` fails, bump-version.sh shows cargo's last
# 15 lines and the CARGO_HOME hint (it used to send them to /dev/null). --mutants re-runs the
# check against the pre-fix line and must see it go red.
set -uo pipefail
cd "$(dirname "$0")/../.."
S=scripts/bump-version.sh

# check <script>: run it in a fixture tree with a failing fake cargo; 0 = stderr surfaced.
check() {
    local d out rc=0
    d=$(mktemp -d "${TMPDIR:-/tmp}/bump-stderr.XXXXXX") || return 2
    mkdir -p "$d/scripts" "$d/bin" "$d/crates/facades"
    cp "$1" "$d/scripts/bump-version.sh"
    printf '[workspace.package]\nversion = "0.1.0"\n' > "$d/Cargo.toml"
    printf '[workspace.package]\nversion = "0.4.0"\n' > "$d/crates/facades/Cargo.toml"
    printf '#!/bin/sh\nfor i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18; do echo "noise-$i" >&2; done\necho "MARKER-NO-MATCHING-PACKAGE" >&2\nexit 1\n' > "$d/bin/cargo"
    chmod +x "$d/bin/cargo"
    out=$(PATH="$d/bin:$PATH" bash "$d/scripts/bump-version.sh" 0.1.1 2>&1) || rc=$?
    rm -rf "${d:?}"
    [ "$rc" = 1 ] || { echo "FAIL  exit $rc, want 1"; return 1; }
    grep -q 'could not refresh Cargo.lock' <<< "$out" || { echo "FAIL  no refresh message"; return 1; }
    grep -q 'MARKER-NO-MATCHING-PACKAGE' <<< "$out" || { echo "FAIL  cargo stderr hidden"; return 1; }
    grep -q 'noise-4$' <<< "$out" && { echo "FAIL  printed more than 15 lines"; return 1; }
    grep -q 'noise-5$' <<< "$out" || { echo "FAIL  tail missing"; return 1; }
    grep -q 'CARGO_HOME' <<< "$out" || { echo "FAIL  CARGO_HOME hint missing"; return 1; }
    echo "PASS  cargo stderr tail + CARGO_HOME hint shown"
}

if [ "${1:-}" = "--mutants" ]; then
    d=$(mktemp -d "${TMPDIR:-/tmp}/bump-stderr-m.XXXXXX"); trap 'rm -rf "${d:?}"' EXIT
    check "$S" || { echo "FAIL: red on the real script"; exit 1; }
    sed 's#>"\$upd_log" 2>&1; then#>/dev/null 2>\&1; then#' "$S" > "$d/old.sh"
    cmp -s "$S" "$d/old.sh" && { echo "FAIL: mutant did not change the script"; exit 1; }
    if check "$d/old.sh" >/dev/null; then echo "FAIL: pre-fix mutant survived"; exit 1; fi
    echo "mutant pre-fix: killed"
    exit 0
fi
check "$S"
