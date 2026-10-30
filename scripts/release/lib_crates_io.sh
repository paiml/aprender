#!/usr/bin/env bash
# lib_crates_io.sh -- SOURCED library: what crates.io says about a crate's versions (PMAT-5306).
# Operator ruling 2026-10-30: "a release = the version is on crates.io and passed the clean-container smoke.
# A tag alone is not a release." Option-neutral (no `set`): every failure is a RETURN STATUS.
#
#   . scripts/release/lib_crates_io.sh || exit 2
#   crates_io_versions CRATE            one version per line; rc 0 read, rc 2 NOT MEASURED (never an empty "no versions")
#   crates_io_has_version CRATE V       rc 0 present, 1 absent (a read that succeeded), 2 not measured
#   crates_io_close_gate CRATE V        rc 0 only when crates.io shows V; prints why otherwise. THE milestone/epic close gate.
#
# Test seam: CRATES_IO_JSON_FILE (a fixture of the versions API response) is honoured ONLY when a file named
# .crates-io-test-seams sits next to this library, i.e. only in a falsifier's sandbox copy. Production never has one.
_CIO_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || return 2
_CIO_UA="aprender-release-andon (https://github.com/paiml/aprender)"

crates_io_versions() {
    local crate=${1:-} json
    [[ $crate =~ ^[A-Za-z0-9_-]+$ ]] || { printf 'crates_io_versions: bad crate name\n' >&2; return 2; }
    if [ -f "$_CIO_HERE/.crates-io-test-seams" ] && [ -n "${CRATES_IO_JSON_FILE:-}" ]; then
        json=$(cat -- "$CRATES_IO_JSON_FILE" 2> /dev/null) || return 2
    else
        json=$(curl -fsS --max-time 30 -H "User-Agent: $_CIO_UA" "https://crates.io/api/v1/crates/$crate/versions" 2> /dev/null) || return 2
    fi
    # a body that is not the versions list is a failed read, not "no versions"; yanked versions do not count as published
    printf '%s' "$json" | jq -e '.versions | type == "array"' > /dev/null 2>&1 || return 2
    printf '%s' "$json" | jq -r '.versions[] | select(.yanked != true) | .num' 2> /dev/null || return 2
    return 0
}

crates_io_has_version() {
    local crate=${1:-} v=${2:-} list
    [ -n "$v" ] || return 2
    list=$(crates_io_versions "$crate") || return 2
    printf '%s\n' "$list" | grep -qxF -- "$v" && return 0
    return 1
}

crates_io_close_gate() {
    local crate=${1:-} v=${2:-} rc
    crates_io_has_version "$crate" "$v"; rc=$?
    case "$rc" in
        0) return 0 ;;
        1) printf 'crates.io does not show %s: milestone and epic stay open\n' "$v" >&2; return 1 ;;
        *) printf 'crates.io read failed (not measured) for %s: milestone and epic stay open\n' "$v" >&2; return 2 ;;
    esac
}
