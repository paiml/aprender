#!/usr/bin/env bash
# lib_crates_io.sh -- SOURCED library: does crates.io show a crate's version? (PMAT-5306)
# Operator ruling 2026-10-30: "a release = the version is on crates.io and passed the clean-container smoke.
# A tag alone is not a release." Option-neutral (no `set`): every failure is a RETURN STATUS.
#
#   . scripts/release/lib_crates_io.sh || exit 2
#   crates_io_has_version CRATE V   GET https://crates.io/api/v1/crates/CRATE/V, status read with curl -w '%{http_code}' (never -f):
#                                   200 and .version.num == V -> rc 0 present; 404 -> rc 1 absent; anything else -> rc 2 NOT MEASURED
#   crates_io_close_gate CRATE V    rc 0 only when crates.io shows V; prints why otherwise. THE milestone/epic close gate.
#
# Test seam: CRATES_IO_RESPONSE_FILE (line 1 = the HTTP status, the rest = the body; "000" = timeout) is honoured ONLY when a file
# named .crates-io-test-seams sits next to this library, i.e. only in a falsifier's sandbox copy. Production never has one.
_CIO_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || return 2
_CIO_UA="aprender-release-andon (https://github.com/paiml/aprender)"

crates_io_has_version() {
    local crate=${1:-} v=${2:-} resp code body
    [[ $crate =~ ^[A-Za-z0-9_-]+$ ]] || { printf 'crates_io_has_version: bad crate name\n' >&2; return 2; }
    [[ $v =~ ^[0-9A-Za-z.+-]+$ ]] || { printf 'crates_io_has_version: bad version\n' >&2; return 2; }
    if [ -f "$_CIO_HERE/.crates-io-test-seams" ] && [ -n "${CRATES_IO_RESPONSE_FILE:-}" ]; then
        resp=$(cat -- "$CRATES_IO_RESPONSE_FILE" 2> /dev/null; printf '\n') || return 2
        code=$(printf '%s' "$resp" | head -n 1)
        body=$(printf '%s' "$resp" | tail -n +2)
    else
        resp=$(curl -sS --max-time 30 -H "User-Agent: $_CIO_UA" -w '\n%{http_code}' "https://crates.io/api/v1/crates/$crate/$v" 2> /dev/null) || return 2
        code=$(printf '%s' "$resp" | tail -n 1)
        body=$(printf '%s' "$resp" | sed '$d')
    fi
    case "$code" in
        200)
            # present only when the body names exactly this version and it is not yanked; any other 200 body is a failed read
            printf '%s' "$body" | jq -e --arg v "$v" '.version.num == $v' > /dev/null 2>&1 || return 2
            printf '%s' "$body" | jq -e '.version.yanked != true' > /dev/null 2>&1 || return 1
            return 0 ;;
        404) return 1 ;;
        *) return 2 ;;
    esac
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
