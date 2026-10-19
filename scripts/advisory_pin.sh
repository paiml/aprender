#!/usr/bin/env bash
# advisory_pin.sh -- P9 option 4: the merge path checks advisories against a PINNED
# RustSec snapshot (ci/advisory-db.pin), not the live DB. Deterministic, still a GATE.
#
#   advisory_pin.sh age <pin-file> [<today YYYY-MM-DD>]   F5: red if the pin is older than MAX_AGE_DAYS
#   advisory_pin.sh needed <base-ref>                     exit 0 if Cargo.toml/Cargo.lock/deny.toml changed vs the merge-base
#   advisory_pin.sh run <base-ref>                        age check; skip unless needed; deny against the pinned snapshot
#   advisory_pin.sh --self-test                           case table (fresh/boundary/stale/malformed + mutation)
#
# Exit: 0 pass, 1 red, 75 could not check (a missing/garbled pin NEVER passes).
set -euo pipefail
MAX_AGE_DAYS="${MAX_AGE_DAYS:-7}"
PIN_FILE_DEFAULT="ci/advisory-db.pin"
NEEDED_RE='(^|/)Cargo\.(toml|lock)$|^deny\.toml$'

age_check() {
    local pin="$1" today="${2:-$(date -u +%F)}" sha d pd td age
    [ -r "$pin" ] || { echo "advisory_pin: COULD NOT CHECK: no pin file $pin" >&2; return 75; }
    sha=$(sed -n 's/^sha=//p' "$pin")
    d=$(sed -n 's/^date=//p' "$pin")
    case "$sha" in *[!0-9a-f]*|"") echo "advisory_pin: COULD NOT CHECK: bad sha in $pin" >&2; return 75 ;; esac
    [ "${#sha}" -eq 40 ] || { echo "advisory_pin: COULD NOT CHECK: sha is not 40 hex" >&2; return 75; }
    pd=$(date -u -d "$d" +%s 2>/dev/null) || { echo "advisory_pin: COULD NOT CHECK: bad date '$d'" >&2; return 75; }
    td=$(date -u -d "$today" +%s 2>/dev/null) || { echo "advisory_pin: COULD NOT CHECK: bad today '$today'" >&2; return 75; }
    age=$(( (td - pd) / 86400 ))
    if [ "$age" -gt "$MAX_AGE_DAYS" ]; then
        echo "advisory_pin: RED: pin $d is $age days old (max $MAX_AGE_DAYS); the bump PR did not land" >&2
        return 1
    fi
    echo "advisory_pin: ok: pin $d is $age days old (max $MAX_AGE_DAYS)"
}

needed() {
    local base="$1" mb tf rc
    mb=$(git merge-base "$base" HEAD 2>/dev/null) || return 0   # cannot tell: run (superset)
    tf=$(mktemp)
    git diff --no-renames --name-only "$mb" HEAD > "$tf"
    rc=0; grep -qE "$NEEDED_RE" "$tf" || rc=1
    rm -f "${tf:?}"
    return "$rc"
}

fetch_db() {
    local sha
    sha=$(sed -n "s/^sha=//p" "$PIN_FILE_DEFAULT")
    ADVISORY_DB_DIR="${ADVISORY_DB_DIR:-${RUNNER_TEMP:-/tmp}/advisory-db-pinned}"
    rm -rf "${ADVISORY_DB_DIR:?}"; mkdir -p "$ADVISORY_DB_DIR"
    git -C "$ADVISORY_DB_DIR" init -q
    git -C "$ADVISORY_DB_DIR" fetch -q --depth 1 https://github.com/RustSec/advisory-db.git "$sha" \
        || { echo "advisory_pin: COULD NOT CHECK: fetch of pinned $sha failed" >&2; return 75; }
    git -C "$ADVISORY_DB_DIR" checkout -q FETCH_HEAD
}

run() {
    local base="$1"
    age_check "$PIN_FILE_DEFAULT" || return $?
    if ! needed "$base"; then
        echo "advisory_pin: no Cargo.toml/Cargo.lock/deny.toml change; advisories skipped (daily live audit covers)"
        return 0
    fi
    fetch_db || return $?
    cargo deny --disable-fetch --db-path "$ADVISORY_DB_DIR" check advisories
}

self_test() {
    local t fail=0 pin rc S=999ec89d5d3f10f20be14750ec8c2d8b44e0ac9b p
    t=$(mktemp -d); pin="$t/pin"
    row() { # name want-rc today sha date
        printf 'sha=%s\ndate=%s\n' "$4" "$5" > "$pin"
        rc=0; age_check "$pin" "$3" >/dev/null 2>&1 || rc=$?
        if [ "$rc" -eq "$2" ]; then echo "ok   $1"; else echo "FAIL $1: want rc $2 got $rc"; fail=1; fi
    }
    row "fresh pin (2 d) passes"            0  2026-10-19 "$S" 2026-10-17
    row "boundary (7 d) passes"             0  2026-10-24 "$S" 2026-10-17
    row "stale (8 d) is RED"                1  2026-10-25 "$S" 2026-10-17
    row "stale (30 d) is RED"               1  2026-11-16 "$S" 2026-10-17
    row "garbled date never passes (75)"    75 2026-10-19 "$S" notadate
    row "short sha never passes (75)"       75 2026-10-19 abc  2026-10-17
    row "empty sha never passes (75)"       75 2026-10-19 ""   2026-10-17
    rc=0; age_check "$t/missing" 2026-10-19 >/dev/null 2>&1 || rc=$?
    if [ "$rc" -eq 75 ]; then echo "ok   missing pin file is 75"; else echo "FAIL missing pin file: rc $rc"; fail=1; fi
    # mutation: with the age comparison disabled a 30-day-old pin must pass, proving the stale rows are live
    sed 's/-gt "\$MAX_AGE_DAYS"/-gt 99999/' "${BASH_SOURCE[0]}" > "$t/mut.sh"
    printf 'sha=%s\ndate=2026-10-17\n' "$S" > "$pin"
    rc=0; bash "$t/mut.sh" age "$pin" 2026-11-16 >/dev/null 2>&1 || rc=$?
    if [ "$rc" -eq 0 ]; then echo "ok   mutation (age check disabled) lets a 30-day pin through"; else echo "FAIL mutation not detected (rc $rc)"; fail=1; fi
    for p in Cargo.toml Cargo.lock crates/x/Cargo.toml deny.toml; do
        if printf '%s\n' "$p" | grep -qE "$NEEDED_RE"; then echo "ok   needed matches $p"; else echo "FAIL needed misses $p"; fail=1; fi
    done
    for p in docs/Cargo.md src/lib.rs crates/x/src/Cargo.tomlx; do
        if printf '%s\n' "$p" | grep -qE "$NEEDED_RE"; then echo "FAIL needed matches $p"; fail=1; else echo "ok   needed ignores $p"; fi
    done
    rm -rf "${t:?}"
    return "$fail"
}

case "${1:-}" in
    age) age_check "${2:?pin file}" "${3:-}" ;;
    needed) needed "${2:?base ref}" ;;
    run) run "${2:?base ref}" ;;
    fetch) fetch_db; echo "$ADVISORY_DB_DIR" ;;
    --self-test) self_test ;;
    *) echo "usage: advisory_pin.sh age|needed|run|--self-test" >&2; exit 2 ;;
esac
