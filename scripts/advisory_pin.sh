#!/usr/bin/env bash
# advisory_pin.sh -- P9 option 4 (PMAT-6110): the merge path reads a PINNED RustSec snapshot
# (ci/advisory-db.pin), not the live DB. Deterministic, still a GATE; the live DB is read by
# advisory-andon.yml (main, every 2 h) and advisory-live-nightly.yml (invocation check).
#
#   advisory_pin.sh age <pin-file> [<today YYYY-MM-DD>]   F5: red if the pin was last verified more than MAX_AGE_DAYS ago
#   advisory_pin.sh place <base-deny.toml> <dir>          fetch the pinned sha into <dir>, print the derived config path
#   advisory_pin.sh deny <config> [cargo-deny args]       cargo deny --all-features check advisories, offline, on that config
#   advisory_pin.sh --self-test                           case table (fresh/boundary/stale/malformed + mutation)
#
# Exit: 0 pass, 1 red, 75 could not check (a missing/garbled pin NEVER passes), 3 cargo-deny version
# whose db directory name was not derived (place only; the caller then keeps the live fetch).
set -euo pipefail
MAX_AGE_DAYS="${MAX_AGE_DAYS:-7}"
PIN_FILE_DEFAULT="${PIN_FILE_DEFAULT:-ci/advisory-db.pin}"

age_check() {
    local pin="$1" today="${2:-$(date -u +%F)}" sha d v pd vd td age
    [ -r "$pin" ] || { echo "advisory_pin: COULD NOT CHECK: no pin file $pin" >&2; return 75; }
    sha=$(sed -n 's/^sha=//p' "$pin")
    d=$(sed -n 's/^date=//p' "$pin")
    v=$(sed -n 's/^verified=//p' "$pin")
    case "$sha" in *[!0-9a-f]*|"") echo "advisory_pin: COULD NOT CHECK: bad sha in $pin" >&2; return 75 ;; esac
    [ "${#sha}" -eq 40 ] || { echo "advisory_pin: COULD NOT CHECK: sha is not 40 hex" >&2; return 75; }
    # date -d "" is today and date -d tomorrow is a relative date: both would read as fresh, so the
    # shape is checked before the parser is trusted.
    for f in "$d" "$v" "$today"; do
        grep -qE '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' <<<"$f" \
            || { echo "advisory_pin: COULD NOT CHECK: '$f' is not YYYY-MM-DD" >&2; return 75; }
    done
    pd=$(date -u -d "$d" +%s 2>/dev/null) && vd=$(date -u -d "$v" +%s 2>/dev/null) && td=$(date -u -d "$today" +%s 2>/dev/null) \
        || { echo "advisory_pin: COULD NOT CHECK: unparseable date in $pin" >&2; return 75; }
    # Staleness is the time since the bump job last CONFIRMED the sha (verified=), not the age of the
    # advisory-db commit: a quiet RustSec week must not turn a current pin red.
    age=$(( (td - vd) / 86400 ))
    if [ "$age" -lt 0 ] || [ "$vd" -lt "$pd" ]; then
        echo "advisory_pin: COULD NOT CHECK: verified=$v is in the future or before date=$d" >&2
        return 75
    fi
    if [ "$age" -gt "$MAX_AGE_DAYS" ]; then
        echo "advisory_pin: RED: pin last verified $v, $age days ago (max $MAX_AGE_DAYS); the bump job did not run" >&2
        return 1
    fi
    echo "advisory_pin: ok: pin verified $v, $age days ago (max $MAX_AGE_DAYS)" >&2
}

# cargo-deny keeps the database in <db-path>/advisory-db-<hash of the default URL>. That hash was
# derived on 0.19.x (3157b0e258782691); other versions are not assumed to share it.
DENY_DB_SUBDIR="advisory-db-3157b0e258782691"
DENY_VERSION_PIN="0.19"

# place BASE_DENY_TOML DIR -> prints DIR/deny.toml, the base config with db-path = DIR
place() {
    local base_cfg="$1" dir="$2" sha db
    case "$(cargo deny --version)" in
        "cargo-deny $DENY_VERSION_PIN."*) ;;
        *) echo "advisory_pin: cargo-deny is not $DENY_VERSION_PIN.x; the pinned db directory name was not derived for it" >&2; return 3 ;;
    esac
    age_check "$PIN_FILE_DEFAULT" || return $?
    sha=$(sed -n 's/^sha=//p' "$PIN_FILE_DEFAULT")
    rm -rf "${dir:?}"
    db="$dir/$DENY_DB_SUBDIR"
    mkdir -p "$db"
    git -C "$db" init -q
    git -C "$db" fetch -q --depth 1 https://github.com/RustSec/advisory-db.git "$sha" \
        || { echo "advisory_pin: COULD NOT CHECK: fetch of pinned $sha failed" >&2; return 75; }
    git -C "$db" checkout -q FETCH_HEAD
    [ "$(git -C "$db" rev-parse HEAD)" = "$sha" ] || { echo "advisory_pin: COULD NOT CHECK: checked-out HEAD is not the pin" >&2; return 75; }
    sed "s|^\[advisories\]\$|[advisories]\ndb-path = \"$dir\"|" "$base_cfg" > "$dir/deny.toml"
    grep -q '^db-path = ' "$dir/deny.toml" \
        || { echo "advisory_pin: COULD NOT CHECK: no [advisories] table to carry db-path" >&2; return 75; }
    echo "$dir/deny.toml"
}

# deny CONFIG [args] -- --disable-fetch, so a missing or mis-named db directory is an error,
# never a silent live fetch.
deny_pinned() {
    local cfg="$1"; shift
    cargo deny --all-features "$@" check --config "$cfg" --disable-fetch advisories
}

self_test() {
    local t fail=0 pin rc S=999ec89d5d3f10f20be14750ec8c2d8b44e0ac9b
    t=$(mktemp -d); pin="$t/pin"
    row() { # name want-rc today sha date [verified, default = date]
        printf 'sha=%s\ndate=%s\nverified=%s\n' "$4" "$5" "${6-$5}" > "$pin"
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
    row "empty date never passes (75)"      75 2026-10-19 "$S" ""
    row "relative date never passes (75)"   75 2026-10-19 "$S" tomorrow
    row "future verified never passes (75)" 75 2026-10-19 "$S" 2026-10-17 2026-10-25
    row "empty verified never passes (75)"  75 2026-10-19 "$S" 2026-10-17 ""
    row "quiet RustSec week: old commit date, verified today, passes" 0 2026-10-19 "$S" 2026-10-01 2026-10-19
    row "verified 8 d ago is RED even if commit is recent" 1 2026-10-27 "$S" 2026-10-17 2026-10-19
    rc=0; age_check "$t/missing" 2026-10-19 >/dev/null 2>&1 || rc=$?
    if [ "$rc" -eq 75 ]; then echo "ok   missing pin file is 75"; else echo "FAIL missing pin file: rc $rc"; fail=1; fi
    # mutation: with the age comparison disabled a 30-day-old pin must pass, proving the stale rows are live
    sed 's/-gt "\$MAX_AGE_DAYS"/-gt 99999/' "${BASH_SOURCE[0]}" > "$t/mut.sh"
    printf 'sha=%s\ndate=2026-10-17\nverified=2026-10-17\n' "$S" > "$pin"
    rc=0; bash "$t/mut.sh" age "$pin" 2026-11-16 >/dev/null 2>&1 || rc=$?
    if [ "$rc" -eq 0 ]; then echo "ok   mutation (age check disabled) lets a 30-day pin through"; else echo "FAIL mutation not detected (rc $rc)"; fail=1; fi
    # the committed pin itself must be well-formed (shape only; its age is the F5 row, judged on the live date)
    rc=0; age_check "$PIN_FILE_DEFAULT" "$(sed -n 's/^verified=//p' "$PIN_FILE_DEFAULT")" >/dev/null 2>&1 || rc=$?
    if [ "$rc" -eq 0 ]; then echo "ok   committed pin is well-formed"; else echo "FAIL committed pin is malformed (rc $rc)"; fail=1; fi
    if [ "${ADVISORY_PIN_LIVE_CASE:-0}" = 1 ]; then
        # the invocation itself, with the real binary on a one-crate fixture against the pinned snapshot
        # (fails if the binary or the fetch is missing; needs network, so never on the merge path)
        local fx="$t/fx" cfg
        mkdir -p "$fx/src"; printf '[package]\nname = "fx"\nversion = "0.1.0"\nedition = "2021"\n' > "$fx/Cargo.toml"
        : > "$fx/src/lib.rs"; printf 'version = 4\n\n[[package]]\nname = "fx"\nversion = "0.1.0"\n' > "$fx/Cargo.lock"
        printf '[advisories]\nignore = []\n' > "$fx/deny.toml"
        rc=0; cfg=$(place "$fx/deny.toml" "$t/db" 2>"$t/place.err") || rc=$?
        if [ "$rc" -eq 0 ]; then rc=0; (cd "$fx" && GIT_CEILING_DIRECTORIES="$t" deny_pinned "$cfg") >"$t/deny.out" 2>&1 || rc=$?; fi
        if [ "$rc" -eq 0 ]; then echo "ok   cargo-deny $(cargo deny --version | cut -d' ' -f2) accepts the pinned invocation"; else echo "FAIL pinned cargo-deny invocation rc $rc"; tail -n 5 "$t/place.err" "$t/deny.out" 2>/dev/null; fail=1; fi
    fi
    rm -rf "${t:?}"
    return "$fail"
}

case "${1:-}" in
    age) age_check "${2:?pin file}" "${3:-}" ;;
    place) place "${2:?base deny.toml}" "${3:?dir}" ;;
    deny) cfg="${2:?config}"; shift 2; deny_pinned "$cfg" "$@" ;;
    --self-test) self_test ;;
    *) echo "usage: advisory_pin.sh age|place|deny|--self-test" >&2; exit 2 ;;
esac
