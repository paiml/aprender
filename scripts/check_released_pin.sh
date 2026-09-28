#!/usr/bin/env bash
# check_released_pin.sh — case table for scripts/released_pin.sh and the
# PV_BIN_REQUIRE=released path of scripts/pv_bin.sh (N-1 gate tools).
#
# Every row runs against fake pv binaries in a temp dir: no cargo, no network.
# Then each planted mutant (a sed edit of a COPY of the lib that deletes one
# rule) must turn at least one row RED, so a weakened rule cannot pass green.
#
#   check_released_pin.sh --self-test
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IDENT='(aprender provable-contracts verifier)'

self_test() {
    td=""
    td="$(mktemp -d)"
    trap 'rm -rf "${td:?}"' EXIT
    mkdir -p "$td/bin" "$td/tree/scripts" "$td/tree/crates/aprender-contracts-cli"
    mkpv() {  # mkpv NAME FIRST_LINE
        printf '#!/bin/sh\nprintf "%%s\\n" %q\n' "$2" > "$td/bin/$1"
        chmod +x "$td/bin/$1"
    }
    mkpv rel "pv 0.69.4 $IDENT"
    mkpv dev "pv 0.70.0 $IDENT"
    mkpv rc "pv 0.69.4-rc.1 $IDENT"
    mkpv other "pv 0.69.4 (pipe viewer)"
    mkpv old "pv 0.69.1 $IDENT"
    printf 'not executable\n' > "$td/bin/noexec"
    local rel_sha
    rel_sha=$(sha256sum "$td/bin/rel" | awk '{print $1}')

    # An end-to-end tree for pv_bin.sh: the three resolver files beside each other.
    cp "$HERE/pv_bin.sh" "$HERE/nightly_pin.sh" "$td/tree/scripts/"
    printf '[package]\nname = "aprender-contracts-cli"\nversion = "0.70.0"\n' \
        > "$td/tree/crates/aprender-contracts-cli/Cargo.toml"

    local lib fail rows
    # row NAME WANT(ok|refuse) DECLARED PIN [PV_BIN] [SHA] [CANDIDATES]
    row() {
        local name="$1" want="$2" decl="$3" pin="$4" bin="${5:-}" sha="${6:-}" cands="${7:-}"
        rm -f "$td/pv.pin" "$td/pv.pin.sha256"
        [ "$pin" = "-" ] || printf '%s\n' "$pin" > "$td/pv.pin"
        [ -z "$sha" ] || printf '%s\n' "$sha" > "$td/pv.pin.sha256"
        local got rc=0
        got=$(env -u PV_BIN_RELEASED_CANDIDATES bash -c '
            . "$1" || exit 9
            PV_BIN="$2" PV_BIN_RELEASED_PIN="$3" \
                PV_BIN_RELEASED_CANDIDATES="${5:-/nonexistent/pv}" \
                released_pin_resolve "$4" "$6"' _ \
            "$lib" "$bin" "$td/pv.pin" "$decl" "$cands" "$IDENT" 2>/dev/null) || rc=$?
        rows=$((rows + 1))
        local res=refuse
        [ "$rc" -eq 0 ] && [ -n "$got" ] && res=ok
        if [ "$res" != "$want" ]; then
            echo "FAIL $name: want $want got $res (rc=$rc out=$got)"; fail=1
        fi
    }
    # e2e NAME WANT PV_BIN — source the tree's pv_bin.sh under PV_BIN_REQUIRE=released.
    e2e() {
        local name="$1" want="$2" bin="$3" rc=0 got
        printf '0.69.4\n' > "$td/pv.pin"; rm -f "$td/pv.pin.sha256"
        got=$(cd "$td/tree" && env -u APR_FLEET_MARKER PV_BIN_REQUIRE=released \
            PV_BIN="$bin" PV_BIN_RELEASED_PIN="$td/pv.pin" \
            bash -c '. scripts/pv_bin.sh >/dev/null && printf "%s|%s\n" "$PV" "$PV_SAT"' 2>/dev/null) || rc=$?
        rows=$((rows + 1))
        local res=refuse
        [ "$rc" -eq 0 ] && [ "$got" = "$bin|$(dirname "$bin")/pv-sat" ] && res=ok
        if [ "$res" != "$want" ]; then
            echo "FAIL e2e $name: want $want got $res (rc=$rc out=$got)"; fail=1
        fi
    }
    run_rows() {
        fail=0; rows=0
        row "released pin, released binary: accepted"       ok     0.70.0 0.69.4 "$td/bin/rel"
        row "released pin under a -dev tree: accepted"      ok     0.70.0-dev 0.69.4 "$td/bin/rel"
        row "binary found via the candidate list"           ok     0.70.0 0.69.4 "" "" "/nonexistent/pv:$td/bin/noexec:$td/bin/rel"
        row "HEAD dev build (reports the tree): refused"    refuse 0.70.0 0.69.4 "$td/bin/dev"
        row "pin == tree version (dev build passes): refused" refuse 0.70.0 0.70.0 "$td/bin/dev"
        row "pin newer than tree: refused"                  refuse 0.69.3 0.69.4 "$td/bin/rel"
        row "rc pin: refused"                               refuse 0.70.0 0.69.4-rc.1 "$td/bin/rc"
        row "rc binary under a bare pin: refused"           refuse 0.70.0 0.69.4 "$td/bin/rc"
        row "+sha pin: refused"                             refuse 0.70.0 0.69.4+abc "$td/bin/rel"
        row "two-field pin: refused"                        refuse 0.70.0 0.69 "$td/bin/rel"
        row "binary older than the pin: refused"            refuse 0.70.0 0.69.4 "$td/bin/old"
        row "wrong identity (pipe viewer): refused"         refuse 0.70.0 0.69.4 "$td/bin/other"
        row "no pin file: refused"                          refuse 0.70.0 - "$td/bin/rel"
        row "empty pin: refused"                            refuse 0.70.0 "" "$td/bin/rel"
        row "no declared version: refused"                  refuse "" 0.69.4 "$td/bin/rel"
        row "non-executable PV_BIN: refused"                refuse 0.70.0 0.69.4 "$td/bin/noexec"
        row "no binary anywhere: refused"                   refuse 0.70.0 0.69.4 ""
        row "sha256 matches: accepted"                      ok     0.70.0 0.69.4 "$td/bin/rel" "$rel_sha"
        row "sha256 mismatch: refused"                      refuse 0.70.0 0.69.4 "$td/bin/rel" "$(printf '%064d' 0)"
        row "sha256 file malformed: refused"                refuse 0.70.0 0.69.4 "$td/bin/rel" "nothex"
        cp "$lib" "$td/tree/scripts/released_pin.sh"
        e2e "pv_bin.sh accepts the released pv, exports PV_SAT beside it" ok "$td/bin/rel"
        e2e "pv_bin.sh refuses a HEAD dev build in released mode" refuse "$td/bin/dev"
    }

    echo "== check_released_pin.sh --self-test =="
    lib="$HERE/released_pin.sh"
    run_rows
    [ "$fail" -eq 0 ] || return 1
    echo "self-test: $rows/$rows rows"

    # HEAD mode unchanged: without the knob, the released binary is still STALE.
    local rc=0
    (cd "$td/tree" && env -u PV_BIN_REQUIRE PV_BIN="$td/bin/rel" \
        APR_FLEET_MARKER="$td/no-marker" bash -c '. scripts/pv_bin.sh' >/dev/null 2>&1) || rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "FAIL: HEAD mode accepted a released 0.69.4 under a 0.70.0 tree (default was weakened)"; return 1
    fi
    echo "HEAD mode: released pv still refused without PV_BIN_REQUIRE=released (rc=$rc)"

    # Planted mutants: each deletes one rule from a copy of the lib and must turn
    # a row RED.
    local m killed=0 total=0
    local -a muts=(
        'R1-suffix|s/[*][[][!]0-9.[]][*] [|] //'
        'R2-older|s/if \[ "\$rp_cmp" -ne 0 \]; then/if false; then/'
        'R4-version|s/if \[ "\$rp_ver" != "\$rp_pin" \]; then/if false; then/'
        'R4-identity|s/\*"\$rp_identity"\*) ;;/*) ;;/'
        'R5-sha|s/if \[ -e "\$rp_pin_file.sha256" \]; then/if false; then/'
    )
    # Not listed, measured EQUIVALENT: deleting the pin-readable check (a missing
    # pin then reads as '' and R1 refuses it) or the executable check (running a
    # non-executable binary yields no version and R4 refuses it). Those two checks
    # exist for their messages; the refusal does not depend on them.
    for m in "${muts[@]}"; do
        total=$((total + 1))
        sed -e "${m#*|}" "$HERE/released_pin.sh" > "$td/mut.sh"
        if cmp -s "$td/mut.sh" "$HERE/released_pin.sh"; then
            echo "MUTANT NOT APPLIED: ${m%%|*} (the sed matched nothing)"; continue
        fi
        lib="$td/mut.sh"
        run_rows > "$td/mut.out" 2>&1 || true
        if [ "$fail" -eq 1 ]; then killed=$((killed + 1)); else echo "SURVIVED mutant: ${m%%|*}"; fi
    done
    echo "mutants: $killed/$total RED"
    [ "$killed" -eq "$total" ]
}

case "${1:-}" in
    --self-test) self_test ;;
    *) echo "usage: check_released_pin.sh --self-test" >&2; exit 64 ;;
esac
