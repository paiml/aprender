#!/usr/bin/env bash
# fleet_cells_gate.sh -- no rc is cut while a fleet cell is RED, unless a dated waiver names it (#4328 C4)
#
#   fleet_cells_gate.sh --cells FILE [--waivers FILE] [--now EPOCH]   judge; rc 0 cut, 1 refuse
#   fleet_cells_gate.sh --self-test                                   case table + mutants
#
# rc.2 was cut with intel on an old apr, mini unable to install anything (no darwin asset)
# and jetson unreachable: every one of those cells was RED and nothing on the cut path read
# them. rc_cut.sh now runs this gate before it writes the tag.
#
# CELLS come from infra-64's andon monitor (#4328 C3), published where the clean-room cut
# job can read it: `fleet/cells.tsv` on the `fleet-state` branch of this repo.
#   # measured 2026-09-24T17:58:00Z
#   <host>\t<binary>\t<GREEN|RED>\t<reason>
# A retired host is REMOVED from the cells, not listed RED (C7: jetson, 2026-09-24).
#
# WAIVERS are recorded in git, dated, and expire: scripts/release/fleet-waivers.tsv
#   <host|*>\t<binary|*>\t<until YYYY-MM-DD>\t<reason>
# `*\t*` is the only key that covers MISSING or STALE cells -- no data is not green data.
#
# Refuses: any unwaived RED cell · a state other than GREEN/RED · no cells · no `# measured`
# stamp · cells older than FLEET_CELLS_MAX_AGE_H (default 6) hours. EXIT 0 cut · 1 refuse · 2 usage.
set -uo pipefail
PROG=fleet_cells_gate

# fleet_cells_verdict <cells text> <waivers text> <now epoch> -> `ok ...` rc 0 | `refuse ...` rc 1
# bash + awk only (ARB-AUD-10: no interpreter beyond the shell on the release path). The stamp is
# parsed here; awk sees only epochs, the UTC date and the two texts.
fleet_cells_verdict() {
    local stamp measured=-1 today
    today=$(date -u -d "@$3" +%F) || return 2
    stamp=$(printf '%s\n' "$1" | awk '/^# measured / && NF >= 3 { sub(/^# measured +/, ""); sub(/[ \t]+$/, ""); print; exit }')
    if [[ $stamp =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]]; then
        measured=$(date -u -d "$stamp" +%s 2>/dev/null) || measured=-1
    fi
    printf '%s\n' "$1" | WAIVERS=$2 awk -F'\t' -v now="$3" -v measured="$measured" -v stamp="$stamp" \
        -v max_age="${FLEET_CELLS_MAX_AGE_H:-6}" -v today="$today" '
    # waived(host, binary): host "*" asks for the `*\t*` key only (missing/stale cells)
    function waived(host, binary,    nw, w, i, f, hit) {
        nw = split(ENVIRON["WAIVERS"], w, "\n")
        for (i = 1; i <= nw; i++) {
            if (w[i] ~ /^[ \t]*$/ || w[i] ~ /^[ \t]*#/) continue
            if (split(w[i], f, "\t") < 4 || f[4] ~ /^[ \t]*$/) continue
            if (host == "*") hit = (f[1] == "*" && f[2] == "*")
            else hit = ((f[1] == host || f[1] == "*") && (f[2] == binary || f[2] == "*"))
            if (hit && (f[3] "") >= today) return f[1] "/" f[2] " until " f[3] ": " f[4]
        }
        return ""
    }
    function add(msg) { bad[++nb] = msg }
    # Data faults (stale, unstamped, empty) are the only ones `*\t*` may waive.
    BEGIN {
        if (measured < 0) { add("cells carry no \x27# measured <YYYY-MM-DDTHH:MM:SSZ>\x27 stamp"); nd++ }
        else if ((now - measured) / 3600 > max_age + 0) {
            add(sprintf("cells are stale: measured %s, %.1f h ago (max %s h)", stamp, (now - measured) / 3600, max_age)); nd++
        }
    }
    /^[ \t]*$/ || /^#/ { next }
    NF < 3 { add("malformed cell: \x27" $0 "\x27"); next }
    {
        n++
        if ($3 == "GREEN") next
        if ($3 == "RED") {
            w = waived($1, $2)
            if (w != "") notes[++nn] = $1 "/" $2 " RED waived (" w ")"
            else add($1 "/" $2 " RED: " ($4 != "" ? $4 : "no reason given"))
            next
        }
        add($1 "/" $2 " state \x27" $3 "\x27 is neither GREEN nor RED")
    }
    END {
        if (n == 0) { add("no fleet cells"); nd++ }
        if (nd > 0 && nd == nb) {
            w = waived("*", "*")
            if (w != "") {
                msg = bad[1]; for (i = 2; i <= nb; i++) msg = msg "; " bad[i]
                notes[++nn] = "cells unusable, waived (" w "): " msg
                nb = 0
            }
        }
        if (nb > 0) {
            msg = bad[1]; for (i = 2; i <= nb; i++) msg = msg "; " bad[i]
            print "refuse " msg
            exit 1
        }
        msg = ""; for (i = 1; i <= nn; i++) msg = msg "; " notes[i]
        print "ok " n + 0 " fleet cells, none RED unwaived" msg
    }'
}

self_test() {
    local fail=0 got rc now d mut
    now=$(date -u -d 2026-09-24T18:00:00Z +%s)
    local S='# measured 2026-09-24T17:58:00Z'
    row() {  # row <want rc> <label> <cells> [waivers]
        got=$(fleet_cells_verdict "$(printf '%b' "$3")" "$(printf '%b' "${4:-}")" "$now"); rc=$?
        if [ "$rc" = "$1" ]; then echo "  ok   $2 -> rc $rc"; else printf '  FAIL %s: want rc %s, got %s\n       %s\n' "$2" "$1" "$rc" "$got"; fail=1; fi
    }
    echo "$PROG self-test"
    row 0 'every cell GREEN' "$S\nlambda\tapr\tGREEN\t\nmini\tapr\tGREEN\t\n"
    row 1 'ONE RED cell refuses the cut' "$S\nlambda\tapr\tGREEN\t\nintel\tapr\tRED\tv0.69.1 on PATH\n"
    row 0 'a dated waiver covers that cell' "$S\nintel\tapr\tRED\told\n" 'intel\tapr\t2026-09-24\tintel reimage, cop 2026-09-24\n'
    row 0 'a host-wide waiver covers any binary on it' "$S\nintel\tpv\tRED\told\n" 'intel\t*\t2026-09-30\treimage\n'
    row 1 'an EXPIRED waiver covers nothing' "$S\nintel\tapr\tRED\told\n" 'intel\tapr\t2026-09-23\tx\n'
    row 1 'a waiver for another host covers nothing' "$S\nintel\tapr\tRED\told\n" 'mini\tapr\t2026-12-31\tx\n'
    row 1 'a waiver with no reason is not recorded' "$S\nintel\tapr\tRED\told\n" 'intel\tapr\t2026-12-31\t\n'
    row 1 'an unknown state refuses' "$S\nyoga\tapr\tAMBER\t\n"
    row 1 'no cells refuses: no data is not green' "$S\n"
    row 1 'no measured stamp refuses' 'lambda\tapr\tGREEN\t\n'
    row 1 'stale cells refuse' '# measured 2026-09-24T09:00:00Z\nlambda\tapr\tGREEN\t\n'
    row 0 'only *\t* waives missing cells' '' '*\t*\t2026-09-24\tC3 not publishing yet, cop 2026-09-24\n'
    row 1 'a host waiver does not waive missing cells' '' 'intel\t*\t2026-09-30\tx\n'
    row 1 'a waiver for another binary covers nothing' "$S\nintel\tapr\tRED\told\n" '*\tpv\t2026-09-30\tx\n'
    # MUTANTS: the refusal made a no-op must let the RED cell through.
    d=$(mktemp -d) || return 2
    for m in 's/            else add(\$1 "\/" \$2 " RED: "/            else ("\/" " RED: "/' 's/^            exit 1$/            exit 0/'; do
        mut=$d/m.sh; sed "$m" "${BASH_SOURCE[0]}" > "$mut"
        if cmp -s "$mut" "${BASH_SOURCE[0]}"; then echo "  FAIL mutant '$m' not built: the anchor moved"; fail=1; continue; fi
        got=$(bash -c ". '$mut' --source-only; fleet_cells_verdict \"\$(printf '%b' '$S\nintel\tapr\tRED\told\n')\" '' $now" 2>&1); rc=$?
        if [ "$rc" = 0 ]; then echo "  ok   mutant lets the RED cell through: the refusal is load-bearing"
        else echo "  FAIL mutant still refuses (rc $rc): $got"; fail=1; fi
    done
    # WIRING: rc_cut.sh runs this gate before it writes the tag, and dies on a refusal.
    local cut; cut="$(dirname -- "${BASH_SOURCE[0]}")/rc_cut.sh"
    local call tagline
    call=$(grep -n 'fleet_cells_gate.sh' "$cut" | grep -v '^\s*[0-9]*:\s*#' | head -n 1 | cut -d: -f1)
    tagline=$(grep -nF 'out=$(api_post git/refs' "$cut" | head -n 1 | cut -d: -f1)
    if [ -n "$call" ] && [ -n "$tagline" ] && [ "$call" -lt "$tagline" ]; then echo "  ok   rc_cut.sh runs the gate (line $call) before it writes the tag (line $tagline)"
    else echo "  FAIL rc_cut.sh does not run fleet_cells_gate.sh before creating the tag (call=${call:-none}, tag=${tagline:-none})"; fail=1; fi
    rm -rf -- "${d:?}"
    if [ "$fail" = 0 ]; then echo "$PROG self-test: PASS"; else echo "$PROG self-test: FAIL"; fi
    return "$fail"
}

main() {
    local cells='' waivers='' now
    now=$(date -u +%s)
    while [ $# -gt 0 ]; do
        case "$1" in
            --self-test) self_test; return $? ;;
            --source-only) return 0 ;;
            --cells) cells=$2; shift 2 ;;
            --waivers) waivers=$2; shift 2 ;;
            --now) now=$2; shift 2 ;;
            *) echo "usage: $PROG --cells FILE [--waivers FILE] [--now EPOCH] | --self-test" >&2; return 2 ;;
        esac
    done
    [ -n "$cells" ] || { echo "usage: $PROG --cells FILE [--waivers FILE]" >&2; return 2; }
    fleet_cells_verdict "$(cat -- "$cells" 2>/dev/null)" "$([ -n "$waivers" ] && cat -- "$waivers" 2>/dev/null)" "$now"
}

if [ "${1:-}" = --source-only ]; then return 0 2>/dev/null || exit 0; fi
main "$@"
