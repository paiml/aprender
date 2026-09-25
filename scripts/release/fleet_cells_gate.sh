#!/usr/bin/env bash
# fleet_cells_gate.sh -- no rc is cut while a fleet cell is RED, unless a dated waiver names it (#4328 C4)
#
#   fleet_cells_gate.sh --cells FILE [--waivers FILE] [--now EPOCH]   judge; rc 0 cut, 1 refuse
#   fleet_cells_gate.sh --self-test                                   case table + mutants
#
# rc.2 was cut with intel on an old apr, mini unable to install anything (no darwin asset)
# and jetson unreachable: every one of those cells was RED and nothing on the cut path read
# them. The rc tagger must run this gate before it writes the tag (see the WIRING row).
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
fleet_cells_verdict() {
    CELLS=$1 WAIVERS=$2 NOW=$3 MAX_AGE_H=${FLEET_CELLS_MAX_AGE_H:-6} python3 - <<'EOF'
import datetime as dt, os, sys
now = dt.datetime.fromtimestamp(int(os.environ["NOW"]), dt.timezone.utc)
today = now.date().isoformat()
def waived(host, binary):  # host "*" asks for the `*\t*` key only (missing/stale cells)
    for line in os.environ["WAIVERS"].splitlines():
        f = line.split("\t")
        if not line.strip() or line.lstrip().startswith("#") or len(f) < 4 or not f[3].strip():
            continue
        if host == "*":
            hit = f[0] == "*" and f[1] == "*"
        else:
            hit = f[0] in (host, "*") and f[1] in (binary, "*")
        if hit and f[2] >= today:
            return f"{f[0]}/{f[1]} until {f[2]}: {f[3]}"
    return None
lines = os.environ["CELLS"].splitlines()
stamp = next((l.split(None, 2)[2] for l in lines if l.startswith("# measured ") and len(l.split()) >= 3), "")
bad, notes, n = [], [], 0
try:
    measured = dt.datetime.strptime(stamp.strip(), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    age_h = (now - measured).total_seconds() / 3600
    if age_h > float(os.environ["MAX_AGE_H"]):
        bad.append(f"cells are stale: measured {stamp.strip()}, {age_h:.1f} h ago (max {os.environ['MAX_AGE_H']} h)")
except ValueError:
    bad.append("cells carry no '# measured <YYYY-MM-DDTHH:MM:SSZ>' stamp")
for l in lines:
    if not l.strip() or l.startswith("#"):
        continue
    f = l.split("\t")
    if len(f) < 3:
        bad.append(f"malformed cell: {l!r}")
        continue
    n += 1
    host, binary, state = f[0], f[1], f[2]
    reason = f[3] if len(f) > 3 else ""
    if state == "GREEN":
        continue
    if state == "RED":
        w = waived(host, binary)
        if w:
            notes.append(f"{host}/{binary} RED waived ({w})")
        else:
            bad.append(f"{host}/{binary} RED: {reason or 'no reason given'}")
        continue
    bad.append(f"{host}/{binary} state {state!r} is neither GREEN nor RED")
if n == 0:
    bad.append("no fleet cells")
data_bad = [b for b in bad if b.startswith(("cells ", "no fleet cells"))]
if data_bad and len(data_bad) == len(bad):
    w = waived("*", "*")
    if w:
        notes.append(f"cells unusable, waived ({w}): " + "; ".join(data_bad))
        bad = []
if bad:
    print("refuse " + "; ".join(bad))
    sys.exit(1)
print(f"ok {n} fleet cells, none RED unwaived" + ("; " + "; ".join(notes) if notes else ""))
EOF
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
    for m in 's/            bad.append(f"{host}\/{binary} RED: /            pass  # /' 's/^    sys.exit(1)$/    sys.exit(0)/'; do
        mut=$d/m.sh; sed "$m" "${BASH_SOURCE[0]}" > "$mut"
        if cmp -s "$mut" "${BASH_SOURCE[0]}"; then echo "  FAIL mutant '$m' not built: the anchor moved"; fail=1; continue; fi
        got=$(bash -c ". '$mut' --source-only; fleet_cells_verdict \"\$(printf '%b' '$S\nintel\tapr\tRED\told\n')\" '' $now" 2>&1); rc=$?
        if [ "$rc" = 0 ]; then echo "  ok   mutant lets the RED cell through: the refusal is load-bearing"
        else echo "  FAIL mutant still refuses (rc $rc): $got"; fail=1; fi
    done
    # WIRING: the rc tagger runs this gate before it writes the tag, and dies on a refusal.
    # rc_cut.sh (#4314) was superseded by the 2026-09-25 16:58 ruling (rc = tag on a queue-green
    # main, e6's tagger). Until that tagger calls this gate, this row is RED on purpose: an
    # unwired gate blocks nothing. RC_TAGGER names the tagger script once it exists.
    local cut; cut="${RC_TAGGER:-$(dirname -- "${BASH_SOURCE[0]}")/rc_cut.sh}"
    local call tagline
    call=$(grep -n 'fleet_cells_gate.sh' "$cut" | grep -v '^\s*[0-9]*:\s*#' | head -n 1 | cut -d: -f1)
    tagline=$(grep -nF 'out=$(api_post git/refs' "$cut" | head -n 1 | cut -d: -f1)
    if [ -n "$call" ] && [ -n "$tagline" ] && [ "$call" -lt "$tagline" ]; then echo "  ok   $(basename -- "$cut") runs the gate (line $call) before it writes the tag (line $tagline)"
    else echo "  FAIL UNWIRED: $(basename -- "$cut") does not run fleet_cells_gate.sh before creating the tag (call=${call:-none}, tag=${tagline:-none})"; fail=1; fi
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
