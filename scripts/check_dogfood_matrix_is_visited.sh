#!/usr/bin/env bash
# check_dogfood_matrix_is_visited.sh -- the dogfood host matrix is DERIVED from the hosts the release
# train visits, and the gate fails in both directions (#3732; #3544 item (c), the fleet ruling).
#
# WHY. check_multiplatform_dogfood.sh demanded receipts for a hand-kept HOSTS list. The fleet lane's
# ruling on #3544 (2026-09-20): a receipt for a host the train never visited is a measurement nobody
# took, and a hand-narrowed list is a constant that drifts; so the matrix is derived from what the
# train actually reaches, and a host it reaches but does not demand is DECLARED absent
# (NA{reason, decided_by, date}), never silently absent. Under #3731 the train's hosts step VISITS
# every HOSTS member (a host receipt over ssh, the train host locally), so demanded-but-not-visited
# is closed by construction and guarded here against regression; visited-but-not-demanded is real:
# the CUDA release asset runs on yoga, which no matrix names (measured 2026-09-22).
#
# VISITED, derived, never re-listed:
#   V1  `scripts/release/autopilot.sh --visited` -- the hosts its hosts step reaches, printed from the
#       variables the step walks (release asset, installer, host receipts)
#   V2  the hosts the release-path workflows NAME: binary-release.yml's matrix `host:` fields and
#       b2-gpu.yml's runs-on host label. A POOL label (clean-room: "intel, yoga or gx10") names no
#       host, so it is REPORTed, never demanded
# DEMANDED: the gate's HOSTS; DECLARED ABSENT: the gate's NA_HOSTS ("host:reason:decided_by:date").
#   R1  every VISITED host is in HOSTS or in NA_HOSTS            (visited-but-not-demanded is RED)
#   R2  every HOSTS member is VISITED                           (demanded-but-not-visited is RED)
#   R3  no NA host is in HOSTS, and every NA host is VISITED     (a stale NA row is RED: it must leave
#                                                                when the host joins the matrix, or
#                                                                when nothing visits it any more)
#   R4  every NA row carries a reason, a decider and a YYYY-MM-DD date
#   R0  VISITED is non-empty and HOSTS is non-empty (vacuity is RED)
#
#   bash scripts/check_dogfood_matrix_is_visited.sh              # the tree
#   bash scripts/check_dogfood_matrix_is_visited.sh --self-test  # the case table, a mutant per row
# exit 0 green; 1 RED; 2 ENV (nothing was judged).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
GATE_REL="scripts/check_multiplatform_dogfood.sh"
AUTOPILOT_REL="scripts/release/autopilot.sh"
env_die() { printf 'ENV   %s -- nothing was judged, not a pass\n' "$*" >&2; exit 2; }
for t in python3 awk sed; do command -v "$t" > /dev/null 2>&1 || env_die "no $t"; done

# gate_value ROOT KEY -> the double-quoted value of KEY="..." in the gate
gate_value() { sed -n "s/^$2=\"\\(.*\\)\"$/\\1/p" "$1/$GATE_REL" | head -n 1; }
# workflow_hosts ROOT -> "host\tvia" for every host a release-path workflow names; "POOL\tlabel" for pools
workflow_hosts() {
    local r=$1 f
    f="$r/.github/workflows/binary-release.yml"
    # a matrix `host:` that names a POOL (clean-room) is a pool, not a host
    [ -f "$f" ] && sed -n 's/^[[:space:]]*host:[[:space:]]*\([a-z0-9-]*\)[[:space:]]*$/\1/p' "$f" | while read -r h; do
        case "$h" in clean-room) printf 'POOL\tclean-room (binary-release.yml matrix host)\n' ;; *) printf '%s\tbinary-release.yml matrix host\n' "$h" ;; esac
    done
    f="$r/.github/workflows/b2-gpu.yml"
    [ -f "$f" ] && sed -n 's/^[[:space:]]*runs-on:.*/&/p' "$f" | tr -d '[],"' | tr ' ' '\n' | grep -xE 'yoga|gx10|intel|mini|lambda' | sed 's/$/\tb2-gpu.yml runs-on label/'
    for f in "$r"/.github/workflows/binary-release.yml "$r"/.github/workflows/install-script.yml "$r"/.github/workflows/b2-gpu.yml; do
        [ -f "$f" ] && grep -qE 'runs-on:.*clean-room' "$f" && printf 'POOL\tclean-room (%s)\n' "${f##*/}"
    done
    return 0
}

# judge ROOT -> ok/FAIL/REPORT rows; rc 0/1; 2 when nothing could be judged
judge() {
    local r=$1 hosts na visited wf rc=0
    [ -f "$r/$GATE_REL" ] || { printf 'FAIL  no %s\n' "$GATE_REL"; return 2; }
    [ -f "$r/$AUTOPILOT_REL" ] || { printf 'FAIL  no %s\n' "$AUTOPILOT_REL"; return 2; }
    hosts=$(gate_value "$r" HOSTS); na=$(gate_value "$r" NA_HOSTS)
    visited=$(cd "$r" && bash "$AUTOPILOT_REL" --visited 2> /dev/null) || { printf 'FAIL  R0 `autopilot.sh --visited` failed: the visited set cannot be derived\n'; return 1; }
    wf=$(workflow_hosts "$r")
    python3 - "$hosts" "$na" "$visited" "$wf" <<'PY'
import re, sys
hosts, na, visited, wf = sys.argv[1:5]
H = hosts.split()
V = {}
for l in visited.splitlines():
    p = l.split()
    if len(p) >= 3 and p[0] == "VISITED": V.setdefault(p[1], set()).add(p[2])
pools = []
for l in wf.splitlines():
    if not l.strip(): continue
    h, via = l.split("\t", 1)
    if h == "POOL": pools.append(via)
    else: V.setdefault(h, set()).add(via)
bad = 0
if not H: print("FAIL  R0 HOSTS is empty"); sys.exit(1)
if not V: print("FAIL  R0 nothing is visited: the derivation read nothing"); sys.exit(1)
NA = {}
for row in [x for x in re.split(r"\s*\|\s*", na) if x.strip()] if "|" in na else ([na] if na.strip() else []):
    f = row.split(":", 3)
    if len(f) != 4 or not all(x.strip() for x in f) or not re.fullmatch(r"20[0-9]{2}-[01][0-9]-[0-3][0-9]", f[3].strip()):
        print(f"FAIL  R4 NA_HOSTS row is not host:reason:decided_by:YYYY-MM-DD: {row!r}"); bad = 1; continue
    NA[f[0].strip()] = (f[1].strip(), f[2].strip(), f[3].strip())
for h in sorted(V):
    if h in H: print(f"ok    {h:<7} visited ({', '.join(sorted(V[h]))}) and demanded")
    elif h in NA: print(f"REPORT {h:<6} visited ({', '.join(sorted(V[h]))}) and DECLARED absent: {NA[h][0]} -- {NA[h][1]}, {NA[h][2]}")
    else: print(f"FAIL  R1 {h} is VISITED ({', '.join(sorted(V[h]))}) but neither in HOSTS nor declared absent in NA_HOSTS: a host the train reaches and nothing demands"); bad = 1
for h in H:
    if h not in V: print(f"FAIL  R2 {h} is DEMANDED (HOSTS) but nothing visits it: its receipt would be a measurement nobody took"); bad = 1
for h in NA:
    if h in H: print(f"FAIL  R3 {h} is declared absent in NA_HOSTS and also in HOSTS: the NA row is stale, delete it"); bad = 1
    elif h not in V: print(f"FAIL  R3 {h} is declared absent in NA_HOSTS but nothing visits it any more: the NA row is stale, delete it"); bad = 1
for p in pools: print(f"REPORT pool   {p}: a pool label names no host; whichever member takes the job is not a matrix visit")
sys.exit(bad)
PY
    rc=$?
    return "$rc"
}

# ---- the case table ------------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
    TMP=$(mktemp -d) || exit 2
    cleanup() { case "${TMP:-}" in ''|/) return 0 ;; *) [ -d "$TMP" ] && rm -rf -- "$TMP" ;; esac; }
    trap cleanup EXIT
    fails=0; rows=0
    # fixture NAME HOSTS NA VISITED-LINES [WF-HOST-LINE] -> $TMP/NAME with a stub autopilot and gate
    fixture() {
        local d="$TMP/$1"
        mkdir -p "$d/scripts/release" "$d/.github/workflows" || return 2
        printf 'HOSTS="%s"\nNA_HOSTS="%s"\n' "$2" "$3" > "$d/scripts/check_multiplatform_dogfood.sh"
        printf '#!/usr/bin/env bash\n[ "${1:-}" = --visited ] || exit 2\nprintf "%%b\\n" "%s"\n' "$4" > "$d/scripts/release/autopilot.sh"
        printf '%s\n' "${5:-}" > "$d/.github/workflows/binary-release.yml"
    }
    row() { rows=$((rows + 1)); if [ "$2" = 0 ]; then printf 'ok    %s\n' "$1"; elif [ "$2" = 2 ]; then printf 'ENV   %s\n' "$1"; exit 2; else printf 'FAIL  %s: %s\n' "$1" "$3"; fails=$((fails + 1)); fi; }
    expect() { # NAME DIR WANT-RC NEEDLE
        local out rc; out=$(judge "$2" 2>&1); rc=$?
        [ "$rc" = "$3" ] && grep -qF -- "$4" <<< "$out"; row "$1" $? "rc=$rc (want $3): $(tr '\n' '|' <<< "$out" | cut -c1-300)"
    }
    V3='VISITED a host-receipt-local\nVISITED b host-receipt-ssh\nVISITED c release-asset'
    fixture f1 "a b" "c:asset smoke only:fleet:2026-09-20" "$V3"
    expect 'S1 control: a b demanded and visited, c visited and declared absent -> green' "$TMP/f1" 0 'REPORT c      visited'
    fixture f2 "a b" "" "$V3"
    expect 'S2 c visited, not demanded, not declared -> R1 (visited-but-not-demanded)' "$TMP/f2" 1 'R1 c is VISITED'
    fixture f3 "a b d" "c:asset smoke only:fleet:2026-09-20" "$V3"
    expect 'S3 d demanded, never visited -> R2 (demanded-but-not-visited)' "$TMP/f3" 1 'R2 d is DEMANDED'
    fixture f4 "a b c" "c:asset smoke only:fleet:2026-09-20" "$V3"
    expect 'S4 c joined HOSTS while still declared absent -> R3 stale NA' "$TMP/f4" 1 'R3 c is declared absent in NA_HOSTS and also in HOSTS'
    fixture f5 "a b" "c:asset smoke only:fleet:2026-09-20|e:gone:fleet:2026-01-01" "$V3"
    expect 'S5 an NA host nothing visits -> R3 stale NA' "$TMP/f5" 1 'R3 e is declared absent in NA_HOSTS but nothing visits it'
    fixture f6 "a b" "c:asset smoke only" "$V3"
    expect 'S6 an NA row without decider/date -> R4' "$TMP/f6" 1 'R4 NA_HOSTS row is not host:reason:decided_by:YYYY-MM-DD'
    fixture f7 "a b" "c:asset smoke only:fleet:2026-09-20" "$V3" '          - target: x
            host: w
            labels: x'
    expect 'S7 a workflow names host w that nothing demands -> R1 via the workflow' "$TMP/f7" 1 'R1 w is VISITED (binary-release.yml matrix host)'
    fixture f8 "" "" "$V3"
    expect 'S8 an empty HOSTS -> R0' "$TMP/f8" 1 'R0 HOSTS is empty'
    fixture f9 "a b" "" ""
    expect 'S9 nothing visited -> R0' "$TMP/f9" 1 'R0 nothing is visited'
    fixture f10 "a b" "c:asset smoke only:fleet:2026-09-20" "$V3" '    runs-on: [self-hosted, Linux, clean-room]'
    expect 'S10 a pool label is REPORTed, not demanded' "$TMP/f10" 0 'REPORT pool   clean-room (binary-release.yml)'

    mutant() { # NAME OLD NEW WANT-RC NEEDLE DIR
        local name=$1 old=$2 new=$3 want=$4 needle=$5 dir=$6
        python3 - "$0" "$TMP/mut-$name.sh" "$old" "$new" <<'PY' || { printf 'FAIL  mutant %s: anchor missing\n' "$name"; fails=$((fails + 1)); return; }
import sys
src, dst, old, new = sys.argv[1:5]
s = open(src).read()
if s.count(old) < 1: sys.exit(1)
open(dst, "w").write(s.replace(old, new, 1))
PY
        local out rc; out=$( . "$TMP/mut-$name.sh" --source-only; judge "$dir" 2>&1 ); rc=$?
        rows=$((rows + 1))
        if [ "$rc" = "$want" ] && grep -qF -- "$needle" <<< "$out"; then printf 'FAIL  mutant %s SURVIVED\n' "$name"; fails=$((fails + 1)); else printf 'ok    mutant %s killed\n' "$name"; fi
    }
    mutant drop-r1 '    else: print(f"FAIL  R1' '    elif False: print(f"FAIL  R1' 1 'R1 c is VISITED' "$TMP/f2"
    mutant drop-r2 '    if h not in V: print(f"FAIL  R2' '    if False: print(f"FAIL  R2' 1 'R2 d is DEMANDED' "$TMP/f3"
    mutant drop-r3 '    if h in H: print(f"FAIL  R3' '    if False: print(f"FAIL  R3' 1 'R3 c is declared absent in NA_HOSTS and also in HOSTS' "$TMP/f4"
    mutant drop-r4 '    if len(f) != 4 or not all(x.strip() for x in f) or not re.fullmatch' '    if False and len(f) != 4 or not all(x.strip() for x in f) and not re.fullmatch' 1 'R4 NA_HOSTS row' "$TMP/f6"
    mutant drop-workflows '    wf=$(workflow_hosts "$r")' '    wf=""' 1 'R1 w is VISITED' "$TMP/f7"

    [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED (%s of %s)\n' "$fails" "$rows"; exit 1; }
    printf '\nSELF-TEST PASSED (%s rows and mutants)\n' "$rows"
    exit 0
fi
[ "${1:-}" = "--source-only" ] && return 0 2> /dev/null

# ---- the tree ------------------------------------------------------------------------------------
printf '=== the dogfood matrix is derived from the hosts the train visits, both directions (#3732) ===\n'
judge "$ROOT"; rc=$?
[ "$rc" -eq 0 ] && printf 'PASS\n' || printf 'FAIL  (rc=%s)\n' "$rc"
exit "$rc"
