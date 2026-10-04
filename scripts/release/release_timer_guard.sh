#!/usr/bin/env bash
# release_timer_guard.sh -- R10 (#4670): no timer installs a tool on a release host while a release is open.
#
# WHY. During the 0.70.1 release a user timer on the train host fired mid-dogfood and installed a new
# llama.cpp build. A tool on a release host changed under an open release, so the measurements taken
# before and after it were taken against two different tools.
#
# The fix has two halves. The TIMERS themselves (provisioned outside this repo) check a release-open
# marker and skip, with one log line, while it is set. THIS script is the other half: the pre-tag check.
# For every release host it lists the user timers named in tool-install-timers.txt and FAILS when one
# of them is armed (enabled, or active) while that host carries the release-open marker.
#
#   marker present + a tool-installing timer armed  -> FAIL (exit 1), the unit named
#   marker present + every one of them disarmed     -> ok
#   no marker                                       -> ok   (no release open on that host)
#   the host cannot be asked                        -> NOT_MEASURED (exit 2): never a pass
#   no marker writer (release_marker.sh) beside it  -> NOT_MEASURED: "no marker" would be vacuous
#   --release V and the marker absent, or for another version on a host
#                                                   -> NOT_MEASURED: the writer should have set it
#
# The marker is ${XDG_STATE_HOME:-$HOME/.local/state}/apr/release-open on each host, read on that host.
# This script only READS: it never enables, disables, starts or stops a timer, and never writes a marker.
#
# Every refusal is ONE line ending in a `# R-<NAME>` marker, so scripts/check_release_timer_guard.sh can
# delete it and prove its table turns red without it.
#
# EXIT  0 every host passes · 1 a tool-installing timer is armed during a release ·
#       2 NOT_MEASURED: a host could not be judged, or no marker writer exists · 3 caller error. Every non-zero stops the caller.
#
# SEAMS (the case table drives every row through them; production sets neither):
#   TIMER_GUARD_PROBE   a command run as `<probe> <local|ssh> <host> <units...>` instead of the real probe
#   TIMER_GUARD_LIST    the unit list (default: tool-install-timers.txt beside this script)
#   TIMER_GUARD_WRITER  the marker writer whose presence makes "no marker" mean something
#                       (default: release_marker.sh beside this script)
#
# USAGE
#   release_timer_guard.sh [--release VERSION] [--local NAME] [--ssh HOST]...
#   --release VERSION   the train calls it so: every host must carry the marker for VERSION
set -uo pipefail

PROG=${0##*/}
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LIST="${TIMER_GUARD_LIST:-$HERE/tool-install-timers.txt}"
WRITER="${TIMER_GUARD_WRITER:-$HERE/release_marker.sh}"
RELEASE=""

caller_error() { printf 'FAIL  R10 %s: caller error: %s\n' "$PROG" "$*"; exit 3; }

# The probe, run ON the host (locally, or as `bash -s` over ssh). Prints:
#   MARKER present <content>|absent
#   TIMER <unit> <is-enabled> <is-active>     one line per listed unit
#   END
# A user manager that cannot be reached prints no END: the caller reads that as could-not-judge.
# shellcheck disable=SC2016
PROBE_BODY='
m="${XDG_STATE_HOME:-$HOME/.local/state}/apr/release-open"
if [ -e "$m" ]; then printf "MARKER present %s\n" "$(head -c 64 "$m" | tr -cd "[:alnum:]._-")"; else echo "MARKER absent"; fi
systemctl --user show-environment >/dev/null 2>&1 || { echo "NOBUS systemctl --user cannot reach the user manager"; exit 0; }
for u in "$@"; do
  e=$(systemctl --user is-enabled "$u" 2>/dev/null); e=${e:-not-found}
  a=$(systemctl --user is-active "$u" 2>/dev/null); a=${a:-unknown}
  printf "TIMER %s %s %s\n" "$u" "$e" "$a"
done
echo END
'

real_probe() { # real_probe <local|ssh> <host> <units...>
    local how=$1 host=$2; shift 2
    case "$how" in
        local) bash -c "$PROBE_BODY" probe "$@" ;;
        ssh)   ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" bash -s -- "$@" <<<"$PROBE_BODY" ;;
    esac
}

probe() { if [ -n "${TIMER_GUARD_PROBE:-}" ]; then "$TIMER_GUARD_PROBE" "$@"; else real_probe "$@"; fi; }

# armed <is-enabled> <is-active>: 0 when the timer will fire (enabled in any form, or running now)
armed() {
    case "$1" in enabled|enabled-runtime|linked|linked-runtime|alias|indirect|generated|transient) return 0 ;; esac
    [ "$2" = active ] || [ "$2" = activating ]
}

# judge_host <local|ssh> <host> <units...> -> prints its rows; returns 0 ok, 1 armed, 2 could not judge
judge_host() {
    local how=$1 host=$2 out rc marker line u e a armed_units="" listed=0 nobus; shift 2
    out=$(probe "$how" "$host" "$@" 2>&1); rc=$?
    [ "$rc" -eq 0 ] || { printf 'NOT_MEASURED R10 %s: the probe exited %s, so its timers cannot be judged: %s\n' "$host" "$rc" "$(tail -n 1 <<<"$out")"; return 2; } # R-PROBE
    marker=$(sed -n 's/^MARKER //p' <<<"$out" | head -n 1)
    [ -n "$marker" ] || { printf 'NOT_MEASURED R10 %s: the probe printed no MARKER line, so whether a release is open is unknown\n' "$host"; return 2; } # R-NOMARKERLINE
    nobus=$(sed -n 's/^NOBUS //p' <<<"$out" | head -n 1)
    [ -z "$nobus" ] || { printf 'NOT_MEASURED R10 %s: %s, so its timers cannot be listed\n' "$host" "$nobus"; return 2; } # R-NOBUS
    grep -qx END <<<"$out" || { printf 'NOT_MEASURED R10 %s: the probe stopped before END, so its timer list is partial\n' "$host"; return 2; } # R-NOEND
    while read -r line; do
        read -r _ u e a <<<"$line"
        listed=$((listed+1))
        armed "$e" "$a" && armed_units="$armed_units $u($e/$a)"
    done < <(grep '^TIMER ' <<<"$out")
    [ "$listed" -eq "$#" ] || { printf 'NOT_MEASURED R10 %s: asked about %s timer(s), the probe answered %s\n' "$host" "$#" "$listed"; return 2; } # R-COUNT
    if [ -n "$RELEASE" ] && [ "$marker" = absent ]; then printf "NOT_MEASURED R10 %s: no release-open marker, but release %s is open: the writer did not set it\n" "$host" "$RELEASE"; return 2; fi # R-UNSET
    if [ -n "$RELEASE" ] && [ "${marker#present }" != "$RELEASE" ]; then printf "NOT_MEASURED R10 %s: the marker is for %s, not the open release %s\n" "$host" "${marker#present }" "$RELEASE"; return 2; fi # R-OTHERVERSION
    case "$marker" in
        absent)
            printf 'ok    R10 %s: no release-open marker; tool-installing timers armed:%s\n' "$host" "${armed_units:- none}"
            return 0 ;;
    esac
    [ -z "$armed_units" ] || { printf 'FAIL  R10 %s: release open (%s) and tool-installing timer(s) armed:%s -- a tool can change under the release\n' "$host" "$marker" "$armed_units"; return 1; } # R-ARMED
    printf 'ok    R10 %s: release open (%s); all %s tool-installing timer(s) disarmed\n' "$host" "$marker" "$listed"
    return 0
}

main() {
    local hosts=() units=() u worst=0 rc
    while [ $# -gt 0 ]; do
        case "$1" in
            --release)
                [[ ${2:-} =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || caller_error "--release needs a version" # R-NORELEASE
                RELEASE=$2; shift 2 ;;
            --local|--ssh)
                [ -n "${2:-}" ] || caller_error "$1 needs a host name" # R-NOHOSTNAME
                hosts+=("${1#--} $2"); shift 2 ;;
            *)
                caller_error "unknown argument '$1' (usage: $PROG [--release VERSION] [--local NAME] [--ssh HOST]...)" ;; # R-BADARG
        esac
    done
    [ "${#hosts[@]}" -gt 0 ] || caller_error "no release host named: a check of zero hosts is vacuous" # R-NOHOSTS
    [ -r "$LIST" ] || caller_error "cannot read the tool-installing timer list $LIST" # R-NOLIST
    while read -r u; do
        u=${u%%#*}; u=${u//[[:space:]]/}
        [ -n "$u" ] || continue
        [[ $u =~ ^[A-Za-z0-9@._-]+\.timer$ ]] || caller_error "'$u' in $LIST is not a timer unit name" # R-BADUNIT
        units+=("$u")
    done <"$LIST"
    [ "${#units[@]}" -gt 0 ] || caller_error "$LIST names no timer: a check of zero timers is vacuous" # R-EMPTYLIST
    [ -f "$WRITER" ] || { printf "NOT_MEASURED R10 no release-open marker writer at %s: an absent marker would mean nothing\n" "$WRITER"; exit 2; } # R-NOWRITER
    for h in "${hosts[@]}"; do
        # shellcheck disable=SC2086
        judge_host $h "${units[@]}"; rc=$?
        if [ "$rc" -eq 2 ] && [ "$worst" -ne 1 ]; then worst=2; fi
        [ "$rc" -eq 1 ] && worst=1
    done
    case "$worst" in
        0) printf 'ok    R10 TIMERS PASS release=%s hosts=%s timers=%s\n' "${RELEASE:-none}" "${#hosts[@]}" "${#units[@]}" ;;
        1) printf 'FAIL  R10 a tool-installing timer is armed on a release host while the release is open\n' ;;
        *) printf 'NOT_MEASURED R10 at least one release host could not be judged; not judged is not a pass\n' ;;
    esac
    exit "$worst"
}

main "$@"
