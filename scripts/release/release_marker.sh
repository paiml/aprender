#!/usr/bin/env bash
# release_marker.sh -- the writer of the release-open marker (#4670, R10).
#
# WHY. R10 (release_timer_guard.sh) refuses the tag while a tool-installing timer is armed on a
# release host that carries the release-open marker, and the timers themselves skip while it is set.
# Both read a marker; without a writer neither ever sees a release open, so R10 would pass everywhere.
# The release autopilot calls this script: `set` at its first step, `clear` when the release is live.
#
#   set <version>     on every host: write <version> into the marker, then READ IT BACK. A marker
#                     already there for another version, or older than 24h, is stale: WARN, then
#                     overwrite. A host where the write or the read-back fails stops the train.
#   clear <version>   on every host: remove the marker if it names <version>, then read back that it
#                     is gone. A marker naming another version is not ours: WARN and leave it.
#
# The marker is ${XDG_STATE_HOME:-$HOME/.local/state}/apr/release-open on each host, written on that
# host; its content is the version. It touches nothing else: never a timer, never a unit.
#
# Every refusal and every warning is ONE line ending in a `# R-<NAME>` marker, so
# scripts/check_release_marker.sh can delete it and prove its table turns red without it.
#
# EXIT  0 every host done (warnings allowed) · 2 a host could not be set/cleared · 3 caller error.
#
# USAGE
#   release_marker.sh <set|clear> <version> [--local NAME] [--ssh HOST]...
set -uo pipefail

PROG=${0##*/}
STALE_S=86400

caller_error() { printf 'FAIL  MARKER %s: caller error: %s\n' "$PROG" "$*"; exit 3; }

# The body, run ON the host (locally, or as `bash -s` over ssh) as `<op> <version>`. Prints:
#   FOUND <content> <age-seconds> | FOUND-NONE      what was there before
#   WRITE-FAILED                                    set could not write
#   READBACK <content> | READBACK-ABSENT            what is there after
#   END
# shellcheck disable=SC2016
MARKER_BODY='
m="${XDG_STATE_HOME:-$HOME/.local/state}/apr/release-open"; op=$1; v=$2; old=""
if [ -e "$m" ]; then
  old=$(head -c 64 "$m" | tr -cd "[:alnum:]._-")
  printf "FOUND %s %s\n" "${old:-empty}" "$(( $(date +%s) - $(stat -c %Y "$m") ))"
else echo FOUND-NONE; fi
case "$op" in
  set)   { mkdir -p "${m%/*}" && printf "%s\n" "$v" > "$m"; } 2>/dev/null || echo WRITE-FAILED ;;
  clear) if [ "$old" = "$v" ]; then rm -f "${m:?}"; fi ;;
esac
if [ -e "$m" ]; then printf "READBACK %s\n" "$(head -c 64 "$m" | tr -cd "[:alnum:]._-")"; else echo READBACK-ABSENT; fi
echo END
'

run_body() { # run_body <local|ssh> <host> <op> <version>
    local how=$1 host=$2; shift 2
    case "$how" in
        local) bash -c "$MARKER_BODY" marker "$@" ;;
        ssh)   ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" bash -s -- "$@" <<<"$MARKER_BODY" ;;
    esac
}

# on_host <local|ssh> <host> <op> <version> -> prints its rows; returns 0 done, 2 not done
on_host() {
    local how=$1 host=$2 op=$3 v=$4 out rc found back old age
    out=$(run_body "$how" "$host" "$op" "$v" 2>&1); rc=$?
    [ "$rc" -eq 0 ] || { printf 'FAIL  MARKER %s: %s exited %s: %s\n' "$host" "$op" "$rc" "$(tail -n 1 <<<"$out")"; return 2; } # R-RUN
    grep -qx END <<<"$out" || { printf 'FAIL  MARKER %s: %s stopped before END\n' "$host" "$op"; return 2; } # R-NOEND
    found=$(grep -m1 -E '^FOUND' <<<"$out"); back=$(grep -m1 -E '^READBACK' <<<"$out")
    read -r _ old age <<<"$found"
    if [ -n "$old" ] && [ "$old" != "$v" ]; then printf 'WARN  MARKER %s: stale release-open marker for %s (not %s) found\n' "$host" "$old" "$v"; fi # R-STALEVERSION
    if [ "$old" = "$v" ] && [ "${age:-0}" -gt "$STALE_S" ]; then printf 'WARN  MARKER %s: stale release-open marker for %s, %ss old (over 24h)\n' "$host" "$old" "$age"; fi # R-STALEAGE
    case "$op" in
        set)
            ! grep -qx WRITE-FAILED <<<"$out" || { printf 'FAIL  MARKER %s: could not write the release-open marker\n' "$host"; return 2; } # R-WRITE
            [ "$back" = "READBACK $v" ] || { printf 'FAIL  MARKER %s: read back "%s", not %s: the release is not marked open\n' "$host" "${back:-nothing}" "$v"; return 2; } # R-READBACK
            printf 'ok    MARKER %s: release %s marked open (read back)\n' "$host" "$v" ;;
        clear)
            if [ -n "$old" ] && [ "$old" != "$v" ]; then
                [ "$back" = "READBACK $old" ] || { printf 'FAIL  MARKER %s: the marker for %s changed under clear %s: read back "%s"\n' "$host" "$old" "$v" "${back:-nothing}"; return 2; } # R-FOREIGN
                printf 'ok    MARKER %s: left the marker for %s in place (not ours)\n' "$host" "$old"
                return 0
            fi
            [ "$back" = READBACK-ABSENT ] || { printf 'FAIL  MARKER %s: the marker for %s is still there after clear: %s\n' "$host" "$v" "${back:-nothing}"; return 2; } # R-CLEAR
            printf 'ok    MARKER %s: release %s marker cleared (read back absent)\n' "$host" "$v" ;;
    esac
    return 0
}

main() {
    local op=${1:-} v=${2:-} hosts=() h worst=0 rc
    case "$op" in set|clear) ;; *) caller_error "first argument must be set or clear, not '$op'" ;; esac # R-BADOP
    [[ $v =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || caller_error "'$v' is not a version" # R-BADVERSION
    shift 2
    while [ $# -gt 0 ]; do
        case "$1" in
            --local|--ssh)
                [ -n "${2:-}" ] || caller_error "$1 needs a host name" # R-NOHOSTNAME
                hosts+=("${1#--} $2"); shift 2 ;;
            *)
                caller_error "unknown argument '$1' (usage: $PROG <set|clear> <version> [--local NAME] [--ssh HOST]...)" ;; # R-BADARG
        esac
    done
    [ "${#hosts[@]}" -gt 0 ] || caller_error "no release host named: marking zero hosts is vacuous" # R-NOHOSTS
    for h in "${hosts[@]}"; do
        # shellcheck disable=SC2086
        on_host $h "$op" "$v"; rc=$?
        [ "$rc" -eq 0 ] || worst=2
    done
    if [ "$worst" -eq 0 ]; then
        printf 'ok    MARKER %s %s hosts=%s\n' "$op" "$v" "${#hosts[@]}"
    else
        printf 'FAIL  MARKER %s %s: at least one host was not done\n' "$op" "$v"
    fi
    exit "$worst"
}

main "$@"
