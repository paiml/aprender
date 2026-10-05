# shellcheck shell=bash
# nightly_pick.sh (lib) — one candidate C per night, published as an immutable ref (see scripts/release/nightly_pick.sh).
#
# SOURCED, so option-neutral: no shell options are changed here, nothing exits; every function fails by return status.
#   . scripts/lib/nightly_pick.sh || exit 1
#
# The night's ref is refs/heads/nightly/<NIGHT>, NIGHT = the UTC date of (pick time - 12 h): a night runs 12:00Z to 12:00Z,
# so a scheduled pick that fires hours late still lands on the night it was meant for. The ref is created once and
# never moved: a second pick the same night keeps the first C, whatever main is by then. Every producer, the train
# and the release read C from that ref, never from main's head.
# It is a BRANCH because repository rulesets cover only branches and tags: a ruleset on refs/heads/nightly/* (update,
# deletion and non-fast-forward refused, creation allowed, no bypass) makes it immutable on the server as well. The
# pick pushes with the workflow's GITHUB_TOKEN, so creating the branch fires no workflow. Only that exact name is C:
# the rolling tag refs/tags/nightly, or any other ref whose name merely ends in the night's name, is never read as C.
#
# Return codes: 0 ok; 1 refused (RED); 2 not_measured (could not read or write the remote); 3 caller error.
#
#   np_night [EPOCH]              print the night of EPOCH (default now)
#   np_date DAYS                  print the UTC date DAYS after 1970-01-01 (no date -d: portable to BSD date)
#   np_days YYYY-MM-DD            print its days after 1970-01-01; 3 for a day the calendar lacks
#   np_epoch YYYY-MM-DDTHH:MM:SSZ print its epoch; 3 for any other form. No date -d anywhere: the darwin runner has BSD date
#   np_pick REPO REMOTE NIGHT SHA create refs/heads/nightly/NIGHT -> SHA unless it exists; prints "C=<sha>" + PICKED|KEPT
#   np_resolve REPO REMOTE NIGHT  print the night's C; 2 when the night has no pick
#   np_verify REPO REMOTE NIGHT SHA
#                                 0 when SHA is the night's C, 1 when it is not, 2 when the night has no pick

np_night() { # [EPOCH]
    local e="${1:-}" s
    [ -n "$e" ] || e="$(date -u +%s)" || return 3
    case "$e" in '' | *[!0-9]*) printf 'NP caller error: epoch %s is not a number\n' "$e" >&2; return 3 ;; esac
    s=$((e - 43200))
    np_date $((s >= 0 ? s / 86400 : -((86399 - s) / 86400)))
}

np_date() { # DAYS since 1970-01-01 -> YYYY-MM-DD, in integer arithmetic: date -d @N is GNU-only and the darwin
    # runner's BSD date refuses it (H. Hinnant's civil_from_days, proleptic Gregorian)
    local z era doe yoe doy mp m
    z=$(($1 + 719468))
    era=$(((z >= 0 ? z : z - 146096) / 146097)); doe=$((z - era * 146097))
    yoe=$(((doe - doe / 1460 + doe / 36524 - doe / 146096) / 365))
    doy=$((doe - (365 * yoe + yoe / 4 - yoe / 100))); mp=$(((5 * doy + 2) / 153))
    m=$((mp < 10 ? mp + 3 : mp - 9))
    printf '%04d-%02d-%02d\n' $((yoe + era * 400 + (m <= 2))) "$m" $((doy - (153 * mp + 2) / 5 + 1))
}

np_days() { # YYYY-MM-DD -> days since 1970-01-01 (H. Hinnant's days_from_civil); 3 for a date the calendar lacks
    local y m d days back
    [[ $1 =~ ^([0-9]{4})-([0-9]{2})-([0-9]{2})$ ]] || return 3
    # a leading 1 keeps "08" decimal: 108 - 100
    y=$((1${BASH_REMATCH[1]} - 10000)); m=$((1${BASH_REMATCH[2]} - 100)); d=$((1${BASH_REMATCH[3]} - 100))
    [ "$m" -gt 2 ] || y=$((y - 1))
    days=$(((y / 400) * 146097 + (y % 400) * 365 + (y % 400) / 4 - (y % 400) / 100 + (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1 - 719468))
    # back through np_date: 2026-02-30 comes back 2026-03-02, so it is refused
    back="$(np_date "$days")"
    [ "$back" = "$1" ] || return 3
    printf '%s\n' "$days"
}

np_epoch() { # YYYY-MM-DDTHH:MM:SSZ (the form workflow_run.created_at takes) -> epoch; 3 for any other form
    local ymd hh mm ss days
    [[ $1 =~ ^([0-9]{4}-[0-9]{2}-[0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})Z$ ]] || return 3
    ymd="${BASH_REMATCH[1]}"; hh=$((1${BASH_REMATCH[2]} - 100)); mm=$((1${BASH_REMATCH[3]} - 100)); ss=$((1${BASH_REMATCH[4]} - 100))
    (( hh < 24 && mm < 60 && ss < 60 )) || return 3
    days="$(np_days "$ymd")" || return 3
    printf '%s\n' $((days * 86400 + hh * 3600 + mm * 60 + ss))
}

np__args() { # NIGHT [SHA]
    # [[ =~ ]] anchors the whole string; `printf | grep -x` matched per line, so "2026-10-09<newline>x" passed
    [[ $1 =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || { printf 'NP caller error: night %s is not YYYY-MM-DD\n' "$1" >&2; return 3; }
    [ "$#" -lt 2 ] || [[ $2 =~ ^[0-9a-f]{40}$ ]] || { printf 'NP caller error: sha %s is not 40 hex\n' "$2" >&2; return 3; }
}

np__read() { # REPO REMOTE NIGHT -> the ref's sha on stdout; 0 found, 1 absent, 2 unreadable
    local out rc=0
    out="$(git -C "$1" ls-remote --refs "$2" "refs/heads/nightly/$3" 2>/dev/null)" || rc=$?
    [ "$rc" -eq 0 ] || return 2
    # exact name only. ls-remote matches a pattern by its tail, and so does a push destination, so another ref ending in
    # refs/heads/nightly/N (a branch refs/heads/refs/heads/nightly/N) would be read as C, and would swallow the pick's push.
    # Such a name makes the night ambiguous: unreadable (2), never a C.
    if printf '%s\n' "$out" | awk -v r="refs/heads/nightly/$3" 'NF && $2 != r { found = 1 } END { exit !found }'; then
        printf 'NP: refs/heads/nightly/%s is ambiguous on the remote (another ref ends in that name)\n' "$3" >&2; return 2
    fi
    out="$(printf '%s\n' "$out" | awk -v r="refs/heads/nightly/$3" '$2 == r { print $1; exit }')"
    [ -n "$out" ] || return 1
    printf '%s\n' "$out"
}

np_resolve() { # REPO REMOTE NIGHT
    local c rc=0
    np__args "$3" || return 3
    c="$(np__read "$1" "$2" "$3")" || rc=$?
    case "$rc" in
        0) printf '%s\n' "$c" ;;
        1) printf 'NP NOT_MEASURED: night %s has no pick\n' "$3" >&2; return 2 ;;
        *) printf 'NP NOT_MEASURED: cannot read refs/heads/nightly/%s\n' "$3" >&2; return 2 ;;
    esac
}

np_verify() { # REPO REMOTE NIGHT SHA
    local c rc=0
    np__args "$3" "$4" || return 3
    c="$(np_resolve "$1" "$2" "$3")" || rc=$?
    [ "$rc" -eq 0 ] || return "$rc"
    if [ "$c" = "$4" ]; then printf 'NP OK: %s is the C of night %s\n' "${4:0:10}" "$3"; return 0; fi
    printf 'NP RED: %s is not the C of night %s (C=%s) -- measure C, not main\n' "${4:0:10}" "$3" "${c:0:10}"
    return 1
}

np_pick() { # REPO REMOTE NIGHT SHA
    local c rc=0
    np__args "$3" "$4" || return 3
    c="$(np__read "$1" "$2" "$3")" || rc=$?
    if [ "$rc" -eq 0 ]; then
        printf 'C=%s KEPT night %s (picked earlier; main now %s does not move it)\n' "$c" "$3" "${4:0:10}"; return 0
    fi
    [ "$rc" -eq 1 ] || { printf 'NP NOT_MEASURED: cannot read refs/heads/nightly/%s; nothing picked\n' "$3"; return 2; }
    # C must be on main: a dispatch from a branch must never become the night
    git -C "$1" fetch -q "$2" main 2>/dev/null || { printf 'NP NOT_MEASURED: cannot fetch main; nothing picked\n'; return 2; }
    rc=0; git -C "$1" merge-base --is-ancestor "$4" FETCH_HEAD 2>/dev/null || rc=$?
    case "$rc" in
        0) ;;
        1) printf 'NP RED: %s is not on main; nothing picked\n' "${4:0:10}"; return 1 ;;
        *) printf 'NP NOT_MEASURED: cannot tell whether %s is on main (merge-base rc=%s); nothing picked\n' "${4:0:10}" "$rc"; return 2 ;;
    esac
    # create-only: the lease with an empty expected value accepts the push only while the ref does not exist. A plain
    # push is NOT create-only: it fast-forwards an existing branch, and main only moves
    # forward, so a racing later pick would move the night's C.
    rc=0; git -C "$1" push -q --force-with-lease="refs/heads/nightly/$3:" "$2" "$4:refs/heads/nightly/$3" 2>/dev/null || rc=$?
    c="$(np__read "$1" "$2" "$3")" || { printf 'NP NOT_MEASURED: refs/heads/nightly/%s absent or unreadable after the push (push rc=%s)\n' "$3" "$rc"; return 2; }
    if [ "$c" = "$4" ]; then printf 'C=%s PICKED night %s\n' "$c" "$3"; return 0; fi
    printf 'C=%s KEPT night %s (another pick won the race)\n' "$c" "$3"; return 0
}
