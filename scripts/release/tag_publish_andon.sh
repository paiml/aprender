#!/usr/bin/env bash
# tag_publish_andon.sh -- a tag that is not on crates.io is not a release (PMAT-5306)
#
#   bash scripts/release/tag_publish_andon.sh --repo DIR --state DIR --inbox FILE --escalate FILE (--page-target T ... | --page-dry) [--window-min 120] [--since ISO8601Z]
#
# Operator ruling 2026-10-30: "a release = the version is on crates.io and passed the clean-container smoke. A tag alone is
# not a release. A tag not on crates.io within 2 h of push is an andon that pages Noah live."
# For every v<semver> tag (no pre-release) pushed at or after --since and older than --window-min: crate `aprender` must show
# that version on crates.io. Absent = ANDON: an inbox line EVERY run while open, an escalate line and a live page ONCE per tag.
# A crates.io read that fails is NOT_MEASURED: inbox line, exit 2, no page, no decision.
# A page is stamped (once per tag) ONLY when at least one --page-target (verified with tmux has-session) accepted it; otherwise an inbox
# line "PAGE UNDELIVERED <tag>" is written and the next run retries.
# Exit 0 all clear, 1 andon fired or open, 2 usage / not measured, 3 another run holds the lock.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
CRATE=aprender
repo="" state="" inbox="" esc="" win=120 since=2026-10-30T00:00:00Z dry=0 targets=()
usage() { printf 'tag_publish_andon: %s\n' "$*" >&2; exit 2; }
say() { printf 'tag_publish_andon: %s\n' "$*"; }
inbox_line() {
    local t; t=$(date -u +%H:%MZ) # bashrs disable-line=DET002
    printf '%s | tag-andon | PMAT-5306 | %s\n' "$t" "$1" | cut -c1-300 >> "$inbox" 2> /dev/null || true
}
while [ $# -gt 0 ]; do
    case "$1" in
        --repo) [ $# -ge 2 ] || usage "--repo needs a value"; repo=$2; shift 2 ;;
        --state) [ $# -ge 2 ] || usage "--state needs a value"; state=$2; shift 2 ;;
        --inbox) [ $# -ge 2 ] || usage "--inbox needs a value"; inbox=$2; shift 2 ;;
        --escalate) [ $# -ge 2 ] || usage "--escalate needs a value"; esc=$2; shift 2 ;;
        --window-min) [ $# -ge 2 ] || usage "--window-min needs a value"; win=$2; shift 2 ;;
        --since) [ $# -ge 2 ] || usage "--since needs a value"; since=$2; shift 2 ;;
        --page-target) [ $# -ge 2 ] || usage "--page-target needs a value"; targets+=("$2"); shift 2 ;;
        --page-dry) dry=1; shift ;;
        *) usage "unknown argument: $1" ;;
    esac
done
[ -d "${repo:-/nonexistent}" ] && [ -n "$state" ] && [ -n "$inbox" ] && [ -n "$esc" ] || usage "--repo DIR --state DIR --inbox FILE --escalate FILE are required"
[ "$dry" = 1 ] || [ "${#targets[@]}" -gt 0 ] || usage "--page-target T (repeatable) is required unless --page-dry"
[[ $win =~ ^[0-9]+$ ]] || usage "bad --window-min '$win'"
since_s=$(date -u -d "$since" +%s 2> /dev/null) || usage "bad --since '$since'"
# shellcheck source=scripts/release/lib_crates_io.sh
. "$HERE/lib_crates_io.sh" || usage "cannot source lib_crates_io.sh"
install -d -- "$state" || usage "cannot create $state"
install -d -- "$(dirname "$esc")" || usage "cannot create the escalate dir"
exec 9> "${state:?}/.lock"
flock -n 9 || { say "another run holds the lock: nothing done"; exit 3; }
G() { git -C "$repo" "$@"; }

# page_text: quotes, backticks, $, ! and backslashes stripped (tmux send-keys / echo safe), <= 240 chars (merge-drought precedent)
page_text() { printf '%s' "$1" | tr -d "'\"\`\$!\\\\" | tr '\n' ' ' | cut -c1-240; }
page() { # page LINE: rc 0 only when at least one --page-target accepted it. Each target is verified with tmux has-session first;
    # a target whose name is "apr:cop" style is a bare shell (echo only), any other is a claude TUI (send-keys -l, then Enter). --page-dry prints.
    # "aprender-flow-relay-free" is the sender label merge-drought.sh pages with, so both andons read alike in the cop window
    local msg tgt delivered=0; msg="OPERATOR-FACING ESCALATION aprender-flow-relay-free $(page_text "$1")"
    if [ "$dry" = 1 ] && [ "${#targets[@]}" -eq 0 ]; then echo "PAGE (dry): $msg"; return 0; fi
    for tgt in "${targets[@]}"; do
        if [ "$dry" = 1 ]; then echo "PAGE $tgt: $msg"; delivered=1; continue; fi
        tmux has-session -t "${tgt%%:*}" > /dev/null 2>&1 || { say "page target $tgt: no such tmux session"; continue; }
        case "$tgt" in
            apr:cop) tmux send-keys -t "$tgt" "echo '$msg'" Enter > /dev/null 2>&1 && delivered=1 ;;
            *) tmux send-keys -t "$tgt" -l "$msg" > /dev/null 2>&1 && tmux send-keys -t "$tgt" Enter > /dev/null 2>&1 && delivered=1 ;;
        esac
    done
    [ "$delivered" = 1 ]
}
# blocking_step TAG: the first missing step visible from here
blocking_step() {
    local tag=$1
    if G remote get-url origin > /dev/null 2>&1; then
        G ls-remote --exit-code --tags origin "refs/tags/$tag" > /dev/null 2>&1
        case $? in
            0) ;;
            2) printf 'tag not on origin'; return ;;
            *) printf 'origin unreadable (tag-on-origin not measured); crates.io publish missing'; return ;;
        esac
    fi
    if ! command -v gh > /dev/null 2>&1; then printf "GitHub release state unreadable (no gh); crates.io publish missing"; return; fi
    if ! gh release view "$tag" --repo paiml/aprender > /dev/null 2>&1; then printf 'no GitHub release (assets/clean-room not started or incomplete)'; return; fi
    printf 'crates.io publish not started or incomplete (GitHub release exists)'
}

# the tag list must be current: a failed fetch is not measured, never "no tags"
if G remote get-url origin > /dev/null 2>&1 && ! G fetch -q --tags origin > /dev/null 2>&1; then
    inbox_line "NOT_MEASURED tag andon: cannot fetch tags from origin"; say "cannot fetch tags: not measured"; exit 2
fi
now=$(date -u +%s) # bashrs disable-line=DET002
tags=$(G tag --list 'v[0-9]*.[0-9]*.[0-9]*' 2> /dev/null) || usage "cannot list tags in $repo"
open=0 nm=0
# A tag is dated by its push, measured as the run that first saw it on origin (state/seen-<tag>; at most one timer period late).
# The first run cannot know when the tags already present were pushed, so it dates each by the newest lower bound it has:
# the tagger date of an annotated tag, the commit date of a lightweight one. Either can only date the push early (an early
# page); a tag older than --since by that bound is skipped. So the timer is installed before the next tag is pushed: a tag
# pushed between --since and the first run is the one case git cannot date (measured empty at install, 2026-10-30).
baseline=0; [ -e "$state/baseline" ] || baseline=1
tag_time() { # tag_time TAG -> the epoch the tag is dated by (see above)
    local t=$1 seen="$state/seen-$1" lw=1
    [ "$(G cat-file -t "refs/tags/$t" 2> /dev/null)" = tag ] && lw=0
    if [ ! -s "$seen" ]; then
        if [ "$baseline" = 1 ] && [ "$lw" = 0 ]; then G for-each-ref --format="%(taggerdate:unix)" "refs/tags/$t" > "$seen" 2> /dev/null || return 1
        elif [ "$baseline" = 1 ]; then G log -1 --format=%ct "refs/tags/$t" > "$seen" 2> /dev/null || return 1
        else
            echo "$now" > "$seen" || return 1
            [ "$lw" = 0 ] || inbox_line "LIGHTWEIGHT TAG $t: not cut by the release path; dated by first sight, not its commit"
        fi
    fi
    cat -- "$seen"
}
while IFS= read -r tag; do
    [[ $tag =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || continue
    ts=$(tag_time "$tag")
    [[ $ts =~ ^[0-9]+$ ]] || { nm=1; inbox_line "NOT_MEASURED tag $tag: cannot date it (state dir unwritable or git read failed); no decision"; continue; }
    [ "$ts" -ge "$since_s" ] || continue
    age=$(( (now - ts) / 60 ))
    [ "$age" -ge "$win" ] || continue
    ver=${tag#v}
    crates_io_has_version "$CRATE" "$ver"; rc=$?
    case "$rc" in
        0) continue ;;
        1) ;;
        *) nm=1; inbox_line "NOT_MEASURED tag $tag: crates.io read failed; no decision, no page"; continue ;;
    esac
    open=1
    step=$(blocking_step "$tag")
    line="ANDON $tag pushed ${age} min ago and $CRATE $ver is NOT on crates.io, absent or yanked (window ${win} min); blocking step: $step"
    say "$line"
    inbox_line "$line"
    if [ ! -e "$state/escalated-$tag" ]; then
        # a failed escalate write never blocks the live page below: inbox line, no stamp, retried next run
        if printf '%s %s\n' "$(date -u +%FT%TZ)" "$line" >> "$esc" 2> /dev/null; then
            : > "$state/escalated-$tag"
        else
            inbox_line "ESCALATE WRITE FAILED $tag: cannot append to $esc; paging anyway, retrying next run"
        fi
    fi
    if [ ! -e "$state/paged-$tag" ]; then
        # stamped ONLY on delivery: an undelivered page is an inbox line and a retry next run, never a silent "paged"
        if page "$line"; then : > "$state/paged-$tag"; else inbox_line "PAGE UNDELIVERED $tag: no page target accepted it; retrying next run"; fi
    fi
done <<< "$tags"
[ "$baseline" = 1 ] && : > "$state/baseline"
[ "$open" = 1 ] && exit 1
[ "$nm" = 1 ] && exit 2
exit 0
