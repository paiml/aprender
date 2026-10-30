#!/usr/bin/env bash
# tag_publish_andon.sh -- a tag that is not on crates.io is not a release (PMAT-5306)
#
#   bash scripts/release/tag_publish_andon.sh --repo DIR --state DIR --inbox FILE --escalate FILE [--window-min 120] [--since ISO8601Z] [--page-dry]
#
# Operator ruling 2026-10-30: "a release = the version is on crates.io and passed the clean-container smoke. A tag alone is
# not a release. A tag not on crates.io within 2 h of push is an andon that pages Noah live."
# For every v<semver> tag (no pre-release) created at or after --since and older than --window-min: crate `aprender` must show
# that version on crates.io. Absent = ANDON: an inbox line EVERY run while open, an escalate line and a live page ONCE per tag.
# A crates.io read that fails is NOT_MEASURED: inbox line, exit 2, no page, no decision.
# Exit 0 all clear, 1 andon fired or open, 2 usage / not measured.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
CRATE=aprender
repo="" state="" inbox="" esc="" win=120 since=2026-10-30T00:00:00Z dry=0
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
        --page-dry) dry=1; shift ;;
        *) usage "unknown argument: $1" ;;
    esac
done
[ -d "${repo:-/nonexistent}" ] && [ -n "$state" ] && [ -n "$inbox" ] && [ -n "$esc" ] || usage "--repo DIR --state DIR --inbox FILE --escalate FILE are required"
[[ $win =~ ^[0-9]+$ ]] || usage "bad --window-min '$win'"
since_s=$(date -u -d "$since" +%s 2> /dev/null) || usage "bad --since '$since'"
# shellcheck source=scripts/release/lib_crates_io.sh
. "$HERE/lib_crates_io.sh" || usage "cannot source lib_crates_io.sh"
install -d -- "$state" || usage "cannot create $state"
install -d -- "$(dirname "$esc")" || usage "cannot create the escalate dir"
exec 9> "${state:?}/.lock"
flock -n 9 || { say "another run holds the lock: nothing done"; exit 0; }
G() { git -C "$repo" "$@"; }

# page_text: quotes, backticks, $, ! and backslashes stripped (tmux send-keys / echo safe), <= 240 chars (merge-drought precedent)
page_text() { printf '%s' "$1" | tr -d "'\"\`\$!\\\\" | tr '\n' ' ' | cut -c1-240; }
page() { # page LINE: tmux page to a3 (send-keys -l then Enter) and the cop window (echo only); --page-dry prints
    local msg tgt; msg="OPERATOR-FACING ESCALATION aprender-flow-relay-free $(page_text "$1")"
    for tgt in apr:aprender-a3 apr:cop; do
        if [ "$dry" = 1 ]; then echo "PAGE $tgt: $msg"; continue; fi
        case "$tgt" in
            apr:cop) tmux send-keys -t "$tgt" "echo '$msg'" Enter 2> /dev/null ;;
            *) tmux send-keys -t "$tgt" -l "$msg" 2> /dev/null && tmux send-keys -t "$tgt" Enter 2> /dev/null ;;
        esac
    done
    return 0
}
# blocking_step TAG: the first missing step visible from here
blocking_step() {
    local tag=$1
    if ! G ls-remote --exit-code --tags origin "refs/tags/$tag" > /dev/null 2>&1 && G remote get-url origin > /dev/null 2>&1; then
        printf 'tag not on origin'; return
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
while IFS= read -r tag; do
    [[ $tag =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || continue
    ts=$(G for-each-ref --format='%(creatordate:unix)' "refs/tags/$tag" 2> /dev/null)
    [[ $ts =~ ^[0-9]+$ ]] || continue
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
    line="ANDON $tag pushed ${age} min ago and $CRATE $ver is NOT on crates.io (window ${win} min); blocking step: $step"
    say "$line"
    inbox_line "$line"
    if [ ! -e "$state/paged-$tag" ]; then
        printf '%s %s\n' "$(date -u +%FT%TZ)" "$line" >> "$esc" || usage "cannot write $esc"
        page "$line"
        : > "$state/paged-$tag"
    fi
done <<< "$tags"
[ "$open" = 1 ] && exit 1
[ "$nm" = 1 ] && exit 2
exit 0
