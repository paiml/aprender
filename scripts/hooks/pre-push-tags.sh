#!/usr/bin/env bash
# pre-push-tags.sh: a pre-push hook that refuses every tag push except an armed release tag.
#
# Refused, for every line git hands the hook whose remote ref is under refs/tags/:
#   - a delete (new sha is all zeros);
#   - a move of a tag the remote already has (old sha set and different), with or without -f;
#   - a create of anything but vX.Y.Z or vX.Y.Z-rc.N (keep/* and every other name,
#     lightweight or annotated);
#   - a release tag create without an unexpired marker naming that tag and sha, or whose
#     commit is not on the remote's main.
# One refused line refuses the whole push. Allowed: branch pushes (other guards own them) and
# a tag the remote already has at the same sha.
#
# The release marker is a one-shot file in this work tree's git dir. --arm-release writes it
# with an expiry (600 s, or SECONDS, at most 3600); the next push that runs the hook removes it
# whatever that push carries. A push with nothing to update never runs the hook, so the expiry
# bounds how long an unspent marker stays armed. Neither the marker nor its expiry is ever read
# from the environment: a variable left exported would arm every later push.
#
# usage:
#   pre-push-tags.sh REMOTE URL < pre-push stdin     run by git as (part of) the pre-push hook
#   pre-push-tags.sh --arm-release vX.Y.Z[-rc.N] [SECONDS]
#                                                    arm the next push for that one tag
#   pre-push-tags.sh --install                       copy this file and pre-push-dispatch.sh into
#                                                    the clone's hooks dir, before any pre-push there
#   pre-push-tags.sh --verify                        RED unless this clone's pre-push is the
#                                                    dispatcher and the guard beside it is this file
# exit: 0 allowed (or verified), 1 refused (or RED), 2 usage or environment error.
set -euo pipefail

SELF="$(cd "$(dirname "$0")" && pwd)/${0##*/}"
DISPATCH_MARK='# written by pre-push-tags.sh --install'
TTL_DEFAULT=600
TTL_MAX=3600

is_zero() {
    case "$1" in '' | *[!0]*) return 1 ;; esac
}

is_count() {
    case "$1" in '' | *[!0-9]*) return 1 ;; esac
}

is_release_name() {
    [[ "$1" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-rc\.[0-9]+)?$ ]]
}

marker_file() {
    printf '%s/pre-push-release-tag\n' "$(git rev-parse --absolute-git-dir)"
}

hooks_dir() {
    printf '%s/hooks\n' "$(git rev-parse --path-format=absolute --git-common-dir)"
}

refuse() {
    printf 'pre-push-tags: REFUSED %s\n' "$1" >&2
    BAD=1
}

# on_main SHA URL: SHA peels to a commit that the remote's main contains
on_main() {
    local commit main
    commit="$(git rev-parse --verify -q "$1^{commit}")" || return 1
    main="$(git ls-remote "$2" refs/heads/main)" || return 1
    main="${main%%[[:space:]]*}"
    [ -n "$main" ] || return 1
    git merge-base --is-ancestor "$commit" "$main" 2>/dev/null
}

hook() {
    local url="${2:-${1:-}}" mfile marker='' mref='' msha='' mexp='' now lref lsha rref rsha
    BAD=0
    mfile="$(marker_file)"
    if [ -e "$mfile" ] || [ -L "$mfile" ]; then
        [ -L "$mfile" ] || marker="$(cat -- "$mfile")"
        rm -f -- "${mfile:?}"
    fi
    read -r mref msha mexp _ <<< "$marker" || true
    # the marker's expiry is a deadline by design, so the clock is an input here
    now="$(date +%s)" # bashrs disable-line=DET005
    while read -r lref lsha rref rsha; do
        case "$rref" in refs/tags/*) ;; *) continue ;; esac
        if is_zero "$lsha"; then refuse "delete of $rref"; continue; fi
        if ! is_zero "$rsha"; then
            [ "$rsha" = "$lsha" ] || refuse "move of $rref from $rsha to $lsha"
            continue
        fi
        case "$rref" in
            refs/tags/v*) ;;
            *) refuse "create of $rref, which is not a release tag"; continue ;;
        esac
        if ! is_release_name "${rref#refs/tags/}"; then
            refuse "create of $rref, which is not vX.Y.Z or vX.Y.Z-rc.N"; continue
        fi
        if [ "$mref $msha" != "$rref $lsha" ]; then
            refuse "create of $rref at $lsha without its release marker (--arm-release)"; continue
        fi
        if ! is_count "$mexp" || [ "$now" -gt "$mexp" ]; then
            refuse "create of $rref: its release marker expired; arm it again"; continue
        fi
        if ! on_main "$lsha" "$url"; then
            refuse "create of $rref: $lsha is not on the remote's main"; continue
        fi
        printf 'pre-push-tags: allowed armed release tag %s\n' "$rref" >&2
    done
    if [ "$BAD" -ne 0 ]; then
        printf 'pre-push-tags: tags are never pushed except an armed release tag; nothing was pushed\n' >&2
        return 1
    fi
}

arm_release() {
    local tag="${1:-}" ttl="${2:-$TTL_DEFAULT}" ref sha
    if ! is_release_name "$tag"; then
        printf 'pre-push-tags: --arm-release takes vX.Y.Z or vX.Y.Z-rc.N, not [%s]\n' "$tag" >&2
        return 2
    fi
    if ! is_count "$ttl" || [ "$ttl" -lt 1 ] || [ "$ttl" -gt "$TTL_MAX" ]; then
        printf 'pre-push-tags: SECONDS is a whole number from 1 to %s, not [%s]\n' "$TTL_MAX" "$ttl" >&2
        return 2
    fi
    ref="refs/tags/$tag"
    if ! sha="$(git rev-parse --verify -q "$ref")"; then
        printf 'pre-push-tags: no local tag %s\n' "$ref" >&2
        return 2
    fi
    printf '%s %s %s\n' "$ref" "$sha" "$(($(date +%s) + ttl))" > "$(marker_file)"
    printf 'pre-push-tags: armed %s %s for the next push only, for %s s\n' "$ref" "$sha" "$ttl"
}

# install_hook: the guard and the dispatcher into the clone's common hooks dir; a foreign
# pre-push there becomes pre-push.chained and still runs after the guard
install_hook() {
    local dir here
    here="$(dirname "$SELF")"
    if git config --get core.hooksPath > /dev/null; then
        printf 'not_installed: core.hooksPath is set; add pre-push-tags.sh to that pre-push by hand\n' >&2
        return 2
    fi
    dir="$(hooks_dir)"
    mkdir -p "${dir:?}"
    if [ -e "${dir:?}/pre-push" ] && ! grep -qxF "$DISPATCH_MARK" "${dir:?}/pre-push"; then
        if [ -e "${dir:?}/pre-push.chained" ]; then
            printf 'not_installed: %s/pre-push and pre-push.chained both exist\n' "$dir" >&2
            return 2
        fi
        (cd "${dir:?}" && mv -- pre-push pre-push.chained)
    fi
    install -m 0755 "$SELF" "${dir:?}/pre-push-tags"
    install -m 0755 "${here:?}/pre-push-dispatch.sh" "${dir:?}/pre-push"
    printf 'installed: %s/pre-push runs pre-push-tags first\n' "$dir"
}

# verify_hook: RED when git would not run this guard on the next push. Run it from the tracked
# scripts/hooks/, whose pre-push-dispatch.sh is what --install wrote; a pmat (or any) reinstall
# that rewrote pre-push, a removed or edited guard, or core.hooksPath each turn it RED
verify_hook() {
    local dir here red=0
    here="$(dirname "$SELF")"
    if [ ! -f "$here/pre-push-dispatch.sh" ]; then
        printf 'not_measured: run --verify from the tracked scripts/hooks/, not from %s\n' "$here" >&2
        return 2
    fi
    dir="$(hooks_dir)"
    if git config --get core.hooksPath > /dev/null; then
        printf 'pre-push-tags: verify RED: core.hooksPath is set, so git never runs %s/pre-push\n' "$dir"
        red=1
    fi
    if [ ! -x "$dir/pre-push" ] || ! cmp -s "$here/pre-push-dispatch.sh" "$dir/pre-push"; then
        printf 'pre-push-tags: verify RED: %s/pre-push is not the dispatcher (missing, not executable, or rewritten by another installer); run --install\n' "$dir"
        red=1
    fi
    if [ ! -x "$dir/pre-push-tags" ] || ! cmp -s "$SELF" "$dir/pre-push-tags"; then
        printf 'pre-push-tags: verify RED: %s/pre-push-tags is missing, not executable, or not this file; run --install\n' "$dir"
        red=1
    fi
    if [ "$red" -ne 0 ]; then return 1; fi
    printf 'pre-push-tags: verified: %s/pre-push runs this guard first\n' "$dir"
}

case "${1:-}" in
    --arm-release) arm_release "${2:-}" "${3:-}" ;;
    --install) install_hook ;;
    --verify) verify_hook ;;
    --help | -h) sed -n '2,28p' "$0" ;;
    -*) sed -n '20,28p' "$0" >&2; exit 2 ;;
    *) hook "$@" ;;
esac
