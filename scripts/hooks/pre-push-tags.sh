#!/usr/bin/env bash
# pre-push-tags.sh: a pre-push hook that refuses every tag push except an armed release tag.
#
# Refused, for every line git hands the hook whose remote ref is under refs/tags/:
#   - a delete (new sha is all zeros);
#   - a move of a tag the remote already has (old sha set and different), with or without -f;
#   - a create of anything but vX.Y.Z or vX.Y.Z-rc.N (keep/* and every other name,
#     lightweight or annotated);
#   - a release tag create without a marker naming that tag and sha, or whose commit is not
#     on the remote's main.
# One refused line refuses the whole push. Allowed: branch pushes (other guards own them) and
# a tag the remote already has at the same sha.
#
# The release marker is a one-shot file in this work tree's git dir. --arm-release writes it;
# the next push removes it whatever that push carries. It is never read from the environment:
# a variable left exported would arm every later push.
#
# usage:
#   pre-push-tags.sh REMOTE URL < pre-push stdin     run by git as (part of) the pre-push hook
#   pre-push-tags.sh --arm-release vX.Y.Z[-rc.N]     arm the next push for that one tag
#   pre-push-tags.sh --install                       copy this file and pre-push-dispatch.sh into
#                                                    the clone's hooks dir, before any pre-push there
# exit: 0 allowed, 1 refused, 2 usage or environment error.
set -euo pipefail

SELF="$(cd "$(dirname "$0")" && pwd)/${0##*/}"
DISPATCH_MARK='# written by pre-push-tags.sh --install'

is_zero() {
    case "$1" in '' | *[!0]*) return 1 ;; esac
}

is_release_name() {
    [[ "$1" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-rc\.[0-9]+)?$ ]]
}

marker_file() {
    printf '%s/pre-push-release-tag\n' "$(git rev-parse --absolute-git-dir)"
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
    local url="${2:-${1:-}}" mfile marker='' lref lsha rref rsha
    BAD=0
    mfile="$(marker_file)"
    if [ -e "$mfile" ] || [ -L "$mfile" ]; then
        [ -L "$mfile" ] || marker="$(cat -- "$mfile")"
        rm -f -- "${mfile:?}"
    fi
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
        if [ "$marker" != "$rref $lsha" ]; then
            refuse "create of $rref at $lsha without its release marker (--arm-release)"; continue
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
    local tag="${1:-}" ref sha
    if ! is_release_name "$tag"; then
        printf 'pre-push-tags: --arm-release takes vX.Y.Z or vX.Y.Z-rc.N, not [%s]\n' "$tag" >&2
        return 2
    fi
    ref="refs/tags/$tag"
    if ! sha="$(git rev-parse --verify -q "$ref")"; then
        printf 'pre-push-tags: no local tag %s\n' "$ref" >&2
        return 2
    fi
    printf '%s %s\n' "$ref" "$sha" > "$(marker_file)"
    printf 'pre-push-tags: armed %s %s for the next push only\n' "$ref" "$sha"
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
    dir="$(git rev-parse --path-format=absolute --git-common-dir)/hooks"
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

case "${1:-}" in
    --arm-release) arm_release "${2:-}" ;;
    --install) install_hook ;;
    --help | -h) sed -n '2,24p' "$0" ;;
    -*) sed -n '17,22p' "$0" >&2; exit 2 ;;
    *) hook "$@" ;;
esac
