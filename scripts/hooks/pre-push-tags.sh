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
# whatever that push carries, even a push with nothing to update. The expiry bounds a marker that
# no push spends, and the hook refuses one that expires later than now + 3600. Neither the
# marker nor its expiry is ever read from the environment: an exported variable would arm every push.
#
# usage:
#   pre-push-tags.sh REMOTE URL < pre-push stdin     run by git as (part of) the pre-push hook
#   pre-push-tags.sh --arm-release vX.Y.Z[-rc.N] [SECONDS]
#                                                    arm the next push for that one tag
#   pre-push-tags.sh --install                       copy this file and pre-push-dispatch.sh into
#                                                    the clone's hooks dir, before any pre-push there
#   pre-push-tags.sh --uninstall                     undo --install; the chained pre-push goes back only
#                                                    if its sha256 matches the one recorded at install
#   pre-push-tags.sh --verify [--all-worktrees]      RED unless git, in this work tree (or in every
#                                                    one), runs the dispatcher and this guard first
# exit: 0 allowed (or verified), 1 refused (or RED), 2 usage or environment error.
set -euo pipefail

SELF="$(cd "$(dirname "$0")" && pwd)/${0##*/}"
HERE="${SELF%/*}"
DISPATCH_MARK='# written by pre-push-tags.sh --install'
TTL_DEFAULT=600
TTL_MAX=3600

is_zero() {
    case "$1" in '' | *[!0]*) return 1 ;; esac
}

is_count() {
    case "$1" in '' | *[!0-9]*) return 1 ;; esac
}

# epoch_now: seconds since the epoch, from the bash builtin clock (no date(1), no environment)
epoch_now() {
    printf '%(%s)T\n' -1
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
    # the expiry is a deadline by design, so the clock is an input here
    now="$(epoch_now)"
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
        if [ "$mexp" -gt "$((now + TTL_MAX))" ]; then
            refuse "create of $rref: its release marker expires after the $TTL_MAX s cap; arm it again"; continue
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
    local tag="${1:-}" ttl="${2:-$TTL_DEFAULT}" ref sha exp
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
    exp="$(($(epoch_now) + ttl))"
    printf '%s %s %s\n' "$ref" "$sha" "$exp" > "$(marker_file)"
    printf 'pre-push-tags: armed %s %s for the next push only, for %s s\n' "$ref" "$sha" "$ttl"
}

# install_hook: the guard and the dispatcher into the clone's common hooks dir; a foreign
# pre-push there becomes pre-push.chained and still runs after the guard
install_hook() {
    local dir here sha
    here="$HERE"
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
    if [ -e "${dir:?}/pre-push.chained" ]; then
        # noclobber: the sha256 recorded when the hook was first chained is never rewritten
        sha="$(sha256sum < "${dir:?}/pre-push.chained")"
        (set -C && printf '%s\n' "${sha%% *}" > "${dir:?}/pre-push.chained.sha256") 2> /dev/null || true
    fi
    install -m 0755 "$SELF" "${dir:?}/pre-push-tags"
    install -m 0755 "${here:?}/pre-push-dispatch.sh" "${dir:?}/pre-push"
    printf 'installed: %s/pre-push runs pre-push-tags first\n' "$dir"
}

# uninstall_hook: undo --install. The pre-push that --install chained goes back as pre-push only
# if its sha256 still matches the one recorded at install; with none chained, pre-push is
# removed (the clone had none). A pre-push that is not the dispatcher is never touched
uninstall_hook() {
    local dir want='' have
    if git config --get core.hooksPath > /dev/null; then
        printf 'not_uninstalled: core.hooksPath is set; this guard was never installed here\n' >&2
        return 2
    fi
    dir="$(hooks_dir)"
    if [ -e "${dir:?}/pre-push" ] && ! grep -qxF "$DISPATCH_MARK" "${dir:?}/pre-push"; then
        printf 'not_uninstalled: %s/pre-push is not the dispatcher (another installer rewrote it); nothing changed\n' "$dir" >&2
        return 2
    fi
    if [ -e "${dir:?}/pre-push.chained" ]; then
        [ ! -f "${dir:?}/pre-push.chained.sha256" ] || want="$(< "${dir:?}/pre-push.chained.sha256")"
        have="$(sha256sum < "${dir:?}/pre-push.chained")"
        have="${have%% *}"
        if [ "$want" != "$have" ]; then
            printf 'not_uninstalled: pre-push.chained has sha256 %s, not the [%s] recorded at install; nothing changed\n' "$have" "$want" >&2
            return 1
        fi
        (cd "${dir:?}" && mv -- pre-push.chained pre-push)
        printf 'restored: %s/pre-push, sha256 %s as recorded at install\n' "$dir" "$have"
    else
        rm -f -- "${dir:?}/pre-push"
        printf 'removed: %s/pre-push; the clone had none before the install\n' "$dir"
    fi
    rm -f -- "${dir:?}/pre-push-tags" "${dir:?}/pre-push.chained.sha256"
}

# verify_in WT [quiet]: RED when git, run in work tree WT, would not run this guard on the next push.
# It judges the hooks dir git itself resolves there (--git-path hooks), so core.hooksPath at any
# scope, a worktree-scoped one included, turns it RED; so does a pre-push that is not the
# dispatcher (a pmat or any reinstall rewrote it) and a guard that is missing or not this file
verify_in() {
    local wt="$1" dir eff red=0
    dir="$(git -C "$wt" rev-parse --path-format=absolute --git-common-dir)/hooks"
    eff="$(git -C "$wt" rev-parse --path-format=absolute --git-path hooks)"
    if [ "$eff" != "$dir" ]; then
        printf 'pre-push-tags: verify RED: %s: git runs hooks from %s (core.hooksPath), not %s\n' "$wt" "$eff" "$dir"
        red=1
    fi
    if [ ! -x "$dir/pre-push" ] || ! cmp -s "$HERE/pre-push-dispatch.sh" "$dir/pre-push"; then
        printf 'pre-push-tags: verify RED: %s/pre-push is not the dispatcher (missing, not executable, or rewritten by another installer); run --install\n' "$dir"
        red=1
    fi
    if [ ! -x "$dir/pre-push-tags" ] || ! cmp -s "$SELF" "$dir/pre-push-tags"; then
        printf 'pre-push-tags: verify RED: %s/pre-push-tags is missing, not executable, or not this file; run --install\n' "$dir"
        red=1
    fi
    if [ "$red" -ne 0 ]; then return 1; fi
    [ -n "${2:-}" ] || printf 'pre-push-tags: verified: %s runs this guard first\n' "$wt"
}

# verify_hook [--all-worktrees]: verify_in this work tree, or in every work tree of the clone.
# Run it from the tracked scripts/hooks/, whose pre-push-dispatch.sh is what --install wrote.
# A listed work tree whose directory is gone cannot push and is counted as skipped
verify_hook() {
    local wt red=0 n=0 skipped=0
    if [ ! -f "$HERE/pre-push-dispatch.sh" ]; then
        printf 'not_measured: run --verify from the tracked scripts/hooks/, not from %s\n' "$HERE" >&2
        return 2
    fi
    case "${1:-}" in
        '') verify_in "$(git rev-parse --show-toplevel)"; return ;;
        --all-worktrees) ;;
        *) printf 'pre-push-tags: --verify takes nothing or --all-worktrees, not [%s]\n' "$1" >&2; return 2 ;;
    esac
    while read -r wt; do
        if [ ! -d "$wt" ]; then skipped=$((skipped + 1)); continue; fi
        n=$((n + 1))
        verify_in "$wt" quiet || red=$((red + 1))
    done < <(git worktree list --porcelain | sed -n 's/^worktree //p')
    printf 'pre-push-tags: verify --all-worktrees: %s checked, %s RED, %s skipped (directory gone)\n' "$n" "$red" "$skipped"
    if [ "$n" -eq 0 ]; then return 2; fi
    if [ "$red" -ne 0 ]; then return 1; fi
}

case "${1:-}" in
    --arm-release) arm_release "${2:-}" "${3:-}" ;;
    --install) install_hook ;;
    --uninstall) uninstall_hook ;;
    --verify) verify_hook "${2:-}" ;;
    --help | -h) sed -n '2,30p' "$0" ;;
    -*) sed -n '20,30p' "$0" >&2; exit 2 ;;
    *) hook "$@" ;;
esac
