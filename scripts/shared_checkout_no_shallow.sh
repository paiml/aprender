#!/usr/bin/env bash
# shared_checkout_no_shallow.sh — a shallow fetch into a checkout that worktrees share is refused (#5025)
#
# WHY. Every linked worktree reads ONE `shallow` file, in the common git dir. A depth-limited fetch run with the
#   shared checkout as its cwd (a harness whose `-C` a mutant dropped, a `cd X && ...; git fetch --depth=1` whose
#   `cd` failed) writes it, and git then breaks in every worktree at once (2026-10-08 15:36Z, 2026-10-10 04:59Z).
#
# THE MECHANISM, measured on git 2.34.1. git writes `shallow` through the lockfile `shallow.lock` (O_CREAT|O_EXCL).
#   A DIRECTORY at that path can never be taken or unlinked by git, so every shallow write fails before it starts:
#     fetch --depth=1 <branch> / <sha>, --deepen, --shallow-since, --shallow-exclude   rc 128, no `shallow` written
#     fetch, commit, gc                                                                  unaffected
#   A `reference-transaction` hook does NOT do this: it runs after `shallow` is written, an abort leaves the file,
#   and a fetch by bare sha (FETCH_HEAD only) runs no ref transaction at all, so the hook never even sees it.
#
# USAGE
#   shared_checkout_no_shallow.sh --install [DIR]   put the lock directory into DIR's common git dir (default: cwd's)
#   shared_checkout_no_shallow.sh --check [DIR]     GREEN = not shallow and the lock directory is in place
#   shared_checkout_no_shallow.sh --uninstall [DIR] remove the lock directory (only one --install wrote)
#   shared_checkout_no_shallow.sh --self-test       the case table, in a fixture: one checkout + one linked worktree
#   shared_checkout_no_shallow.sh --mutants         each mutant of this script must turn the case table red
# EXIT 0 green · 1 red · 3 a caller error.
set -euo pipefail

SELF="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
MARK='written by shared_checkout_no_shallow.sh --install (#5025): this checkout is shared by worktrees and is never shallow'

common_dir() { # common_dir DIR -> the absolute common git dir of DIR
    local d
    d="$(git -C "$1" rev-parse --git-common-dir)" || return 1
    case "$d" in /*) printf '%s\n' "$d" ;; *) printf '%s/%s\n' "$(cd -- "$1" && pwd)" "$d" ;; esac
}

install_lock() {
    local cd
    cd="$(common_dir "$1")" || { echo "not_installed: $1 is not a git checkout" >&2; return 3; }
    if [ -e "$cd/shallow" ]; then
        echo "not_installed: $cd/shallow exists (boundaries: $(tr '\n' ' ' < "$cd/shallow")); unshallow it first" >&2
        return 1
    fi
    if [ -e "$cd/shallow.lock" ] && ! [ -f "$cd/shallow.lock/README" ]; then
        echo "not_installed: $cd/shallow.lock exists and is not ours (a shallow write in progress?)" >&2
        return 1
    fi
    mkdir -p -- "$cd/shallow.lock"
    printf '%s\n' "$MARK" > "$cd/shallow.lock/README"
    echo "installed: $cd/shallow.lock/ -- a shallow fetch into this checkout is refused"
}

uninstall_lock() {
    local cd
    cd="$(common_dir "$1")" || { echo "not_uninstalled: $1 is not a git checkout" >&2; return 3; }
    if ! [ -f "$cd/shallow.lock/README" ] || [ "$(cat "$cd/shallow.lock/README")" != "$MARK" ]; then
        echo "not_uninstalled: $cd/shallow.lock is not one --install wrote; nothing changed" >&2
        return 1
    fi
    rm -f -- "$cd/shallow.lock/README"
    rmdir -- "$cd/shallow.lock"
    echo "uninstalled: $cd/shallow.lock/"
}

check_lock() {
    local cd
    cd="$(common_dir "$1")" || { echo "RED: $1 is not a git checkout" >&2; return 3; }
    if [ -f "$cd/shallow" ]; then
        echo "RED: $cd/shallow exists -- every worktree of this checkout is shallow (boundaries: $(tr '\n' ' ' < "$cd/shallow"))"
        return 1
    fi
    if ! [ -d "$cd/shallow.lock" ]; then
        echo "RED: $cd/shallow.lock/ is missing -- a shallow fetch here would be accepted (run --install)"
        return 1
    fi
    echo "GREEN: $cd is not shallow and refuses a shallow fetch"
}

self_test() {
    local td ok=0 bad=0 o m w l s rc f
    td="$(mktemp -d "${TMPDIR:-/tmp}/no-shallow.XXXXXX")"
    trap 'rm -rf -- "${td:?}"' RETURN
    # A row whose `-C` were lost reads "not a git repository" here, never the caller's checkout.
    cd -- "$td" || return 1
    export GIT_CEILING_DIRECTORIES="${td%/*}" GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
    o="$td/origin"
    g() { git -c user.name=st -c user.email=st@example.invalid -c commit.gpgsign=false "$@"; }
    row() { # row LABEL GOT WANT
        if [ "$2" = "$3" ]; then printf 'ok    %s\n' "$1"; ok=$((ok + 1))
        else printf 'FAIL  %s: got [%s], want [%s]\n' "$1" "$2" "$3"; bad=$((bad + 1)); fi
    }
    shallow() { if [ -e "$m/.git/shallow" ]; then printf shallow; else printf full; fi; }
    fetch_in_wt() { # fetch_in_wt ARGS... -> "rc=N/full|shallow"; any shallow written is removed for the next row
        rc=0; (cd -- "$w" && g fetch -q "$@" > /dev/null 2>&1) || rc=$?
        printf 'rc=%s/%s' "$([ "$rc" = 0 ] && echo 0 || echo nz)" "$(shallow)"
        rm -f -- "$m/.git/shallow"
    }
    g init -q -b main "$o"
    for i in 1 2 3; do g -C "$o" commit -q --allow-empty -m "c$i"; done
    for f in ctl main; do
        g clone -q "file://$o" "$td/$f"
        g -C "$td/$f" worktree add -q "$td/$f-wt" -b w
    done
    for i in 4 5; do g -C "$o" commit -q --allow-empty -m "c$i"; done
    s="$(git -C "$o" rev-parse HEAD)"

    # Control, in its OWN fixture: without the lock a shallow write really happens (else every row below is
    # vacuous). Deleting the `shallow` it wrote does not undo it -- origin/main then names a commit whose parents
    # are absent and gc fails -- which is why the lock refuses the write instead of cleaning up after it.
    m="$td/ctl" w="$td/ctl-wt"
    row "control: unguarded, a --depth=1 fetch from the worktree writes the common shallow" \
        "$(fetch_in_wt --depth=1 origin main)" "rc=0/shallow"
    row "control: unguarded, a --depth=1 fetch by bare sha writes it too" "$(fetch_in_wt --depth=1 origin "$s")" "rc=0/shallow"
    row "check is RED on an unguarded checkout" "$(check_lock "$w" > /dev/null 2>&1; echo $?)" 1
    m="$td/main" w="$td/main-wt"

    row "install from the linked worktree" "$(install_lock "$w" > /dev/null 2>&1; echo $?)" 0
    row "the lock is a directory in the COMMON git dir, not the worktree's own" \
        "$([ -d "$m/.git/shallow.lock" ] && echo common)" common
    row "check is GREEN once installed" "$(check_lock "$m" > /dev/null 2>&1; echo $?)" 0

    # must-fail: every shallow-writing form is refused and leaves no shallow file.
    row "refused: --depth=1 <branch>" "$(fetch_in_wt --depth=1 origin main)" "rc=nz/full"
    row "refused: --depth=1 <bare sha> (no ref transaction)" "$(fetch_in_wt --depth=1 origin "$s")" "rc=nz/full"
    row "refused: --deepen=1" "$(fetch_in_wt --deepen=1 origin)" "rc=nz/full"
    row "refused: --shallow-since" "$(fetch_in_wt --shallow-since=2000-01-01 origin)" "rc=nz/full"
    l="$( (cd -- "$m" && g fetch -q --depth=1 origin main 2>&1) || true)"; rm -f -- "$m/.git/shallow"
    row "refused in the main checkout as well, and git names the lock" \
        "$(case "$l" in *shallow.lock*) echo named ;; *) echo "$l" ;; esac)" named

    # must-pass: ordinary work is untouched.
    row "passes: a plain fetch" "$(fetch_in_wt origin)" "rc=0/full"
    row "passes: a commit in the worktree" \
        "$( (cd -- "$w" && g commit -q --allow-empty -m wt) > /dev/null 2>&1; echo $?)" 0
    row "passes: gc, and the lock survives it" \
        "$(g -C "$m" gc -q > /dev/null 2>&1; echo $?)/$([ -d "$m/.git/shallow.lock" ] && echo kept)" 0/kept

    # install refuses a checkout that is already shallow; check is RED on one.
    printf '%s\n' "$s" > "$m/.git/shallow"
    row "check is RED on a shallow checkout" "$(check_lock "$m" > /dev/null 2>&1; echo $?)" 1
    rm -f -- "$m/.git/shallow"

    row "uninstall removes only its own lock" "$(uninstall_lock "$m" > /dev/null 2>&1; echo $?)/$([ -e "$m/.git/shallow.lock" ] && echo left || echo gone)" 0/gone
    printf '%s\n' "$s" > "$m/.git/shallow"
    row "install refuses an already-shallow checkout and changes nothing" \
        "$(install_lock "$m" > /dev/null 2>&1; echo $?)/$([ -e "$m/.git/shallow.lock" ] && echo lock || echo none)" 1/none
    rm -f -- "$m/.git/shallow"
    mkdir "$m/.git/shallow.lock"
    row "uninstall refuses a lock it did not write" "$(uninstall_lock "$m" > /dev/null 2>&1; echo $?)/$([ -d "$m/.git/shallow.lock" ] && echo kept)" 1/kept

    printf 'shared_checkout_no_shallow self-test: %s ok, %s bad\n' "$ok" "$bad"
    [ "$bad" -eq 0 ] && [ "$ok" -ge 18 ]
}

mutants() {
    local n=0 bad=0 name expr rc td
    td="$(mktemp -d "${TMPDIR:-/tmp}/no-shallow-mut.XXXXXX")"
    trap 'rm -rf -- "${td:?}"' RETURN
    bash "$SELF" --self-test > /dev/null 2>&1 || { echo "BROKEN: the unmutated self-test is red"; return 1; }
    while IFS=$'\t' read -r name expr; do
        [ -n "$name" ] || continue
        n=$((n + 1))
        sed -E "$expr" "$SELF" > "$td/$name.sh"
        if cmp -s "$SELF" "$td/$name.sh"; then echo "BROKEN: $name changed nothing"; bad=$((bad + 1)); continue; fi
        rc=0; bash "$td/$name.sh" --self-test > /dev/null 2>&1 || rc=$?
        if [ "$rc" != 0 ]; then echo "killed: $name"; else echo "SURVIVED: $name"; bad=$((bad + 1)); fi
    done <<'MUTANTS'
m1_lock_in_worktree_gitdir	s/rev-parse --git-common-dir/rev-parse --absolute-git-dir/
m2_lock_is_a_file	s/^    mkdir -p -- "\$cd\/shallow.lock"$/    : > "$cd\/shallow.lock.x"/
m3_check_ignores_shallow	s/if \[ -f "\$cd\/shallow" \]/if false/
m4_check_ignores_missing_lock	s/if ! \[ -d "\$cd\/shallow.lock" \]; then/if false; then/
m5_install_accepts_shallow	s/if \[ -e "\$cd\/shallow" \]; then/if false; then/
MUTANTS
    echo "--- $((n - bad))/$n mutants killed ---"
    [ "$bad" = 0 ] && [ "$n" -gt 0 ]
}

case "${1:-}" in
    --install) install_lock "${2:-.}" ;;
    --uninstall) uninstall_lock "${2:-.}" ;;
    --check) check_lock "${2:-.}" ;;
    --self-test) self_test ;;
    --mutants) mutants ;;
    -h | --help) sed -n '2,23p' "$SELF" ;;
    *) echo "usage: shared_checkout_no_shallow.sh --install|--check|--uninstall [DIR] | --self-test | --mutants" >&2; exit 3 ;;
esac
