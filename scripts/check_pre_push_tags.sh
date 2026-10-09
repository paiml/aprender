#!/usr/bin/env bash
# check_pre_push_tags.sh: proves scripts/hooks/pre-push-tags.sh refuses every tag push except
# an armed release tag, with real `git push`es from a throwaway clone to a throwaway bare remote.
#
# REPORT-ONLY in a bare run: it prints one SUMMARY line, saying whether this clone has the
# guard installed, and exits 0 either way (2 = not_measured outside a work tree). The local
# hook itself refuses from the day it is installed.
#
# usage:
#   check_pre_push_tags.sh                    SUMMARY of this clone's install state
#   check_pre_push_tags.sh --self-test [HOOK] the case table against HOOK (default: the
#                                             tracked hook; pre-push-dispatch.sh beside it).
#                                             Every push row is real `git push`es; a refused
#                                             row needs a non-zero push, the hook's REFUSED
#                                             line with the row's reason, and an unchanged
#                                             remote ref. Unsets every `git rev-parse
#                                             --local-env-vars` variable and reads no global
#                                             or system git config. A fixed row count; fewer
#                                             rows is UNMEASURED and fails.
#   check_pre_push_tags.sh --mutants          each mutant of the hooks must turn --self-test RED
set -euo pipefail

SELF="$(cd "$(dirname "$0")" && pwd)/${0##*/}"
HOOK_DIR="$(dirname "$SELF")/hooks"
WANT_ROWS=54
ROWS=0
FAILS=0
HOOK=''
TP=''
C1=''
SIDE=''
WORK=''

bare() {
    local top dir
    if ! top="$(git rev-parse --show-toplevel 2>/dev/null)"; then
        printf 'not_measured: not inside a git work tree\n'
        return 2
    fi
    dir="$(git -C "$top" rev-parse --path-format=absolute --git-common-dir)/hooks"
    if [ -f "$dir/pre-push-tags" ] && grep -qxF '# written by pre-push-tags.sh --install' "$dir/pre-push" 2>/dev/null; then
        printf 'SUMMARY the pre-push tag guard is installed in this clone (report-only)\n'
    else
        printf 'SUMMARY the pre-push tag guard is not installed in this clone (report-only)\n'
    fi
}

ok() {
    printf '  ok    %s\n' "$1"
}

fail() {
    printf '  FAIL  %s: %s\n' "$1" "$2"
    FAILS=$((FAILS + 1))
}

remote_sha() {
    git -C "$TP/remote.git" rev-parse -q --verify "$1" || printf 'absent\n'
}

# push_row NAME WANT REF STEP: run STEP in the clone. WANT is "allowed" or the reason the
# hook must print after REFUSED; REF is the remote ref STEP targets
push_row() {
    local name="$1" want="$2" ref="$3" step="$4" before after rc=0 out local_sha
    ROWS=$((ROWS + 1))
    before="$(remote_sha "$ref")"
    "$step" > "$TP/row.log" 2>&1 || rc=$?
    out="$(< "$TP/row.log")"
    after="$(remote_sha "$ref")"
    if [ "$want" != allowed ]; then
        if [ "$rc" -eq 0 ]; then fail "$name" "the push exited 0"; return 0; fi
        case "$out" in
            *"pre-push-tags: REFUSED $want"*) ;;
            *) fail "$name" "rc $rc without [REFUSED $want]"; return 0 ;;
        esac
        if [ "$before" != "$after" ]; then fail "$name" "$ref moved on the remote ($before to $after)"; return 0; fi
    else
        local_sha="$(git rev-parse -q --verify "$ref" || printf 'absent\n')"
        if [ "$rc" -ne 0 ]; then fail "$name" "rc $rc: ${out//$'\n'/ }"; return 0; fi
        case "$out" in *'pre-push-tags: REFUSED'*) fail "$name" "the hook printed REFUSED"; return 0 ;; esac
        if [ "$after" != "$local_sha" ]; then fail "$name" "$ref is $after on the remote, $local_sha here"; return 0; fi
    fi
    ok "$name"
}

# check_row NAME WHAT STEP: a non-push assertion; STEP must exit 0
check_row() {
    local name="$1" what="$2" step="$3"
    ROWS=$((ROWS + 1))
    if "$step" > "$TP/row.log" 2>&1; then ok "$name"; else fail "$name" "$what"; fi
}

# the clock as the hook reads it, for marker expiries
epoch() { printf '%(%s)T\n' -1; }
write_marker() {
    local exp="${3:-$(($(epoch) + 600))}"
    printf '%s %s %s\n' "$1" "$2" "$exp" > .git/pre-push-release-tag
}

# the steps, each run from the throwaway clone
step_install() { bash "$HOOK" --install && grep -q chained.log .git/hooks/pre-push.chained; }
step_install_under_hookspath() {
    local rc=0
    GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.hooksPath GIT_CONFIG_VALUE_0=/nonexistent-hooks \
        bash "$HOOK" --install || rc=$?
    [ "$rc" -eq 2 ]
}
# a second clone that already has both pre-push and pre-push.chained: install exits 2 and
# leaves both byte-identical, with no guard copied in
step_install_both_exist() {
    local rc=0 h="$TP/w2/.git/hooks"
    git clone -q "$TP/remote.git" "$TP/w2" || return 1
    printf '#!/bin/sh\n# one\n' > "$TP/one"
    printf '#!/bin/sh\n# two\n' > "$TP/two"
    printf '#!/bin/sh\n# one\n' > "$h/pre-push"
    printf '#!/bin/sh\n# two\n' > "$h/pre-push.chained"
    cd "$TP/w2" || return 1
    bash "$HOOK" --install 2> /dev/null || rc=$?
    cd "$TP/w" || return 1
    [ "$rc" -eq 2 ] && cmp -s "$TP/one" "$h/pre-push" && cmp -s "$TP/two" "$h/pre-push.chained" \
        && [ ! -e "$h/pre-push-tags" ]
}
step_branch() { git branch feat && git push -q origin feat; }
step_chained_saw_stdin() { grep -q '^refs/heads/feat [0-9a-f]* refs/heads/feat 0*$' "$TP/chained.log"; }
step_same_sha() { git push origin keep/old; }
step_tags_clean() { git push --tags origin; }
step_release() { git tag -a -m r v0.1.0 && bash "$HOOK" --arm-release v0.1.0 && git push origin v0.1.0; }
step_release_rc() { git tag v0.1.1-rc.1 && bash "$HOOK" --arm-release v0.1.1-rc.1 60 && git push origin v0.1.1-rc.1; }
step_keep() { git tag keep/new && git push origin keep/new; }
step_keep_annotated() { git tag -a -m n keep/new-ann && git push origin keep/new-ann; }
step_raw() { git push origin HEAD:refs/tags/keep/raw; }
step_no_marker() { git tag -a -m r v0.2.0 && git push origin v0.2.0; }
step_env_marker() {
    PRE_PUSH_RELEASE_TAG="refs/tags/v0.2.0 $(git rev-parse v0.2.0)" git push origin v0.2.0
}
step_other_sha() { write_marker refs/tags/v0.2.0 "$(git rev-parse main)" && git push origin v0.2.0; }
step_not_a_release_name() {
    git tag vnext && write_marker refs/tags/vnext "$(git rev-parse vnext)" && git push origin vnext
}
step_off_main() { git tag v0.3.0 side && bash "$HOOK" --arm-release v0.3.0 && git push origin v0.3.0; }
step_expired() {
    write_marker refs/tags/v0.2.0 "$(git rev-parse v0.2.0)" "$(($(epoch) - 1))" && git push origin v0.2.0
}
step_capped() {
    write_marker refs/tags/v0.2.0 "$(git rev-parse v0.2.0)" "$(($(epoch) + 7200))" && git push origin v0.2.0
}
# release/0.4.0 (remote head R2, R1 its parent, both off main) and release/0.5.0 (head R5):
# a release tag is accepted off main only at exactly the head of its own release branch
step_rel_ancestor() { git tag v0.4.0 "$R1" && bash "$HOOK" --arm-release v0.4.0 && git push origin v0.4.0; }
step_rel_other_branch() { git tag -f v0.4.0 "$R5" > /dev/null && bash "$HOOK" --arm-release v0.4.0 && git push origin v0.4.0; }
step_rel_absent() { git tag v0.6.0 "$R2" && bash "$HOOK" --arm-release v0.6.0 && git push origin v0.6.0; }
step_rel_unmarked() { git tag -f v0.4.0 "$R2" > /dev/null && git push origin v0.4.0; }
step_rel_expired() {
    write_marker refs/tags/v0.4.0 "$R2" "$(($(epoch) - 1))" && git push origin v0.4.0
}
step_rel_capped() {
    write_marker refs/tags/v0.4.0 "$R2" "$(($(epoch) + 7200))" && git push origin v0.4.0
}
step_rel_rc() { git tag v0.4.0-rc.1 "$R2" && bash "$HOOK" --arm-release v0.4.0-rc.1 && git push origin v0.4.0-rc.1; }
step_rel_head() {
    git tag -f -a -m r v0.4.0 "$R2" > /dev/null && bash "$HOOK" --arm-release v0.4.0 && git push origin v0.4.0
}
step_rel_move() { git tag -f v0.4.0 "$R1" > /dev/null && bash "$HOOK" --arm-release v0.4.0 && git push -f origin v0.4.0; }
step_rel_delete() { bash "$HOOK" --arm-release v0.4.0 && git push origin :refs/tags/v0.4.0; }
# a linked work tree of w3 with a worktree-scoped core.hooksPath: --verify there is RED, plain
# --verify in w3 stays green, and --all-worktrees from w3 is RED
step_verify_worktree_hookspath() {
    local rc=0 all=0 main=0
    cd "$TP/w3" || return 1
    bash "$HOOK" --install > /dev/null || return 1
    git config extensions.worktreeConfig true && git worktree add -q "$TP/w3b" 2> /dev/null || return 1
    git -C "$TP/w3b" config --worktree core.hooksPath "$TP/elsewhere"
    bash "$HOOK" --verify > /dev/null || main=$?
    bash "$HOOK" --verify --all-worktrees > /dev/null || all=$?
    cd "$TP/w3b" || return 1
    bash "$HOOK" --verify > /dev/null || rc=$?
    cd "$TP/w" || return 1
    [ "$main" -eq 0 ] && [ "$all" -eq 1 ] && [ "$rc" -eq 1 ]
}
# --arm-release with TTL-like variables exported: the expiry is still now + 600, and SECONDS
# above 3600 is still refused, so nothing in the environment sets or widens the TTL
step_ttl_not_from_env() {
    local rc=0 left e
    env TTL=5 TTL_DEFAULT=5 TTL_MAX=99999 PRE_PUSH_TTL=5 PRE_PUSH_TAGS_TTL=5 ARM_TTL=5 \
        bash "$HOOK" --arm-release v0.2.0 > /dev/null || return 1
    read -r _ _ e < .git/pre-push-release-tag || return 1
    rm -f .git/pre-push-release-tag
    left=$((e - $(epoch)))
    env TTL_MAX=99999 PRE_PUSH_TAGS_TTL_MAX=99999 bash "$HOOK" --arm-release v0.2.0 7200 2> /dev/null || rc=$?
    [ "$left" -ge 595 ] && [ "$left" -le 600 ] && [ "$rc" -eq 2 ] && [ ! -e .git/pre-push-release-tag ]
}
step_bad_ttl() {
    local t rc
    for t in 0 3601 abc -5 1.5; do
        rc=0
        bash "$HOOK" --arm-release v0.2.0 "$t" 2> /dev/null || rc=$?
        [ "$rc" -eq 2 ] || return 1
    done
    [ ! -e .git/pre-push-release-tag ]
}
step_verify_ok() { bash "$HOOK" --verify; }
# sha256 of a file, for the byte-for-byte restore rows
sum_of() { local s; s="$(sha256sum < "$1")" && printf '%s\n' "${s%% *}"; }
# a fourth clone with its own pre-push: install, uninstall, and that pre-push is back byte for byte
step_uninstall_restores() {
    local rc=9 h="$TP/w4/.git/hooks" before
    git clone -q "$TP/remote.git" "$TP/w4" || return 1
    printf '#!/bin/sh\n# four\nexit 0\n' > "$TP/four" && install -m 0755 "$TP/four" "$h/pre-push"
    before="$(sum_of "$h/pre-push")"
    cd "$TP/w4" || return 1
    if bash "$HOOK" --install > /dev/null && bash "$HOOK" --verify > /dev/null; then
        rc=0
        bash "$HOOK" --uninstall > /dev/null || rc=$?
    fi
    cd "$TP/w" || return 1
    [ "$rc" -eq 0 ] && [ "$(sum_of "$h/pre-push")" = "$before" ] && [ -x "$h/pre-push" ] \
        && [ ! -e "$h/pre-push-tags" ] && [ ! -e "$h/pre-push.chained" ] && [ ! -e "$h/pre-push.chained.sha256" ]
}
# the same clone reinstalled, then its chained hook edited: uninstall restores nothing
step_uninstall_refuses_a_changed_chain() {
    local rc=0 h="$TP/w4/.git/hooks"
    cd "$TP/w4" || return 1
    if bash "$HOOK" --install > /dev/null; then
        printf '# edited after install\n' >> "$h/pre-push.chained"
        bash "$HOOK" --uninstall 2> /dev/null || rc=$?
    fi
    cd "$TP/w" || return 1
    [ "$rc" -eq 1 ] && grep -q 'edited after install' "$h/pre-push.chained" \
        && cmp -s "$(dirname "$HOOK")/pre-push-dispatch.sh" "$h/pre-push" && [ -e "$h/pre-push-tags" ]
}
# a fifth clone with no pre-push: install, another installer rewrites pre-push, uninstall leaves it
step_uninstall_keeps_a_rewritten_pre_push() {
    local rc=0 h="$TP/w5/.git/hooks"
    git clone -q "$TP/remote.git" "$TP/w5" || return 1
    printf '#!/usr/bin/env bash\n# PMAT Pre-Push Quality Gate\nexit 0\n' > "$TP/pmat"
    cd "$TP/w5" || return 1
    if bash "$HOOK" --install > /dev/null; then
        printf '#!/usr/bin/env bash\n# PMAT Pre-Push Quality Gate\nexit 0\n' > "$h/pre-push"
        bash "$HOOK" --uninstall 2> /dev/null || rc=$?
    fi
    cd "$TP/w" || return 1
    [ "$rc" -eq 2 ] && cmp -s "$TP/pmat" "$h/pre-push"
}
# a sixth clone with no pre-push: install then uninstall leaves no pre-push at all
step_uninstall_with_none_before() {
    local rc=9 h="$TP/w6/.git/hooks"
    git clone -q "$TP/remote.git" "$TP/w6" || return 1
    cd "$TP/w6" || return 1
    if bash "$HOOK" --install > /dev/null; then
        rc=0
        bash "$HOOK" --uninstall > /dev/null || rc=$?
    fi
    cd "$TP/w" || return 1
    [ "$rc" -eq 0 ] && [ ! -e "$h/pre-push" ] && [ ! -e "$h/pre-push-tags" ]
}
# a third clone: installed and verified, then another installer rewrites pre-push
step_verify_overwritten() {
    local rc=0
    git clone -q "$TP/remote.git" "$TP/w3" || return 1
    cd "$TP/w3" || return 1
    if bash "$HOOK" --install && bash "$HOOK" --verify; then
        printf '#!/usr/bin/env bash\n# PMAT Pre-Push Quality Gate\nexit 0\n' > .git/hooks/pre-push
        bash "$HOOK" --verify || rc=$?
    fi
    cd "$TP/w" || return 1
    [ "$rc" -eq 1 ]
}
# the same clone reinstalled (the rewritten pre-push is chained), then the guard removed
# a seventh clone whose linked work tree, with a worktree-scoped core.hooksPath, was moved without
# git: --all-worktrees still judges it from its admin dir and is RED (1), not "skipped" and 0.
# An admin dir git cannot read is then added: --all-worktrees exits 2, never 0
step_verify_moved_worktree() {
    local moved=0 junk=0
    git clone -q "$TP/remote.git" "$TP/w7" || return 1
    cd "$TP/w7" || return 1
    bash "$HOOK" --install > /dev/null || return 1
    git config extensions.worktreeConfig true && git worktree add -q "$TP/w7b" 2> /dev/null || return 1
    git -C "$TP/w7b" config --worktree core.hooksPath "$TP/elsewhere"
    (cd "$TP" && mv -- w7b w7moved) || return 1
    bash "$HOOK" --verify --all-worktrees > /dev/null || moved=$?
    git -C "$TP/w7moved" config --worktree --unset core.hooksPath
    mkdir -p .git/worktrees/junk || return 1
    bash "$HOOK" --verify --all-worktrees > /dev/null || junk=$?
    cd "$TP/w" || return 1
    [ "$moved" -eq 1 ] && [ "$junk" -eq 2 ]
}
step_verify_no_guard() {
    local rc=0
    cd "$TP/w3" || return 1
    if bash "$HOOK" --install && bash "$HOOK" --verify; then
        rm -f -- "${TP:?}/w3/.git/hooks/pre-push-tags"
        bash "$HOOK" --verify || rc=$?
    fi
    cd "$TP/w" || return 1
    [ "$rc" -eq 1 ]
}
step_spent() {
    bash "$HOOK" --arm-release v0.2.0 && git push -q origin HEAD:refs/heads/spend && git push origin v0.2.0
}
step_one_bad_line() { bash "$HOOK" --arm-release v0.2.0 && git push origin v0.2.0 keep/new; }
step_move() { git tag -f keep/old main > /dev/null && git push origin keep/old; }
step_move_forced() { git push -f origin keep/old; }
step_move_unrelated() { git tag -f keep/old "$SIDE" > /dev/null && git push -f origin keep/old; }
step_move_annotated() { git tag -f -a -m moved keep/old-ann main > /dev/null && git push -f origin keep/old-ann; }
step_delete() { git tag -f keep/old "$C1" > /dev/null && git push origin :refs/tags/keep/old; }
step_delete_flag() { git push --delete origin keep/old-ann; }
step_delete_release() { bash "$HOOK" --arm-release v0.1.0 && git push origin :refs/tags/v0.1.0; }
step_push_tags() { git push --tags origin; }
step_tags_refspec() { git push origin 'refs/tags/*:refs/tags/*'; }
step_mirror() { git push --mirror origin; }
step_arm_keep() {
    local rc=0
    bash "$HOOK" --arm-release keep/new || rc=$?
    [ "$rc" -eq 2 ] && [ ! -e .git/pre-push-release-tag ]
}

# fixture: a bare remote with main and two tags it already has; a clone with a foreign pre-push
# that logs its stdin. Leaves the cwd in the clone.
fixture() {
    git init -q --bare -b main "$TP/remote.git"
    git init -q -b main "$TP/w"
    cd "$TP/w" || return 1
    git config user.name t
    git config user.email t@t
    git config commit.gpgsign false
    git config tag.gpgsign false
    git commit -q --allow-empty -m c1
    git remote add origin "$TP/remote.git"
    git push -q origin main
    git tag keep/old
    git tag -a -m old keep/old-ann
    git push -q origin keep/old keep/old-ann
    git checkout -q -b side
    git commit -q --allow-empty -m side
    git checkout -q main
    git commit -q --allow-empty -m c2
    git push -q origin main
    C1="$(git rev-parse main~1)"
    SIDE="$(git rev-parse side)"
    git checkout -q -b rel side
    git commit -q --allow-empty -m r1
    git commit -q --allow-empty -m r2
    git push -q origin rel:refs/heads/release/0.4.0
    R1="$(git rev-parse rel~1)"
    R2="$(git rev-parse rel)"
    git checkout -q -b rel5 "$C1"
    git commit -q --allow-empty -m r5
    git push -q origin rel5:refs/heads/release/0.5.0
    R5="$(git rev-parse rel5)"
    git checkout -q main
    printf '#!/usr/bin/env bash\ncat >> "%s/chained.log"\n' "$TP" > .git/hooks/pre-push
    chmod 0755 .git/hooks/pre-push
}

self_test() {
    local v has_dir=0 has_common=0
    local -a gv
    HOOK="${1:-$HOOK_DIR/pre-push-tags.sh}"
    HOOK="$(cd "$(dirname "$HOOK")" || exit 1; pwd)/${HOOK##*/}"
    if [ ! -f "$HOOK" ]; then
        printf 'UNMEASURED no hook at %s\n' "$HOOK"
        return 1
    fi
    mapfile -t gv < <(git rev-parse --local-env-vars)
    for v in "${gv[@]}"; do
        [ "$v" != GIT_DIR ] || has_dir=1
        [ "$v" != GIT_COMMON_DIR ] || has_common=1
    done
    if [ "$has_dir" -ne 1 ] || [ "$has_common" -ne 1 ]; then
        printf 'UNMEASURED git rev-parse --local-env-vars did not list GIT_DIR and GIT_COMMON_DIR\n'
        return 1
    fi
    unset "${gv[@]}" PRE_PUSH_RELEASE_TAG
    export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 GIT_TERMINAL_PROMPT=0
    TP="$(mktemp -d)"
    trap 'rm -rf -- "${TP:?}"' EXIT
    fixture > "$TP/fixture.log" 2>&1 || {
        printf 'UNMEASURED the fixture failed: %s\n' "$(tail -n 1 "$TP/fixture.log")"
        return 1
    }
    cd "$TP/w" || return 1

    check_row install_chains_the_existing_pre_push "install failed or the old hook is not pre-push.chained" step_install
    check_row a_second_install_keeps_the_chain "a reinstall chained the dispatcher to itself" step_install
    check_row install_refuses_under_core_hookspath "install did not exit 2 with core.hooksPath set" step_install_under_hookspath
    check_row install_never_replaces_an_existing_chain "install did not exit 2, or changed pre-push or pre-push.chained" step_install_both_exist
    check_row verify_passes_once_installed "--verify was not 0 in an installed clone" step_verify_ok
    check_row verify_is_red_when_another_installer_rewrote_pre_push "--verify did not exit 1 after pre-push was rewritten" step_verify_overwritten
    check_row verify_is_red_when_the_guard_is_gone "--verify did not exit 1 with pre-push-tags removed" step_verify_no_guard
    check_row verify_judges_the_hooks_path_of_each_worktree "--verify missed a worktree-scoped core.hooksPath, or --all-worktrees did not exit 1" step_verify_worktree_hookspath
    check_row verify_judges_a_moved_worktree_from_its_admin_dir "--all-worktrees skipped a moved work tree, or passed with an admin dir it could not read" step_verify_moved_worktree
    check_row uninstall_restores_the_old_pre_push_byte_for_byte "uninstall failed, or pre-push differs from the one before install" step_uninstall_restores
    check_row uninstall_refuses_a_chained_hook_changed_since_install "uninstall did not exit 1, or changed something" step_uninstall_refuses_a_changed_chain
    check_row uninstall_never_touches_a_rewritten_pre_push "uninstall did not exit 2, or changed the rewritten pre-push" step_uninstall_keeps_a_rewritten_pre_push
    check_row uninstall_with_no_old_hook_leaves_none "uninstall left a pre-push or the guard behind" step_uninstall_with_none_before

    push_row a_branch_push_is_untouched allowed refs/heads/feat step_branch
    check_row the_chained_pre_push_saw_the_same_stdin "chained.log lacks the refs/heads/feat line" step_chained_saw_stdin
    push_row an_existing_tag_at_the_same_sha_is_a_no_op allowed refs/tags/keep/old step_same_sha
    push_row tags_the_remote_already_has_push_clean allowed refs/tags/keep/old-ann step_tags_clean
    push_row an_armed_release_tag_on_main_is_pushed allowed refs/tags/v0.1.0 step_release
    push_row an_armed_lightweight_rc_tag_is_pushed allowed refs/tags/v0.1.1-rc.1 step_release_rc

    push_row a_keep_tag_create_is_refused 'create of refs/tags/keep/new,' refs/tags/keep/new step_keep
    push_row an_annotated_keep_tag_create_is_refused 'create of refs/tags/keep/new-ann,' refs/tags/keep/new-ann step_keep_annotated
    push_row a_commit_pushed_to_a_tag_ref_is_refused 'create of refs/tags/keep/raw,' refs/tags/keep/raw step_raw
    push_row a_release_tag_without_a_marker_is_refused 'create of refs/tags/v0.2.0 at' refs/tags/v0.2.0 step_no_marker
    push_row a_marker_in_the_environment_arms_nothing 'create of refs/tags/v0.2.0 at' refs/tags/v0.2.0 step_env_marker
    push_row a_marker_for_another_sha_is_refused 'create of refs/tags/v0.2.0 at' refs/tags/v0.2.0 step_other_sha
    push_row an_expired_marker_is_refused 'create of refs/tags/v0.2.0: its release marker expired' refs/tags/v0.2.0 step_expired
    push_row a_marker_past_the_ttl_cap_is_refused 'create of refs/tags/v0.2.0: its release marker expires after the 3600 s cap' refs/tags/v0.2.0 step_capped
    check_row arm_release_refuses_a_bad_ttl "--arm-release took a TTL outside 1..3600, or left a marker" step_bad_ttl
    check_row the_ttl_is_never_read_from_the_environment "exported TTL-like variables changed the 600 s expiry or let 7200 through" step_ttl_not_from_env
    push_row a_marked_v_tag_that_is_not_a_release_name_is_refused 'create of refs/tags/vnext, which is not vX' refs/tags/vnext step_not_a_release_name
    push_row an_armed_release_tag_off_main_is_refused 'create of refs/tags/v0.3.0:' refs/tags/v0.3.0 step_off_main
    push_row a_release_tag_at_an_ancestor_of_its_release_head_is_refused "create of refs/tags/v0.4.0: $R1 is neither" refs/tags/v0.4.0 step_rel_ancestor
    push_row a_release_tag_at_another_release_head_is_refused "create of refs/tags/v0.4.0: $R5 is neither" refs/tags/v0.4.0 step_rel_other_branch
    push_row a_release_tag_with_no_release_branch_is_refused "create of refs/tags/v0.6.0: $R2 is neither" refs/tags/v0.6.0 step_rel_absent
    push_row an_unmarked_tag_at_the_release_head_is_refused "create of refs/tags/v0.4.0 at" refs/tags/v0.4.0 step_rel_unmarked
    push_row an_expired_marker_at_the_release_head_is_refused "create of refs/tags/v0.4.0: its release marker expired" refs/tags/v0.4.0 step_rel_expired
    push_row an_over_cap_marker_at_the_release_head_is_refused "create of refs/tags/v0.4.0: its release marker expires after the 3600 s cap" refs/tags/v0.4.0 step_rel_capped
    push_row an_armed_rc_tag_at_the_release_head_is_pushed allowed refs/tags/v0.4.0-rc.1 step_rel_rc
    push_row an_armed_release_tag_at_the_release_head_is_pushed allowed refs/tags/v0.4.0 step_rel_head
    push_row moving_a_release_tag_off_the_release_head_is_refused "move of refs/tags/v0.4.0" refs/tags/v0.4.0 step_rel_move
    push_row deleting_a_release_tag_at_the_release_head_is_refused "delete of refs/tags/v0.4.0" refs/tags/v0.4.0 step_rel_delete
    push_row the_marker_is_spent_by_one_push 'create of refs/tags/v0.2.0 at' refs/tags/v0.2.0 step_spent
    push_row one_bad_line_refuses_an_armed_release_too 'create of refs/tags/keep/new,' refs/tags/v0.2.0 step_one_bad_line

    push_row a_plain_tag_move_is_refused 'move of refs/tags/keep/old' refs/tags/keep/old step_move
    push_row a_forced_fast_forward_tag_move_is_refused 'move of refs/tags/keep/old' refs/tags/keep/old step_move_forced
    push_row a_forced_move_to_an_unrelated_commit_is_refused 'move of refs/tags/keep/old' refs/tags/keep/old step_move_unrelated
    push_row a_forced_annotated_tag_move_is_refused 'move of refs/tags/keep/old-ann' refs/tags/keep/old-ann step_move_annotated
    push_row a_tag_delete_is_refused 'delete of refs/tags/keep/old' refs/tags/keep/old step_delete
    push_row a_delete_flag_is_refused 'delete of refs/tags/keep/old-ann' refs/tags/keep/old-ann step_delete_flag
    push_row an_armed_release_tag_delete_is_refused 'delete of refs/tags/v0.1.0' refs/tags/v0.1.0 step_delete_release

    push_row push_tags_is_refused 'create of refs/tags/keep/new,' refs/tags/keep/new step_push_tags
    push_row a_tags_refspec_is_refused 'create of refs/tags/keep/new,' refs/tags/keep/new step_tags_refspec
    push_row mirror_is_refused 'create of refs/tags/keep/new,' refs/tags/keep/new step_mirror
    check_row arm_release_refuses_a_keep_name "--arm-release keep/new did not exit 2, or left a marker" step_arm_keep

    printf -- '--- %s/%s rows ---\n' "$((ROWS - FAILS))" "$ROWS"
    if [ "$ROWS" -ne "$WANT_ROWS" ]; then
        printf 'UNMEASURED %s of %s rows ran\n' "$ROWS" "$WANT_ROWS"
        return 1
    fi
    [ "$FAILS" -eq 0 ]
}

# NAME<TAB>FILE<TAB>sed expression; FILE is under scripts/hooks/
MUTANTS='m01_the_delete_arm_is_dropped	pre-push-tags.sh	s/^        if is_zero "\$lsha"; then refuse .*$/        :/
m02_the_move_arm_is_dropped	pre-push-tags.sh	s/^\(            . "\$rsha" = "\$lsha" .\) || refuse /\1 || true /
m03_keep_tags_are_accepted	pre-push-tags.sh	s/^            refs\/tags\/v\*) ;;$/&\n            refs\/tags\/keep\/*) continue ;;/
m04_a_forced_fast_forward_move_is_accepted	pre-push-tags.sh	s/^\(            . "\$rsha" = "\$lsha" .\) || refuse /\1 || git merge-base --is-ancestor "$rsha" "$lsha" || refuse /
m05_the_marker_is_read_from_the_environment	pre-push-tags.sh	s/^        . -L "\$mfile" . || marker=.*$/        marker="${PRE_PUSH_RELEASE_TAG:-}"/
m06_the_marker_is_not_spent	pre-push-tags.sh	s/^        rm -f -- "\${mfile:?}"$/        true/
m07_main_is_not_checked	pre-push-tags.sh	s/^        if ! on_main "\$lsha" "\$url" && /        if false \&\& /
m08_the_marker_sha_is_not_compared	pre-push-tags.sh	s/^        if . "\$mref \$msha" != "\$rref \$lsha" .; then$/        if ! test "$mref" = "$rref"; then/
m09_the_release_name_is_not_checked	pre-push-tags.sh	s/^        if ! is_release_name "\${rref#refs\/tags\/}"; then$/        if false; then/
m10_the_dispatcher_ignores_the_guard	pre-push-dispatch.sh	s/^printf .%s. "\$in" | bash "\$d\/pre-push-tags" .*$/& || true/
m11_a_foreign_pre_push_is_overwritten	pre-push-tags.sh	s/^        (c. "\${dir:?}" .. m. .. pre-push pre-push.chained)$/        true/
m12_an_existing_chain_is_overwritten	pre-push-tags.sh	s/^            return 2$/            :/
m13_the_marker_expiry_is_not_checked	pre-push-tags.sh	s/^        if ! is_count "\$mexp" || .*$/        if false; then/
m14_verify_ignores_a_rewritten_pre_push	pre-push-tags.sh	s/^    if . ! -x "\$dir\/pre-push" . || ! cmp .*$/    if false; then/
m15_verify_ignores_a_missing_guard	pre-push-tags.sh	s/^    if . ! -x "\$dir\/pre-push-tags" . || .*$/    if false; then/
m16_any_ttl_is_accepted	pre-push-tags.sh	s/^    if ! is_count "\$ttl" || .*$/    if false; then/
m17_uninstall_skips_the_sha256_check	pre-push-tags.sh	s/^        if . "\$want" != "\$have" .; then$/        if false; then/
m18_uninstall_touches_a_rewritten_pre_push	pre-push-tags.sh	/^uninstall_hook() {$/,/^}$/s/^    if . -e "\${dir:?}\/pre-push" . .. ! grep -qxF .*$/    if false; then/
m19_install_records_no_sha256	pre-push-tags.sh	s/^        (set -C .. printf .*$/        true/
m20_the_ttl_cap_is_not_rechecked	pre-push-tags.sh	s/^        if . "\$mexp" .gt .*$/        if false; then/
m21_verify_ignores_the_effective_hooks_path	pre-push-tags.sh	s/^    if . "\$eff" != "\$dir" .; then$/    if false; then/
m22_the_ttl_is_read_from_the_environment	pre-push-tags.sh	s/ttl="\${2:-\$TTL_DEFAULT}"/ttl="\${2:-\${PRE_PUSH_TAGS_TTL:-\$TTL_DEFAULT}}"/
m23_the_ttl_cap_is_read_from_the_environment	pre-push-tags.sh	s/^TTL_MAX=3600$/TTL_MAX="\${TTL_MAX:-3600}"/
m24_verify_walks_only_the_main_admin_dir	pre-push-tags.sh	s/^    for gd in "\$common" "\$common".worktrees...; do$/    for gd in "$common"; do/
m25_an_unreadable_admin_dir_is_not_counted	pre-push-tags.sh	s/elif . "\$rc" .ne 0 .; then unjudged/elif false; then unjudged/
m26_the_release_head_equality_is_dropped	pre-push-tags.sh	s/^    \[ "\$commit" = "\$head" \]$/    true/
m27_the_release_head_is_an_ancestor_test	pre-push-tags.sh	s/^    \[ "\$commit" = "\$head" \]$/    git merge-base --is-ancestor "$commit" "$head"/
m28_the_rc_suffix_is_kept	pre-push-tags.sh	s/^    ver="\${3#v}"; ver="\${ver%%-rc.\*}"$/    ver="${3#v}"/'

mutants() {
    local name file expr rc killed=0 total=0 t
    t="$(printf '\t')"
    WORK="$(mktemp -d)"
    trap 'rm -rf -- "${WORK:?}"' EXIT
    while IFS="$t" read -r name file expr; do
        total=$((total + 1))
        cp -- "$HOOK_DIR/pre-push-tags.sh" "$HOOK_DIR/pre-push-dispatch.sh" "$WORK/"
        sed -e "$expr" "$HOOK_DIR/$file" > "$WORK/$file"
        if cmp -s "$HOOK_DIR/$file" "$WORK/$file" || ! bash -n "$WORK/$file"; then
            printf '  FAIL  %s did not apply or does not parse\n' "$name"
            continue
        fi
        rc=0
        bash "$SELF" --self-test "$WORK/pre-push-tags.sh" > /dev/null 2>&1 || rc=$?
        if [ "$rc" -ne 0 ]; then
            printf '  ok    %s killed\n' "$name"
            killed=$((killed + 1))
        else
            printf '  FAIL  %s survived\n' "$name"
        fi
    done <<< "$MUTANTS"
    printf -- '--- %s/%s mutants killed ---\n' "$killed" "$total"
    [ "$killed" -eq "$total" ]
}

case "${1:-}" in
    --self-test) self_test "${2:-}" ;;
    --mutants) mutants ;;
    --help | -h) sed -n '2,20p' "$SELF" ;;
    '') bare ;;
    *) sed -n '9,20p' "$SELF" >&2; exit 2 ;;
esac
