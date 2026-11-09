#!/usr/bin/env bash
# pin_build_base.sh — pin refs/remotes/origin/main to the commit this run's tree was BUILT on.
# It is the comparand every ratchet in guard-tree and guard-cargo reads (#4861).
#
# Both sections used to fetch the origin/main TIP when they started. A section can start hours
# after the merge commit it judges was built, so a branch that never touched a baseline went red
# when main shrank that baseline in between. #4813: built on 358e9d4d11, where
# pipe_grep_q_baseline.txt records 62; the section started when main's tip was ece9e563bd, after
# da7059c5c7 shrank it to 60; `GREW` 62 vs 60, in a file #4813 does not touch. A re-run builds
# the same merge commit, so the false red survives every re-run. #4983 is the same defect on a
# RE-RUN: #4970's m17 complexity step, attempt 2, judged its unchanged merge commit against main's
# moved tip acd599e52 and went red where attempt 1 had not. The shared ratchet library also takes
# the merge commit's first parent on its own (scripts/lib_baseline_ratchet.sh FIRSTPARENT); this
# pin covers every reader of origin/main, library or not, and the merge queue.
#
#   pull_request  the merge commit's FIRST parent. HEAD must be a two-parent merge commit whose
#                 second parent is the PR head (PR_HEAD_SHA); any other HEAD names no build base,
#                 and this refuses rather than guess one.
#   merge_group   merge_group.base_sha (MG_BASE_SHA), as the refinement-gate pin does. The queue
#                 builds on main as it merges, so an entry main already deleted that a PR adds
#                 back is still refused there.
#   otherwise     the origin/main tip, deepened by one on push: scripts/lib/resolve_base.sh judges
#                 a push against HEAD's first parent, and a depth-1 checkout has not fetched it.
#
# EXIT 0 pinned · 1 refused (no build base can be named) or a fetch failed · 64 usage
#
#   PR_HEAD_SHA=<sha> MG_BASE_SHA=<sha> bash scripts/ci/pin_build_base.sh [--self-test]
set -euo pipefail

ZERO=0000000000000000000000000000000000000000
# #4936 (P6, operator C343 #9): every fetch below goes through fetch_p6.sh, as the sections' fetches
# do. A read that did not answer (curl 92, curl 56, a stall) is read again, up to 3 reads in all; a
# read that answered, an object origin does not have among them, is never read again.
P6="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/fetch_p6.sh"

fetch() { # fetch <repo dir> <git fetch args...>: git fetch in <repo dir>, read under P6
    local dir="$1"; shift
    bash "$P6" -C "$dir" "$@"
}

is_sha() { [[ ${1:-} =~ ^[0-9a-f]{40}$ ]] && [ "$1" != "$ZERO" ]; }

refuse() {
    printf '::error::pin_build_base: %s\n' "$1"
    return 1
}

pin() { # pin <repo dir>: pins origin/main there for $GITHUB_EVENT_NAME
    local dir="$1" ev="${GITHUB_EVENT_NAME:-}" base how
    local parents=()
    case "$ev" in
        pull_request) # PIN-BUILD-BASE-MUTATION-POINT
            mapfile -t parents < <(git -C "$dir" cat-file -p 'HEAD^{commit}' 2>/dev/null | awk '/^$/{exit} /^parent /{print $2}')
            [ "${#parents[@]}" -eq 2 ] \
                || refuse "pull_request: HEAD has ${#parents[@]} parent(s), not 2. Only refs/pull/N/merge names its build base." || return 1
            is_sha "${PR_HEAD_SHA:-}" || refuse "pull_request: no PR head sha (PR_HEAD_SHA='${PR_HEAD_SHA:-}')" || return 1
            [ "${parents[1]}" = "$PR_HEAD_SHA" ] \
                || refuse "pull_request: HEAD's second parent ${parents[1]} is not the PR head $PR_HEAD_SHA, so HEAD is not this PR's merge commit" || return 1
            base="${parents[0]}"; how="first parent of the merge commit"
            ;;
        merge_group)
            is_sha "${MG_BASE_SHA:-}" || refuse "merge_group: no base_sha (MG_BASE_SHA='${MG_BASE_SHA:-}')" || return 1
            base="$MG_BASE_SHA"; how="merge_group.base_sha"
            ;;
        *)
            fetch "$dir" -q --no-tags --depth=1 origin +refs/heads/main:refs/remotes/origin/main || return 1
            if [ "$ev" = push ]; then
                fetch "$dir" -q --no-tags --deepen=1 origin +refs/heads/main:refs/remotes/origin/main || return 1
            fi
            printf 'comparand: %s pinned as origin/main (the tip; event %s)\n' "$(git -C "$dir" rev-parse 'origin/main^{commit}')" "${ev:-none}"
            return 0
            ;;
    esac
    fetch "$dir" -q --no-tags --depth=1 origin "$base" || refuse "cannot fetch the build base $base" || return 1
    git -C "$dir" update-ref refs/remotes/origin/main "$base"
    [ "$(git -C "$dir" rev-parse 'origin/main^{commit}')" = "$base" ] || refuse "origin/main does not read back as $base" || return 1
    printf 'comparand: %s pinned as origin/main (%s; event %s)\n' "$base" "$how" "$ev"
}

# ---------------------------------------------------------------------------
# The case table. Scratch origin + depth-1 clones, the shape actions/checkout leaves. The #4813
# rows run the REAL scripts/lib_baseline_ratchet.sh: the old tip comparand is RED on a PR that
# never touched the baseline, the build base is GREEN, and a PR that grows it is still RED.
self_test() {
    local here; here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
    local lib="$here/scripts/lib_baseline_ratchet.sh"
    [ -f "$lib" ] || { printf 'FAIL  self-test: %s is missing\n' "$lib"; return 1; }
    TD=$(mktemp -d); trap 'rm -rf "${TD:?}"' EXIT
    # #4936: the table runs in TD, and git finds no repository above it. A fetch that lost its -C (a
    # mutant, a slip) then reads "not a git repository" here, never the caller's checkout.
    # The ceiling is TD's parent: git never stops at the directory it starts in.
    cd -- "$TD" || return 1
    export GIT_CEILING_DIRECTORIES="${TD%/*}"
    local o="$TD/origin.git" w="$TD/w" u="file://$TD/origin.git"
    local c0 prh m c2 grh m2 ok=0 bad=0
    g() { git -C "$w" -c user.name=st -c user.email=st@example.invalid -c commit.gpgsign=false "$@"; }
    git init -q --template= --bare -b main "$o"
    git -C "$o" config uploadpack.allowReachableSHA1InWant true
    git init -q --template= -b main "$w"
    printf '# count\n62\n' > "$w/r_baseline.txt"; g add -A; g commit -q -m c0; c0=$(g rev-parse HEAD)
    g push -q "$u" main
    # The PR: off c0, and it never touches the baseline. GitHub's merge commit: c0 first, PR head second.
    g checkout -q -b pr; printf 'pr\n' > "$w/pr.txt"; g add -A; g commit -q -m pr; prh=$(g rev-parse HEAD)
    g checkout -q --detach "$c0"; g merge -q --no-ff --no-edit pr; m=$(g rev-parse HEAD)
    g push -q "$u" "$m:refs/pull/1/merge"
    # main moves on AFTER that build: it shrinks the baseline 62 -> 60 (#4813).
    g checkout -q main; printf '# count\n60\n' > "$w/r_baseline.txt"; g add -A; g commit -q -m c2; c2=$(g rev-parse HEAD)
    g push -q "$u" main
    # A PR built on c2 that GROWS the baseline 60 -> 61.
    g checkout -q -b grow; printf '# count\n61\n' > "$w/r_baseline.txt"; g add -A; g commit -q -m grow; grh=$(g rev-parse HEAD)
    g checkout -q --detach "$c2"; g merge -q --no-ff --no-edit grow; m2=$(g rev-parse HEAD)
    g push -q "$u" "$m2:refs/pull/2/merge"


    clone_at() { # clone_at <name> <sha>: a depth-1 clone checked out at <sha>, the shape actions/checkout leaves
        local d="$TD/$1"
        [ ! -e "$d" ] || { printf 'FAIL  self-test: clone %s reused; a row would judge another row'"'"'s pin\n' "$1" >&2; return 1; }
        git init -q --template= "$d" && git -C "$d" remote add origin "$u" \
            && git -C "$d" fetch -q --no-tags --depth=1 origin "$2" && git -C "$d" checkout -q --detach FETCH_HEAD \
            || { printf 'FAIL  self-test: cannot build clone %s at %s\n' "$1" "$2" >&2; return 1; }
        printf '%s' "$d"
    }
    run_pin() { # run_pin <dir> <event> <pr head> <mg base> -> pin's rc; PINNED = origin/main after, or "unset"
        local rc=0
        ( GITHUB_EVENT_NAME="$2" PR_HEAD_SHA="$3" MG_BASE_SHA="$4" pin "$1" ) > "$TD/last.log" 2>&1 || rc=$?
        PINNED=$(git -C "$1" rev-parse -q --verify 'refs/remotes/origin/main^{commit}' 2>/dev/null) || PINNED=unset
        return "$rc"
    }
    ratchet() { # ratchet <dir> [VAR=value ...] -> the real library's verdict on r_baseline.txt: "<rc> <GREW|ok|other>"
        # The runner's own GITHUB_EVENT_NAME/GITHUB_SHA never leak in: a row names the event it judges.
        local dir="$1" rc=0 says=other; shift
        ( unset BASELINE_RATCHET_BASE_REF GITHUB_EVENT_NAME GITHUB_SHA; [ "$#" -eq 0 ] || export "$@"
          . "$lib" || exit 2; baseline_ratchet_check "$dir" r_baseline.txt count ) \
            > "$TD/last.log" 2>&1 || rc=$?
        if grep -q 'GREW' "$TD/last.log"; then says="GREW:$(sed -nE 's/.*count rose ([0-9]+ -> [0-9]+).*/\1/p' "$TD/last.log" | tr ' ' _)"
        elif grep -q '^ok    ratchet' "$TD/last.log"; then says=ok; fi
        printf '%s %s' "$rc" "$says"
    }
    row() { # row LABEL GOT WANT
        if [ "$2" = "$3" ]; then ok=$((ok + 1)); printf 'ok    %s\n' "$1"
        else bad=$((bad + 1)); printf 'FAIL  %s: got <%s>, want <%s>\n' "$1" "$2" "$3"; sed 's/^/        /' "$TD/last.log"; fi
    }
    said() { # said <text>: did the last run name its cause? A refusal is an andon, and an andon names why
        if grep -qF -- "$1" "$TD/last.log"; then printf 'says-cause'; else printf 'cause-missing'; fi
    }
    local d rc bogus PINNED nbad=0

    d=$(clone_at pr1 "$m"); rc=0; run_pin "$d" pull_request "$prh" "" || rc=$?
    row "pull_request pins the merge commit's first parent" "$rc $PINNED" "0 $c0"
    row "#4813 shape, build base: the PR never touched the baseline, ratchet GREEN" "$(ratchet "$d")" "0 ok"
    row "the library's own first-parent path (#4983) gives the pin's verdict" \
        "$(ratchet "$d" GITHUB_EVENT_NAME=pull_request "GITHUB_SHA=$m") $(said "first parent of this pull request's merge commit, $c0")" "0 ok says-cause"
    d=$(clone_at tip1 "$m"); rc=0; run_pin "$d" workflow_dispatch "" "" || rc=$?
    row "#4813 shape, the old tip comparand: the same PR is RED, 62 vs main's 60" "$rc $PINNED $(ratchet "$d")" "0 $c2 1 GREW:60_->_62"

    d=$(clone_at pr2 "$m2"); rc=0; run_pin "$d" pull_request "$grh" "" || rc=$?
    row "pull_request on a PR built on c2 pins c2" "$rc $PINNED" "0 $c2"
    row "a PR that grows the baseline is still RED against its build base" "$(ratchet "$d")" "1 GREW:60_->_61"
    row "... and still RED on the library's first-parent path" \
        "$(ratchet "$d" GITHUB_EVENT_NAME=pull_request "GITHUB_SHA=$m2")" "1 GREW:60_->_61"

    d=$(clone_at single "$prh"); rc=0; run_pin "$d" pull_request "$prh" "" || rc=$?
    row "pull_request on a single-parent HEAD (checkout of the PR head) refuses and pins nothing" \
        "$rc $PINNED $(said 'HEAD has 1 parent(s), not 2')" "1 unset says-cause"
    d=$(clone_at wronghead "$m"); rc=0; run_pin "$d" pull_request "$c2" "" || rc=$?
    row "pull_request whose second parent is not the PR head refuses" \
        "$rc $PINNED $(said 'is not the PR head')" "1 unset says-cause"
    d=$(clone_at swapped "$m"); rc=0; run_pin "$d" pull_request "$c0" "" || rc=$?
    row "pull_request naming the FIRST parent as the PR head refuses" \
        "$rc $PINNED $(said 'is not the PR head')" "1 unset says-cause"
    d=$(clone_at nohead "$m"); rc=0; run_pin "$d" pull_request "" "" || rc=$?
    row "pull_request with no PR head sha refuses" "$rc $PINNED $(said 'no PR head sha')" "1 unset says-cause"

    d=$(clone_at mg "$m2"); rc=0; run_pin "$d" merge_group "" "$c2" || rc=$?
    row "merge_group pins base_sha" "$rc $PINNED" "0 $c2"
    for bogus in "" "$ZERO" main "${c2:0:39}" "${c2^^}"; do
        nbad=$((nbad + 1)); d=$(clone_at "mgbad$nbad" "$m2"); rc=0; run_pin "$d" merge_group "" "$bogus" || rc=$?
        row "merge_group with base_sha <${bogus:0:12}> refuses" "$rc $PINNED $(said 'merge_group: no base_sha')" "1 unset says-cause"
    done
    d=$(clone_at mgmissing "$m2"); rc=0; run_pin "$d" merge_group "" "$(printf '%040d' 7)" || rc=$?
    row "merge_group whose base_sha origin does not have refuses" \
        "$rc $PINNED $(said 'cannot fetch the build base')" "1 unset says-cause"

    d=$(clone_at push "$c2"); rc=0; run_pin "$d" push "" "" || rc=$?
    row "push pins the tip" "$rc $PINNED" "0 $c2"
    row "push deepens by one: the tip's first parent is fetched" \
        "$(git -C "$d" rev-parse -q --verify 'origin/main^1^{commit}' 2>/dev/null || printf none)" "$c0"
    d=$(clone_at wd "$m"); rc=0; run_pin "$d" workflow_dispatch "" "" || rc=$?
    row "workflow_dispatch pins the tip" "$rc $PINNED" "0 $c2"
    rc=0; run_pin "$TD/pr1" pull_request "$prh" "" || rc=$?
    row "pinning twice is idempotent" "$rc $PINNED" "0 $c0"

    # #4936 P6: a planted git counts every fetch read and answers each read whose number is in
    # SHIM_DIE ("1 3": the first and the third) as a stalled read (curl 92); a planted sleep skips the back-off. The real git does the rest.
    local shim="$TD/shim" real_git READS
    real_git=$(command -v git)
    mkdir -p "$shim"
    printf '%s\n' '#!/usr/bin/env bash' \
        'case " $* " in *" fetch "*)' \
        '    echo fetch >> "$SHIM_LOG"' \
        '    case " $SHIM_DIE " in *" $(wc -l < "$SHIM_LOG" | tr -d " ") "*)' \
        "        printf '%s\\n' 'error: RPC failed; curl 92 HTTP/2 stream 5 was not closed cleanly: CANCEL (err 8)' 'fatal: early EOF' >&2; exit 128" \
        '    esac ;;' \
        'esac' \
        'exec "$SHIM_REAL" "$@"' > "$shim/git"
    printf '%s\n' '#!/usr/bin/env bash' 'exit 0' > "$shim/sleep"
    chmod +x "$shim/git" "$shim/sleep"
    run_pin_p6() { # run_pin_p6 "<dead read numbers>" <dir> <event> <pr head> <mg base> -> as run_pin; READS = fetch reads git saw
        local dies="$1" rc=0; shift
        : > "$TD/reads.log"
        PATH="$shim:$PATH" SHIM_REAL="$real_git" SHIM_LOG="$TD/reads.log" SHIM_DIE="$dies" run_pin "$@" || rc=$?
        READS=$(wc -l < "$TD/reads.log" | tr -d ' ')
        return "$rc"
    }
    d=$(clone_at p6mg "$m2"); rc=0; run_pin_p6 1 "$d" merge_group "" "$c2" || rc=$?
    row "P6: merge_group's base fetch, first read dead (curl 92), is read again and pins" \
        "$rc $PINNED $READS $(said 'read 1/3 did not answer')" "0 $c2 2 says-cause"
    d=$(clone_at p6pr "$m"); rc=0; run_pin_p6 "1 2" "$d" pull_request "$prh" "" || rc=$?
    row "P6: pull_request's base fetch, two reads dead, pins on the third" "$rc $PINNED $READS" "0 $c0 3"
    d=$(clone_at p6push "$c2"); rc=0; run_pin_p6 1 "$d" push "" "" || rc=$?
    row "P6: push's tip fetch, first read dead, is read again; the deepen still runs" \
        "$rc $PINNED $READS $(git -C "$d" rev-parse -q --verify 'origin/main^1^{commit}' 2>/dev/null || printf none)" "0 $c2 3 $c0"
    d=$(clone_at p6deepen "$c2"); rc=0; run_pin_p6 2 "$d" push "" "" || rc=$?
    row "P6: push's deepen fetch, first read dead, is read again; the tip's first parent is fetched" \
        "$rc $PINNED $READS $(git -C "$d" rev-parse -q --verify 'origin/main^1^{commit}' 2>/dev/null || printf none)" "0 $c2 3 $c0"
    d=$(clone_at p6cap "$m2"); rc=0; run_pin_p6 "1 2 3" "$d" merge_group "" "$c2" || rc=$?
    row "P6: three dead reads are the cap, and the pin refuses" \
        "$rc $PINNED $READS $(said 'no answer under P6') $(said 'cannot fetch the build base')" "1 unset 3 says-cause says-cause"
    d=$(clone_at p6answer "$m2"); rc=0; run_pin_p6 "" "$d" merge_group "" "$(printf '%040d' 7)" || rc=$?
    row "P6: a read that answered (origin has no such base) is read once, never again" \
        "$rc $PINNED $READS $(said 'cannot fetch the build base')" "1 unset 1 says-cause"

    row "the table runs in a cwd that is no repository, so a fetch without -C reaches no checkout" \
        "$(if git rev-parse --git-dir >/dev/null 2>&1; then printf repo; else printf none; fi)" none
    row "git looks no higher than TD, even if TD sits inside a checkout" \
        "${GIT_CEILING_DIRECTORIES:-unset}" "${TD%/*}"

    printf 'pin_build_base self-test: %s ok, %s bad\n' "$ok" "$bad"
    [ "$bad" -eq 0 ] && [ "$ok" -ge 30 ]
}

case "${1:-}" in
    --self-test) self_test ;;
    "") pin "$(git rev-parse --show-toplevel)" ;;
    *) printf 'usage: %s [--self-test]\n' "$0" >&2; exit 64 ;;
esac
