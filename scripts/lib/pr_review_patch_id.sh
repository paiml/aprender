# shellcheck shell=bash
# scripts/lib/pr_review_patch_id.sh - the diff patch-id a pr-review receipt binds to (#4421).
#
# SOURCED, so it is option-neutral: no `set`, no `exit`. Every function reports by
# return status (CLAUDE.md, "a SOURCED library must be option-neutral").
#
# WHY A PATCH-ID. The merge queue lands a PR as a SQUASH with ONE parent, so the
# reviewed head_sha is never an ancestor of the queue sha: the head binding matched
# 0 of 6 queue merges, a tree binding 4 of 6, and the diff's patch-id 6 of 6
# (merge-queue evaluation, operator Y5, 2026-09-25). So the receipt records the
# patch-id of the diff it reviewed, and Arm 4 recomputes it from the subject.
#
# WHY --verbatim AND NOT --stable. --stable strips all whitespace before hashing, so
# a whitespace-only change (an indentation fix in Python/YAML, a changed string
# literal's spaces) would keep the reviewed id. --verbatim hashes every byte of
# every line; a 1-byte change to the patch is a different id.
#
# WHAT IS EXCLUDED. evidence/pr-review/<pr>/ only: the receipt and its signature are
# committed INTO the PR, after the review, so they cannot be part of the diff they
# attest. Nothing else is excluded.
#
# THE DIFF IS PINNED. A patch-id is a hash of `git diff` output, and that output
# depends on config: diff.algorithm, diff.renames, diff.orderFile, color.diff,
# diff.noprefix, diff.external, diff.suppressBlankEmpty, textconv drivers. Every one
# of them is overridden here, so the signer and the verifier compute the same bytes
# on any box.
#
# OLD GIT. `git patch-id --verbatim` arrived in git 2.40; lambda and intel run
# 2.34.1 (rc 129). Where the native flag is missing, scripts/lib/git_patch_id.py, a
# line-for-line port of git v2.53 get_one_patchid(), computes it. The port is checked
# against native git by prpid_self_test on every box that has a native command:
# --stable/--unstable against 2.34, --verbatim as well against 2.40+. Measured
# 2026-09-25: 120/120 on lambda (2.34.1), 189/189 on yoga (2.53.0) and gx10 (2.43.0).
#
# FAILS CLOSED. prpid_compute prints a 40-hex id and returns 0, or prints nothing
# and returns non-zero: 1 for a diff with no patch-id (empty, unparseable, more than
# one id) or an unresolvable commit, 2 when the box has neither a native
# --verbatim nor python3.
#
#   PRPID_IMPL  auto (default) | native | reimpl - which implementation computes it.
#               auto takes native when the flag exists. No value skips the check.

PRPID_ALGO='git-patch-id-verbatim/pinned-diff-v1'
PRPID_LIB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# prpid_native_ok - 0 when this git has `patch-id --verbatim`.
prpid_native_ok() {
    git patch-id --verbatim </dev/null >/dev/null 2>&1
}

# prpid_diff <gitdir> <base> <head> <pr> - the pinned diff, on stdout.
prpid_diff() {
    local gd=$1 base=$2 head=$3 pr=$4
    case "$pr" in ''|*[!0-9]*) echo "prpid: PR number '$pr' is not a number" >&2; return 1 ;; esac
    git -C "$gd" \
        -c core.quotePath=true -c diff.noprefix=false -c diff.mnemonicPrefix=false \
        -c diff.suppressBlankEmpty=false -c diff.relative=false \
        diff --no-color --no-ext-diff --no-textconv --no-renames --full-index \
        --diff-algorithm=myers --indent-heuristic --inter-hunk-context=0 -U3 \
        --src-prefix=a/ --dst-prefix=b/ -O/dev/null \
        "$base" "$head" -- . ":(exclude)evidence/pr-review/$pr"
}

# prpid_of_stdin - patch-id of the diff on stdin, verbatim, as 40 hex.
prpid_of_stdin() {
    local impl=${PRPID_IMPL:-auto} out rc=0
    case "$impl" in
      auto)   if prpid_native_ok; then impl=native; else impl=reimpl; fi ;;
      native|reimpl) ;;
      *) echo "prpid: PRPID_IMPL='$impl' is not auto|native|reimpl" >&2; return 2 ;;
    esac
    if [ "$impl" = native ]; then
        prpid_native_ok || { echo "prpid: this git ($(git --version)) has no patch-id --verbatim" >&2; return 2; }
        out=$(git patch-id --verbatim) || rc=$?
    else
        command -v python3 >/dev/null 2>&1 \
            || { echo "prpid: git lacks patch-id --verbatim and python3 is absent" >&2; return 2; }
        out=$(python3 "$PRPID_LIB_DIR/git_patch_id.py" --verbatim) || rc=$?
    fi
    [ "$rc" -eq 0 ] || { echo "prpid: the $impl patch-id failed (rc $rc)" >&2; return 1; }
    # Exactly one id, for exactly one patch. Empty output is an EMPTY diff: nothing
    # was reviewed, and nothing binds.
    if ! printf '%s\n' "$out" | grep -Eqx '[0-9a-f]{40} 0{40}' || [ "$(printf '%s\n' "$out" | wc -l)" -ne 1 ]; then
        echo "prpid: no single patch-id in the $impl output (empty diff?): '${out:0:100}'" >&2
        return 1
    fi
    printf '%s\n' "${out%% *}"
}

# prpid_compute <gitdir> <base> <head> <pr> - the id the receipt binds, as 40 hex.
prpid_compute() {
    local gd=$1 base=$2 head=$3 pr=$4 c diff rc=0
    for c in "$base" "$head"; do
        git -C "$gd" rev-parse --verify --quiet "${c}^{commit}" >/dev/null \
            || { echo "prpid: '$c' is not a commit in $gd" >&2; return 1; }
    done
    diff=$(prpid_diff "$gd" "$base" "$head" "$pr" && echo x) || rc=$?
    [ "$rc" -eq 0 ] || { echo "prpid: git diff $base $head failed (rc $rc)" >&2; return 1; }
    # The trailing x keeps the final newline, which is part of the hashed bytes.
    printf '%s' "${diff%x}" | prpid_of_stdin
}

# prpid_self_test <gitdir> - the port, against GOLDEN ids and against native git.
#
# git 2.34 is NOT an oracle for the port. Its legacy --stable/--unstable predate the
# binary-oid hashing and the "\ No newline" handling of git 2.39/2.40, and on the
# whitespace/binary/no-EOL fixture below it disagrees with git 2.53 itself (measured
# 2026-09-25 on lambda: 4 of 4 fixture comparisons differ; yoga 2.53 and gx10 2.43
# agree with the port on all of them). So:
#   - GOLDEN: the fixture's ids, measured with native git 2.53.0 on yoga, are
#     asserted on every box. Its blob oids depend on content only, so the diff is
#     byte-identical everywhere (md5 7e15f43c... on lambda and yoga).
#   - NATIVE: where this git has --verbatim (>= 2.40, the same algorithm era), the
#     port must equal it in all three modes on the fixture plus the last 20 commits
#     of <gitdir>. On an older git the native comparison is SKIPPED and says so.
# Returns 0 when everything compared agrees, 1 on a disagreement, 2 when the box
# cannot run it (no python3, fixture not buildable).
prpid_self_test() {
    local gd=$1 td n=0 bad=0 m c a b f want native=0
    command -v python3 >/dev/null 2>&1 || { echo "prpid self-test: python3 absent" >&2; return 2; }
    prpid_native_ok && native=1
    td=$(mktemp -d "${TMPDIR:-/tmp}/prpid-selftest.XXXXXX") || return 2
    (
        cd "$td" && git init -q ws && cd ws || exit 2
        ci() { git -c user.name=t -c user.email=t@t -c core.hooksPath=/dev/null commit -qm "$1"; }
        printf 'a \n\tb\r\nc' > f; printf '\000\001bin' > b.bin
        git add . && ci 1 || exit 2
        printf 'a  \n\tb\nc\n' > f; printf '\000\002bin' > b.bin; echo new > g
        git add . && ci 2 || exit 2
        git diff --no-color --full-index HEAD^ HEAD > "$td/ws.diff"
        git diff --no-color --full-index --binary HEAD^ HEAD > "$td/wsbin.diff"
    ) || { echo "prpid self-test: could not build the whitespace fixture" >&2; case "$td" in *prpid-selftest.*) rm -rf -- "${td:?}" ;; esac; return 2; }

    # Golden ids (native git 2.53.0, yoga, 2026-09-25). --binary changes nothing:
    # a binary hunk is hashed as its two blob oids, never its bytes.
    for f in ws.diff wsbin.diff; do
        for m in --verbatim:4866d77a6281c72194256e3826f262bcf982de28 \
                 --stable:1379117dba9a33896e4076e98048a064488ae8db \
                 --unstable:92a2dd2834633681d83627e08e08cd2b6b4c6ab4; do
            want=${m#*:}; m=${m%%:*}
            b=$(python3 "$PRPID_LIB_DIR/git_patch_id.py" "$m" < "$td/$f")
            n=$((n + 1))
            if [ "${b%% *}" != "$want" ]; then
                bad=$((bad + 1)); echo "prpid self-test: GOLDEN $f $m want=$want port=[$b]" >&2
            fi
        done
    done

    if [ "$native" -eq 1 ]; then
        for c in $(git -C "$gd" rev-list --max-count=20 --no-merges HEAD 2>/dev/null); do
            git -C "$gd" diff --no-color --full-index "$c^" "$c" > "$td/h-$c.diff" 2>/dev/null \
                || rm -f -- "${td:?}/h-${c:?}.diff"
        done
        for f in "$td"/*.diff; do
            for m in --verbatim --stable --unstable; do
                a=$(git patch-id "$m" < "$f")
                b=$(python3 "$PRPID_LIB_DIR/git_patch_id.py" "$m" < "$f")
                n=$((n + 1))
                if [ "$a" != "$b" ]; then
                    bad=$((bad + 1)); echo "prpid self-test: NATIVE ${f##*/} $m native=[$a] port=[$b]" >&2
                fi
            done
        done
    else
        echo "prpid self-test: $(git --version) has no patch-id --verbatim; native comparison SKIPPED, golden only"
    fi
    case "$td" in *prpid-selftest.*) rm -rf -- "${td:?}" ;; esac
    echo "prpid self-test: $n comparisons, $bad disagreements"
    [ "$bad" -eq 0 ] && [ "$n" -gt 0 ]
}
