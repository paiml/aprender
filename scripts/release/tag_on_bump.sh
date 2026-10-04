#!/usr/bin/env bash
# tag_on_bump.sh -- the release tag v<V> sits ON the commit that bumped the version to V, with no
# commit between the bump and the tag (#4692, BLD-002 R3 as signed by the operator).
#
#   bash scripts/release/tag_on_bump.sh <V>                       # judge refs/tags/v<V>
#   bash scripts/release/tag_on_bump.sh <V> --commit <rev>        # judge a commit before it is tagged
#   bash scripts/release/tag_on_bump.sh <V> --root <repo> ...     # another checkout (default: .)
#   bash scripts/release/tag_on_bump.sh --self-test               # planted rows + mutants, scratch repo
#
# WHY. Signed for 0.70.2: "At release, with the tag on the bump commit." A commit between the bump
# and the tag is code no gate measured under the released version: 0.70.1 had 3 (bump 291c4f5ccc,
# 21.95 h before crates.io), 0.67.0 had 13, 0.69.1 had 541. The 0.70.2 ledger's P5 measured it after
# the fact with a pickaxe -- `git log --reverse -S'version = "V"' vV -- Cargo.toml | head -n 1` --
# which takes the FIRST commit whose Cargo.toml text gains the string. That reads a dependency table's
# `version = "V"` line, or a bump that was reverted and redone, as the bump. This keeps P5's count
# (`git rev-list --count B..T`, must be 0) and makes B exact.
#
# THE BUMP COMMIT B of a target T: walking T's first-parent history from T, the newest commit whose
# root Cargo.toml version is V while its parent's is not. "Version" is [workspace.package] version,
# or [package] version when the manifest has no [workspace.package]. Only commits that touch a
# `version = "` line of Cargo.toml are read (git log -G), so the walk costs one read per bump.
#
# VERDICTS (one line, then `bump=<B|none> between=<N|NA>` for the ledger):
#   0  PASS          T carries V and T is B: 0 commits between the bump and the tag
#   1  REFUSE        N >= 1 commits between B and T; or no bump commit (T does not carry V)
#   2  NOT_MEASURED  V is not a version, T does not resolve (no tag), or T's version is unreadable.
#                    Unknown is never a pass (L25).
# This script decides nothing by itself: cut_tag() in autopilot.sh logs its line, report-only, and
# (once wired) the release-gates nightly runs its self-test and judges the newest final tag; until then
# no nightly measures it, and three green nights are what make it a refusal (L31).
set -uo pipefail
PROG=tag_on_bump

# ws_version <root> <rev> -- the root manifest's version at <rev>, or nothing
ws_version() {
    git -C "$1" show "$2:Cargo.toml" 2>/dev/null | awk '
        /^\[/ { sec = $0; sub(/[ \t]*(#.*)?$/, "", sec); next }
        /^version[ \t]*=[ \t]*"/ {
            v = $0; sub(/^version[ \t]*=[ \t]*"/, "", v); sub(/".*$/, "", v)
            if (sec == "[workspace.package]") w = v; else if (sec == "[package]") p = v
        }
        END { if (w != "") print w; else if (p != "") print p }'
}

# judge <root> <V> <rev> <label> -- prints the verdict and returns 0/1/2
judge() {
    local root=$1 v=$2 rev=$3 label=$4 t tv c b="" pv n
    if ! [[ "$v" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.+-]+)?$ ]]; then
        echo "NOT_MEASURED $PROG: '$v' is not a version"; echo "bump=none between=NA"; return 2
    fi
    if ! t=$(git -C "$root" rev-parse -q --verify "$rev^{commit}" 2>/dev/null) || [ -z "$t" ]; then
        echo "NOT_MEASURED $PROG: $label does not resolve to a commit (no tag yet?)"; echo "bump=none between=NA"; return 2
    fi
    tv=$(ws_version "$root" "$t")
    if [ -z "$tv" ]; then
        echo "NOT_MEASURED $PROG: no version readable from Cargo.toml at $label ${t:0:12}"; echo "bump=none between=NA"; return 2
    fi
    if [ "$tv" = "$v" ]; then
        local cands nm="" sh shq
        # git log failing is Unknown, not "no bump commit"
        if ! cands=$(git -C "$root" log --first-parent --format=%H -G'^version[[:space:]]*=' "$t" -- Cargo.toml 2>/dev/null); then
            echo "NOT_MEASURED $PROG: the history under $label ${t:0:12} could not be read"; echo "bump=none between=NA"; return 2
        fi
        # a shallow repository whose shallow file cannot be located (git < 2.31 has no --path-format)
        # treats every parentless candidate as a boundary: Unknown, never a root
        shq=$(git -C "$root" rev-parse --is-shallow-repository 2>/dev/null)
        sh=$(git -C "$root" rev-parse --path-format=absolute --git-path shallow 2>/dev/null)
        while IFS= read -r c; do
            [ -n "$c" ] || continue
            [ "$(ws_version "$root" "$c")" = "$v" ] || continue
            if git -C "$root" rev-parse -q --verify "$c^{commit}^" > /dev/null 2>&1; then
                pv=$(ws_version "$root" "$c^")
                # an empty parent version is "not V" only when the parent has no Cargo.toml at all
                if [ -z "$pv" ] && git -C "$root" cat-file -e "$c^:Cargo.toml" 2>/dev/null; then
                    nm="the parent of ${c:0:12} has a Cargo.toml with no readable version"; break
                fi
            elif [ "$shq" != false ] && { [ -z "$sh" ] || [ ! -f "$sh" ] || grep -qx "$c" "$sh"; }; then
                nm="${c:0:12} is a shallow-clone boundary: its parent is not in this checkout"; break
            else
                pv=""   # a true root commit: nothing before it carried V
            fi
            [ "$pv" != "$v" ] && { b=$c; break; }   # bump: newest V-commit whose parent is not V
        done <<< "$cands"
        if [ -n "$nm" ]; then
            echo "NOT_MEASURED $PROG: $nm"; echo "bump=none between=NA"; return 2
        fi
    fi
    if [ -z "$b" ]; then
        echo "REFUSE $PROG: no bump commit -- $label ${t:0:12} carries version $tv, and no commit under it bumps to $v"
        echo "bump=none between=NA"; return 1
    fi
    n=$(git -C "$root" rev-list --count "$b..$t" 2>/dev/null) || n=""
    if [ -z "$n" ]; then
        echo "NOT_MEASURED $PROG: commits between bump ${b:0:12} and $label could not be counted"; echo "bump=$b between=NA"; return 2
    fi
    if [ "$n" -eq 0 ]; then
        echo "PASS $PROG: $label ${t:0:12} is the bump commit to $v (0 commits between bump and tag)"
        echo "bump=$b between=0"; return 0
    fi
    echo "REFUSE $PROG: $n commit(s) between the bump to $v (${b:0:12}) and $label ${t:0:12}"
    echo "bump=$b between=$n"; return 1
}

main() {
    local v="" root=. rev="" label
    v=${1:-}; [ "$#" -gt 0 ] && shift
    while [ "$#" -gt 0 ]; do
        case $1 in
            --root) root=${2:?--root needs a path}; shift 2 ;;
            --commit) rev=${2:?--commit needs a rev}; shift 2 ;;
            *) echo "$PROG: usage: <V> [--commit <rev>] [--root <repo>] | --self-test" >&2; return 2 ;;
        esac
    done
    [ -n "$v" ] || { echo "$PROG: usage: <V> [--commit <rev>] [--root <repo>] | --self-test" >&2; return 2; }
    if [ -n "$rev" ]; then label="commit $rev"; else rev="refs/tags/v$v"; label="tag v$v"; fi
    judge "$root" "$v" "$rev" "$label"
}

# ---------------------------------------------------------------------------------------------
# self-test: planted rows on a scratch repository, then mutants of THIS file (each must turn a
# row red). Prints ok/FAIL rows and "mutants killed K/N".
mkrepo() {   # mkrepo <dir>: a repo whose root manifest is at 1.2.2, plus one work commit
    local d=$1
    git init -q "$d" && git -C "$d" config user.email t@example.invalid && git -C "$d" config user.name t \
        && git -C "$d" config core.hooksPath /dev/null \
        && git -C "$d" config commit.gpgsign false && git -C "$d" config tag.gpgsign false || return 1
    setv "$d" 1.2.2 "init"; work "$d" a
}
setv() {     # setv <dir> <version> <msg>: commit the manifest at <version>
    printf '[workspace]\nmembers = []\n\n[workspace.package]\nversion = "%s"\n%s' "$2" "${EXTRA:-}" > "$1/Cargo.toml"
    git -C "$1" add Cargo.toml && git -C "$1" commit -qm "$3"
}
work() { printf '%s\n' "$2" >> "$1/w.txt"; git -C "$1" add w.txt && git -C "$1" commit -qm "work $2"; }

self_test() {
    local subj=${1:-${BASH_SOURCE[0]}} quiet=${2:-} d fail=0 got rc row want name expect
    command -v git > /dev/null || { echo "$PROG self-test: needs git"; return 2; }
    d=$(mktemp -d) || return 2
    run() { got=$(bash "$subj" "$@" 2>&1); rc=$?; }
    check() {   # check <name> <want-rc> <want-substring>
        if [ "$rc" = "$2" ] && [[ "$got" == *"$3"* ]]; then [ -n "$quiet" ] || echo "  ok   $1 (rc=$rc)"
        else echo "  FAIL $1: want rc=$2 '$3', got rc=$rc: ${got//$'\n'/ | }"; fail=1; fi
    }
    [ -n "$quiet" ] || echo "$PROG self-test: planted rows"
    # 1. the tag on the bump commit passes
    mkrepo "$d/r1" && setv "$d/r1" 1.2.3 bump && git -C "$d/r1" tag v1.2.3
    run 1.2.3 --root "$d/r1"; check tag_on_bump_passes 0 "between=0"
    # 2. one commit between bump and tag refuses
    mkrepo "$d/r2" && setv "$d/r2" 1.2.3 bump && work "$d/r2" b && git -C "$d/r2" tag v1.2.3
    run 1.2.3 --root "$d/r2"; check one_commit_between_refuses 1 "between=1"
    # 3. no bump commit refuses: v1.2.3 tagged on a commit that is still 1.2.2
    mkrepo "$d/r3" && git -C "$d/r3" tag v1.2.3
    run 1.2.3 --root "$d/r3"; check no_bump_commit_refuses 1 "no bump commit"
    # 4. an unreadable tag prints not_measured
    mkrepo "$d/r4" && setv "$d/r4" 1.2.3 bump
    run 1.2.3 --root "$d/r4"; check unreadable_tag_not_measured 2 "NOT_MEASURED"
    # 5. an unreadable version prints not_measured: the argument, then the manifest
    run 1.2 --root "$d/r1"; check unreadable_version_arg_not_measured 2 "NOT_MEASURED"
    mkrepo "$d/r5" && printf '[workspace]\nmembers = []\n' > "$d/r5/Cargo.toml" && git -C "$d/r5" commit -qam nover \
        && git -C "$d/r5" tag v1.2.3
    run 1.2.3 --root "$d/r5"; check unreadable_manifest_version_not_measured 2 "NOT_MEASURED"
    # 6. P5's pickaxe takes the FIRST commit gaining the string; a reverted-then-redone bump is judged
    #    by the redo, which here IS the tag
    mkrepo "$d/r6" && setv "$d/r6" 1.2.3 early-bump && work "$d/r6" b && setv "$d/r6" 1.2.2 revert \
        && work "$d/r6" c && setv "$d/r6" 1.2.3 bump && git -C "$d/r6" tag v1.2.3
    run 1.2.3 --root "$d/r6"; check rebump_judged_by_the_redo 0 "between=0"
    # 7. a dependency table's `version = "1.2.3"` line before the bump is not the bump
    mkrepo "$d/r7" && EXTRA=$'\n[workspace.dependencies.x]\nversion = "1.2.3"\n' setv "$d/r7" 1.2.2 dep \
        && work "$d/r7" b && EXTRA=$'\n[workspace.dependencies.x]\nversion = "1.2.3"\n' setv "$d/r7" 1.2.3 bump \
        && git -C "$d/r7" tag v1.2.3
    run 1.2.3 --root "$d/r7"; check dep_line_is_not_the_bump 0 "between=0"
    # 8. --commit judges a commit before it is tagged (autopilot cut_tag's call)
    run 1.2.3 --root "$d/r4" --commit HEAD; check commit_mode_on_bump_passes 0 "between=0"
    run 1.2.3 --root "$d/r2" --commit HEAD; check commit_mode_after_bump_refuses 1 "between=1"
    # 9. a commit after the bump that edits a `version = "` line but keeps V is not the bump
    mkrepo "$d/r9" && setv "$d/r9" 1.2.3 bump \
        && EXTRA=$'\n[workspace.dependencies.x]\nversion = "9.9.9"\n' setv "$d/r9" 1.2.3 dep-after && git -C "$d/r9" tag v1.2.3
    run 1.2.3 --root "$d/r9"; check version_line_edit_after_bump_refuses 1 "between=1"
    # 10. a shallow clone whose boundary is the tag commit cannot see the parent: NOT_MEASURED, never PASS
    git clone -q --depth 1 --no-local "file://$d/r2" "$d/r10" 2>/dev/null && git -C "$d/r10" fetch -q --depth 1 origin tag v1.2.3 2>/dev/null
    run 1.2.3 --root "$d/r10" --commit HEAD; check shallow_boundary_not_measured 2 "shallow-clone boundary"
    # 11. a parent manifest with no readable version is Unknown, not "not V"
    mkrepo "$d/r11" && printf '[workspace]\nmembers = []\n' > "$d/r11/Cargo.toml" && git -C "$d/r11" commit -qam nover \
        && setv "$d/r11" 1.2.3 bump && git -C "$d/r11" tag v1.2.3
    run 1.2.3 --root "$d/r11"; check unreadable_parent_version_not_measured 2 "no readable version"

    if [ -z "$quiet" ]; then
        # MUTANTS of THIS file: each removes one refusal (or the exact bump) and must turn a row red
        local src a1 b1 k=0 n=0
        src=$(cat -- "$subj"; printf x); src=${src%x}; src=${src%%$'\n'"# ---------"*}
        for row in 'between refusal|if [ "$n" -eq 0 ]; then|if [ "$n" -ge 0 ]; then' \
                   'no-bump refusal|echo "bump=none between=NA"; return 1|echo "bump=none between=0"; return 0' \
                   'unreadable tag|(no tag yet?)"; echo "bump=none between=NA"; return 2|"; echo "bump=none between=0"; return 0' \
                   'unreadable version arg|is not a version"; echo "bump=none between=NA"; return 2|"; echo "bump=none between=0"; return 0' \
                   'unreadable manifest|$label ${t:0:12}"; echo "bump=none between=NA"; return 2|"; echo "bump=none between=0"; return 0' \
                   'exact bump (first match, as the pickaxe)|{ b=$c; break; }|{ b=$c; }' \
                   'shallow boundary (read as a root)|nm="${c:0:12} is a shallow-clone boundary: its parent is not in this checkout"; break|pv=""' \
                   'unreadable parent (read as not V)|nm="the parent of ${c:0:12} has a Cargo.toml with no readable version"; break|:' \
                   'parent check (any V-commit is the bump)|[ "$pv" != "$v" ] && |'; do
            IFS='|' read -r name a1 b1 <<< "$row"; n=$((n + 1))
            case $src in *"$a1"*) ;; *) echo "  FAIL mutant $name: anchor moved, re-anchor it"; fail=1; continue ;; esac
            { printf '%s' "${src/"$a1"/"$b1"}"; printf '\nmain "$@"\n'; } > "$d/mut.sh"
            if got=$(self_test "$d/mut.sh" quiet 2>&1); then echo "  FAIL mutant ($name) survived the table"; fail=1
            else got=${got#*FAIL }; echo "  ok   mutant ($name) killed by row ${got%%:*}"; k=$((k + 1)); fi
        done
        echo "$PROG self-test: mutants killed $k/$n"
    fi
    rm -rf -- "${d:?}"
    if [ "$fail" -eq 0 ]; then [ -n "$quiet" ] || echo "$PROG self-test: PASS"; return 0; fi
    [ -n "$quiet" ] || echo "$PROG self-test: FAIL"; return 1
}

case "${1:-}" in
    --self-test) self_test ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}" ;;
    *) main "$@" ;;
esac
