#!/usr/bin/env bash
# tag_on_bump_sandbox.sh — the nightly lane of the tag-on-bump judge (#4692).
#
#   tag_on_bump_sandbox.sh <V> [--root <repo>] [--rev <rev>]   |   --self-test
#
# Real origin tags are judged only at release (cut_tag in autopilot.sh), and the newest final tag
# is fixed until the next release, so a nightly that judged it would print the same verdict every
# night. This lane rehearses release day instead, on the commit under test (default HEAD):
#   1. a shared clone of <repo> in a scratch directory, detached at <rev>, with NO remote: nothing
#      here can push, and no ref of <repo> is written;
#   2. the release bump procedure of THAT commit, verbatim: bash scripts/bump-version.sh <V>;
#   3. one commit of what it changed, and an annotated tag v<V> on it, in the clone only;
#   4. tag_on_bump.sh <V> judges that tag.
# <V> is the train version: the lowest open X.Y.Z milestone above the newest final vX.Y.Z tag, as
# release-gates-nightly derives it. The caller passes it; this script reads no milestone.
#
# Exit: the judge's — 0 PASS, 1 REFUSE (e.g. a bump procedure that does not move the root version),
# 2 NOT_MEASURED. Every step that cannot run (no V, V already the version of <rev>, clone, bump or
# commit failed) is 2, never a pass (L25). Report-only until three green nights (L31).
set -uo pipefail
PROG=tag_on_bump_sandbox
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JUDGE=${TOB_JUDGE:-$HERE/tag_on_bump.sh}

nm() { echo "NOT_MEASURED $PROG: $*"; echo "bump=none between=NA"; return 2; }

sandbox() {   # sandbox <root> <V> <rev>
    local root=$1 v=$2 rev=$3 h hv tmp c rc
    [[ "$v" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { nm "'$v' is not a train version X.Y.Z"; return; }
    [ -f "$JUDGE" ] || { nm "the judge $JUDGE is not there"; return; }
    h=$(git -C "$root" rev-parse -q --verify "$rev^{commit}" 2>/dev/null) || { nm "$rev does not resolve to a commit"; return; }
    hv=$(git -C "$root" show "$h:Cargo.toml" 2>/dev/null | awk -F'"' '/^\[workspace\.package\]/{w=1;next} /^\[/{w=0} w&&/^version[ \t]*=/{print $2; exit}')
    [ -n "$hv" ] || { nm "no [workspace.package] version at ${h:0:12}"; return; }
    [ "$hv" != "$v" ] || { nm "${h:0:12} already carries $v: there is nothing to bump"; return; }
    tmp=$(mktemp -d) || { nm "no scratch directory"; return; }
    c="$tmp/r"
    if ! { git clone -q --shared --no-checkout "$root" "$c" 2>/dev/null && git -C "$c" remote remove origin \
            && git -C "$c" -c advice.detachedHead=false checkout -q --detach "$h" \
            && git -C "$c" config core.hooksPath /dev/null && git -C "$c" config commit.gpgsign false \
            && git -C "$c" config tag.gpgsign false && git -C "$c" config user.name "sandbox bump" \
            && git -C "$c" config user.email sandbox@example.invalid; }; then
        rm -rf "${tmp:?}"; nm "the sandbox clone of ${h:0:12} could not be made"; return
    fi
    if ! (cd "$c" && bash scripts/bump-version.sh "$v") > "$tmp/bump.log" 2>&1; then
        tail -n 5 "$tmp/bump.log" | sed 's/^/  bump: /'
        rm -rf "${tmp:?}"; nm "the bump procedure of ${h:0:12} failed for $v"; return
    fi
    if ! { git -C "$c" add -A && git -C "$c" commit -qm "chore(release): v$v (sandbox bump, never pushed)" \
            && git -C "$c" tag -a "v$v" -m "v$v (sandbox)"; }; then
        rm -rf "${tmp:?}"; nm "the sandbox bump commit or tag could not be made (did the bump change nothing?)"; return
    fi
    rc=0
    bash "$JUDGE" "$v" --root "$c" || rc=$?
    echo "sandbox: ${h:0:12} at $hv, bumped to $v by scripts/bump-version.sh, tagged locally"
    rm -rf "${tmp:?}"
    return "$rc"
}

main() {
    local v="" root=. rev=HEAD
    v=${1:-}; [ "$#" -gt 0 ] && shift
    while [ "$#" -gt 0 ]; do
        case $1 in
            --root) root=${2:?--root needs a path}; shift 2 ;;
            --rev) rev=${2:?--rev needs a rev}; shift 2 ;;
            *) echo "$PROG: usage: <V> [--root <repo>] [--rev <rev>] | --self-test" >&2; return 2 ;;
        esac
    done
    [ -n "$v" ] || { echo "$PROG: usage: <V> [--root <repo>] [--rev <rev>] | --self-test" >&2; return 2; }
    sandbox "$root" "$v" "$rev"
}

# ---------------------------------------------------------------------------------------------
# self-test: planted rows on scratch repositories whose scripts/bump-version.sh is a stub, then
# mutants of THIS file (each must turn a row red). Prints ok/FAIL rows and "mutants killed K/N".
mkrepo() {   # mkrepo <dir> <bump-stub-body>: root manifest at 1.2.2, a stub bump procedure
    local d=$1
    git init -q "$d" && git -C "$d" config user.email t@example.invalid && git -C "$d" config user.name t \
        && git -C "$d" config core.hooksPath /dev/null && git -C "$d" config commit.gpgsign false || return 1
    printf '[workspace]\nmembers = []\n\n[workspace.package]\nversion = "1.2.2"\n' > "$d/Cargo.toml"
    mkdir -p "$d/scripts" && printf '%s\n' "$2" > "$d/scripts/bump-version.sh"
    git -C "$d" add -A && git -C "$d" commit -qm init
}
STUB_OK='sed -i "s/^version = \".*\"/version = \"$1\"/" Cargo.toml'

self_test() {
    local subj=${1:-${BASH_SOURCE[0]}} quiet=${2:-} d fail=0 got rc
    command -v git > /dev/null || { echo "$PROG self-test: needs git"; return 2; }
    [ -f "$JUDGE" ] || { echo "$PROG self-test: no judge at $JUDGE"; return 2; }
    d=$(mktemp -d) || return 2
    run() { got=$(TOB_JUDGE="$JUDGE" bash "$subj" "$@" 2>&1); rc=$?; }
    check() {   # check <name> <want-rc> <want-substring>
        if [ "$rc" = "$2" ] && [[ "$got" == *"$3"* ]]; then [ -n "$quiet" ] || echo "  ok   $1 (rc=$rc)"
        else echo "  FAIL $1: want rc=$2 '$3', got rc=$rc: ${got//$'\n'/ | }"; fail=1; fi
    }
    [ -n "$quiet" ] || echo "$PROG self-test: planted rows"
    # 1. a bump procedure that moves the root version, tagged on its commit: PASS
    mkrepo "$d/r1" "$STUB_OK"
    run 1.2.3 --root "$d/r1"; check sandbox_bump_passes 0 "between=0"
    # 2. the source repository is untouched: no tag, HEAD unchanged, no new commit
    if [ -z "$(git -C "$d/r1" tag -l)" ] && [ "$(git -C "$d/r1" rev-list --count HEAD)" = 1 ] \
        && [ -z "$(git -C "$d/r1" status --porcelain)" ]; then rc=0; got="untouched"; else rc=1; got="source written: $(git -C "$d/r1" tag -l) $(git -C "$d/r1" log --oneline | head -2)"; fi
    check source_repo_untouched 0 "untouched"
    # 3. a bump procedure that changes a file but not the root version: the tag carries no bump, REFUSE
    mkrepo "$d/r3" 'echo touched >> notes.txt'
    run 1.2.3 --root "$d/r3"; check bump_that_misses_the_root_version_refuses 1 "no bump commit"
    # 4. a bump procedure that fails: NOT_MEASURED
    mkrepo "$d/r4" 'exit 3'
    run 1.2.3 --root "$d/r4"; check failed_bump_not_measured 2 "bump procedure"
    # 5. a bump procedure that changes nothing: no commit, NOT_MEASURED
    mkrepo "$d/r5" 'true'
    run 1.2.3 --root "$d/r5"; check empty_bump_not_measured 2 "could not be made"
    # 6. no train version, or the version the commit already carries: NOT_MEASURED
    run 1.2 --root "$d/r1"; check bad_train_version_not_measured 2 "not a train version"
    run 1.2.2 --root "$d/r1"; check train_equals_current_not_measured 2 "nothing to bump"
    # 7. an unresolvable rev: NOT_MEASURED
    run 1.2.3 --root "$d/r1" --rev nope; check unresolvable_rev_not_measured 2 "does not resolve"

    if [ -z "$quiet" ]; then
        local src a1 b1 k=0 n=0 name row
        # row = name|anchor|replacement: the anchor must hold no `|` (read splits on it); the
        # replacement may, since read hands it the rest of the line
        src=$(cat -- "$subj"; printf x); src=${src%x}; src=${src%%$'\n'"# ---------"*}
        for row in 'failed bump read as done|if ! (cd "$c" && bash scripts/bump-version.sh "$v")|if (cd "$c" && bash scripts/bump-version.sh "$v") || true; then :; fi; if false' \
                   'tag in the source repo|git -C "$c" tag -a "v$v"|git -C "$root" tag -a "v$v"' \
                   'judge rc dropped|return "$rc"|return 0' \
                   'current version accepted|[ "$hv" != "$v" ]|true' \
                   'empty bump committed|git -C "$c" commit -qm|git -C "$c" commit --allow-empty -qm'; do
            IFS='|' read -r name a1 b1 <<< "$row"; n=$((n + 1))
            case $src in *"$a1"*) ;; *) echo "  FAIL mutant $name: anchor moved, re-anchor it"; fail=1; continue ;; esac
            { printf '%s' "${src/"$a1"/"$b1"}"; printf '\nmain "$@"\n'; } > "$d/mut.sh"
            if got=$(self_test "$d/mut.sh" quiet 2>&1); then echo "  FAIL mutant ($name) survived the table"; fail=1
            else got=${got#*FAIL }; echo "  ok   mutant ($name) killed by row ${got%%:*}"; k=$((k + 1)); fi
        done
        echo "$PROG self-test: mutants killed $k/$n"
    fi
    rm -rf "${d:?}"
    if [ "$fail" -eq 0 ]; then [ -n "$quiet" ] || echo "$PROG self-test: PASS"; return 0; fi
    [ -n "$quiet" ] || echo "$PROG self-test: FAIL"; return 1
}

if [ "${1:-}" = "--self-test" ]; then self_test; exit $?; fi
main "$@"
