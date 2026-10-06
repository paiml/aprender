#!/usr/bin/env bash
# check_dogfood_reads_ci.sh -- in --phase pre-publish, dogfood READS the commit's CI for fmt,
# clippy, test and coverage, and no green receipt for that exact sha is RED (#4672, C316 2c).
#
#   bash scripts/check_dogfood_reads_ci.sh              # R1-R3 over scripts/dogfood.sh
#   bash scripts/check_dogfood_reads_ci.sh --self-test  # planted runner defects must go RED
#
#   R1  scripts/release/sha_checks.sh --self-test is green (its planted rows stay red);
#   R2  STRUCTURE: the four rows are `ci_mark` calls inside the pre-publish branch, and their
#       measuring commands (`gate fmt|clippy|test`, `coverage-check`) sit only in its else;
#   R3  BEHAVIOUR: the runner's own CI-read block and ci_mark(), lifted and driven against
#       planted check rows for a real commit, give PASS only for a green check on THAT sha,
#       and FAIL for a red check, a missing check, a green check on another sha, a failed
#       read, a missing reader and a dirty tree.
# EXIT 0 all hold · 1 a finding · 2 the guard cannot judge (not a pass).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
SRC="$ROOT/scripts/dogfood.sh"
SHA_CHECKS="$ROOT/scripts/release/sha_checks.sh"

# r2 <runner> -> 0 when the row sites are wired as stated
r2() {
    awk '
        /^if \[ "\$DOGFOOD_PHASE" = pre-publish \]; then$/ { inpp = 1; inelse = 0; next }
        inpp && /^else$/ { inelse = 1; next }
        inpp && /^fi( |$)/ { inpp = 0; inelse = 0; next }
        /^[[:space:]]*ci_mark fmt +"\$CI_SPEC_GATE"$/      { if (inpp && !inelse) f++ }
        /^[[:space:]]*ci_mark clippy +"\$CI_SPEC_GATE"$/   { if (inpp && !inelse) c++ }
        /^[[:space:]]*ci_mark test +"\$CI_SPEC_TEST"$/     { if (inpp && !inelse) t++ }
        /^[[:space:]]*ci_mark coverage +"\$CI_SPEC_COV"$/  { if (inpp && !inelse) v++ }
        /^[[:space:]]*gate (fmt|clippy|test) / || /make .*coverage-check/ {
            if (!inelse) { printf "FAIL  R2 line %d measures outside the pre-publish else: %s\n", NR, $0; bad = 1 } }
        END {
            if (f != 1 || c != 1 || t != 1 || v != 1) {
                printf "FAIL  R2 ci_mark rows in the pre-publish branch: fmt=%d clippy=%d test=%d coverage=%d (want 1 each)\n", f, c, t, v; bad = 1 }
            exit bad }' "$1"
}

# r3 <runner> -> 0 when the lifted read block decides every planted case right
r3() {
    local src=$1 d blk sha other fail=0
    d=$(mktemp -d) || return 2
    blk="$d/block.sh"
    awk '/^# ── C316 item 2c/ { on = 1 } on { print } on && /^}$/ { exit }' "$src" > "$blk"
    grep -q '^ci_mark()' "$blk" || { echo "ENV   R3 no C316 read block ending in ci_mark() in $src"; rm -rf "${d:?}"; return 2; }
    git -C "$d" init -q repo && git -C "$d/repo" -c user.name=t -c user.email=t@t commit -q --allow-empty -m one > /dev/null 2>&1 \
        || { echo "ENV   R3 cannot make a scratch commit"; rm -rf "${d:?}"; return 2; }
    printf 'x\n' > "$d/repo/f"; git -C "$d/repo" add f
    git -C "$d/repo" -c user.name=t -c user.email=t@t commit -q -m two > /dev/null 2>&1
    sha=$(git -C "$d/repo" rev-parse HEAD); other=$(git -C "$d/repo" rev-parse HEAD~1)
    # case <name> <want: PASS|FAIL> <dirty 0|1> <gh> <rows...>  (rows "sha|wf|check|status|concl")
    case_() {
        local name=$1 want=$2 dirty=$3 gh=$4 got; shift 4
        printf '%s\n' "$@" | awk -F'|' 'NF { print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5 "\t2026-10-06T10:00:00Z" }' > "$d/rows.tsv"
        if [ "$dirty" = 1 ]; then printf 'y\n' > "$d/repo/f"; else git -C "$d/repo" checkout -q -- f; fi
        local tsv=""; [ "$gh" = gh ] && tsv="$d/rows.tsv"
        got=$(cd "$d/repo" && GH="$gh" SHA_CHECKS_TSV="$tsv" DOGFOOD_PHASE=pre-publish REPO_ROOT="$d/repo" \
            SKILL_DIR="$([ "$gh" = noreader ] && echo "$d" || echo "$ROOT/scripts")" WORKLOG="$d" \
            bash -c 'mark() { printf "%s=%s\n" "$1" "$2"; }; . "$1"; ci_mark fmt "$CI_SPEC_GATE"; ci_mark clippy "$CI_SPEC_GATE"; ci_mark test "$CI_SPEC_TEST"; ci_mark coverage "$CI_SPEC_COV"' _ "$blk" 2>&1 | tr '\n' ' ')
        local exp="fmt=$want clippy=$want test=$want coverage=$want "
        if [ "$got" = "$exp" ]; then printf 'ok    R3 %-28s %s\n' "$name" "$got"
        else printf 'FAIL  R3 %-28s got [%s] want [%s]\n' "$name" "$got" "$exp"; fail=1; fi
    }
    local G="CI|ci / gate|COMPLETED" T="CI|workspace-test|COMPLETED" C="Coverage Nightly|coverage|COMPLETED"
    case_ green_control             PASS 0 gh "$sha|$G|SUCCESS" "$sha|$T|SUCCESS" "$sha|$C|SUCCESS"
    case_ red_check_for_the_sha     FAIL 0 gh "$sha|$G|FAILURE" "$sha|$T|FAILURE" "$sha|$C|FAILURE"
    case_ missing_check             FAIL 0 gh "$sha|CI|gpu-touched|COMPLETED|SUCCESS"
    case_ green_for_a_different_sha FAIL 0 gh "$other|$G|SUCCESS" "$other|$T|SUCCESS" "$other|$C|SUCCESS"
    case_ dirty_tree                FAIL 1 gh "$sha|$G|SUCCESS" "$sha|$T|SUCCESS" "$sha|$C|SUCCESS"
    case_ github_read_failed        FAIL 0 false
    case_ the_reader_is_missing     FAIL 0 noreader
    rm -rf "${d:?}"
    return "$fail"
}

judge() { # judge <runner> -> 0 green, 1 finding, 2 env
    local src=$1 rc=0 r
    [ -f "$src" ] || { echo "ENV   no runner at $src"; return 2; }
    if bash "$SHA_CHECKS" --self-test > /dev/null 2>&1; then echo "ok    R1 sha_checks.sh --self-test green"
    else echo "FAIL  R1 sha_checks.sh --self-test is red"; rc=1; fi
    if r2 "$src"; then echo "ok    R2 fmt/clippy/test/coverage are ci_mark rows in pre-publish; measuring only in its else"
    else rc=1; fi
    r3 "$src"; r=$?
    [ "$r" -eq 2 ] && return 2
    [ "$r" -ne 0 ] && rc=1
    return "$rc"
}

self_test() {
    local d fail=0
    d=$(mktemp -d) || return 2
    judge "$SRC" > "$d/out" 2>&1 || { echo "FAIL  pristine runner is not green:"; cat "$d/out"; rm -rf "${d:?}"; return 1; }
    echo "ok    pristine runner green"
    planted() { # planted <name> <why> <sed expr>
        sed -e "$3" "$SRC" > "$d/$1.sh"
        if cmp -s "$SRC" "$d/$1.sh"; then echo "FAIL  plant $1 changed nothing"; fail=1; return; fi
        if judge "$d/$1.sh" > "$d/$1.out" 2>&1; then echo "FAIL  plant $1 stayed green: $2"; fail=1
        else echo "ok    plant $1 RED: $2"; fi
    }
    planted rerun     "pre-publish re-runs fmt instead of reading CI" 's/^  ci_mark fmt    "\$CI_SPEC_GATE"$/  gate fmt true/'
    planted nodirty   "a dirty tree still reads HEAD's CI"            's/--untracked-files=no 2>\/dev\/null)" \]; then$/--untracked-files=no 2>\/dev\/null)" ] \&\& false; then/'
    planted skipread  "a failed read becomes a pass"                  's/\*)       mark "\$1" FAIL "no receipt/*)       mark "$1" PASS "no receipt/'
    planted wrongspec "test reads ci / gate instead of workspace-test" 's/^  ci_mark test     "\$CI_SPEC_TEST"$/  ci_mark test     "$CI_SPEC_GATE"/'
    rm -rf "${d:?}"
    [ "$fail" -eq 0 ] && { echo "check_dogfood_reads_ci self-test: PASS"; return 0; }
    echo "check_dogfood_reads_ci self-test: FAIL"; return 1
}

case "${1:-}" in
    --self-test) self_test; exit $? ;;
    "") judge "$SRC"; rc=$?; [ "$rc" -eq 0 ] && echo PASS; exit "$rc" ;;
    *) sed -n '2,6p' "$0" >&2; exit 2 ;;
esac
