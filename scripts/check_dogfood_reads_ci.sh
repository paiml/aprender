#!/usr/bin/env bash
# check_dogfood_reads_ci.sh -- in --phase pre-publish, dogfood READS the commit's CI for fmt and
# test and its coverage receipt, and no green receipt for that exact sha is RED (#4672, C316 2c).
#
#   bash scripts/check_dogfood_reads_ci.sh              # R1-R3 over scripts/dogfood.sh
#   bash scripts/check_dogfood_reads_ci.sh --self-test  # planted runner defects must go RED
#
#   R1  sha_checks.sh and tag_coverage_gate.sh --self-test are green (their planted rows stay red);
#   R2  STRUCTURE: fmt and test are `ci_mark` calls and coverage is `cov_mark` (tag_coverage_gate.sh
#       --resolve, the receipt cut_tag requires) inside the pre-publish branch, and their
#       measuring commands (`gate fmt|test`, `coverage-check`) sit only in its else; clippy is
#       measured in every phase and never read from CI (its CI lint is narrower); pre-publish
#       never PASSes coverage it did not read;
#   R3  BEHAVIOUR: the runner's own CI-read block and ci_mark(), lifted and driven against
#       planted check rows for a real commit, give PASS only for a green check on THAT sha,
#       and FAIL for a red check, a missing check, a green check on another sha, a failed
#       read, a missing reader and a dirty tree.
# EXIT 0 all hold · 1 a finding · 2 the guard cannot judge (not a pass).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
SRC="$ROOT/scripts/dogfood.sh"
SHA_CHECKS="$ROOT/scripts/release/sha_checks.sh"
TCG="$ROOT/scripts/release/tag_coverage_gate.sh"

# r2 <runner> -> 0 when the row sites are wired as stated
r2() {
    awk '
        /^if \[ "\$DOGFOOD_PHASE" = pre-publish \]; then$/ { inpp = 1; inelse = 0; next }
        inpp && /^else$/ { inelse = 1; next }
        inpp && /^fi( |$)/ { inpp = 0; inelse = 0; next }
        /^[[:space:]]*ci_mark fmt +"\$CI_SPEC_GATE"$/   { if (inpp && !inelse) f++ }
        /^[[:space:]]*ci_mark test +"\$CI_SPEC_TEST"$/  { if (inpp && !inelse) t++ }
        /^[[:space:]]*cov_mark$/                        { if (inpp && !inelse) v++ }
        /^[[:space:]]*ci_mark clippy/ { printf "FAIL  R2 line %d: clippy reads CI, whose lint is narrower than the row: %s\n", NR, $0; bad = 1 }
        /^gate clippy / { if (!inpp) c++ }
        /^[[:space:]]*(ci_)?mark coverage / { if (inpp && !inelse) { printf "FAIL  R2 line %d: pre-publish records coverage without reading the receipt: %s\n", NR, $0; bad = 1 } }
        /^[[:space:]]*gate (fmt|test) / || /make .*coverage-check/ {
            if (!inelse) { printf "FAIL  R2 line %d measures outside the pre-publish else: %s\n", NR, $0; bad = 1 } }
        END {
            if (f != 1 || t != 1 || v != 1 || c != 1) {
                printf "FAIL  R2 want fmt, test and coverage read in pre-publish once each, clippy measured once in every phase: fmt=%d test=%d coverage=%d clippy=%d\n", f, t, v, c; bad = 1 }
            exit bad }' "$1"
}

# r3 <runner> -> 0 when the lifted read block decides every planted case right
r3() {
    local src=$1 d blk sha other fail=0
    d=$(mktemp -d) || return 2
    blk="$d/block.sh"
    awk '/^# ── C316 item 2c/ { on = 1 } on { print } on && /^}$/ && ++k == 2 { exit }' "$src" > "$blk"
    { grep -q '^ci_mark()' "$blk" && grep -q '^cov_mark()' "$blk"; } \
        || { echo "ENV   R3 no C316 read block with ci_mark() and cov_mark() in $src"; rm -rf "${d:?}"; return 2; }
    git -C "$d" init -q repo && git -C "$d/repo" -c user.name=t -c user.email=t@t commit -q --allow-empty -m one > /dev/null 2>&1 \
        || { echo "ENV   R3 cannot make a scratch commit"; rm -rf "${d:?}"; return 2; }
    printf 'x\n' > "$d/repo/f"; git -C "$d/repo" add f
    git -C "$d/repo" -c user.name=t -c user.email=t@t commit -q -m two > /dev/null 2>&1
    sha=$(git -C "$d/repo" rev-parse HEAD); other=$(git -C "$d/repo" rev-parse HEAD~1)
    # The real sha_checks.sh, and a coverage reader that answers rc $TCG_RC for --resolve $TCG_SHA
    # only (the real tag_coverage_gate.sh's own reds are R1's).
    mkdir -p "$d/skill/release" && cp "$SHA_CHECKS" "$d/skill/release/sha_checks.sh"
    printf '%s\n' '[ "$1" = --resolve ] && [ "$2" = "$TCG_SHA" ] || { echo "no receipt for $2"; exit 1; }' \
        'echo "receipt rc $TCG_RC"; exit "$TCG_RC"' > "$d/skill/release/tag_coverage_gate.sh"
    # case <name> <want fmt+test> <want coverage> <dirty 0|1> <gh> <cov rc> <cov sha> <rows...>
    case_() {
        local name=$1 want=$2 wcov=$3 dirty=$4 gh=$5 crc=$6 csha=$7 got; shift 7
        printf '%s\n' "$@" | awk -F'|' 'NF { print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5 "\t2026-10-06T10:00:00Z" }' > "$d/rows.tsv"
        if [ "$dirty" = 1 ]; then printf 'y\n' > "$d/repo/f"; else git -C "$d/repo" checkout -q -- f; fi
        local tsv=""; [ "$gh" = gh ] && tsv="$d/rows.tsv"
        got=$(cd "$d/repo" && GH="$gh" SHA_CHECKS_TSV="$tsv" DOGFOOD_PHASE=pre-publish REPO_ROOT="$d/repo" \
            TCG_RC="$crc" TCG_SHA="$csha" SKILL_DIR="$([ "$gh" = noreader ] && echo "$d" || echo "$d/skill")" WORKLOG="$d" \
            bash -c 'mark() { printf "%s=%s\n" "$1" "$2"; }; . "$1"; ci_mark fmt "$CI_SPEC_GATE"; ci_mark test "$CI_SPEC_TEST"; cov_mark' _ "$blk" 2>&1 | tr '\n' ' ')
        local exp="fmt=$want test=$want coverage=$wcov "
        if [ "$got" = "$exp" ]; then printf 'ok    R3 %-30s %s\n' "$name" "$got"
        else printf 'FAIL  R3 %-30s got [%s] want [%s]\n' "$name" "$got" "$exp"; fail=1; fi
    }
    local G="CI|ci / gate|COMPLETED" T="CI|workspace-test|COMPLETED" GG GT
    GG="$sha|$G|SUCCESS"; GT="$sha|$T|SUCCESS"
    case_ green_control                PASS PASS 0 gh 0 "$sha" "$GG" "$GT"
    case_ red_check_for_the_sha        FAIL PASS 0 gh 0 "$sha" "$sha|$G|FAILURE" "$sha|$T|FAILURE"
    case_ missing_check                FAIL PASS 0 gh 0 "$sha" "$sha|CI|gpu-touched|COMPLETED|SUCCESS"
    case_ green_for_a_different_sha    FAIL PASS 0 gh 0 "$sha" "$other|$G|SUCCESS" "$other|$T|SUCCESS"
    case_ coverage_gate_refused        PASS FAIL 0 gh 1 "$sha" "$GG" "$GT"
    case_ coverage_gate_other_sha     PASS FAIL 0 gh 0 "$other" "$GG" "$GT"
    case_ dirty_tree                   FAIL FAIL 1 gh 0 "$sha" "$GG" "$GT"
    case_ github_read_failed           FAIL PASS 0 false 0 "$sha"
    case_ the_readers_are_missing      FAIL FAIL 0 noreader 0 "$sha"
    rm -rf "${d:?}"
    return "$fail"
}

judge() { # judge <runner> -> 0 green, 1 finding, 2 env
    local src=$1 rc=0 r
    [ -f "$src" ] || { echo "ENV   no runner at $src"; return 2; }
    if bash "$SHA_CHECKS" --self-test > /dev/null 2>&1; then echo "ok    R1 sha_checks.sh --self-test green"
    else echo "FAIL  R1 sha_checks.sh --self-test is red"; rc=1; fi
    if bash "$TCG" --self-test > /dev/null 2>&1; then echo "ok    R1 tag_coverage_gate.sh --self-test green (no receipt, other sha, below floor, gh failing: red)"
    else echo "FAIL  R1 tag_coverage_gate.sh --self-test is red"; rc=1; fi
    if r2 "$src"; then echo "ok    R2 fmt/test are ci_mark rows and coverage a cov_mark row in pre-publish, measured only in its else; clippy measured in every phase"
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
    planted clippyread "clippy reads ci / gate, a narrower lint"      's/^gate clippy .*/ci_mark clippy "$CI_SPEC_GATE"/'
    planted covpass   "pre-publish passes coverage it never read"     's/^  cov_mark$/  mark coverage PASS later/'
    planted covmissing "pre-publish drops the coverage row"          '/^  cov_mark$/d'
    planted covskip   "a refused coverage receipt becomes a pass"   's/^  else mark coverage FAIL "no coverage receipt/  else mark coverage PASS "no coverage receipt/'
    planted covwrongsha "coverage resolves a receipt for another sha" 's/--resolve "\$CI_SHA"/--resolve "${CI_SHA}~1"/'
    rm -rf "${d:?}"
    [ "$fail" -eq 0 ] && { echo "check_dogfood_reads_ci self-test: PASS"; return 0; }
    echo "check_dogfood_reads_ci self-test: FAIL"; return 1
}

case "${1:-}" in
    --self-test) self_test; exit $? ;;
    "") judge "$SRC"; rc=$?; [ "$rc" -eq 0 ] && echo PASS; exit "$rc" ;;
    *) sed -n '2,6p' "$0" >&2; exit 2 ;;
esac
