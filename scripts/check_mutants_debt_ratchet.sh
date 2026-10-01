#!/usr/bin/env bash
# check_mutants_debt_ratchet.sh -- the mutants gate in RATCHET mode (#4646; operator rulings C187(a)/C188(a),
# 2026-10-01: "new gates are adopted as RATCHETS, never as an instant absolute bar").
#
# ci/mutants-debt.tsv is the committed baseline of every known surviving mutant. This guard holds it
# SHRINK-ONLY: every row on HEAD must already be a row of the merge base's copy. So:
#   - deleting a row (a survivor was killed)          -> GREEN
#   - adding a row, or swapping one row for another   -> RED (a new survivor cannot be ledgered away)
#   - a missing/unreadable file, a malformed row,
#     a duplicate row, an unresolvable merge base     -> RED (never a pass; L25)
#   - no copy at the merge base                       -> GREEN once: the baseline is being introduced
#
# Usage: bash scripts/check_mutants_debt_ratchet.sh            (compare HEAD with merge-base(HEAD, $MUTANTS_DEBT_BASE))
#        bash scripts/check_mutants_debt_ratchet.sh --self-test (case table R1-R12 in a scratch repo)
# MUTANTS_DEBT_BASE unset = resolve_base (merge-base with origin/main, or the CI-shape fallbacks); set = merge-base(HEAD, it). MUTANTS_DEBT_FILE defaults to ci/mutants-debt.tsv.
set -euo pipefail

FILE=${MUTANTS_DEBT_FILE:-ci/mutants-debt.tsv}
BASE=${MUTANTS_DEBT_BASE:-}
# Default base = scripts/lib/resolve_base.sh, the base every differential guard uses on every CI checkout shape
# (depth-1 pull_request merge commit, merge_group squash head, push to main). It refuses rather than judge HEAD vs HEAD.
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd); PROG=check_mutants_debt_ratchet
# shellcheck source=scripts/lib/resolve_base.sh
. "$(dirname -- "${BASH_SOURCE[0]}")/lib/resolve_base.sh" || exit 1

rows() { grep -v '^#' | grep -v '^[[:space:]]*$' || true; }

check_rows() { # stdin = rows; prints each malformed row, returns 1 if any
    awk -F'\t' '
        NF != 5 { print "  malformed (need 5 tab fields): " $0; bad = 1; next }
        $3 != "MissedMutant" && $3 != "Timeout" { print "  bad outcome \"" $3 "\": " $0; bad = 1; next }
        $4 !~ /^[0-9a-f]{40}$/ { print "  bad sha \"" $4 "\": " $0; bad = 1; next }
        $5 !~ /^[0-9]+$/ { print "  bad run id \"" $5 "\": " $0; bad = 1; next }
        index($2, $1) != 1 { print "  mutant does not name its file: " $0; bad = 1; next }
        END { exit bad }'
}

run() {
    [ -r "$FILE" ] || { echo "FAIL  $FILE missing or unreadable -- the ratchet has no baseline (RED, never a pass)"; return 1; }
    local head base mb bad dup added
    head=$(rows < "$FILE")
    if ! bad=$(check_rows <<<"$head"); then echo "FAIL  $FILE has malformed rows:"; echo "$bad"; return 1; fi
    dup=$(sort <<<"$head" | uniq -d)
    [ -z "$dup" ] || { echo "FAIL  $FILE has duplicate rows:"; sed 's/^/  /' <<<"$dup"; return 1; }
    if [ -n "$BASE" ]; then
        mb=$(git merge-base HEAD "$BASE" 2>/dev/null) || { echo "FAIL  merge-base(HEAD, $BASE) unresolvable -- NOT_MEASURED is not a pass"; return 1; }
    else
        resolve_base HEAD 2>&1 || { echo "FAIL  base unresolvable (resolve_base) -- NOT_MEASURED is not a pass"; return 1; }
        mb=$BASE_REF; echo "base: ${mb:0:10} ($BASE_HOW)"
    fi
    if ! git cat-file -e "$mb:$FILE" 2>/dev/null; then
        echo "PASS  $FILE introduced: $(grep -c . <<<"$head") row(s), no copy at merge base ${mb:0:10}"
        return 0
    fi
    base=$(git show "$mb:$FILE" | rows)
    added=$(comm -23 <(sort <<<"$head") <(sort <<<"$base"))
    if [ -n "$added" ]; then
        echo "FAIL  $FILE is shrink-only; $(grep -c . <<<"$added") row(s) not in merge base ${mb:0:10}:"
        sed 's/^/  + /' <<<"$added"
        echo "  A new survivor is fixed by a test that kills it, never by adding it to the debt file."
        return 1
    fi
    echo "PASS  $FILE shrink-only: $(grep -c . <<<"$base") -> $(grep -c . <<<"$head") row(s) vs merge base ${mb:0:10}"
}

self_test() {
    local tmp me fails=0
    me=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
    tmp=$(mktemp -d)
    case "$tmp" in /tmp/*|"${TMPDIR:-/tmp}"/*) ;; *) echo "SELF-TEST: refusing scratch dir '$tmp'"; return 1;; esac
    trap 'rm -rf -- "${tmp:?}"' RETURN
    local sha=0123456789abcdef0123456789abcdef01234567
    local r1="src/a.rs	src/a.rs:1:1: replace + with -	MissedMutant	$sha	1"
    local r2="src/b.rs	src/b.rs:2:2: replace > with <	Timeout	$sha	2"
    local r3="src/c.rs	src/c.rs:3:3: replace && with ||	MissedMutant	$sha	3"
    git -C "$tmp" init -q -b main
    git -C "$tmp" config user.email t@t; git -C "$tmp" config user.name t; git -C "$tmp" config core.hooksPath /dev/null
    mkdir -p "$tmp/ci"
    printf '# header\n%s\n%s\n' "$r1" "$r2" > "$tmp/ci/mutants-debt.tsv"
    git -C "$tmp" add -A; git -C "$tmp" commit -qm base
    git -C "$tmp" branch base
    # case <id> <want rc> <want output substring> <description> <file content|__MISSING__> [base ref]
    # The substring pins WHICH rule decided: a RED for the wrong reason is a FAIL (each rule is mutation-tested).
    case_row() {
        local id=$1 want=$2 pat=$3 what=$4 content=$5 b=${6:-base} rc=0 out
        git -C "$tmp" checkout -q -B "c$id" base
        if [ "$content" = __MISSING__ ]; then git -C "$tmp" rm -q ci/mutants-debt.tsv
        else printf '%s' "$content" > "$tmp/ci/mutants-debt.tsv"; git -C "$tmp" add -A; fi
        git -C "$tmp" commit -qm "case $id" --allow-empty
        out=$(cd "$tmp" && MUTANTS_DEBT_BASE=$b bash "$me" 2>&1) || rc=$?
        if [ "$rc" -eq "$want" ] && grep -qF -- "$pat" <<<"$out"; then echo "ok    $id rc=$rc  $what"
        else echo "FAIL  $id rc=$rc want=$want '$pat'  $what"; fails=$((fails + 1)); fi
    }
    case_row R1 1 "row(s) not in merge base" "a grown debt file is RED"           "$(printf '%s\n%s\n%s\n' "$r1" "$r2" "$r3")"$'\n'
    case_row R2 1 "missing or unreadable"    "a missing debt file is RED, never a pass" __MISSING__
    case_row R3 0 "shrink-only: 2 -> 1"      "a shrink is GREEN"                   "$r1"$'\n'
    case_row R4 1 "need 5 tab fields"        "a 4-field row is RED"                "${r1%	*}"$'\n'
    case_row R5 1 "row(s) not in merge base" "a swap (same count, new row) is RED" "$(printf '%s\n%s\n' "$r1" "$r3")"$'\n'
    case_row R6 0 "shrink-only: 2 -> 2"      "an unchanged file is GREEN"          "$(printf '# header\n%s\n%s\n' "$r1" "$r2")"$'\n'
    case_row R7 1 "duplicate rows"           "a duplicate row is RED"              "$(printf '%s\n%s\n' "$r1" "$r1")"$'\n'
    case_row R8 1 "unresolvable"             "an unresolvable merge base is RED"   "$r1"$'\n' no-such-ref
    case_row R9 1 "bad outcome"              "a Caught outcome is not debt (RED)"  "${r1/MissedMutant/CaughtMutant}"$'\n'
    # R10: introduction -- the merge base has no copy
    git -C "$tmp" checkout -q --orphan intro; git -C "$tmp" rm -rq --cached . ; rm -rf -- "${tmp:?}/ci"
    : > "$tmp/keep"; git -C "$tmp" add keep; git -C "$tmp" commit -qm intro-base; git -C "$tmp" branch intro-base
    case_row_intro() {
        local rc=0
        mkdir -p "$tmp/ci"; printf '%s\n' "$r1" > "$tmp/ci/mutants-debt.tsv"; git -C "$tmp" add -A; git -C "$tmp" commit -qm intro
        (cd "$tmp" && MUTANTS_DEBT_BASE=intro-base bash "$me" >/dev/null 2>&1) || rc=$?
        if [ "$rc" -eq 0 ]; then echo "ok    R10 rc=0  the first introduction is GREEN"; else echo "FAIL  R10 rc=$rc want=0  introduction"; fails=$((fails + 1)); fi
    }
    case_row_intro
    # R11: no explicit base and no origin/main (a branch commit with no nameable base) -> resolve_base refuses -> RED
    local rc=0 out
    git -C "$tmp" checkout -q -B c11 base; printf '%s\n' "$r1" > "$tmp/ci/mutants-debt.tsv"; git -C "$tmp" add -A; git -C "$tmp" commit -qm c11
    out=$(cd "$tmp" && env -u MUTANTS_DEBT_BASE bash "$me" 2>&1) || rc=$?
    if [ "$rc" -eq 1 ] && grep -qF "base unresolvable (resolve_base)" <<<"$out"; then echo "ok    R11 rc=1  no nameable base is RED (resolve_base refuses)"
    else echo "FAIL  R11 rc=$rc want=1  no nameable base"; fails=$((fails + 1)); fi
    # R12: default base via origin/main: a grown file vs origin/main is RED
    rc=0; git -C "$tmp" update-ref refs/remotes/origin/main base
    git -C "$tmp" checkout -q -B c12 base; printf '%s\n%s\n%s\n' "$r1" "$r2" "$r3" > "$tmp/ci/mutants-debt.tsv"; git -C "$tmp" add -A; git -C "$tmp" commit -qm c12
    out=$(cd "$tmp" && env -u MUTANTS_DEBT_BASE bash "$me" 2>&1) || rc=$?
    if [ "$rc" -eq 1 ] && grep -qF "row(s) not in merge base" <<<"$out"; then echo "ok    R12 rc=1  default base (origin/main via resolve_base): grown file is RED"
    else echo "FAIL  R12 rc=$rc want=1  default base grown"; fails=$((fails + 1)); fi
    [ "$fails" -eq 0 ] || { echo "SELF-TEST FAILED: $fails case(s)"; return 1; }
    echo "SELF-TEST PASSED (12 cases)"
}

case "${1:-}" in
    --self-test) self_test ;;
    "") run ;;
    *) echo "usage: $0 [--self-test]" >&2; exit 2 ;;
esac
