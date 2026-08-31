#!/usr/bin/env bash
# check_pr_review_counts.sh - the counts PR-REVIEW-SKILL-002's own files state as
# MEASURED must equal what the tree derives.
#
# WHY
# ---
# The skill reviewed its own PR and found five counts stated as measured that the very
# commit shipping them falsified:
#
#   * the S3.B trigger "fires on five paths"                 -> 8
#   * the guard's mutation set "119/119", in FOUR places     -> 185 (the PR TITLE said 185)
#   * "this file is 65 tests"                                -> 121
#   * "the 22-row fixture table", twice                      -> 26
#   * "Not yet done: the backtest", in the commit landing results-v3.md at 3 of 3
#
# None of them was a lie anyone told. Each was true when written and went stale when the
# thing it counted grew, and nothing recomputed it. That is precisely the defect class
# this whole skill exists to catch - a stale count stated as measured - reproduced inside
# the artifact that defines it. So the three counts that are DERIVABLE FROM THE TREE are
# derived here, on every run, and the files that state them are checked against the
# derivation.
#
# HOW, AND WHY NOT WITH A CLEVER REGEX
# ------------------------------------
# Each row of the table below is (id, file, occurrences, template). The template is
# rendered with the DERIVED value and counted with `grep -c -F` - a fixed string, no
# regex, no window, no anchor word. The count must match EXACTLY:
#
#   too few  => the number went stale, or the sentence was deleted. Both are defects:
#              a claim that vanishes is not a claim that was checked, and a guard whose
#              universe can be emptied is a guard that passes over nothing.
#   too many => a new site states the number and the table does not know about it. The
#              table is the record of where this repository claims these numbers; a site
#              it has never seen has never been checked.
#
# The first draft of this guard used a keyword window ("any N/N within 200 characters of
# `mutate-guard`") and it was wrong twice before a single line of it ran: SKILL.md breaks
# `reports` and `119/119` across a newline, and ci.yml's "58 cgp tests" is a legitimate
# `[0-9]+ tests` two lines from the one under test. Five of this repository's
# `apr`-invocation patterns were wrong in exactly that way. A fixed string cannot be.
#
# THE FOURTH COUNT IS NOT HERE, ON PURPOSE. "The S3.B trigger fires on N paths" counts a
# DIFF, and a diff moves with every commit, so no derivation of it is stable enough to
# check a written number against. SKILL.md now pins that one to the commit it was measured
# at and prints the command that recomputes it, which is the honest form for a number that
# is a snapshot rather than an invariant.
#
# EVIDENCE FILES ARE DELIBERATELY OUT OF SCOPE. evidence/ is the append-only record of
# what a run measured WHEN IT RAN; "correcting" a past transcript to match today's tree is
# how a measurement becomes folklore.
#
#   bash scripts/check_pr_review_counts.sh              # check
#   bash scripts/check_pr_review_counts.sh --show       # print the derived values
#   bash scripts/check_pr_review_counts.sh --self-test  # both polarities, on copies
#
# EXIT: 0 every row matches; 1 any row does not; 2 the box cannot derive a value.

set -uo pipefail

PROG=${0##*/}
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

die_env() { echo "$PROG: ENV - $*" >&2; exit 2; }

# ---------------------------------------------------------------------------
# The three derivations. Each reads the TREE, never a written-down number.
# ---------------------------------------------------------------------------

# The mutation set is a DERIVATION inside mutate-guard.sh (one `drop` and one `flip` per
# `reject B<n>` site, rescanned from the guard on every run, plus the named `text` edits),
# so its size is whatever that script says today. --list runs no mutants.
derive_mutants() {
    local root=$1 err rows tally rc=0
    err=$(mktemp "${TMPDIR:-/tmp}/prcounts-list.XXXXXX") || return 1
    # TWO derivations of the same number, from the two streams --list writes:
    # the catalogue rows on stdout and its own tally on stderr. They must agree.
    # A single one of them is a number this guard would be trusting rather than
    # deriving -- and the whole point of this file is that a number nothing
    # recomputes goes stale without anybody lying.
    rows=$( (cd "$root" && bash scripts/mutate-guard.sh --list 2>"$err") \
            | awk -F'\t' '$2 == "drop" || $2 == "flip" || $2 == "text" { n += 1 } END { print n + 0 }' )
    rc=${PIPESTATUS[0]}
    tally=$(sed -n 's/^\([0-9][0-9]*\) mutants$/\1/p' "$err" | tail -1)
    rm -f -- "$err"
    [ "$rc" -eq 0 ] || return 1
    [ -n "$tally" ] || return 1
    [ "$rows" -gt 0 ] || return 1
    [ "$rows" = "$tally" ] || {
        echo "$PROG: mutate-guard.sh --list printed $rows catalogue rows but tallied $tally" >&2
        return 1
    }
    printf '%s\n' "$rows"
}

# Fixture rows: directories, not a list. tests/pr-review.bats asserts this number too;
# that assertion and this one are the same fact checked from two sides.
derive_fixture_rows() {
    local root=$1 n
    n=$(find "$root/tests/fixtures/pr-review" -maxdepth 1 -type d -name 'row-*' 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] || return 1
    printf '%s\n' "$n"
}

# bats tests: `@test` at column 0 in the file the CI step actually runs.
derive_bats_tests() {
    local root=$1 n
    [ -f "$root/tests/pr-review.bats" ] || return 1
    n=$(grep -c '^@test ' "$root/tests/pr-review.bats")
    [ "$n" -gt 0 ] || return 1
    printf '%s\n' "$n"
}

# ---------------------------------------------------------------------------
# The site table. id | file | occurrences | template (@N@ -> derived value)
# ---------------------------------------------------------------------------
SITES='
mutants|.claude/skills/pr-review/SKILL.md|2|@N@/@N@
mutants|contracts/binding.yaml|1|@N@/@N@
mutants|.github/workflows/ci.yml|1|@N@/@N@
fixture_rows|.github/workflows/ci.yml|2|@N@-row
fixture_rows|tests/pr-review.bats|1|@N@ row
fixture_rows|tests/pr-review.bats|1|-eq @N@ ]
bats_tests|.github/workflows/ci.yml|2|@N@ tests
'

# ---------------------------------------------------------------------------
# check <root> - run every row against the tree at <root>. Prints a table.
# ---------------------------------------------------------------------------
check() {
    local root=$1 quiet=${2:-} fails=0
    local -A derived=()
    local v
    v=$(derive_mutants      "$root") || die_env "cannot derive the mutation-set size (mutate-guard.sh --list)"
    derived[mutants]=$v
    v=$(derive_fixture_rows "$root") || die_env "cannot derive the fixture-row count"
    derived[fixture_rows]=$v
    v=$(derive_bats_tests   "$root") || die_env "cannot derive the bats test count"
    derived[bats_tests]=$v

    [ -n "$quiet" ] || {
        echo "=== the counts these files state as measured, against the tree ($PROG) ==="
        echo "derived:  mutants=${derived[mutants]}  fixture_rows=${derived[fixture_rows]}  bats_tests=${derived[bats_tests]}"
    }

    local row id file want tmpl needle got
    while IFS='|' read -r id file want tmpl; do
        [ -n "$id" ] || continue
        if [ ! -f "$root/$file" ]; then
            echo "FAIL  $id  $file is missing" >&2
            fails=$((fails + 1)); continue
        fi
        needle=${tmpl//@N@/${derived[$id]}}
        got=$(grep -c -F -- "$needle" "$root/$file")
        if [ "$got" -eq "$want" ]; then
            [ -n "$quiet" ] || printf 'ok    %-13s %-40s %s x%s\n' "$id" "$file" "$needle" "$got"
        else
            fails=$((fails + 1))
            printf 'FAIL  %-13s %-40s "%s" occurs %s time(s), the table says %s\n' \
                "$id" "$file" "$needle" "$got" "$want" >&2
            if [ "$got" -lt "$want" ]; then
                echo "      The tree derives ${derived[$id]}. Either the file states a" >&2
                echo "      STALE number, or the sentence that stated it was deleted." >&2
                # Show the neighbourhood using the LONGER of the template's two
                # literal halves. `${tmpl%%@N@*}` is EMPTY when the template starts
                # with @N@, and `grep -F ""` matches every line in the file - a
                # diagnostic that prints the top of the file and calls it a hint.
                local pre=${tmpl%%@N@*} post=${tmpl##*@N@} hint
                if [ ${#pre} -ge ${#post} ]; then hint=$pre; else hint=$post; fi
                if [ -n "$hint" ]; then
                    echo "      Lines carrying \"$hint\":" >&2
                    grep -nF -- "$hint" "$root/$file" 2>/dev/null | head -4 | sed 's/^/        /' >&2
                fi
            else
                echo "      A site states this number and the table has never seen it." >&2
                echo "      Add the row rather than widening the count." >&2
            fi
        fi
    done <<< "$(printf '%s\n' "$SITES" | sed '/^[[:space:]]*$/d')"

    if [ "$fails" -ne 0 ]; then
        echo "--- $fails row(s) disagree with the tree ---" >&2
        return 1
    fi
    [ -n "$quiet" ] || echo "PASS  every stated count equals the derived one"
    return 0
}

# ---------------------------------------------------------------------------
# safe_rm_scratch <path> <required-substring> - a recursive delete, guarded.
# SEC011, and not decoration: the two ways `rm -rf -- "$x"` goes wrong are an
# EMPTY x and an x that is not ours. Both are checked before the expansion, and
# a path that fails either check is left alone rather than deleted "carefully".
# ---------------------------------------------------------------------------
safe_rm_scratch() {
    local victim=${1:-} must=${2:-}
    [ -n "$victim" ] || return 0
    [ -n "$must" ]   || return 0
    [ "$victim" != "/" ] || return 0
    case "$victim" in
      *"$must"*) if [ -n "$victim" ] && [ "$victim" != "/" ]; then rm -rf -- "$victim"; fi ;;
      *) return 0 ;;
    esac
}

ST_ROOT=''
cleanup() { safe_rm_scratch "$ST_ROOT" 'prcounts-selftest.'; }
trap cleanup EXIT

# ---------------------------------------------------------------------------
# --self-test: MUTATE THE TREE, not the guard's opinion of it. Each row copies
# the repository, makes one edit, and requires the verdict to flip. A guard that
# is only ever run against a passing tree has never been shown to fail.
# ---------------------------------------------------------------------------
self_test() {
    local fails=0
    ST_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/prcounts-selftest.XXXXXX") || die_env "mktemp failed"

    local base="$ST_ROOT/tree"
    mkdir -p "$base"
    # Only the files the table names, plus what the derivations read.
    ( cd "$REPO_ROOT" && \
      tar -cf - .claude/skills/pr-review/SKILL.md contracts/binding.yaml \
                .github/workflows/ci.yml tests/pr-review.bats \
                scripts/mutate-guard.sh scripts/check_pr_review_receipt.sh \
                scripts/check_pr_review_counts.sh tests/fixtures/pr-review ) \
      | ( cd "$base" && tar -xf - ) || die_env "could not stage a copy of the tree"

    row() {
        local id=$1 want=$2 desc=$3; shift 3
        local dir="$ST_ROOT/$id" got=0
        safe_rm_scratch "$dir" 'prcounts-selftest.'
        cp -a "$base" "$dir"
        "$@" "$dir" || die_env "row $id could not apply its edit"
        # A MUTATION THAT MATCHED NOTHING IS A BROKEN HARNESS, NOT A DEAD MUTANT.
        # Every edit below is a `sed` against an anchor, and an anchor that has
        # drifted changes nothing while the row still reports a verdict: the
        # unmutated tree passes, the row wanted RED, and the table would read
        # "the guard failed to catch it" when in fact nothing was ever done to
        # it. mutate-guard.sh learned this the same way (its note 1: a probe
        # once reported "34 passed" over a file it had not touched). So the tree
        # must actually differ, and a row that did not change it stops the run
        # with a distinct diagnosis instead of a misattributed failure.
        if [ "$id" != baseline ] && diff -rq "$base" "$dir" >/dev/null 2>&1; then
            die_env "HARNESS-BROKEN: row $id's edit changed nothing (its anchor has drifted)"
        fi
        check "$dir" quiet >/dev/null 2>&1 || got=$?
        if [ "$got" -eq "$want" ]; then
            printf 'ok    %-26s rc=%s  %s\n' "$id" "$got" "$desc"
        else
            printf 'FAIL  %-26s rc=%s (wanted %s)  %s\n' "$id" "$got" "$want" "$desc"
            fails=$((fails + 1))
        fi
    }

    noop()          { :; }
    stale_mutants() { sed -i 's#185/185#119/119#' "$1/.claude/skills/pr-review/SKILL.md"; }
    stale_rows()    { sed -i 's#26-row#22-row#'   "$1/.github/workflows/ci.yml"; }
    stale_tests()   { sed -i 's#121 tests#65 tests#' "$1/.github/workflows/ci.yml"; }
    delete_claim()  { sed -i '0,\#185/185#{\#185/185#d}' "$1/contracts/binding.yaml"; }
    grow_bats()     { printf '\n@test "a new test the docs do not count" {\n  true\n}\n' >> "$1/tests/pr-review.bats"; }
    grow_rows()     { mkdir -p "$1/tests/fixtures/pr-review/row-27-invented"; }
    grow_mutants()  { sed -i 's#^  \[ -n "$head" \] || reject B1 "predicate.head_sha is absent" || return 1#&\n  [ -n "$head" ] || reject B1 "an invented rule nothing documents" || return 1#' \
                             "$1/scripts/check_pr_review_receipt.sh"; }

    echo "--- check_pr_review_counts.sh --self-test ---"
    row baseline                0 "the tree as committed"                                   noop
    row stale-mutation-score    1 "185/185 written back to 119/119 (the shipped defect)"    stale_mutants
    row stale-fixture-rows      1 "26-row written back to 22-row (the shipped defect)"      stale_rows
    row stale-bats-count        1 "121 tests written back to 65 tests (the shipped defect)" stale_tests
    row claim-deleted           1 "the sentence stating the count is deleted"               delete_claim
    row tree-grew-a-test        1 "a bats test is added and no file says so"                grow_bats
    row tree-grew-a-fixture-row 1 "a row-* fixture is added and no file says so"            grow_rows
    row tree-grew-a-mutant      1 "a reject site is added and no file says so"              grow_mutants

    if [ "$fails" -ne 0 ]; then
        echo "--- $fails row(s) did not produce the required verdict ---" >&2
        return 1
    fi
    echo "--- 8/8 rows, both polarities ---"
    return 0
}

for t in grep find sed tar diff; do
    command -v "$t" >/dev/null 2>&1 || die_env "$t is not on PATH"
done

case "${1:-}" in
  --self-test) self_test; exit $? ;;
  --show)
      printf 'mutants       %s\n' "$(derive_mutants      "$REPO_ROOT")"
      printf 'fixture_rows  %s\n' "$(derive_fixture_rows "$REPO_ROOT")"
      printf 'bats_tests    %s\n' "$(derive_bats_tests   "$REPO_ROOT")"
      exit 0 ;;
  -h|--help) sed -n '2,55p' "$0"; exit 0 ;;
  '') check "$REPO_ROOT"; exit $? ;;
  *) echo "$PROG: unknown argument: $1" >&2; exit 1 ;;
esac
